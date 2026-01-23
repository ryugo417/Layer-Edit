from diffusers import (
    AutoencoderKL,
    FluxPipeline,
)
from transformers import AutoTokenizer
import argparse
import torch
from PIL import Image
import os

# このスクリプトは、事前学習済みのモデルとLoRA重みを使って
# テキストから画像を生成し、出力ディレクトリに保存する推論コードです。
def parse_args(input_args=None):
    # コマンドライン引数の定義とパースを行う
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model",
        type=str,
        default=None,
        required=False,
        help="Path to RGBA VAE model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        nargs="+",
        required=False,
        help="Prompts for the image generation.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        required=False,
        help="Path to the file containing the prompts for the image generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Output directory for the generated images.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        required=True,
        help="Path to the LORA weights.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        required=False,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--noise_add",
        type=float,
        default=0.0,
        required=False,
        help="Add constant value to a single channel of initial noise (latents).",
    )
    parser.add_argument(
        "--noise_add_channel",
        type=int,
        default=0,
        required=False,
        help="Target channel index for --noise_add.",
    )
    # 引数をパースして返却
    return parser.parse_args(input_args)

def main(args):
    # VAEモデルを明示指定している場合はそれを読み込み、
    # 未指定の場合はベースモデル内のVAEを使用する
    if args.pretrained_vae_model is not None:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model,
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
        )

    # 推論時のdtypeをbfloat16に設定（GPUメモリ節約/高速化）
    weight_dtype = torch.bfloat16
    # 2系統のトークナイザーを読み込む（モデル仕様に合わせる）
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=True,
    )
    # FluxPipelineを構築し、VAEとトークナイザーを差し替える
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        vae=vae,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
    )
    # LoRA重みを読み込み（注意：Attention Processorとして適用）
    pipeline.load_lora_weights(args.lora_path)
    # GPUへ転送して推論準備
    pipeline.to(device='cuda', dtype=weight_dtype)
    # 再現性のためにseedを設定（未指定ならNone）
    generator = torch.Generator('cuda').manual_seed(args.seed) if args.seed is not None else None

    # プロンプトはコマンドライン/ファイルのどちらかから取得
    prompts = args.prompts
    if args.prompt_file is not None:
        with open(args.prompt_file, 'r') as f:
            prompts = f.readlines()
    if prompts is None:
        raise ValueError("Either --prompts or --prompt_file must be provided.")
        
    # 出力先ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    # 生成サイズはパイプライン既定値を使う
    height = pipeline.default_sample_size * pipeline.vae_scale_factor
    width = pipeline.default_sample_size * pipeline.vae_scale_factor
    # 各プロンプトごとに指定枚数の画像を生成して保存
    for idx, prompt in enumerate(prompts):
        for i in range(args.num_images_per_prompt):
            # 初期ノイズ（latents）を生成し、指定チャンネルに定数を加算
            num_channels_latents = pipeline.transformer.config.in_channels // 4
            latents, _ = pipeline.prepare_latents(
                batch_size=1,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=weight_dtype,
                device="cuda",
                generator=generator,
                latents=None,
            )
            if args.noise_add != 0.0:
                if not (0 <= args.noise_add_channel < num_channels_latents):
                    raise ValueError(
                        f"--noise_add_channel must be in [0, {num_channels_latents - 1}]"
                    )
                unpacked = pipeline._unpack_latents(
                    latents, height, width, pipeline.vae_scale_factor
                )
                unpacked[:, args.noise_add_channel, :, :] += args.noise_add
                latent_h, latent_w = unpacked.shape[2], unpacked.shape[3]
                latents = pipeline._pack_latents(
                    unpacked, 1, num_channels_latents, latent_h, latent_w
                )
            image = pipeline(
                prompt=prompt,
                prompt_2=prompt,
                generator=generator,
                latents=latents,
                height=height,
                width=width,
            ).images[0]
            # ファイル名はプロンプト番号とサンプル番号で決定
            if args.num_images_per_prompt == 1:
                output_path = f"{args.output_dir}/{idx:05d}.png"
            else:
                output_path = f"{args.output_dir}/{idx:05d}_prompt_{i}_sample.png"
            image.save(output_path)
            print(f"generated image is save in {output_path}")

if __name__ == "__main__":
    # エントリーポイント
    args = parse_args()
    main(args)
    
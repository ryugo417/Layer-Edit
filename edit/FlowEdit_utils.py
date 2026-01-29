from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps



def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar



def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred
    
def _shift_stream_id(img_ids: torch.Tensor, offset: int = 1) -> torch.Tensor:
    """
    FG/BG を区別したい場合に “定数っぽい列” をoffsetする（安全版）
    """
    if img_ids is None:
        return None
    x = img_ids.clone()

    # どの列が定数か自動で探す（y/xを壊さないため）
    sample = x[0] if x.ndim == 3 else x
    # (Seq, D) を想定
    std = sample.float().std(dim=0)
    col = int(torch.argmin(std).item())
    x[..., col] = x[..., col] + offset
    return x


def _concat_img_ids(ids_fg: torch.Tensor, ids_bg: torch.Tensor) -> torch.Tensor:
    """
    latents を seq方向にcatしたのと同じ軸で img_ids もcatする
    """
    if ids_fg is None or ids_bg is None:
        return None
    if ids_fg.ndim == 3:
        # (B, Seq, D)
        return torch.cat([ids_fg, ids_bg], dim=1)
    elif ids_fg.ndim == 2:
        # (Seq, D)
        return torch.cat([ids_fg, ids_bg], dim=0)
    else:
        raise ValueError(f"Unexpected img_ids shape: {ids_fg.shape}")


def calc_v_flux_dual(
    pipe,
    latents_fg, latents_bg,
    prompt_embeds, pooled_prompt_embeds,
    guidance, text_ids,
    latent_image_ids_fg, latent_image_ids_bg,
    t,
):
    latents = torch.cat([latents_fg, latents_bg], dim=1)

    # latentsと同じ「seq方向」に合わせてimg_idsを結合
    latent_image_ids = _concat_img_ids(latent_image_ids_fg, latent_image_ids_bg)

    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    fg_len = latents_fg.shape[1]
    return noise_pred[:, :fg_len], noise_pred[:, fg_len:]


@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar) 

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar



@torch.no_grad()
def FlowEditFLUX(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,):

    device = x_src.device
    latent_h, latent_w = x_src.shape[2], x_src.shape[3]
    orig_height = latent_h * pipe.vae_scale_factor
    orig_width = latent_w * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, latent_h, latent_w)
    latent_src_image_ids = pipe._prepare_latent_image_ids(
        x_src.shape[0],
        latent_h // 2,
        latent_w // 2,
        device,
        x_src.dtype,
    )
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
        )
    
    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    
    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                                    

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                
                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(pipe,
                                                    latents=zt_src,
                                                    prompt_embeds=src_prompt_embeds, 
                                                    pooled_prompt_embeds=src_pooled_prompt_embeds, 
                                                    guidance=src_guidance,
                                                    text_ids=src_text_ids, 
                                                    latent_image_ids=latent_src_image_ids, 
                                                    t=t)
                
                Vt_tar = calc_v_flux(pipe,
                                                    latents=zt_tar,
                                                    prompt_embeds=tar_prompt_embeds, 
                                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                                    guidance=tar_guidance,
                                                    text_ids=tar_text_ids, 
                                                    latent_image_ids=latent_tar_image_ids, 
                                                    t=t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            Vt_tar = calc_v_flux(pipe,
                                    latents=xt_tar,
                                    prompt_embeds=tar_prompt_embeds, 
                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                    guidance=tar_guidance,
                                    text_ids=tar_text_ids, 
                                    latent_image_ids=latent_tar_image_ids, 
                                    t=t)


            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out


@torch.no_grad()
def FlowEditFLUX_DUAL(
    pipe,
    scheduler,
    x_fg, x_bg,                    # (B,C,H,W) latent grids（同形状）
    src_prompt,
    tar_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
):
    device = x_fg.device
    latent_h, latent_w = x_fg.shape[2], x_fg.shape[3]
    orig_height = latent_h * pipe.vae_scale_factor
    orig_width  = latent_w * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    # pack
    fg_packed = pipe._pack_latents(x_fg, x_fg.shape[0], num_channels_latents, latent_h, latent_w)
    bg_packed = pipe._pack_latents(x_bg, x_bg.shape[0], num_channels_latents, latent_h, latent_w)

    def _seq_len(ids):
        return ids.shape[1] if ids.ndim == 3 else ids.shape[0]

    print("fg_packed:", fg_packed.shape, "bg_packed:", bg_packed.shape)

    # img_ids（まずFGを作り、BGは stream id をずらす）
    ids_fg = pipe._prepare_latent_image_ids(
        x_fg.shape[0],
        latent_h // 2,
        latent_w // 2,
        device,
        x_fg.dtype,
    )
    ids_bg = pipe._prepare_latent_image_ids(
        x_bg.shape[0],
        latent_h // 2,
        latent_w // 2,
        device,
        x_bg.dtype,
    )
    ids_bg = _shift_stream_id(ids_bg, offset=1)

    print("ids_fg:", ids_fg.shape, "ids_bg:", ids_bg.shape)

    assert fg_packed.shape[1] == _seq_len(ids_fg), "FG: packed seq and img_ids seq mismatch"
    assert bg_packed.shape[1] == _seq_len(ids_bg), "BG: packed seq and img_ids seq mismatch"

    ids_all = _concat_img_ids(ids_fg, ids_bg)
    assert fg_packed.shape[1] + bg_packed.shape[1] == _seq_len(ids_all), "Concat: latents seq and img_ids seq mismatch"


    # timesteps（transformer に渡す総seq長に合わせる）
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = fg_packed.shape[1] + bg_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    # prompts
    src_prompt_embeds, src_pooled, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )
    tar_prompt_embeds, tar_pooled, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(fg_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(fg_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    zt_edit_fg = fg_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if scheduler.step_index + 1 < len(scheduler.sigmas):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        # ---- ODE edit phase ----
        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(fg_packed)

            for _ in range(n_avg):
                noise_fg = torch.randn_like(fg_packed)
                noise_bg = torch.randn_like(bg_packed)

                zt_src_fg = (1 - t_i) * fg_packed + t_i * noise_fg
                zt_ctx_bg = (1 - t_i) * bg_packed + t_i * noise_bg

                zt_tar_fg = zt_edit_fg + zt_src_fg - fg_packed

                V_src_fg, _ = calc_v_flux_dual(
                    pipe,
                    latents_fg=zt_src_fg,
                    latents_bg=zt_ctx_bg,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids_fg=ids_fg,
                    latent_image_ids_bg=ids_bg,
                    t=t,
                )
                V_tar_fg, _ = calc_v_flux_dual(
                    pipe,
                    latents_fg=zt_tar_fg,
                    latents_bg=zt_ctx_bg,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids_fg=ids_fg,
                    latent_image_ids_bg=ids_bg,
                    t=t,
                )

                V_delta_avg += (V_tar_fg - V_src_fg) / n_avg

            zt_edit_fg = zt_edit_fg.to(torch.float32)
            zt_edit_fg = zt_edit_fg + (t_im1 - t_i) * V_delta_avg
            zt_edit_fg = zt_edit_fg.to(V_delta_avg.dtype)

        # ---- SDEdit phase ----
        else:
            if i == T_steps - n_min:
                noise_fg = torch.randn_like(fg_packed)
                noise_bg = torch.randn_like(bg_packed)
                xt_src_fg = scale_noise(scheduler, fg_packed, t, noise=noise_fg)
                xt_ctx_bg = scale_noise(scheduler, bg_packed, t, noise=noise_bg)
                xt_tar_fg = zt_edit_fg + xt_src_fg - fg_packed

            V_tar_fg, _ = calc_v_flux_dual(
                pipe,
                latents_fg=xt_tar_fg,
                latents_bg=xt_ctx_bg,
                prompt_embeds=tar_prompt_embeds,
                pooled_prompt_embeds=tar_pooled,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids_fg=ids_fg,
                latent_image_ids_bg=ids_bg,
                t=t,
            )

            xt_tar_fg = xt_tar_fg.to(torch.float32)
            xt_tar_fg = xt_tar_fg + (t_im1 - t_i) * V_tar_fg
            xt_tar_fg = xt_tar_fg.to(V_tar_fg.dtype)

    out_fg = zt_edit_fg if n_min == 0 else xt_tar_fg
    out_fg = pipe._unpack_latents(out_fg, orig_height, orig_width, pipe.vae_scale_factor)
    return out_fg


@torch.no_grad()
def FlowEditFLUX_DUAL_BISTREAM(
    pipe,
    scheduler,
    x_fg, x_scene,                 # (B,C,H,W)
    src_prompt,
    tar_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    fg_only_start_step: Optional[int] = None,
):
    device = x_fg.device
    latent_h, latent_w = x_fg.shape[2], x_fg.shape[3]
    orig_height = latent_h * pipe.vae_scale_factor
    orig_width  = latent_w * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    # pack
    fg_packed = pipe._pack_latents(x_fg, x_fg.shape[0], num_channels_latents, latent_h, latent_w)
    sc_packed = pipe._pack_latents(x_scene, x_scene.shape[0], num_channels_latents, latent_h, latent_w)

    # img_ids
    ids_fg = pipe._prepare_latent_image_ids(
        x_fg.shape[0],
        latent_h // 2,
        latent_w // 2,
        device,
        x_fg.dtype,
    )
    ids_sc = pipe._prepare_latent_image_ids(
        x_scene.shape[0],
        latent_h // 2,
        latent_w // 2,
        device,
        x_scene.dtype,
    )
    ids_sc = _shift_stream_id(ids_sc, offset=1)

    # timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = fg_packed.shape[1] + sc_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    # prompts
    src_prompt_embeds, src_pooled, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )
    tar_prompt_embeds, tar_pooled, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(fg_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(fg_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    zt_edit_fg = fg_packed.clone()
    zt_edit_sc = sc_packed.clone()

    xt_tar_fg = None
    xt_tar_sc = None

    for i, t in tqdm(enumerate(timesteps)):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if scheduler.step_index + 1 < len(scheduler.sigmas):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        use_fg_only = (fg_only_start_step is not None) and (i >= fg_only_start_step)

        # ---- ODE edit phase ----
        if T_steps - i > n_min:
            Vd_fg = torch.zeros_like(fg_packed)
            Vd_sc = torch.zeros_like(sc_packed)

            for _ in range(n_avg):
                noise = torch.randn_like(fg_packed)

                zt_src_fg = (1 - t_i) * fg_packed + t_i * noise
                zt_src_sc = (1 - t_i) * sc_packed + t_i * noise

                zt_tar_fg = zt_edit_fg + zt_src_fg - fg_packed
                zt_tar_sc = zt_edit_sc + zt_src_sc - sc_packed

                if use_fg_only:
                    V_src_fg = calc_v_flux(
                        pipe,
                        latents=zt_src_fg,
                        prompt_embeds=src_prompt_embeds,
                        pooled_prompt_embeds=src_pooled,
                        guidance=src_guidance,
                        text_ids=src_text_ids,
                        latent_image_ids=ids_fg,
                        t=t,
                    )
                    V_tar_fg = calc_v_flux(
                        pipe,
                        latents=zt_tar_fg,
                        prompt_embeds=tar_prompt_embeds,
                        pooled_prompt_embeds=tar_pooled,
                        guidance=tar_guidance,
                        text_ids=tar_text_ids,
                        latent_image_ids=ids_fg,
                        t=t,
                    )
                    V_src_sc = torch.zeros_like(zt_src_sc)
                    V_tar_sc = torch.zeros_like(zt_tar_sc)
                else:
                    V_src_fg, V_src_sc = calc_v_flux_dual(
                        pipe,
                        latents_fg=zt_src_fg,
                        latents_bg=zt_src_sc,
                        prompt_embeds=src_prompt_embeds,
                        pooled_prompt_embeds=src_pooled,
                        guidance=src_guidance,
                        text_ids=src_text_ids,
                        latent_image_ids_fg=ids_fg,
                        latent_image_ids_bg=ids_sc,
                        t=t,
                    )
                    V_tar_fg, V_tar_sc = calc_v_flux_dual(
                        pipe,
                        latents_fg=zt_tar_fg,
                        latents_bg=zt_tar_sc,
                        prompt_embeds=tar_prompt_embeds,
                        pooled_prompt_embeds=tar_pooled,
                        guidance=tar_guidance,
                        text_ids=tar_text_ids,
                        latent_image_ids_fg=ids_fg,
                        latent_image_ids_bg=ids_sc,
                        t=t,
                    )

                Vd_fg += (V_tar_fg - V_src_fg) / n_avg
                Vd_sc += (V_tar_sc - V_src_sc) / n_avg

            zt_edit_fg = (zt_edit_fg.to(torch.float32) + (t_im1 - t_i) * Vd_fg).to(Vd_fg.dtype)
            zt_edit_sc = (zt_edit_sc.to(torch.float32) + (t_im1 - t_i) * Vd_sc).to(Vd_sc.dtype)

        # ---- SDEdit phase ----
        else:
            if xt_tar_fg is None:
                noise = torch.randn_like(fg_packed)
                xt_src_fg = scale_noise(scheduler, fg_packed, t, noise=noise)
                xt_src_sc = scale_noise(scheduler, sc_packed, t, noise=noise)

                xt_tar_fg = zt_edit_fg + xt_src_fg - fg_packed
                xt_tar_sc = zt_edit_sc + xt_src_sc - sc_packed
            if use_fg_only:
                V_tar_fg = calc_v_flux(
                    pipe,
                    latents=xt_tar_fg,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=ids_fg,
                    t=t,
                )
                xt_tar_fg = (xt_tar_fg.to(torch.float32) + (t_im1 - t_i) * V_tar_fg).to(V_tar_fg.dtype)
            else:
                V_tar_fg, V_tar_sc = calc_v_flux_dual(
                    pipe,
                    latents_fg=xt_tar_fg,
                    latents_bg=xt_tar_sc,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids_fg=ids_fg,
                    latent_image_ids_bg=ids_sc,
                    t=t,
                )
                xt_tar_fg = (xt_tar_fg.to(torch.float32) + (t_im1 - t_i) * V_tar_fg).to(V_tar_fg.dtype)
                xt_tar_sc = (xt_tar_sc.to(torch.float32) + (t_im1 - t_i) * V_tar_sc).to(V_tar_sc.dtype)

    out_fg = zt_edit_fg if n_min == 0 else xt_tar_fg
    out_sc = zt_edit_sc if n_min == 0 else xt_tar_sc

    out_fg = pipe._unpack_latents(out_fg, orig_height, orig_width, pipe.vae_scale_factor)
    out_sc = pipe._unpack_latents(out_sc, orig_height, orig_width, pipe.vae_scale_factor)
    return out_fg, out_sc

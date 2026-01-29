#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate dataset.yaml for 3-layer scene editing from metadata.jsonl
- Reads DreamLayer-style metadata.jsonl: {file_name, text}
- Groups by group_id (prefix before "_layer_")
- For each group, generates edit tasks for layer_1 and layer_2:
    style_change: only attributes (color/material/texture/pattern/finish/lighting appearance) within same object category
    object_change: replace the object/category/shape while keeping scene consistency from layer_whole

Output:
- YAML list of entries:
  - group_id, layer (layer_1/layer_2), edit_type, source_prompt, target_prompts

Design goals:
- Cost efficient: use gpt-5-mini by default + batch multiple groups per request
- Reliable: Structured Outputs (json_schema) => always valid JSON
- Safety: forbid artist/IP/brands; if present in source, must generalize
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import yaml
from openai import OpenAI


# -----------------------------
# Pricing/Model choice guidance
# -----------------------------
DEFAULT_MODEL = "gpt-5.1"
# If you need slightly better quality, try: "gpt-5.2" (more expensive)


# -----------------------------
# JSON schema for structured outputs
# -----------------------------
SCHEMA = {
    "name": "layer_edit_dataset",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "group_id": {"type": "string"},
                        "layer": {"type": "string", "enum": ["layer_1", "layer_2"]},
                        "edit_type": {"type": "string", "enum": ["style_change", "object_change"]},
                        "source_prompt": {"type": "string"},
                        "target_prompts": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["group_id", "layer", "edit_type", "source_prompt", "target_prompts"],
                },
            }
        },
        "required": ["entries"],
    },
}


# -----------------------------
# Parsing helpers
# -----------------------------
LAYER_SUFFIXES = ["_layer_0", "_layer_1", "_layer_2", "_layer_whole"]


def parse_group_id(file_name: str) -> str | None:
    marker = "_layer_"
    if marker not in file_name:
        return None
    return file_name.split(marker, 1)[0]


def detect_layer(file_name: str) -> str | None:
    for suf in LAYER_SUFFIXES:
        if file_name.endswith(suf):
            return "layer" + suf.split("_layer_", 1)[1]
    return None


def file_to_layer_key(file_name: str) -> str | None:
    if file_name.endswith("_layer_0"):
        return "layer_0"
    if file_name.endswith("_layer_1"):
        return "layer_1"
    if file_name.endswith("_layer_2"):
        return "layer_2"
    if file_name.endswith("_layer_whole"):
        return "layer_whole"
    return None


# -----------------------------
# Text cleanup (cheap token reduction + IP/artist generalization)
# -----------------------------
_IP_PATTERNS = [
    r"\btrending on artstation\b",
    r"\bartstation\b",
    r"\bfortnite\b",
    r"\bby\s+[A-Z][\w\-]+\s+[A-Z][\w\-]+\b",   # "by First Last"
    r"\bdisney\b|\bmarvel\b|\bpixar\b|\bnintendo\b|\bsony\b|\bnetflix\b",
]
_IP_RE = re.compile("|".join(_IP_PATTERNS), flags=re.IGNORECASE)


def clean_text(s: str) -> str:
    """
    Reduce tokens and remove obvious IP/artist/platform mentions.
    This is intentionally simple (cheap) — instructions also enforce the rule.
    """
    if not s:
        return ""
    s = s.strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)

    # remove "by <artist>" fragments and platform tags
    s = _IP_RE.sub("", s)

    # remove trailing punctuation noise
    s = s.strip(" .;:,")
    return s


# -----------------------------
# Prompting
# -----------------------------
def build_instructions(style_k: int, obj_k: int) -> str:
    return (
        "You generate edit tasks for a 3-layer scene (bg=layer_0, fg1=layer_1, fg2=layer_2). "
        "Return ONLY valid JSON that matches the provided JSON schema.\n\n"
        "Safety / IP constraints:\n"
        "- Do NOT use any artist names, art platform tags (e.g., “trending on ArtStation”), or artist references.\n"
        "- Do NOT use copyrighted IP, character names, brand names, game/movie titles, or franchise-specific terms.\n"
        "- If the provided prompts contain such names, rewrite them into a generic description and proceed.\n\n"
        "Edit intent constraints (VERY IMPORTANT):\n"
        f"- For each group_id, for layer_1 and layer_2, create exactly two entries: one style_change and one object_change.\n"
        "- style_change must keep the SAME object category/identity and ONLY change appearance: "
        "color, material, texture, pattern, finish, wear/age, or lighting appearance on that object.\n"
        "  Do NOT change the object category in style_change.\n"
        "- object_change must replace the object itself (category/shape/form), e.g., cat→dog, guitar→violin, chair→stool. "
        "The replacement should be concrete and visually distinct.\n\n"
        "Scene-consistency constraint:\n"
        "- Use layer_whole as the global context. Target prompts must be plausible and not out-of-place in that scene.\n"
        "- Do NOT introduce new background elements or scene changes; only modify the specified layer content.\n\n"
        "Output constraints:\n"
        f"- For style_change, produce about {style_k} target prompts.\n"
        f"- For object_change, produce about {obj_k} target prompts.\n"
        "- Target prompts should be short and concrete (noun phrase level is OK).\n"
    )


def build_batch_input(batch: List[Dict[str, Any]]) -> str:
    """
    Keep input short: JSON lines-ish block. The model will read this.
    """
    # minimal keys to reduce tokens
    lines = []
    lines.append("DATA:")
    for g in batch:
        gid = g["group_id"]
        t0 = g.get("layer_0", "")
        t1 = g.get("layer_1", "")
        t2 = g.get("layer_2", "")
        tw = g.get("layer_whole", "")
        lines.append(
            json.dumps(
                {
                    "group_id": gid,
                    "layer_0": t0,
                    "layer_1": t1,
                    "layer_2": t2,
                    "layer_whole": tw,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


# -----------------------------
# OpenAI call
# -----------------------------
def call_model_structured(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    """
    Responses API + Structured Outputs
    """
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": SCHEMA["name"],
                "schema": SCHEMA["schema"],
            }
        },
        max_output_tokens=max_output_tokens,
        store=False,
    )
    return json.loads(resp.output_text)


# -----------------------------
# Load metadata and group
# -----------------------------
def load_groups(metadata_jsonl: Path) -> Dict[str, Dict[str, str]]:
    """
    groups[group_id]["layer_0"|"layer_1"|"layer_2"|"layer_whole"] = cleaned_text
    """
    groups: Dict[str, Dict[str, str]] = defaultdict(dict)
    with metadata_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fn = obj.get("file_name", "")
            gid = parse_group_id(fn)
            if gid is None:
                continue
            layer = file_to_layer_key(fn)
            if layer is None:
                continue
            txt = clean_text(obj.get("text", "") or "")
            groups[gid][layer] = txt
    return dict(groups)


def iter_batches(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# -----------------------------
# Validation + post filters (cheap safety net)
# -----------------------------
def looks_like_style_only(p: str) -> bool:
    """
    Very lightweight heuristic:
    If prompt contains obvious replacement words "replaced with", it's likely object_change.
    """
    p_low = p.lower()
    if "replaced with" in p_low:
        return False
    return True


def dedup_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x2 = x.strip()
        if not x2:
            continue
        if x2 in seen:
            continue
        seen.add(x2)
        out.append(x2)
    return out


def sanitize_entries(entries: List[Dict[str, Any]], style_k: int, obj_k: int) -> List[Dict[str, Any]]:
    """
    Enforce:
    - exactly 2 entries per (group_id, layer): style_change + object_change
    - trim target_prompts length
    - dedup prompts
    """
    out = []
    bucket = defaultdict(list)
    for e in entries:
        key = (e["group_id"], e["layer"], e["edit_type"])
        bucket[key].append(e)

    # merge duplicates of the same key (just in case)
    merged = {}
    for (gid, layer, et), lst in bucket.items():
        base = lst[0]
        tp = []
        for it in lst:
            tp.extend(it.get("target_prompts", []))
        tp = dedup_keep_order(tp)
        base["target_prompts"] = tp
        merged[(gid, layer, et)] = base

    # rebuild per (gid, layer)
    by_gl = defaultdict(dict)
    for (gid, layer, et), e in merged.items():
        by_gl[(gid, layer)][et] = e

    for (gid, layer), d in by_gl.items():
        if "style_change" not in d or "object_change" not in d:
            # skip incomplete; caller may re-run later if needed
            continue

        st = d["style_change"]
        ob = d["object_change"]

        st["target_prompts"] = dedup_keep_order(st.get("target_prompts", []))[:style_k]
        ob["target_prompts"] = dedup_keep_order(ob.get("target_prompts", []))[:obj_k]

        # ensure non-empty
        if len(st["target_prompts"]) < 3 or len(ob["target_prompts"]) < 3:
            continue

        out.append(st)
        out.append(ob)

    return out


def append_entries_yaml(path: Path, entries: List[Dict[str, Any]]) -> int:
    """
    Append entries as YAML list items to an existing file.
    Returns number of entries written.
    """
    if not entries:
        return 0
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            dumped = yaml.safe_dump(entry, allow_unicode=True, sort_keys=False).strip()
            if not dumped:
                continue
            lines = dumped.splitlines()
            lines[0] = "- " + lines[0]
            for i in range(1, len(lines)):
                lines[i] = "  " + lines[i]
            f.write("\n".join(lines) + "\n")
    return len(entries)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, required=True, help="Path to metadata.jsonl")
    ap.add_argument("--out", type=str, required=True, help="Output dataset.yaml path")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=50, help="Number of groups per request (start with 50)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of groups (0 = all)")
    ap.add_argument("--offset", type=int, default=0, help="Skip first N groups")
    ap.add_argument("--style-k", type=int, default=4, help="Target prompts per style_change")
    ap.add_argument("--obj-k", type=int, default=3, help="Target prompts per object_change")
    ap.add_argument("--max-output-tokens", type=int, default=4000, help="Cap output tokens per request")
    ap.add_argument("--resume", action="store_true", help="Append to existing output (do not truncate)")
    ap.add_argument("--seed", type=int, default=0, help="(Optional) not used; placeholder")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    groups = load_groups(Path(args.metadata))
    gids = sorted(groups.keys())

    if args.offset > 0:
        gids = gids[args.offset:]
    if args.limit > 0:
        gids = gids[:args.limit]

    # Build list of group dicts to feed the model
    items = []
    for gid in gids:
        g = groups[gid]
        # require layer_1 and layer_2 prompts (your study focuses on them)
        if not g.get("layer_1") or not g.get("layer_2"):
            continue
        items.append(
            {
                "group_id": gid,
                "layer_0": g.get("layer_0", ""),
                "layer_1": g.get("layer_1", ""),
                "layer_2": g.get("layer_2", ""),
                "layer_whole": g.get("layer_whole", ""),
            }
        )

    if not items:
        raise RuntimeError("No valid groups found (need at least layer_1 and layer_2).")

    instructions = build_instructions(style_k=args.style_k, obj_k=args.obj_k)

    total_groups = len(items)
    total_batches = math.ceil(total_groups / args.batch_size)
    processed_groups = 0
    total_entries = 0

    def process_batch(batch: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
        input_text = build_batch_input(batch)
        try:
            data = call_model_structured(
                client=client,
                model=args.model,
                instructions=instructions,
                input_text=input_text,
                max_output_tokens=args.max_output_tokens,
            )
        except json.JSONDecodeError:
            if len(batch) == 1:
                raise RuntimeError(
                    "Model output was truncated/invalid JSON for a single group. "
                    "Try increasing --max-output-tokens or simplifying prompts."
                )
            mid = len(batch) // 2
            print(f"[WARN] JSON decode failed for batch {label}; splitting into {mid} + {len(batch)-mid}.")
            left = process_batch(batch[:mid], f"{label}a")
            right = process_batch(batch[mid:], f"{label}b")
            return left + right

        entries = data.get("entries", [])
        return sanitize_entries(entries, style_k=args.style_k, obj_k=args.obj_k)

    for bi, batch in enumerate(iter_batches(items, args.batch_size), start=1):
        if bi == 1:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.resume:
                out_path.write_text("", encoding="utf-8")
            elif not out_path.exists():
                out_path.write_text("", encoding="utf-8")
            elif args.resume:
                print(f"[INFO] Resuming: appending to existing {out_path}")

        entries = process_batch(batch, str(bi))
        total_entries += append_entries_yaml(out_path, entries)

        processed_groups += len(batch)
        pct = (processed_groups / total_groups) * 100 if total_groups else 100.0
        print(
            f"[{bi}/{total_batches}] "
            f"groups={len(batch)} "
            f"(done {processed_groups}/{total_groups} = {pct:.1f}%) "
            f"-> entries={len(entries)} (cumulative {total_entries})"
        )

    print(f"Saved: {out_path}  (entries={total_entries})")


if __name__ == "__main__":
    main()

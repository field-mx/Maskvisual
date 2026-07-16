#!/usr/bin/env python
"""Download the trained control LoRAs from the Hugging Face Hub.

Files in the HF repo (two experts of the Wan2.2 MoE):

    HadiZayer/masked-visual-actions/
      masked_world_lora_high.safetensors   # high-noise expert
      masked_world_lora_low.safetensors    # low-noise expert

The base model (PAI/Wan2.2-Fun-A14B-Control) is downloaded automatically by
`infer.py` via ModelScope; this script only fetches the LoRA adapters.

Usage:
    python download_weights.py --out ./checkpoints

Then pass the printed paths to infer.py:
    --lora-high checkpoints/masked_world_lora_high.safetensors
    --lora-low  checkpoints/masked_world_lora_low.safetensors
"""
import argparse
import os

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "HadiZayer/masked-visual-actions"

FILES = ["masked_world_lora_high.safetensors", "masked_world_lora_low.safetensors"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO_ID, help="HF repo id hosting the LoRAs")
    ap.add_argument("--out", default="./checkpoints", help="Local directory to download into")
    ap.add_argument("--revision", default=None, help="Optional git revision / tag")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = [
        hf_hub_download(repo_id=args.repo, filename=f, revision=args.revision, local_dir=args.out)
        for f in FILES
    ]
    print("\ndownloaded:")
    print(f"    --lora-high {paths[0]}")
    print(f"    --lora-low  {paths[1]}")


if __name__ == "__main__":
    main()

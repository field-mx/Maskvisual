#!/usr/bin/env python
"""Inference for the Wan2.2-Fun-A14B control model with our trained LoRAs.

The base model (PAI/Wan2.2-Fun-A14B-Control) is a two-expert MoE: a high-noise
DiT and a low-noise DiT, split at timestep boundary 0.358. We therefore train and
load *two* LoRAs -- one per expert -- into `pipe.dit` (high noise) and
`pipe.dit2` (low noise).

The model is conditioned on a single `control_video` (e.g. the rendered URDF
robot), a first-frame `reference_image`, and a text `prompt`, and generates the
corresponding RGB video.

Example
-------
    python infer.py \
        --lora-high $LORA/control_high_noise.safetensors \
        --lora-low  $LORA/control_low_noise.safetensors \
        --control-video robot_render.mp4 \
        --reference-image first_frame.png \
        --prompt "a robot arm picks up a mug" \
        --output out.mp4
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# Default Wan negative prompt (as used during our experiments).
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

BASE_MODEL_ID = "PAI/Wan2.2-Fun-A14B-Control"


def build_pipeline(base_model_id, low_vram=False):
    """Load the Wan2.2-Fun-A14B-Control pipeline (both experts + T5 + VAE)."""
    kwargs = {}
    if low_vram:
        # Offload weights to disk; keeps peak VRAM low at the cost of speed.
        kwargs = dict(
            offload_dtype="disk", offload_device="disk",
            onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda",
            computation_dtype=torch.bfloat16, computation_device="cuda",
        )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=base_model_id, origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", **kwargs),
            ModelConfig(model_id=base_model_id, origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", **kwargs),
            ModelConfig(model_id=base_model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **kwargs),
            ModelConfig(model_id=base_model_id, origin_file_pattern="Wan2.1_VAE.pth", **kwargs),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        **({"vram_limit": torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2} if low_vram else {}),
    )
    return pipe


def load_frames(path, height, width, num_frames):
    """Load a video and uniformly resample it to `num_frames` frames."""
    vid = VideoData(path, height=height, width=width)
    indices = np.linspace(0, len(vid) - 1, num_frames).astype(int)
    return [vid[i] for i in indices]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora-high", required=True, help="High-noise expert LoRA (.safetensors)")
    ap.add_argument("--lora-low", required=True, help="Low-noise expert LoRA (.safetensors)")
    ap.add_argument("--control-video", required=True, help="Control video (e.g. URDF robot render)")
    ap.add_argument("--reference-image", default=None, help="First-frame reference image; defaults to control video frame 0")
    ap.add_argument("--prompt", required=True, help="Text prompt / task description")
    ap.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    ap.add_argument("--output", required=True, help="Output .mp4 path")
    ap.add_argument("--base-model-id", default=BASE_MODEL_ID)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--lora-alpha", type=float, default=1.0)
    ap.add_argument("--low-vram", action="store_true", help="Offload weights to disk to reduce peak VRAM")
    ap.add_argument("--save-inputs", action="store_true", help="Also save control/reference alongside the output")
    args = ap.parse_args()

    pipe = build_pipeline(args.base_model_id, low_vram=args.low_vram)
    pipe.load_lora(pipe.dit, args.lora_high, alpha=args.lora_alpha)
    pipe.load_lora(pipe.dit2, args.lora_low, alpha=args.lora_alpha)

    control = load_frames(args.control_video, args.height, args.width, args.num_frames)

    if args.reference_image and os.path.exists(args.reference_image):
        reference_image = Image.open(args.reference_image).convert("RGB").resize((args.width, args.height))
    else:
        if args.reference_image:
            print(f"[warn] reference image {args.reference_image} not found; using control frame 0")
        reference_image = control[0]

    output_video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        control_video=control,
        reference_image=reference_image,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_video(output_video, args.output, fps=args.fps, quality=5)
    print(f"Saved {args.output}")

    if args.save_inputs:
        stem = os.path.splitext(args.output)[0]
        save_video(control, f"{stem}_control.mp4", fps=args.fps, quality=5)
        reference_image.save(f"{stem}_reference.png")


if __name__ == "__main__":
    main()

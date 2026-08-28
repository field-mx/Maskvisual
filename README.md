# masked-visual-actions

Code for finetuning and running our robot-video **control model**: a LoRA on top
of [`PAI/Wan2.2-Fun-A14B-Control`](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control).
Given a **control video** (a rendered URDF robot), a **reference image** (the
first real frame), and a **text prompt**, it generates the corresponding RGB video.

We did not modify the video model or its trainer — we used
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) to train a LoRA
on our data and to run inference. This repo is the thin layer on top: the
inference script, the training recipe, and our weights.

```
inference/   infer.py, download_weights.py   — run the model with our LoRAs
training/    train_control.sh                — finetune a control LoRA on your CSV
```

Weights are on the Hugging Face Hub at
[`HadiZayer/masked-visual-actions`](https://huggingface.co/HadiZayer/masked-visual-actions).

## Setup

Install DiffSynth-Studio at the pinned commit, then this repo's light deps:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
git checkout 3743b1307caf2562af60d475b22d4b6be68e7cd0
pip install -e .
pip install huggingface_hub
```

A CUDA GPU is required (the base model is 14B; `infer.py --low-vram` offloads to
disk if you are memory constrained).

## Inference

```bash
python inference/download_weights.py --out ./checkpoints

python inference/infer.py \
    --lora-high checkpoints/masked_world_lora_high.safetensors \
    --lora-low  checkpoints/masked_world_lora_low.safetensors \
    --control-video robot_render.mp4 \
    --reference-image first_frame.png \
    --prompt "a robot arm picks up a mug" \
    --output out.mp4
```

`Wan2.2-Fun-A14B-Control` is a two-expert MoE (a **high-noise** and a **low-noise**
DiT, split at timestep boundary 0.358), so there are two LoRAs — one loaded into
`pipe.dit`, one into `pipe.dit2`. `--reference-image` is optional (defaults to
frame 0 of the control video). See `infer.py --help` for resolution/seed/steps.

## Training

Provide a CSV with columns `prompt, reference_image, video, control_video` and run
the two-expert recipe (stock DiffSynth params, run from the DiffSynth-Studio root):

```bash
cd DiffSynth-Studio
DATASET_CSV=/path/to/train.csv OUTPUT_DIR=/path/to/out \
    bash /path/to/masked-visual-actions/training/train_control.sh
```

This writes `<OUTPUT_DIR>_high_noise/` and `<OUTPUT_DIR>_low_noise/`; point
`infer.py` at the `step-*.safetensors` checkpoints you want.

## Rendering control videos

Tools for rendering the URDF robot control videos from DROID episodes are coming
soon.

## Local reproduction on a 24GB Turing GPU

This checkout pins DiffSynth-Studio to commit `3743b1307caf2562af60d475b22d4b6be68e7cd0`. The local compatibility patch selects FP16 on pre-Ampere GPUs; use `--low-vram` on a 24GB card.

A minimal pipeline smoke test using the locally generated ORCA control clip is:

```bash
source /home/muxiang/work-maskvisual/.venv/bin/activate
cd /home/muxiang/work-maskvisual
export DIFFSYNTH_MODEL_BASE_PATH=/home/muxiang/work-maskvisual/models
CUDA_VISIBLE_DEVICES=0 python inference/infer.py \
    --lora-high checkpoints/masked_world_lora_high.safetensors \
    --lora-low checkpoints/masked_world_lora_low.safetensors \
    --control-video sample_inputs/orca_control/episode_000.mp4 \
    --reference-image sample_inputs/orca_control/reference.png \
    --prompt "a robotic hand manipulates a cube" \
    --output outputs/orca_smoke.mp4 \
    --height 240 --width 320 --num-frames 17 \
    --num-inference-steps 2 --dtype auto --low-vram
```

The smoke settings only validate execution. Use the upstream defaults (`480x832`, 81 frames, 50 steps) for a quality run. Exact training-result reproduction is not currently possible from this repository alone because the authors have not published their training CSV/media or the DROID-to-URDF control-video renderer. Training on a new dataset remains available through `training/train_control.sh`.

## License

Apache-2.0 (inherited from DiffSynth-Studio). See `LICENSE`.

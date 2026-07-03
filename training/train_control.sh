#!/bin/bash
# Finetune a control LoRA for Wan2.2-Fun-A14B-Control on your own data.
#
# This is just the stock DiffSynth-Studio training recipe
# (examples/wanvideo/model_training/train.py) pointed at your CSV -- we did not
# modify the trainer. The base model is a two-expert MoE (high-noise + low-noise
# DiT) split at timestep boundary 0.358, so we train TWO LoRAs, one per expert.
#
# Prereq: clone DiffSynth-Studio at the pinned commit and install it:
#   git clone https://github.com/modelscope/DiffSynth-Studio.git
#   cd DiffSynth-Studio && git checkout 3743b1307caf2562af60d475b22d4b6be68e7cd0
#   pip install -e .
#
# Your DATASET_CSV must have columns: prompt, reference_image, video, control_video
#   video          = the real target clip
#   control_video  = the conditioning clip (e.g. a rendered URDF robot; see ../rendering)
#   reference_image = the first real frame
#
# Run from the DiffSynth-Studio repo root (or set DIFFSYNTH_DIR):
#   DATASET_CSV=/path/to/train.csv OUTPUT_DIR=/path/to/out bash train_control.sh
set -e

DATASET_CSV=${DATASET_CSV:?set DATASET_CSV to your training metadata csv}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/control}
DIFFSYNTH_DIR=${DIFFSYNTH_DIR:-.}
TRAIN_PY="${DIFFSYNTH_DIR}/examples/wanvideo/model_training/train.py"
NUM_GPUS=${NUM_GPUS:-4}
BASE=PAI/Wan2.2-Fun-A14B-Control
# Which expert(s) to train: high, low, or both.
EXPERT=${EXPERT:-both}

train_expert () {
  local noise=$1 max_b min_b
  if [ "$noise" = "high" ]; then max_b=0.358; min_b=0; else max_b=1; min_b=0.358; fi
  echo "=== training ${noise}-noise expert (timestep boundary ${min_b}-${max_b}) ==="
  accelerate launch --mixed_precision=bf16 --num_processes=${NUM_GPUS} --multi_gpu "${TRAIN_PY}" \
    --dataset_base_path "" \
    --dataset_metadata_path "${DATASET_CSV}" \
    --data_file_keys "video,control_video,reference_image" \
    --height 480 --width 832 \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "${BASE}:${noise}_noise_model/diffusion_pytorch_model*.safetensors,${BASE}:models_t5_umt5-xxl-enc-bf16.pth,${BASE}:Wan2.1_VAE.pth" \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${OUTPUT_DIR}_${noise}_noise" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 256 \
    --save_steps 250 \
    --extra_inputs "control_video,reference_image" \
    --max_timestep_boundary ${max_b} \
    --min_timestep_boundary ${min_b}
}

if [ "$EXPERT" = "both" ] || [ "$EXPERT" = "high" ]; then train_expert high; fi
if [ "$EXPERT" = "both" ] || [ "$EXPERT" = "low" ];  then train_expert low;  fi

echo "Done. LoRAs in ${OUTPUT_DIR}_high_noise/ and ${OUTPUT_DIR}_low_noise/ (use the step-*.safetensors with ../inference/infer.py)."

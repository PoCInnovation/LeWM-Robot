#!/bin/bash
#SBATCH --job-name=lewm-lora
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --threads-per-core=1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/lora_%j.out
#
# Fine-tune LoRA sur les démos réelles pré-encodées (1 GCD, rapide).
#
# Soumission :
#   sbatch --export=ALL,WORK_ROOT=...,ENCODED_DIR=...,PREDICTOR_CKPT=... \
#       slurm/train_lora.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source slurm/env.sh

ENCODED_DIR="${ENCODED_DIR:?Passer ENCODED_DIR=<latents réels encodés> via --export}"
PREDICTOR_CKPT="${PREDICTOR_CKPT:-$WORK_ROOT/checkpoints/predictor_sim/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/checkpoints/predictor_real}"
LORA_ARGS="${LORA_ARGS:---lora-rank 8 --n-epochs 20 --batch-size 32}"

python scripts/00_check_env.py ${PY_CONFIG:+--config "$PY_CONFIG"} --skip-model --skip-dataset \
    --expect-gpu --expect-offline

srun python scripts/05_train_lora.py ${PY_CONFIG:+--config "$PY_CONFIG"} \
    --encoded-data "$ENCODED_DIR" \
    --predictor-ckpt "$PREDICTOR_CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --resume \
    $LORA_ARGS

echo "[lora] terminé → $OUTPUT_DIR"

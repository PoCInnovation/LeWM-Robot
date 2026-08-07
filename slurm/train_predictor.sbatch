#!/bin/bash
#SBATCH --job-name=lewm-train
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --threads-per-core=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/train_%j.out
#
# Training du predictor sur 1 GCD, bf16, avec resume automatique :
# --requeue + --resume => le job survit au mur des 24 h et aux préemptions
# (il repart de last.pt).
#
# Soumission :
#   sbatch --export=ALL,WORK_ROOT=...,ENCODED_DIR=...[,TRAIN_ARGS=...] \
#       slurm/train_predictor.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source slurm/env.sh

ENCODED_DIR="${ENCODED_DIR:?Passer ENCODED_DIR=<dossier latents encodés> via --export}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/checkpoints/predictor_sim}"
TRAIN_ARGS="${TRAIN_ARGS:---fusion cross_attn_bd --delta 2 --n-epochs 100 --batch-size 64}"

# Poids de fusion entraînés par le job finalize (03) : chargés + gelés si dispo
if [ -n "${FUSION_CKPT:-}" ] && [ -f "$FUSION_CKPT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --fusion-ckpt $FUSION_CKPT"
    echo "[train] Fusion pré-entraînée : $FUSION_CKPT"
fi

# Latents pré-encodés uniquement : ni modèle ni dataset à vérifier
python scripts/00_check_env.py ${PY_CONFIG:+--config "$PY_CONFIG"} --skip-model --skip-dataset \
    --expect-gpu --expect-offline

srun python scripts/04_train_predictor.py ${PY_CONFIG:+--config "$PY_CONFIG"} \
    --encoded-data "$ENCODED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --resume \
    $TRAIN_ARGS

echo "[train] terminé → $OUTPUT_DIR"

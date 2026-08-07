#!/bin/bash
#SBATCH --job-name=lewm-finalize
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --threads-per-core=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/finalize_%j.out
#
# Après l'array d'encodage : merge des shards + comparaison des fusions.
# Produit meta.json et les poids de fusion pour le training.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source slurm/env.sh

ENCODED_DIR="${ENCODED_DIR:?ENCODED_DIR manquant}"
FUSION_DIR="${FUSION_DIR:-$WORK_ROOT/fusion}"

echo "=== [finalize 1/2] merge des shards ==="
python scripts/02_encode_dataset.py ${PY_CONFIG:+--config "$PY_CONFIG"} --output "$ENCODED_DIR" --finalize

echo "=== [finalize 2/2] comparaison des fusions ==="
python scripts/03_compare_fusion.py ${PY_CONFIG:+--config "$PY_CONFIG"} --encoded-data "$ENCODED_DIR" \
    --output-dir "$FUSION_DIR"

echo "[finalize] OK → $ENCODED_DIR (meta.json) + $FUSION_DIR (poids fusion)"

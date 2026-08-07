#!/bin/bash
#SBATCH --job-name=lewm-smoke
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --threads-per-core=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/smoke_%j.out
#
# SMOKE TEST — première étape de la chaîne run_all.sh.
# Déroule le pipeline COMPLET en miniature (4 épisodes tronqués) sur 1 GCD :
# encode → train 2 epochs → resume → démo CEM. Si ça casse, les vrais jobs
# (qui dépendent de celui-ci en afterok) ne partent jamais.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source slurm/env.sh

DATASET_DIR="${DATASET_DIR:?DATASET_DIR manquant}"
SMOKE_DIR="$WORK_ROOT/smoke"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

echo "=== [smoke 1/5] check env ==="
python scripts/00_check_env.py ${PY_CONFIG:+--config "$PY_CONFIG"} --dataset-id "$DATASET_DIR" \
    --expect-gpu --expect-offline

echo "=== [smoke 2/5] encode (4 épisodes tronqués, 2 shards) ==="
for i in 0 1; do
    python scripts/02_encode_dataset.py ${PY_CONFIG:+--config "$PY_CONFIG"} --dataset-id "$DATASET_DIR" \
        --output "$SMOKE_DIR/encoded" --max-episodes 4 \
        --max-frames-per-episode 40 --shard-index $i --num-shards 2 \
        --batch-size 32 --num-workers 4
done
python scripts/02_encode_dataset.py ${PY_CONFIG:+--config "$PY_CONFIG"} --output "$SMOKE_DIR/encoded" --finalize

echo "=== [smoke 3/5] train predictor (2 epochs) ==="
python scripts/04_train_predictor.py ${PY_CONFIG:+--config "$PY_CONFIG"} --encoded-data "$SMOKE_DIR/encoded" \
    --output-dir "$SMOKE_DIR/ckpt" --n-epochs 2 --batch-size 16 --n-layers 2

echo "=== [smoke 4/5] resume ==="
python scripts/04_train_predictor.py ${PY_CONFIG:+--config "$PY_CONFIG"} --encoded-data "$SMOKE_DIR/encoded" \
    --output-dir "$SMOKE_DIR/ckpt" --n-epochs 3 --batch-size 16 --n-layers 2 \
    --resume

echo "=== [smoke 5/5] démo CEM sur latents ==="
python scripts/06_inference_demo.py ${PY_CONFIG:+--config "$PY_CONFIG"} --from-encoded "$SMOKE_DIR/encoded" \
    --ckpt "$SMOKE_DIR/ckpt/best.pt" --n-samples 64 --n-iter 2 --horizon 5

echo "=== SMOKE TEST OK — la chaîne réelle peut partir ==="

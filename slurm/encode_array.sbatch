#!/bin/bash
#SBATCH --job-name=lewm-encode
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --threads-per-core=1
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/encode_%A_%a.out
#
# Encodage shardé du dataset en job array : 1 tâche = 1 GCD = 1 shard.
# NB : ne PAS spécifier de --qos (les queues Adastra sont transparentes,
# SLURM place le job selon durée/ressources — doc CINES).
#
# Soumission :
#   sbatch --export=ALL,WORK_ROOT=...,DATASET_DIR=... \
#       [--array=0-N-1] slurm/encode_array.sbatch
#
# Puis, quand l'array est terminé, le merge (léger, frontal ou job court) :
#   python scripts/02_encode_dataset.py --output $ENCODED_DIR --finalize

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source slurm/env.sh

DATASET_DIR="${DATASET_DIR:?Passer DATASET_DIR=<chemin local du dataset> via --export}"
NUM_SHARDS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
ENCODED_DIR="${ENCODED_DIR:-$WORK_ROOT/encoded/$(basename "$DATASET_DIR")}"

# Validation avant travail (échoue vite si un prérequis manque)
python scripts/00_check_env.py --dataset-id "$DATASET_DIR" \
    --expect-gpu --expect-offline

srun python scripts/02_encode_dataset.py \
    --dataset-id "$DATASET_DIR" \
    --output "$ENCODED_DIR" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS" \
    --batch-size 64 \
    --num-workers "$SLURM_CPUS_PER_TASK"

echo "[encode] shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS terminé → $ENCODED_DIR"

#!/usr/bin/env bash
# Setup complet à lancer UNE FOIS sur le NŒUD DE LOGIN d'Adastra
# (le seul endroit avec accès internet, via le proxy CINES).
#
# Usage :
#   export WORK_ROOT=$SCRATCHDIR/lewm-robot     # espace scratch du projet
#   bash slurm/setup_offline.sh [dataset_hf_id] [encoder_model_id...]
#
# Ce script :
#   1. crée le venv + installe les dépendances (torch ROCm inclus)
#   2. pré-télécharge le(s) modèle(s) encodeur dans HF_HOME
#   3. pré-télécharge le dataset LeRobot dans HF_LEROBOT_HOME
#   4. lance 00_check_env.py en mode offline pour valider le tout
#
# ACTIONS UTILISATEUR PRÉALABLES (cf. ADASTRA_SETUP.md) :
#   - token HF : `huggingface-cli login` (DINOv3 est GATED : demander
#     l'accès aux modèles facebook/dinov3-* sur huggingface.co d'abord)
#   - proxy : normalement déjà configuré sur les nœuds de login ; sinon
#     export http_proxy=http://proxy-l-adastra.cines.fr:3128
#     export https_proxy=$http_proxy

set -euo pipefail

DATASET_ID="${1:-divisio74/duck_dataset_v3}"
shift || true
MODELS=("${@:-facebook/dinov3-vits16-pretrain-lvd1689m}")

WORK_ROOT="${WORK_ROOT:?Définir WORK_ROOT (ex: export WORK_ROOT=\$SCRATCHDIR/lewm-robot)}"
VENV_DIR="${VENV_DIR:-$WORK_ROOT/venv}"
export HF_HOME="$WORK_ROOT/hf_home"
export HF_LEROBOT_HOME="$WORK_ROOT/lerobot_home"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.2}"

mkdir -p "$WORK_ROOT" "$HF_HOME" "$HF_LEROBOT_HOME"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== 1/4 venv + dépendances ==="
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# PyTorch ROCm d'abord (sinon pip résout un torch CUDA depuis PyPI)
pip install torch torchvision --index-url "$ROCM_INDEX"
pip install -r "$REPO_DIR/requirements.txt"
pip install "lerobot>=0.5"

echo "=== 2/4 modèles encodeur → $HF_HOME ==="
for model in "${MODELS[@]}"; do
    echo "  - $model"
    huggingface-cli download "$model" >/dev/null
done

echo "=== 3/4 dataset $DATASET_ID → $HF_LEROBOT_HOME ==="
# Téléchargement direct du snapshot ; les jobs y accèdent par CHEMIN LOCAL
# (en offline, lerobot ne peut pas résoudre un repo_id Hub, même en cache).
DATASET_DIR="$HF_LEROBOT_HOME/$DATASET_ID"
huggingface-cli download --repo-type dataset "$DATASET_ID" \
    --local-dir "$DATASET_DIR" >/dev/null
echo "  → $DATASET_DIR"

echo "=== 4/4 validation offline ==="
cd "$REPO_DIR"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/00_check_env.py \
    --dataset-id "$DATASET_DIR" --expect-offline

cat <<EOF

Setup terminé.
  WORK_ROOT      = $WORK_ROOT
  Dataset local  = $DATASET_DIR

Soumettre ensuite (depuis la racine du repo) :
  sbatch --export=ALL,WORK_ROOT=$WORK_ROOT,DATASET_DIR=$DATASET_DIR \\
      slurm/encode_array.sbatch
EOF

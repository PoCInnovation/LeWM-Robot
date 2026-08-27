#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# CHAÎNE COMPLÈTE EN LOCAL — machine avec une RTX 4090 (ou tout GPU NVIDIA).
# Les étapes s'enchaînent dans ce shell, `set -e` arrête tout à la
# première erreur, chaque étape a son log dans logs/.
#
#   1. sanity check encodeur (débit bf16, cos-sim bf16/fp32)
#   2. encodage DINOv3 du dataset (bf16, workers auto) → results/encoded/encoded_data.pt
#   3. comparaison des fusions                          → results/fusion_comparison.{json,png}
#   4. training predictor (fusion gagnante)             → results/checkpoints/predictor_simu.pt
#   5. (optionnel) LoRA si REAL_DATASET est posé        → results/checkpoints/predictor_real.pt
#   6. démo CEM end-to-end depuis les images du dataset
#
# Usage minimal (depuis la racine du repo, venv activé) :
#   bash run_local.sh
#
# Variables optionnelles :
#   DATASET_ID     dataset HF ou chemin local   (défaut: dataset.hf_id de la config)
#   REAL_DATASET   démos réelles (HF id/chemin) → active l'étape LoRA (défaut: aucune)
#   PY_CONFIG      config YAML                  (défaut: configs/default.yaml)
#   FUSION         forcer une fusion            (défaut: gagnante de l'étape 3)
#   TRAIN_ARGS     args de 04  (défaut: --n-epochs 30 --batch-size 64)
#   LORA_ARGS      args de 05  (défaut: --lora-rank 8 --n-epochs 20 --batch-size 32)
#   SMOKE=1        pipeline miniature (200 paires, 2 epochs) — à lancer en
#                  premier sur une nouvelle machine (~2-5 min)
#   SKIP_ENCODE=1  réutiliser results/encoded/encoded_data.pt existant
#   SKIP_FUSION=1  sauter la comparaison (FUSION doit alors être posé)
#   LOG_DIR        dossier des logs (défaut: logs/)
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")"

PY_CONFIG="${PY_CONFIG:-configs/default.yaml}"
REAL_DATASET="${REAL_DATASET:-}"
TRAIN_ARGS="${TRAIN_ARGS:---n-epochs 30 --batch-size 64}"
LORA_ARGS="${LORA_ARGS:---lora-rank 8 --n-epochs 20 --batch-size 32}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

DATASET_ID="${DATASET_ID:-$(python - "$PY_CONFIG" <<'EOF'
import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["dataset"]["hf_id"])
EOF
)}"

ENCODED="results/encoded/encoded_data.pt"
REAL_ENCODED="results/encoded/real_encoded_data.pt"
CKPT_SIM="results/checkpoints/predictor_simu.pt"
CKPT_REAL="results/checkpoints/predictor_real.pt"
ENC_ARGS=""
FUSION_EPOCHS=""
DEMO_ARGS="--n-samples 1000 --rollout-chunk 500"
if [ "${SMOKE:-0}" = "1" ]; then
    ENCODED="results/encoded/_smoke_encoded_data.pt"
    REAL_ENCODED="results/encoded/_smoke_real_encoded_data.pt"
    CKPT_SIM="results/checkpoints/_smoke_predictor_simu.pt"
    CKPT_REAL="results/checkpoints/_smoke_predictor_real.pt"
    ENC_ARGS="--max-pairs 200"
    FUSION_EPOCHS="--n-epochs 2"
    TRAIN_ARGS="--n-epochs 2 --batch-size 16 --n-layers 2"
    LORA_ARGS="--lora-rank 4 --n-epochs 2 --batch-size 16"
    DEMO_ARGS="--n-samples 64 --n-iter 2 --horizon 5"
fi

run() {  # run <étape> <commande...>  — log dans $LOG_DIR/<étape>_<stamp>.log
    local step="$1"; shift
    echo; echo "══════ [$step] $*"
    "$@" 2>&1 | tee "$LOG_DIR/${step}_${STAMP}.log"
    return "${PIPESTATUS[0]}"
}

echo "══════════════════════════════════════════════════════"
echo " LeWM-Robot — chaîne locale (GPU NVIDIA)"
echo "   DATASET  : $DATASET_ID"
echo "   CONFIG   : $PY_CONFIG"
echo "   ENCODED  : $ENCODED"
echo "   SMOKE    : ${SMOKE:-0}"
echo "══════════════════════════════════════════════════════"

# ── 1. sanity check encodeur ──────────────────────────────────────────
run test_encoder python scripts/01_test_dinov3.py --config "$PY_CONFIG"

# ── 2. encodage ───────────────────────────────────────────────────────
if [ "${SKIP_ENCODE:-0}" != "1" ] || [ ! -f "$ENCODED" ]; then
    run encode python scripts/02_encode_dataset.py --config "$PY_CONFIG" \
        --dataset-id "$DATASET_ID" --output "$ENCODED" $ENC_ARGS
else
    echo "[run_local] SKIP_ENCODE=1 — encodage sauté ($ENCODED)."
fi

# ── 3. comparaison des fusions ────────────────────────────────────────
if [ "${SKIP_FUSION:-0}" != "1" ]; then
    run fusion python scripts/03_compare_fusion.py --config "$PY_CONFIG" \
        --encoded-data "$ENCODED" $FUSION_EPOCHS
    if [ -z "${FUSION:-}" ]; then
        FUSION=$(python - "$PY_CONFIG" <<'EOF'
import json, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
r = json.load(open(cfg["paths"].get("fusion_results", "results/fusion_comparison.json")))
print(min(r, key=lambda k: r[k]["best_val_mae"]))
EOF
)
    fi
else
    FUSION="${FUSION:-concat_view}"
    echo "[run_local] SKIP_FUSION=1 — fusion : $FUSION"
fi
echo "[run_local] Fusion retenue : $FUSION"

# ── 4. training predictor ─────────────────────────────────────────────
run train python scripts/04_train_predictor.py --config "$PY_CONFIG" \
    --encoded-data "$ENCODED" --output "$CKPT_SIM" --fusion "$FUSION" $TRAIN_ARGS

# ── 5. LoRA (optionnel) ───────────────────────────────────────────────
DEMO_CKPT="$CKPT_SIM"
if [ -n "$REAL_DATASET" ]; then
    if [ "${SKIP_ENCODE:-0}" != "1" ] || [ ! -f "$REAL_ENCODED" ]; then
        run encode_real python scripts/02_encode_dataset.py --config "$PY_CONFIG" \
            --dataset-id "$REAL_DATASET" --output "$REAL_ENCODED" $ENC_ARGS
    fi
    run lora python scripts/05_train_lora.py --config "$PY_CONFIG" \
        --real-data "$REAL_ENCODED" --predictor-ckpt "$CKPT_SIM" \
        --output "$CKPT_REAL" $LORA_ARGS
    DEMO_CKPT="$CKPT_REAL"
fi

# ── 6. démo CEM ───────────────────────────────────────────────────────
run demo python scripts/06_inference_demo.py --config "$PY_CONFIG" \
    --dataset-id "$DATASET_ID" --predictor-ckpt "$DEMO_CKPT" $DEMO_ARGS

cat <<EOF

══════════════════════════════════════════════════════
 Chaîne terminée.
   Latents      : $ENCODED
   Fusions      : results/fusion_comparison.json (gagnante : $FUSION)
   Predictor    : $CKPT_SIM
$( [ -n "$REAL_DATASET" ] && echo "   LoRA         : $CKPT_REAL" )
   Logs         : $LOG_DIR/*_${STAMP}.log
══════════════════════════════════════════════════════
EOF

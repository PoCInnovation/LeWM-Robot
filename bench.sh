#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# UNE SEULE COMMANDE — benchmark de la RTX 5090 depuis un clone vierge :
#
#     make
#
# À la fin : results/benchmark_report_<date>.tar.gz — C'EST CE FICHIER QU'IL
# FAUT NOUS RENVOYER (estimations + log + infos GPU/versions).
#
# Fait tout, dans l'ordre, sans rien demander :
#   1. trouve un Python 3.10+ ;
#   2. crée .venv et installe torch CUDA + dépendances + lerobot
#      (sauté si les dépendances sont déjà installées) ;
#   3. login HuggingFace si HF_TOKEN est posé (DINOv3 est gated ; sans
#      token l'encodeur est simplement sauté, le reste est mesuré) ;
#   4. vérifie que le GPU est visible ;
#   5. lance scripts/07_benchmark.py : mini-trains chronométrés →
#      estimation du temps de chaque étape + TOTAL, results/benchmark.json,
#      log dans logs/benchmark_<date>.log ;
#   6. empaquette le tout dans results/benchmark_report_<date>.tar.gz.
#
# Options (variables d'environnement, toutes facultatives) :
#   HF_TOKEN=hf_xxx      token HuggingFace (accès DINOv3)
#   HF_TOKEN_FILE=...    fichier contenant le token (défaut configs/hf_token.txt)
#   N_EPOCHS=100         epochs visés pour le predictor   (défaut 30)
#   LORA_EPOCHS=20       epochs visés pour le LoRA        (défaut 20)
#   BATCH_SIZES=64,128   batch sizes des mini-trains      (défaut 32,64,128,256)
#   DATASET_ID=...       dataset pour mesurer le décodage vidéo réel
#                        (défaut : celui de configs/default.yaml)
#   NO_DATASET=1         ne pas télécharger/mesurer le dataset
#   BENCH_ARGS="..."     args supplémentaires pour 07_benchmark.py
#   PYTHON=/chemin/python  utiliser cet interpréteur (pas de venv)
#   CUDA_INDEX=...       index pip torch (défaut cu128)
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
CONFIG="${CONFIG:-configs/default.yaml}"
N_EPOCHS="${N_EPOCHS:-30}"
LORA_EPOCHS="${LORA_EPOCHS:-20}"
BATCH_SIZES="${BATCH_SIZES:-32,64,128,256}"
BENCH_ARGS="${BENCH_ARGS:-}"
mkdir -p logs results
STAMP_DATE="$(date +%Y%m%d_%H%M%S)"
LOG="logs/benchmark_${STAMP_DATE}.log"

say() { echo; echo "══════ $*"; }

# ── 1. Python ───────────────────────────────────────────────────────────
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$VENV/bin/python" ]; then
        PYTHON="$VENV/bin/python"
    else
        BASE_PY=""
        for c in python3.12 python3.11 python3.10 python3; do
            if command -v "$c" >/dev/null 2>&1 && \
               "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
                BASE_PY="$c"; break
            fi
        done
        [ -n "$BASE_PY" ] || { echo "ERREUR : Python >= 3.10 introuvable (sudo apt install python3.12 python3.12-venv)" >&2; exit 1; }
        say "[1/6] venv $VENV avec $BASE_PY"
        "$BASE_PY" -m venv "$VENV"
        PYTHON="$VENV/bin/python"
    fi
fi
echo "Python : $PYTHON ($("$PYTHON" -c 'import sys;print(sys.version.split()[0])'))"

# ── 2. Dépendances (idempotent : marqueur = hash de requirements) ───────
STAMP="$VENV/.installed"
WANT="$(cat requirements_wm.txt | md5sum | cut -c1-12)-$CUDA_INDEX"
if [ -n "${PYTHON_NO_INSTALL:-}" ] || { [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ]; }; then
    echo "Dépendances : déjà installées."
else
    say "[2/6] Installation des dépendances (torch CUDA, transformers, lerobot…)"
    "$PYTHON" -m pip install --upgrade pip -q
    "$PYTHON" -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url "$CUDA_INDEX"
    "$PYTHON" -m pip install -r requirements_wm.txt
    "$PYTHON" -m pip install "lerobot>=0.5" -c requirements_wm.txt
    mkdir -p "$VENV"
    echo "$WANT" > "$STAMP"
fi
command -v ffmpeg >/dev/null 2>&1 || \
    echo "[avertissement] ffmpeg absent : la mesure du décodage vidéo réel sera sautée (sudo apt install ffmpeg)."

# ── 3. HuggingFace ──────────────────────────────────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_FILE:-configs/hf_token.txt}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    HF_TOKEN="$(< "$HF_TOKEN_FILE")"
    HF_TOKEN="${HF_TOKEN//$'\r'/}"
fi
export HF_TOKEN
if [ -n "${HF_TOKEN:-}" ]; then
    say "[3/6] Login HuggingFace"
    "$PYTHON" -c 'import os; from huggingface_hub import login; login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)' \
        && echo "Token HF enregistré." || echo "[avertissement] login HF échoué — l'encodeur DINOv3 sera sauté."
else
    if ! "$PYTHON" -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
        echo "[info] Pas de token HF (HF_TOKEN) : DINOv3 est gated → l'étape encodeur sera"
        echo "       sautée ou mesurée avec dinov2 si présent ; le reste est complet."
    fi
fi

# ── 4. GPU ──────────────────────────────────────────────────────────────
say "[4/6] GPU"
"$PYTHON" scripts/00_check_gpu.py --config "$CONFIG"

# ── 5. Benchmark ────────────────────────────────────────────────────────
DS_ARGS=""
if [ "${NO_DATASET:-0}" != "1" ]; then
    DATASET_ID="${DATASET_ID:-$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['dataset']['hf_id'])")}"
    DS_ARGS="--dataset-id $DATASET_ID"
fi
say "[5/6] Benchmark (mini-trains chronométrés) — log : $LOG"
set +e
"$PYTHON" scripts/07_benchmark.py --config "$CONFIG" $DS_ARGS \
    --n-epochs "$N_EPOCHS" --lora-epochs "$LORA_EPOCHS" \
    --batch-sizes "$BATCH_SIZES" --output results/benchmark.json $BENCH_ARGS \
    2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" -ne 0 ]; then
    echo; echo "ERREUR : le benchmark s'est arrêté (code $STATUS). Détails : $LOG" >&2
    exit "$STATUS"
fi

# ── 6. Rapport à renvoyer ───────────────────────────────────────────────
say "[6/6] Empaquetage du rapport"
REPORT_DIR="results/benchmark_report_${STAMP_DATE}"
mkdir -p "$REPORT_DIR"
cp results/benchmark.json "$REPORT_DIR/benchmark.json"
cp "$LOG" "$REPORT_DIR/benchmark.log"
cp "$CONFIG" "$REPORT_DIR/config.yaml"
{ command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi; } > "$REPORT_DIR/nvidia-smi.txt" 2>&1 || true
{ uname -a; echo; nproc; free -g 2>/dev/null; echo; "$PYTHON" -m pip freeze; } \
    > "$REPORT_DIR/environment.txt" 2>&1 || true
git rev-parse HEAD > "$REPORT_DIR/git_commit.txt" 2>/dev/null || true
"$PYTHON" scripts/bench_report.py "$REPORT_DIR"
ARCHIVE="results/benchmark_report_${STAMP_DATE}.tar.gz"
tar -czf "$ARCHIVE" -C results "benchmark_report_${STAMP_DATE}"

echo
echo "══════════════════════════════════════════════════════"
echo " TERMINÉ."
echo
echo "   >>> Fichier à nous renvoyer :  $ARCHIVE"
echo
echo " (contenu : summary.md, benchmark.json, benchmark.log, nvidia-smi.txt,"
echo "  environment.txt, config.yaml, git_commit.txt)"
echo " Pour relancer avec d'autres cibles : N_EPOCHS=100 bash bench.sh"
echo "══════════════════════════════════════════════════════"

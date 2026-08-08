#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# COMMANDE UNIQUE — à lancer sur le NŒUD DE LOGIN d'Adastra, depuis la
# racine du repo. Fait TOUT :
#
#   1. setup offline (venv + torch ROCm, modèle, dataset, validation)
#   2. soumet la chaîne SLURM avec dépendances afterok :
#        smoke test (45 min) → encodage (array 16 GCD) → merge + fusions
#        → training predictor (24 h, resume auto)
#      → si une étape échoue, les suivantes ne partent JAMAIS.
#
# Usage minimal :
#   HF_TOKEN=hf_xxx bash run_all.sh
#
# Variables optionnelles (défauts sensés) :
#   WORK_ROOT      espace de travail        (défaut: $SCRATCHDIR/lewm-robot)
#   DATASET_ID     dataset HF à télécharger (défaut: divisio74/duck_dataset_v3)
#   ENCODER_MODEL  modèle HF encodeur       (défaut: facebook/dinov3-vits16-pretrain-lvd1689m)
#   N_SHARDS       taille de l'array encode (défaut: 16)
#   TRAIN_ARGS     args de 04               (défaut: cf. train_predictor.sbatch)
#   SKIP_SETUP=1   sauter l'étape 1 (déjà fait)
#   SKIP_SMOKE=1   sauter le smoke test (déconseillé)
#   DRY_RUN=1      afficher les commandes sans soumettre
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")"

DATASET_ID="${DATASET_ID:-divisio74/duck_dataset_v3}"
ENCODER_MODEL="${ENCODER_MODEL:-facebook/dinov3-vits16-pretrain-lvd1689m}"
N_SHARDS="${N_SHARDS:-16}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCHDIR:?SCRATCHDIR non défini — passer WORK_ROOT=...}/lewm-robot}"

DATASET_DIR="$WORK_ROOT/lerobot_home/$DATASET_ID"
ENCODED_DIR="$WORK_ROOT/encoded/$(basename "$DATASET_ID")"
FUSION_DIR="$WORK_ROOT/fusion"

echo "══════════════════════════════════════════════════════"
echo " LeWM-Robot — chaîne complète Adastra"
echo "   WORK_ROOT   : $WORK_ROOT"
echo "   DATASET     : $DATASET_ID"
echo "   ENCODER     : $ENCODER_MODEL"
echo "   SHARDS      : $N_SHARDS"
echo "══════════════════════════════════════════════════════"

# ── 1. Setup offline (login node, internet via proxy) ──────────────────
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
        huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
    fi
    if [[ "$ENCODER_MODEL" == *dinov3* ]] && \
       ! huggingface-cli whoami >/dev/null 2>&1; then
        echo "[run_all] ERREUR : $ENCODER_MODEL est GATED et aucun token HF" >&2
        echo "  actif. Lancer : HF_TOKEN=hf_xxx bash run_all.sh" >&2
        echo "  (ou huggingface-cli login au préalable)" >&2
        echo "  Fallback public sans token :" >&2
        echo "    ENCODER_MODEL=facebook/dinov2-small bash run_all.sh" >&2
        exit 1
    fi
    bash slurm/setup_offline.sh "$DATASET_ID" "$ENCODER_MODEL"
else
    echo "[run_all] SKIP_SETUP=1 — setup sauté."
fi

# Si l'encodeur n'est pas le dinov3 par défaut de la config, générer une
# config dédiée pour que TOUS les jobs utilisent le bon modèle.
PY_CONFIG=""
if [[ "$ENCODER_MODEL" == *dinov2* ]]; then
    size=$(echo "$ENCODER_MODEL" | grep -oE "small|base|large|giant")
    sed -e 's/family: "dinov3"/family: "dinov2"/' \
        -e "s/size: \"small\"/size: \"$size\"/" \
        configs/default.yaml > configs/run_all.yaml
    PY_CONFIG="configs/run_all.yaml"
    echo "[run_all] Config générée : configs/run_all.yaml (dinov2-$size)"
fi

# ── 2. Chaîne SLURM avec dépendances ───────────────────────────────────
mkdir -p logs
EXPORTS="ALL,WORK_ROOT=$WORK_ROOT,DATASET_DIR=$DATASET_DIR,ENCODED_DIR=$ENCODED_DIR,FUSION_DIR=$FUSION_DIR"
[ -n "$PY_CONFIG" ] && EXPORTS="$EXPORTS,PY_CONFIG=$PY_CONFIG"

# submit <description> <sbatch args...> — pose l'ID dans $SUBMIT_JID
# (pas de $(...) : la substitution de commande casserait le compteur DRY)
DRY_COUNT=0
submit() {
    local desc="$1"; shift
    if [ "${DRY_RUN:-0}" = "1" ]; then
        DRY_COUNT=$((DRY_COUNT + 1))
        SUBMIT_JID="DRY-$DRY_COUNT"
        echo "[DRY] sbatch $*  → $SUBMIT_JID"
    else
        SUBMIT_JID=$(sbatch --parsable "$@")
        echo "[run_all] $desc → job $SUBMIT_JID"
    fi
}

DEP=""
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    submit "smoke test (45 min)" --export="$EXPORTS" slurm/smoke.sbatch
    DEP="--dependency=afterok:$SUBMIT_JID"
fi

submit "encodage array ($N_SHARDS shards)" \
    $DEP --array=0-$((N_SHARDS - 1)) --export="$EXPORTS" \
    slurm/encode_array.sbatch
JID_ENC=$SUBMIT_JID

submit "merge + comparaison fusions" \
    --dependency=afterok:"$JID_ENC" --export="$EXPORTS" \
    slurm/finalize.sbatch
JID_FIN=$SUBMIT_JID

submit "training predictor (24 h, resume auto)" \
    --dependency=afterok:"$JID_FIN" --export="$EXPORTS" \
    slurm/train_predictor.sbatch

cat <<EOF

══════════════════════════════════════════════════════
 Chaîne soumise. Rien d'autre à faire.
   Suivi        : squeue --me
   Logs         : tail -f logs/*.out
   Si échec     : le job fautif est dans logs/, les suivants
                  restent en attente (annuler : scancel --me)
   Résultats    : $WORK_ROOT/checkpoints/predictor_sim/
                  (best.pt + last.pt + history.json)
══════════════════════════════════════════════════════
EOF

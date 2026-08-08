# Environnement commun aux jobs Adastra — à sourcer en tête de chaque sbatch.
#
#   source "$SLURM_SUBMIT_DIR/slurm/env.sh"
#
# PRÉREQUIS (une fois, sur le nœud de login) : slurm/setup_offline.sh

# ── Chemins projet (adapter WORK_ROOT à ton espace scratch) ──────────────
export WORK_ROOT="${WORK_ROOT:-$SCRATCHDIR/lewm-robot}"
export VENV_DIR="${VENV_DIR:-$WORK_ROOT/venv}"

# ── Caches HuggingFace sur le scratch (PAS le home, quota + lenteur) ─────
export HF_HOME="$WORK_ROOT/hf_home"
export HF_LEROBOT_HOME="$WORK_ROOT/lerobot_home"

# ── MODE OFFLINE : les nœuds de calcul n'ont pas d'accès internet direct.
#    Tout doit être dans les caches ci-dessus AVANT de soumettre
#    (cf. setup_offline.sh). Le code échoue immédiatement avec un message
#    clair si quelque chose manque (00_check_env.py le vérifie avant).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Threads CPU par tâche (8 cœurs/GCD sur les nœuds MI250X) ─────────────
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ── Modules CINES ────────────────────────────────────────────────────────
# La pile logicielle exacte dépend de la session Adastra en cours ;
# le plus simple est le PyTorch ROCm du venv (installé par setup_offline.sh
# depuis l'index rocm). Si tu utilises les modules/containers CINES à la
# place, charger ici, p.ex. :
#   module purge
#   module load cpe/23.12 craype-accel-amd-gfx90a rocm
#
# NOTE : si le PyTorch du venv embarque ses propres libs ROCm (cas des
# wheels download.pytorch.org/whl/rocmX.Y), NE PAS charger le module rocm
# (doc CINES : conflit de libamdhip64).

# ── Venv ─────────────────────────────────────────────────────────────────
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[env.sh] ERREUR : venv absent ($VENV_DIR) — lancer slurm/setup_offline.sh" >&2
    exit 1
fi

# ── Sanity minimal (rapide, avant tout travail) ──────────────────────────
python - <<'EOF' || exit 1
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "CPU ONLY"
print(f"[env.sh] torch {torch.__version__} | GPU: {name}")
EOF

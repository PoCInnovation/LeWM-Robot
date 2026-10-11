#!/usr/bin/env bash
#
# Sonde de la machine GPU — aller-retour #1.
#
# Enchaine : activation de l'environnement conda, reinstallation du package,
# verification rapide des scenes, puis la sonde Isaac Sim. Produit deux fichiers
# et les regroupe dans une archive a renvoyer.
#
#   ./run_probe.sh
#
# Variables d'environnement acceptees :
#   CONDA_ENV       nom de l'environnement conda      (defaut : leisaac)
#   PROBE_SAMPLES   configurations echantillonnees    (defaut : 200)
#
# Options :
#   --skip-install  ne pas refaire le pip install -e
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CONDA_ENV="${CONDA_ENV:-leisaac}"
PROBE_SAMPLES="${PROBE_SAMPLES:-200}"
SKIP_INSTALL=0

for arg in "$@"; do
    case "$arg" in
        --skip-install) SKIP_INSTALL=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Option inconnue : $arg (voir --help)"; exit 2 ;;
    esac
done

OUT_DIR="outputs"
JSON="$OUT_DIR/isaac_probe.json"
LOG="$OUT_DIR/isaac_probe.log"
BUNDLE="$OUT_DIR/isaac_probe_bundle.tar.gz"

mkdir -p "$OUT_DIR"

step() { echo; echo "=========== $* ==========="; }
fail() { echo; echo "[ECHEC] $*"; echo; return 1; }


activate_conda() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
        echo "Environnement '$CONDA_ENV' deja actif."
        return 0
    fi

    # 'conda activate' n'est pas disponible dans un shell non interactif tant
    # que le hook n'a pas ete charge — d'ou cette gymnastique.
    if command -v conda >/dev/null 2>&1; then
        local hook
        if hook="$(conda shell.bash hook 2>/dev/null)"; then
            eval "$hook"
        fi
    else
        local base
        for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                    "$HOME/mambaforge" "/opt/conda" "/usr/local/conda"; do
            if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
                # shellcheck disable=SC1091
                source "$base/etc/profile.d/conda.sh"
                break
            fi
        done
    fi

    if ! command -v conda >/dev/null 2>&1; then
        fail "conda est introuvable. Activer l'environnement a la main, puis relancer :
    conda activate $CONDA_ENV
    ./run_probe.sh"
        return 1
    fi

    if conda activate "$CONDA_ENV" 2>/dev/null; then
        echo "Environnement '$CONDA_ENV' active."
        return 0
    fi

    # Le nom exact n'existe pas. Les environnements Isaac sont souvent nommes
    # avec un suffixe (leisaac_envhub, leisaac_2, ...) : si un seul correspond,
    # autant le prendre plutot que de renvoyer l'utilisateur a la doc.
    local candidates count
    candidates="$(conda env list 2>/dev/null \
        | awk '{print $1}' \
        | grep -v '^#' \
        | grep -v '^$' \
        | grep -i -- "$CONDA_ENV")"
    count="$(printf '%s\n' "$candidates" | grep -c . )"

    if [[ "$count" == "1" ]]; then
        local guess
        guess="$(printf '%s\n' "$candidates" | tr -d '[:space:]')"
        echo "'$CONDA_ENV' n'existe pas ; un seul environnement y ressemble."
        if conda activate "$guess" 2>/dev/null; then
            echo "Environnement '$guess' active."
            return 0
        fi
    fi

    echo "Environnements disponibles :"
    conda env list
    fail "impossible d'activer '$CONDA_ENV'. Relancer en donnant le bon nom :
    CONDA_ENV=<le_bon_nom> ./run_probe.sh"
    return 1
}


main() {
    echo "Sonde SO-101 — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Dossier : $SCRIPT_DIR"

    step "1/4  Environnement conda"
    activate_conda || return 1
    echo "python : $(command -v python)"
    python --version

    step "2/4  Installation du package"
    if [[ "$SKIP_INSTALL" == "1" ]]; then
        echo "Ignoree (--skip-install)."
    elif ! pip install -e source/sim_to_real_so101 --no-deps; then
        fail "le pip install a echoue."
        return 1
    fi

    step "3/4  Verification des scenes (sans Isaac, instantane)"
    if ! python -m sim_to_real_so101.utils.scene_validation; then
        fail "la verification des scenes a echoue — inutile de lancer Isaac.
Renvoyer cette sortie, le probleme est en amont."
        return 1
    fi

    step "4/4  Sonde Isaac Sim"
    echo "Isaac compile ses shaders au premier lancement : compter plusieurs"
    echo "minutes. Ne pas interrompre meme si l'affichage semble fige."
    echo
    # Chaque section de la sonde est isolee : meme en cas d'erreur partielle,
    # le JSON est ecrit et reste exploitable.
    python -m sim_to_real_so101.scripts.isaac_probe \
        --out "$JSON" \
        --samples "$PROBE_SAMPLES"
}


main 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

echo
echo "======================================================================="
if [[ -f "$JSON" ]]; then
    if tar czf "$BUNDLE" -C "$OUT_DIR" \
            "$(basename "$JSON")" "$(basename "$LOG")" 2>/dev/null; then
        echo "A RENVOYER — un seul fichier :"
        echo "    $SCRIPT_DIR/$BUNDLE"
        echo
        echo "Il contient le rapport JSON et le journal complet."
    else
        echo "A RENVOYER — ces deux fichiers :"
        echo "    $SCRIPT_DIR/$JSON"
        echo "    $SCRIPT_DIR/$LOG"
    fi
    if [[ "$STATUS" != "0" ]]; then
        echo
        echo "Note : le script s'est termine en erreur, mais le rapport a bien"
        echo "ete ecrit. Le renvoyer quand meme, l'erreur est l'information utile."
    fi
else
    echo "Aucun rapport n'a ete produit."
    echo "A RENVOYER — le journal, qui contient la cause :"
    echo "    $SCRIPT_DIR/$LOG"
fi
echo "======================================================================="

exit "$STATUS"

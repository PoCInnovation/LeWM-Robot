"""
Validation d'environnement AVANT de lancer un job (SLURM ou local).

À lancer en tête de chaque job sbatch : échoue immédiatement (exit != 0) si
un prérequis critique manque, au lieu de brûler des heures d'allocation sur
un plantage à la 30e minute.

Vérifie :
    1. torch + device (sur ROCm, l'API HIP se présente comme "cuda")
    2. bf16 fonctionnel sur le device
    3. Flags offline (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE) et HF_HOME
    4. Modèle encodeur présent dans le cache HF local (SANS réseau)
    5. Dataset LeRobot accessible (metadata seulement, rapide)
    6. Répertoire de sortie inscriptible

Usage:
    python scripts/00_check_env.py [--config configs/default.yaml]
                                    [--dataset-id <hf_id | chemin local>]
                                    [--skip-dataset] [--skip-model]
                                    [--expect-gpu] [--expect-offline]
"""

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = []          # liste de (niveau, nom, message)
HAS_CRITICAL = False


def report(ok: bool, name: str, message: str, critical: bool = True):
    global HAS_CRITICAL
    if ok:
        status = "OK  "
    elif critical:
        status = "FAIL"
        HAS_CRITICAL = True
    else:
        status = "WARN"
    RESULTS.append((status, name, message))
    print(f"  [{status}] {name:<22} {message}")


def check_torch(expect_gpu: bool):
    print("\n--- 1. PyTorch & device ---")
    try:
        import torch
    except ImportError as e:
        report(False, "torch", f"import impossible : {e}")
        return None

    report(True, "torch", f"version {torch.__version__}")

    cuda = torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        is_amd = "AMD" in name or "MI" in name or getattr(torch.version, "hip", None)
        report(True, "device",
               f"{name} ({vram:.0f} GB) "
               f"[{'ROCm/HIP' if is_amd else 'CUDA'}]")
    else:
        report(not expect_gpu, "device",
               "aucun GPU visible" + (" — attendu !" if expect_gpu else " (mode CPU)"),
               critical=expect_gpu)
    return torch


def check_bf16(torch, expect_gpu: bool):
    print("\n--- 2. bf16 ---")
    if torch is None:
        report(False, "bf16", "sauté (torch manquant)")
        return
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(64, 64, device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = a @ a
        ok = out.dtype == torch.bfloat16 or device == "cpu"
        report(True, "bf16", f"autocast fonctionnel sur {device} (out={out.dtype})")
    except Exception as e:
        report(not expect_gpu, "bf16", f"échec : {e}", critical=expect_gpu)


def check_offline_env(expect_offline: bool):
    print("\n--- 3. Environnement offline ---")
    hub_off = os.environ.get("HF_HUB_OFFLINE", "")
    tr_off = os.environ.get("TRANSFORMERS_OFFLINE", "")
    hf_home = os.environ.get("HF_HOME", "")
    lerobot_home = os.environ.get("HF_LEROBOT_HOME", "")

    offline = hub_off.strip().lower() in ("1", "true", "yes")
    if expect_offline:
        report(offline, "HF_HUB_OFFLINE",
               f"'{hub_off or '<non posé>'}'" +
               ("" if offline else " — doit être 1 sur les nœuds de calcul !"))
        report(bool(tr_off), "TRANSFORMERS_OFFLINE",
               f"'{tr_off or '<non posé>'}'", critical=False)
    else:
        report(True, "HF_HUB_OFFLINE", f"'{hub_off or '<non posé>'}' (offline non requis)")

    report(True, "HF_HOME", hf_home or "<défaut ~/.cache/huggingface>", critical=False)
    report(True, "HF_LEROBOT_HOME", lerobot_home or "<défaut>", critical=False)


def check_model_cache(cfg):
    print("\n--- 4. Modèle encodeur dans le cache HF ---")
    from src.encoders import DINOv3Config

    enc = DINOv3Config(
        family=cfg["encoder"].get("family", "dinov3"),
        size=cfg["encoder"]["size"],
    )
    model_id = enc.model_id
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id, local_files_only=True)
        report(True, "modèle", f"{model_id} → {path}")
    except Exception:
        report(False, "modèle",
               f"{model_id} ABSENT du cache local.\n"
               f"         Sur une machine avec internet :\n"
               f"           huggingface-cli download {model_id}\n"
               f"         (DINOv3 est gated : token HF + accès Meta requis)")


def check_dataset(cfg, dataset_id: str | None):
    print("\n--- 5. Dataset LeRobot ---")
    ds_id = dataset_id or cfg["dataset"]["hf_id"]
    if not ds_id:
        report(False, "dataset", "dataset.hf_id non défini dans la config")
        return
    try:
        from src.data import open_lerobot_dataset
        ds = open_lerobot_dataset(ds_id, cfg["dataset"].get("cache_dir"))
        report(True, "dataset",
               f"{ds_id} — {ds.num_episodes} episodes, {ds.num_frames} frames")
    except Exception as e:
        # open_lerobot_dataset fournit déjà un message actionnable
        first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
        report(False, "dataset", f"{ds_id} inaccessible : {first_line}\n"
               f"         (détails complets en relançant sans ce check wrapper)")


def check_output_dirs(cfg):
    print("\n--- 6. Répertoires de sortie ---")
    for key in ("encoded_data",):
        raw = cfg.get("paths", {}).get(key)
        if not raw:
            continue
        parent = (ROOT / raw).parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            probe = parent / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            report(True, f"paths.{key}", f"{parent} inscriptible")
        except OSError as e:
            report(False, f"paths.{key}", f"{parent} NON inscriptible : {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-id", default=None,
                        help="Surcharge dataset.hf_id (hf_id ou chemin local)")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Ne pas vérifier le dataset (jobs training purs)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Ne pas vérifier le modèle (jobs training purs, "
                             "qui lisent des latents pré-encodés)")
    parser.add_argument("--expect-gpu", action="store_true",
                        help="Échouer si aucun GPU n'est visible")
    parser.add_argument("--expect-offline", action="store_true",
                        help="Échouer si HF_HUB_OFFLINE n'est pas posé "
                             "(à utiliser dans les jobs SLURM)")
    args = parser.parse_args()

    print("=" * 64)
    print("CHECK ENVIRONNEMENT — pré-job")
    print("=" * 64)

    from src.config import load_config
    cfg = load_config(ROOT / args.config)

    torch = check_torch(args.expect_gpu)
    check_bf16(torch, args.expect_gpu)
    check_offline_env(args.expect_offline)
    if not args.skip_model:
        check_model_cache(cfg)
    if not args.skip_dataset:
        check_dataset(cfg, args.dataset_id)
    check_output_dirs(cfg)

    print("\n" + "=" * 64)
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
    if HAS_CRITICAL:
        print(f"RÉSULTAT : {n_fail} FAIL, {n_warn} WARN — NE PAS lancer le job.")
        sys.exit(1)
    print(f"RÉSULTAT : tout OK ({n_warn} warnings) — environnement prêt.")
    sys.exit(0)


if __name__ == "__main__":
    main()

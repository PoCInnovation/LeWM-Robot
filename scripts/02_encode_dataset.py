"""
Encode tes démos LeRobot HF avec DINOv3 et sauvegarde les patches sur disque.

Pourquoi pré-encoder : une fois les latents sur disque, les expériences
suivantes (fusions, predictor, LoRA) prennent quelques secondes/minutes.

GPU (RTX 4090) :
    - encodeur en bf16 + SDPA, batch 64 par défaut : le goulot devient le
      DÉCODAGE VIDÉO côté CPU → --num-workers auto (= cœurs - 2, max 8),
      pin_memory + prefetch pour recouvrir le transfert PCIe ;
    - le preprocessing tourne sur le GPU (images transférées brutes).
    Ordre de grandeur : ~1-2 min pour 8 000 images en DINOv3-small (CPU :
    ~2 h).

Usage:
    python scripts/02_encode_dataset.py [--config configs/default.yaml]
                                         [--dataset-id user/so101_pick_drop_duck]
                                         [--size small] [--batch-size 64]
                                         [--num-workers auto] [--output PATH]
                                         [--max-pairs N]

Surcharge la config YAML avec les arguments CLI si fournis.
"""

import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.encoders import DINOv3Config, DINOv3Encoder
from src.data import LeRobotDataConfig, LeRobotPairsDataset
from src.config import load_config, set_seed, log_environment, setup_hardware
from src.device import resolve_num_workers, peak_vram_gb, reset_peak_vram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Chemin vers le YAML de config")
    parser.add_argument("--dataset-id", default=None,
                        help="Surcharge dataset.hf_id de la config")
    parser.add_argument("--size", default=None,
                        choices=["small", "base", "large", "giant"],
                        help="Surcharge encoder.size de la config")
    parser.add_argument("--dtype", default=None,
                        help="dtype des poids de l'encodeur : auto (bf16 sur "
                             "GPU) / bfloat16 / float16 / float32. "
                             "Défaut : encoder.dtype")
    parser.add_argument("--output", default=None,
                        help="Fichier .pt de sortie (défaut : paths.encoded_data)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Défaut : 64 sur GPU, 8 sur CPU")
    parser.add_argument("--num-workers", default=None,
                        help="Workers DataLoader (décodage vidéo = goulot). "
                             "Défaut : hardware.num_workers ('auto')")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limite le nombre de paires (pour test rapide). "
                             "None = toutes les paires.")
    args = parser.parse_args()

    # Charger config
    cfg = load_config(ROOT / args.config)
    if args.dataset_id:
        cfg["dataset"]["hf_id"] = args.dataset_id
    if args.size:
        cfg["encoder"]["size"] = args.size

    if not cfg["dataset"]["hf_id"]:
        print("[ERREUR] dataset.hf_id n'est pas défini.")
        print("Solution 1 : edit configs/default.yaml et set dataset.hf_id")
        print("Solution 2 : python scripts/02_encode_dataset.py --dataset-id <id>")
        sys.exit(1)

    set_seed(cfg["seed"])
    hw = setup_hardware(cfg)
    log_environment()
    on_gpu = torch.cuda.is_available()
    batch_size = args.batch_size or (64 if on_gpu else 8)
    num_workers = resolve_num_workers(
        args.num_workers if args.num_workers is not None else hw["num_workers"])

    family = cfg["encoder"].get("family", "dinov3")
    print(f"\nDataset      : {cfg['dataset']['hf_id']}")
    print(f"Encoder      : {family}-{cfg['encoder']['size']}")
    print(f"Batch size   : {batch_size}  |  Workers : {num_workers}\n")

    # === Setup ===
    output_path = Path(args.output) if args.output else ROOT / cfg["paths"]["encoded_data"]
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enc_cfg = DINOv3Config(
        family=family,
        size=cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
        dtype=args.dtype or cfg["encoder"].get("dtype", "auto"),
        attn_implementation=cfg["encoder"].get("attn_implementation", "sdpa"),
    )
    reset_peak_vram()
    encoder = DINOv3Encoder(enc_cfg)

    data_cfg = LeRobotDataConfig(
        dataset_id=cfg["dataset"]["hf_id"],
        wrist_key=cfg["dataset"]["wrist_key"],
        global_key=cfg["dataset"]["global_key"],
        action_key=cfg["dataset"]["action_key"],
        proprio_key=cfg["dataset"]["proprio_key"],
        delta_timesteps=cfg["dataset"]["delta_timesteps"],
        cache_dir=cfg["dataset"]["cache_dir"],
    )
    dataset = LeRobotPairsDataset(data_cfg)
    if args.max_pairs is not None and args.max_pairs < len(dataset):
        from torch.utils.data import Subset
        # Subsample en gardant des paires de plusieurs épisodes (stride)
        step = len(dataset) // args.max_pairs
        dataset = Subset(dataset, list(range(0, len(dataset), step))[:args.max_pairs])
        print(f"[MAX-PAIRS] Sous-échantillonnage : {len(dataset)} paires (stride={step})")

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         shuffle=False, pin_memory=on_gpu)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(dataset, **loader_kwargs)

    print(f"Total paires : {len(dataset)}")
    print(f"Batches      : {len(loader)}\n")

    # === Encode ===
    buf_z_wrist_t, buf_z_global_t = [], []
    buf_z_wrist_t1, buf_z_global_t1 = [], []
    buf_action, buf_proprio = [], []
    buf_episode, buf_frame = [], []

    t0 = time.time()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Encoding"):
            buf_z_wrist_t.append(encoder.encode(batch["wrist_t"]).cpu())
            buf_z_global_t.append(encoder.encode(batch["global_t"]).cpu())
            buf_z_wrist_t1.append(encoder.encode(batch["wrist_t1"]).cpu())
            buf_z_global_t1.append(encoder.encode(batch["global_t1"]).cpu())

            buf_action.append(batch["action"])
            buf_proprio.append(batch["proprio"])
            buf_episode.append(batch["episode_idx"])
            buf_frame.append(batch["frame_idx"])

    dt = time.time() - t0
    n_images = 4 * len(dataset)
    print(f"\nEncodage terminé en {dt:.1f}s "
          f"({dt / len(dataset) * 1000:.1f} ms/paire, {n_images / dt:.0f} img/s)")
    if on_gpu:
        print(f"VRAM pic : {peak_vram_gb():.2f} GB  "
              f"(si le GPU est loin de 100 % dans nvidia-smi, augmenter "
              f"--num-workers : le décodage vidéo est le goulot)")

    # === Save ===
    torch.save(
        {
            "z_wrist_t":    torch.cat(buf_z_wrist_t),
            "z_global_t":   torch.cat(buf_z_global_t),
            "z_wrist_t1":   torch.cat(buf_z_wrist_t1),
            "z_global_t1":  torch.cat(buf_z_global_t1),
            "action":       torch.cat(buf_action),
            "proprio":      torch.cat(buf_proprio),
            "episode_idx":  torch.cat(buf_episode),
            "frame_idx":    torch.cat(buf_frame),
            "embed_dim":    encoder.embed_dim,
            "num_patches":  encoder.num_patches,
            "encoder_family": family,
            "encoder_size": cfg["encoder"]["size"],
            "encoder_dtype": str(encoder.dtype_).replace("torch.", ""),
            "dataset_id":   cfg["dataset"]["hf_id"],
            "seed":         cfg["seed"],
        },
        output_path,
    )
    print(f"Sauvé : {output_path}")
    print(f"\nProchaine étape : python scripts/03_compare_fusion.py "
          f"--encoded-data {output_path}")


if __name__ == "__main__":
    main()

"""
Encode tes 40 démos LeRobot HF avec DINOv3 et sauvegarde les patches sur disque.

Pourquoi pré-encoder : DINOv3 sur CPU c'est ~1s/image, donc 40 démos × ~100 frames
× 2 caméras = 8000 images = ~2h sur CPU. Une fois pré-encodé, les expériences
suivantes prennent quelques secondes.

Usage:
    python scripts/02_encode_dataset.py [--config configs/default.yaml]
                                         [--dataset-id user/so101_pick_drop_duck]
                                         [--size small]

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
from src.config import load_config, set_seed, log_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Chemin vers le YAML de config")
    parser.add_argument("--dataset-id", default=None,
                        help="Surcharge dataset.hf_id de la config")
    parser.add_argument("--size", default=None,
                        choices=["small", "base", "large", "giant"],
                        help="Surcharge encoder.size de la config")
    parser.add_argument("--batch-size", type=int, default=8)
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
    log_environment()

    print(f"\nDataset      : {cfg['dataset']['hf_id']}")
    print(f"Encoder size : DINOv3-{cfg['encoder']['size']}")
    print(f"Batch size   : {args.batch_size}\n")

    # === Setup ===
    output_dir = ROOT / cfg["paths"]["encoded_data"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    enc_cfg = DINOv3Config(
        family=cfg["encoder"].get("family", "dinov3"),
        size=cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
    )
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
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    print(f"Total paires : {len(dataset)}")
    print(f"Batches      : {len(loader)}\n")

    # === Encode ===
    buf_z_wrist_t, buf_z_global_t = [], []
    buf_z_wrist_t1, buf_z_global_t1 = [], []
    buf_action, buf_proprio = [], []
    buf_episode, buf_frame = [], []

    t0 = time.time()
    for batch in tqdm(loader, desc="Encoding"):
        with torch.no_grad():
            buf_z_wrist_t.append(encoder.encode(batch["wrist_t"]).cpu())
            buf_z_global_t.append(encoder.encode(batch["global_t"]).cpu())
            buf_z_wrist_t1.append(encoder.encode(batch["wrist_t1"]).cpu())
            buf_z_global_t1.append(encoder.encode(batch["global_t1"]).cpu())

        buf_action.append(batch["action"])
        buf_proprio.append(batch["proprio"])
        buf_episode.append(batch["episode_idx"])
        buf_frame.append(batch["frame_idx"])

    dt = time.time() - t0
    print(f"\nEncodage terminé en {dt:.1f}s "
          f"({dt / len(dataset) * 1000:.1f} ms/paire)")

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
            "encoder_size": cfg["encoder"]["size"],
            "dataset_id":   cfg["dataset"]["hf_id"],
            "seed":         cfg["seed"],
        },
        output_dir,
    )
    print(f"Sauvé : {output_dir}")
    print("\nProchaine étape : python scripts/03_compare_fusion.py")


if __name__ == "__main__":
    main()

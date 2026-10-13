"""
Encode tes démos LeRobot HF avec DINOv3 et sauvegarde les patches sur disque.

Pourquoi pré-encoder : une fois les latents sur disque, les expériences
suivantes (fusions, predictor, LoRA) prennent quelques secondes/minutes.

GPU (RTX 4090 24 GB / 5090 32 GB) :
    - encodeur en bf16 + SDPA, batch 64 par défaut : le goulot devient le
      DÉCODAGE VIDÉO côté CPU → --num-workers auto (= cœurs - 2, max 8),
      pin_memory + prefetch pour recouvrir le transfert PCIe ;
    - images transférées en uint8, preprocessing sur le GPU.

Mémoire (format v2, cf. src/latents.py) :
    - passe propre sur les FRAMES uniques : la frame t+1 d'une paire est la
      frame t de la suivante, elle n'est décodée/encodée/stockée qu'une fois ;
    - latents stockés en bf16 (sortie native de l'encodeur bf16, sans perte) ;
    - tables préallouées en VRAM si elles tiennent : la RAM ne porte ni les
      latents ni les copies intermédiaires. Dataset complet (17 478 paires,
      1 passe augmentée) : ~16 GB au lieu de 42 GB.

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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.encoders import DINOv3Config, DINOv3Encoder
from src.data import LeRobotDataConfig, LeRobotPairsDataset, LeRobotFramesDataset
from src.latents import FORMAT_VERSION, build_frame_index, latent_nbytes
from src.augmentation import OnlineAugConfig
from src.config import load_config, set_seed, log_environment, setup_hardware
from src.device import (resolve_num_workers, peak_vram_gb, reset_peak_vram,
                        choose_data_device)


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
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Garde N épisodes entiers tirés au hasard (seed de la "
                             "config) : frames consécutives conservées (08) et "
                             "toutes les tâches représentées. Pour un gros dataset "
                             "(simulation) dont les latents ne tiendraient pas en VRAM.")
    parser.add_argument("--aug-passes", type=int, default=None,
                        help="Passes AUGMENTÉES en plus de la passe propre. "
                             "Chaque passe re-tire des paramètres au hasard, "
                             "donc produit des variations différentes. "
                             "Défaut : augmentation.passes de la config.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Force l'encodage sans augmentation (une seule "
                             "passe propre), quelle que soit la config.")
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

    aug_cfg = OnlineAugConfig.from_dict(cfg.get("augmentation"), seed=cfg["seed"])
    if args.aug_passes is not None:
        aug_cfg.passes = max(0, args.aug_passes)
    if args.no_augment:
        aug_cfg.enabled = False
    augment_on = aug_cfg.enabled and aug_cfg.passes > 0
    n_passes = 1 + (aug_cfg.passes if augment_on else 0)
    print(aug_cfg.describe())
    if not augment_on:
        print("[Augmentation] désactivée — une seule passe propre.")
    print()

    data_cfg = LeRobotDataConfig(
        dataset_id=cfg["dataset"]["hf_id"],
        wrist_key=cfg["dataset"]["wrist_key"],
        global_key=cfg["dataset"]["global_key"],
        action_key=cfg["dataset"]["action_key"],
        proprio_key=cfg["dataset"]["proprio_key"],
        delta_timesteps=cfg["dataset"]["delta_timesteps"],
        cache_dir=cfg["dataset"]["cache_dir"],
        augment=aug_cfg,
        uint8_images=True,
        video_backend=cfg["dataset"].get("video_backend"),
    )
    base_dataset = LeRobotPairsDataset(data_cfg)
    dataset = base_dataset
    if args.max_episodes is not None:
        episodes = sorted({base_dataset.pair_episode(i) for i in range(len(base_dataset))})
        if args.max_episodes < len(episodes):
            g = torch.Generator().manual_seed(cfg["seed"])
            keep = {episodes[j] for j in
                    torch.randperm(len(episodes), generator=g)[:args.max_episodes].tolist()}
            dataset = Subset(base_dataset, [i for i in range(len(base_dataset))
                                            if base_dataset.pair_episode(i) in keep])
            print(f"[MAX-EPISODES] {len(keep)} épisodes sur {len(episodes)} "
                  f"→ {len(dataset)} paires")
    if args.max_pairs is not None and args.max_pairs < len(dataset):
        # Subsample en gardant des paires de plusieurs épisodes (stride)
        step = len(dataset) // args.max_pairs
        dataset = Subset(dataset, list(range(0, len(dataset), step))[:args.max_pairs])
        print(f"[MAX-PAIRS] Sous-échantillonnage : {len(dataset)} paires (stride={step})")

    # === Paires retenues → frames uniques de la passe propre ===
    pair_ids = (list(dataset.indices) if isinstance(dataset, Subset)
                else list(range(len(dataset))))
    delta = data_cfg.delta_timesteps
    pair_frames = [base_dataset.pair_frame(i) for i in pair_ids]
    frames, clean_idx_t, clean_idx_t1 = build_frame_index(pair_frames, delta)
    n_pairs = len(pair_ids)
    n_aug_passes = n_passes - 1
    n_frame_rows = len(frames) + n_aug_passes * 2 * n_pairs

    latent_dtype = (encoder.dtype_ if encoder.dtype_ in (torch.bfloat16, torch.float16)
                    else torch.float32)
    nbytes = latent_nbytes(n_frame_rows, encoder.num_patches, encoder.embed_dim,
                           latent_dtype)
    # Tables en VRAM si elles tiennent (réserve pour l'encodeur et les batchs) :
    # pas de copie GPU→CPU par batch, et la RAM ne porte pas les latents.
    table_device = choose_data_device(nbytes, hw["data_device"],
                                      "cuda" if on_gpu else "cpu", reserve_gb=4.0)

    print(f"Total paires : {n_pairs}")
    print(f"Passes       : {n_passes} (1 propre"
          f"{f' + {n_aug_passes} augmentées' if n_aug_passes else ''})")
    print(f"Frames       : {len(frames)} uniques (passe propre) + "
          f"{n_aug_passes * 2 * n_pairs} augmentées = {n_frame_rows} lignes")
    print(f"Latents      : {nbytes / 1e9:.1f} GB "
          f"(2 caméras x {encoder.num_patches} patches x {encoder.embed_dim} dims x "
          f"{str(latent_dtype).replace('torch.', '')}), tables sur {table_device}\n")

    shape = (n_frame_rows, encoder.num_patches, encoder.embed_dim)
    z_wrist = torch.empty(shape, dtype=latent_dtype, device=table_device)
    z_global = torch.empty(shape, dtype=latent_dtype, device=table_device)

    def encode_into(table, row, images):
        z = encoder.encode(images, out_dtype=latent_dtype)
        table[row:row + z.shape[0]] = z.to(table_device)
        return row + z.shape[0]

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         shuffle=False, pin_memory=on_gpu)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=False, prefetch_factor=2)

    meta = {k: [] for k in ("action", "proprio", "episode_idx", "frame_idx",
                            "pair_id", "is_augmented", "recipe_idx",
                            "idx_t", "idx_t1")}

    t0 = time.time()
    with torch.inference_mode():
        # ── Passe propre : chaque frame une seule fois ──
        frame_action, frame_proprio = [], []
        row = 0
        loader = DataLoader(LeRobotFramesDataset(base_dataset, frames), **loader_kwargs)
        for batch in tqdm(loader, desc="Encoding (propre, frames uniques)"):
            encode_into(z_wrist, row, batch["wrist"])
            row = encode_into(z_global, row, batch["global"])
            frame_action.append(batch["action"])
            frame_proprio.append(batch["proprio"])
        del loader
        frame_action = torch.cat(frame_action)
        frame_proprio = torch.cat(frame_proprio)

        meta["action"].append(frame_action[clean_idx_t])
        meta["proprio"].append(frame_proprio[clean_idx_t])
        meta["episode_idx"].append(torch.tensor(
            [base_dataset._pairs_index[i][0] for i in pair_ids], dtype=torch.long))
        meta["frame_idx"].append(torch.tensor(pair_frames, dtype=torch.long))
        meta["pair_id"].append(torch.tensor(pair_ids, dtype=torch.long))
        meta["is_augmented"].append(torch.zeros(n_pairs, dtype=torch.long))
        meta["recipe_idx"].append(torch.zeros(n_pairs, dtype=torch.long))
        meta["idx_t"].append(clean_idx_t)
        meta["idx_t1"].append(clean_idx_t1)

        # ── Passes augmentées : paramètres tirés par paire → 2 frames par paire ──
        base_dataset.set_augment(True)
        for pass_idx in range(1, n_passes):
            loader = DataLoader(dataset, **loader_kwargs)
            for batch in tqdm(loader, desc=f"Encoding (augmentée {pass_idx}/{n_aug_passes})"):
                b = batch["action"].shape[0]
                encode_into(z_wrist, row, torch.cat([batch["wrist_t"], batch["wrist_t1"]]))
                encode_into(z_global, row, torch.cat([batch["global_t"], batch["global_t1"]]))
                meta["idx_t"].append(torch.arange(row, row + b))
                meta["idx_t1"].append(torch.arange(row + b, row + 2 * b))
                row += 2 * b
                for key in ("action", "proprio"):
                    meta[key].append(batch[key])
                meta["episode_idx"].append(batch["episode_idx"])
                meta["frame_idx"].append(batch["frame_idx"])
                meta["pair_id"].append(batch["pair_id"])
                meta["is_augmented"].append(batch["is_augmented"])
                meta["recipe_idx"].append(batch["recipe_idx"])
            del loader
    assert row == n_frame_rows, (row, n_frame_rows)

    dt = time.time() - t0
    n_images = 2 * n_frame_rows
    n_rows = n_pairs * n_passes
    print(f"\nEncodage terminé en {dt:.1f}s "
          f"({dt / max(1, n_rows) * 1000:.1f} ms/paire, {n_images / dt:.0f} img/s)")
    if on_gpu:
        print(f"VRAM pic : {peak_vram_gb():.2f} GB  "
              f"(si le GPU est loin de 100 % dans nvidia-smi, augmenter "
              f"--num-workers : le décodage vidéo est le goulot)")

    # === Save ===
    # torch.save copie les tables GPU vers la RAM une par une (pic RAM = une
    # table, ~8 GB sur le dataset complet), pas les deux en même temps.
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "z_wrist":      z_wrist,
            "z_global":     z_global,
            **{k: torch.cat(v) for k, v in meta.items()},
            "aug_passes":   n_aug_passes,
            "aug_config":   aug_cfg.__dict__ if augment_on else None,
            "embed_dim":    encoder.embed_dim,
            "num_patches":  encoder.num_patches,
            "latent_dtype": str(latent_dtype).replace("torch.", ""),
            "delta_timesteps": delta,
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

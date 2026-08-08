"""
Pré-encode un dataset LeRobot avec DINOv3 → séquences par épisode (format v2).

Différences avec l'ancien script (paires en vrac dans un seul .pt) :
    - chaque frame est encodée UNE seule fois (l'ancien encodait tout 2x) ;
    - sortie par épisode (ep_XXXXX.pt) → delta/multi-step choisis au training ;
    - écriture incrémentale (RAM bornée à ~1 épisode) ;
    - stockage fp16 par défaut (2x plus petit, aucun impact mesurable) ;
    - shardable en job array SLURM : --shard-index i --num-shards N ;
    - stats de normalisation action/proprio calculées et sauvées (meta.json).

Usage:
    # Mono-process (merge automatique) :
    python scripts/02_encode_dataset.py [--config configs/default.yaml]
        [--dataset-id user/dataset|/chemin/local] [--size small]
        [--output results/encoded/mon_dataset] [--batch-size 8]
        [--num-workers 4] [--max-episodes N] [--fp32]

    # Job array SLURM (N shards en parallèle, puis merge) :
    python scripts/02_encode_dataset.py --shard-index $SLURM_ARRAY_TASK_ID \\
                                         --num-shards N ...
    python scripts/02_encode_dataset.py --output <dir> --finalize
"""

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.encoders import DINOv3Config, DINOv3Encoder
from src.data import LeRobotDataConfig, LeRobotFramesDataset
from src.encoded_data import partial_stats, merge_shard_metas
from src.config import load_config, set_seed, log_environment


def default_output_dir(cfg, dataset_id: str) -> Path:
    name = Path(dataset_id).name.replace("/", "_")
    fam = cfg["encoder"].get("family", "dinov3")
    size = cfg["encoder"]["size"]
    return ROOT / "results" / "encoded" / f"{name}_{fam}-{size}"


def save_episode(output_dir: Path, ep_idx: int, z_wrist: torch.Tensor,
                 z_global: torch.Tensor, action: torch.Tensor,
                 proprio: torch.Tensor, frame_start: int,
                 storage_dtype: torch.dtype) -> dict:
    """Écrit un épisode sur disque, renvoie son entrée pour le meta."""
    path = output_dir / f"ep_{ep_idx:05d}.pt"
    torch.save({
        "episode_idx": ep_idx,
        "z_wrist":  z_wrist.to(storage_dtype),
        "z_global": z_global.to(storage_dtype),
        "action":   action.float(),
        "proprio":  proprio.float(),
        "frame_start": frame_start,
    }, path)
    return {"idx": ep_idx, "length": int(action.shape[0]), "file": path.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-id", default=None,
                        help="Surcharge dataset.hf_id (HF id ou CHEMIN LOCAL — "
                             "obligatoire en mode offline)")
    parser.add_argument("--size", default=None,
                        choices=["small", "base", "large", "giant"])
    parser.add_argument("--output", default=None,
                        help="Dossier de sortie (défaut: results/encoded/"
                             "<dataset>_<family>-<size>)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Workers du DataLoader (décodage vidéo = goulot)")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limite le nombre d'épisodes (test rapide)")
    parser.add_argument("--max-frames-per-episode", type=int, default=None,
                        help="Tronque chaque épisode (smoke test uniquement)")
    parser.add_argument("--fp32", action="store_true",
                        help="Stocker les latents en float32 (défaut: float16)")
    parser.add_argument("--finalize", action="store_true",
                        help="Ne rien encoder : fusionner les shard_meta_*.json "
                             "de --output en meta.json")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.dataset_id:
        cfg["dataset"]["hf_id"] = args.dataset_id
    if args.size:
        cfg["encoder"]["size"] = args.size

    dataset_id = cfg["dataset"]["hf_id"]
    output_dir = Path(args.output) if args.output else default_output_dir(cfg, dataset_id)

    # ── Mode merge seul ──
    if args.finalize:
        merge_shard_metas(output_dir)
        return

    if not dataset_id:
        print("[ERREUR] dataset.hf_id non défini (config ou --dataset-id).")
        sys.exit(1)
    if not (0 <= args.shard_index < args.num_shards):
        print(f"[ERREUR] --shard-index {args.shard_index} hors [0, {args.num_shards}).")
        sys.exit(1)

    set_seed(cfg["seed"])
    log_environment()
    storage_dtype = torch.float32 if args.fp32 else torch.float16

    print(f"\nDataset      : {dataset_id}")
    print(f"Encoder      : {cfg['encoder'].get('family', 'dinov3')}-{cfg['encoder']['size']}")
    print(f"Output       : {output_dir}")
    print(f"Shard        : {args.shard_index}/{args.num_shards}")
    print(f"Batch size   : {args.batch_size}  |  Workers : {args.num_workers}")
    print(f"Storage      : {storage_dtype}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # === Dataset frames (1 frame par item — encodage one-pass) ===
    data_cfg = LeRobotDataConfig(
        dataset_id=dataset_id,
        wrist_key=cfg["dataset"]["wrist_key"],
        global_key=cfg["dataset"]["global_key"],
        action_key=cfg["dataset"]["action_key"],
        proprio_key=cfg["dataset"]["proprio_key"],
        cache_dir=cfg["dataset"]["cache_dir"],
    )
    frames_ds = LeRobotFramesDataset(data_cfg)

    episodes = frames_ds.episodes                     # [(ep_idx, start, length)]
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    if args.max_frames_per_episode is not None:
        episodes = [(e, s, min(l, args.max_frames_per_episode))
                    for (e, s, l) in episodes]
        print(f"[SMOKE TEST] Épisodes tronqués à "
              f"{args.max_frames_per_episode} frames.")

    # Attribution round-robin des épisodes à ce shard (équilibre les longueurs)
    my_episodes = episodes[args.shard_index::args.num_shards]
    if not my_episodes:
        print(f"[shard {args.shard_index}] Aucun épisode assigné, rien à faire.")
        # Écrire un shard_meta vide pour que le merge ne bloque pas
        my_episodes = []

    print(f"[shard {args.shard_index}] {len(my_episodes)} épisodes assignés "
          f"sur {len(episodes)}.")

    # === Encoder (chargé APRÈS le check dataset : fail-fast dans le bon ordre) ===
    enc_cfg = DINOv3Config(
        family=cfg["encoder"].get("family", "dinov3"),
        size=cfg["encoder"]["size"],
        image_size=cfg["encoder"]["image_size"],
    )
    encoder = DINOv3Encoder(enc_cfg)

    # === Boucle d'encodage : DataLoader séquentiel sur les frames du shard,
    #     flush d'un épisode dès que toutes ses frames sont passées ===
    frame_indices = [start + i
                     for (_, start, length) in my_episodes
                     for i in range(length)]
    loader = DataLoader(
        Subset(frames_ds, frame_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    ep_meta_entries = []
    action_parts, proprio_parts = [], []
    cur = {"ep": None, "zw": [], "zg": [], "act": [], "prop": [], "start": None}

    def flush():
        if cur["ep"] is None:
            return
        zw = torch.cat(cur["zw"])
        zg = torch.cat(cur["zg"])
        act = torch.cat(cur["act"])
        prop = torch.cat(cur["prop"])
        entry = save_episode(output_dir, cur["ep"], zw, zg, act, prop,
                             cur["start"], storage_dtype)
        ep_meta_entries.append(entry)
        action_parts.append(partial_stats(act))
        proprio_parts.append(partial_stats(prop))
        cur.update({"ep": None, "zw": [], "zg": [], "act": [], "prop": [],
                    "start": None})

    t0 = time.time()
    n_frames = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding shard {args.shard_index}"):
            z_wrist = encoder.encode(batch["wrist"]).cpu()
            z_global = encoder.encode(batch["global"]).cpu()
            eps = batch["episode_idx"].tolist()
            frs = batch["frame_idx"].tolist()

            # Découper le batch aux frontières d'épisodes
            b0 = 0
            for b in range(1, len(eps) + 1):
                if b == len(eps) or eps[b] != eps[b0]:
                    if cur["ep"] is not None and cur["ep"] != eps[b0]:
                        flush()
                    if cur["ep"] is None:
                        cur["ep"] = eps[b0]
                        cur["start"] = frs[b0]
                    cur["zw"].append(z_wrist[b0:b])
                    cur["zg"].append(z_global[b0:b])
                    cur["act"].append(batch["action"][b0:b])
                    cur["prop"].append(batch["proprio"][b0:b])
                    b0 = b
            n_frames += len(eps)
    flush()

    dt = time.time() - t0
    if n_frames:
        print(f"\nEncodage terminé : {n_frames} frames en {dt:.1f}s "
              f"({dt / n_frames * 1000:.1f} ms/frame — 1 seul passage par frame)")

    # === Shard meta (stats partielles + inventaire) ===
    fps = getattr(getattr(frames_ds._dataset, "meta", None), "fps", None) \
        or getattr(frames_ds._dataset, "fps", None)
    shard_meta = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "embed_dim": encoder.embed_dim,
        "num_patches": encoder.num_patches,
        "encoder_family": enc_cfg.family,
        "encoder_size": enc_cfg.size,
        "image_size": enc_cfg.image_size,
        "dataset_id": dataset_id,
        "fps": fps,
        "seed": cfg["seed"],
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "episodes": ep_meta_entries,
        "action_partial": _merge_partials(action_parts),
        "proprio_partial": _merge_partials(proprio_parts),
    }
    shard_path = output_dir / f"shard_meta_{args.shard_index}.json"
    with open(shard_path, "w") as f:
        json.dump(shard_meta, f, indent=2)
    print(f"Shard meta : {shard_path}")

    # === Merge automatique si mono-shard ===
    if args.num_shards == 1:
        merge_shard_metas(output_dir)
        print(f"\nProchaine étape : python scripts/03_compare_fusion.py "
              f"--encoded-data {output_dir}")
    else:
        print(f"\nQuand les {args.num_shards} shards sont finis :"
              f"\n  python scripts/02_encode_dataset.py --output {output_dir} --finalize")


def _merge_partials(parts: list) -> dict:
    """Agrège les stats partielles par épisode en un seul partial par shard."""
    if not parts:
        return {"sum": [], "sumsq": [], "min": [], "max": [], "count": 0}
    t = {k: torch.tensor([p[k] for p in parts]) for k in ("sum", "sumsq", "min", "max")}
    return {
        "sum":   t["sum"].sum(0).tolist(),
        "sumsq": t["sumsq"].sum(0).tolist(),
        "min":   t["min"].min(0).values.tolist(),
        "max":   t["max"].max(0).values.tolist(),
        "count": sum(p["count"] for p in parts),
    }


if __name__ == "__main__":
    main()

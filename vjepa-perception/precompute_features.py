"""Precompute V-JEPA 2 features over a LeRobot dataset and cache them per-episode.

Usage:
    python precompute_features.py \\
        --src_repo lerobot/libero_object_image \\
        --dst_dir ./cached_features/libero_object \\
        --vjepa2_repo facebook/vjepa2-vitl-fpc64-256 \\
        --batch_size 32 --dtype float16

Output structure:
    <dst_dir>/
        metadata.json
        episode_000000.safetensors
        episode_000001.safetensors
        ...

Each episode file contains:
    features:       (T_ep, P, P, D)        cache dtype  (float16 by default)
    states:         (T_ep, state_dim)      float32
    actions:        (T_ep, action_dim)     float32
    frame_indices:  (T_ep,)                int64

We pre-extract states and actions next to the features so that downstream code can
load everything from a single file per episode without re-reading the source dataset.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoVideoProcessor

from config import VJepa2EncoderConfig
from encoder import FrozenVJepa2Encoder


class _PerFrameDataset(Dataset):
    """Thin wrapper that returns only the fields we need (stackable types)."""

    def __init__(self, src, camera_key, state_key, action_key):
        self.src = src
        self.camera_key = camera_key
        self.state_key = state_key
        self.action_key = action_key

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        s = self.src[idx]
        return {
            "frame": s[self.camera_key],
            "state": torch.as_tensor(s[self.state_key], dtype=torch.float32),
            "action": torch.as_tensor(s[self.action_key], dtype=torch.float32),
            "episode_index": torch.as_tensor(s["episode_index"], dtype=torch.int64),
            "frame_index": torch.as_tensor(s["frame_index"], dtype=torch.int64),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_repo", default="lerobot/libero_object_image",
                   help="Source LeRobot dataset on HF Hub")
    p.add_argument("--dst_dir", required=True,
                   help="Output directory for per-episode .safetensors files")
    p.add_argument("--vjepa2_repo", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--camera_key", default="observation.images.image",
                   help="Which camera to encode (some datasets have multiple)")
    p.add_argument("--state_key", default="observation.state")
    p.add_argument("--action_key", default="action")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Frames per encoder forward")
    p.add_argument("--dtype", default="float16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Cap on number of episodes (useful for a smoke test).")
    return p.parse_args()


def main():
    args = parse_args()
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    out_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    # ---- Encoder ----
    config = VJepa2EncoderConfig(vjepa2_hf_repo=args.vjepa2_repo)
    encoder = FrozenVJepa2Encoder(config).to(device).eval()

    proc = AutoVideoProcessor.from_pretrained(args.vjepa2_repo)
    mean = torch.tensor(proc.image_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(proc.image_std, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    # ---- Source dataset ----
    src_kwargs = {}
    if args.max_episodes:
        src_kwargs["episodes"] = list(range(args.max_episodes))
    src = LeRobotDataset(args.src_repo, **src_kwargs)

    P = config.n_patches_per_side
    n_frames = len(src)
    n_episodes = src.num_episodes
    bytes_per = P * P * config.encoder_dim * (2 if args.dtype == "float16" else 4)
    print(f"Source:   {args.src_repo}  ({n_episodes} episodes, {n_frames} frames)")
    print(f"Encoder:  {args.vjepa2_repo}  ({P}×{P}×{config.encoder_dim} per frame)")
    print(f"Dtype:    {args.dtype}  →  ~{bytes_per * n_frames / 1e9:.1f} GB total")
    print(f"Output:   {dst}")
    print()

    wrapped = _PerFrameDataset(src, args.camera_key, args.state_key, args.action_key)
    dl = DataLoader(wrapped, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # ---- Encode & flush per episode ----
    current_ep = -1
    ep_buf = None
    episode_lengths = {}
    n_done = 0
    t0 = time.time()
    last_log = t0

    def flush(ep, buf):
        out_path = dst / f"episode_{ep:06d}.safetensors"
        save_file({
            "features": torch.stack(buf["features"]),
            "states": torch.stack(buf["states"]),
            "actions": torch.stack(buf["actions"]),
            "frame_indices": torch.stack(buf["frame_indices"]),
        }, str(out_path))
        episode_lengths[ep] = len(buf["features"])

    for batch in dl:
        frames = batch["frame"].to(device, non_blocking=True)
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif frames.dtype != torch.float32:
            frames = frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = (frames - mean) / std

        with torch.no_grad():
            feats = encoder(frames.unsqueeze(1)).squeeze(1)  # (B, P, P, D)
        feats_cpu = feats.cpu().to(out_dtype)

        eps = batch["episode_index"]
        for i in range(len(frames)):
            ep = int(eps[i].item())
            if ep != current_ep:
                if current_ep >= 0 and ep_buf is not None:
                    flush(current_ep, ep_buf)
                current_ep = ep
                ep_buf = {"features": [], "states": [], "actions": [], "frame_indices": []}
            ep_buf["features"].append(feats_cpu[i])
            ep_buf["states"].append(batch["state"][i])
            ep_buf["actions"].append(batch["action"][i])
            ep_buf["frame_indices"].append(batch["frame_index"][i])

        n_done += len(frames)
        now = time.time()
        if now - last_log > 10:
            rate = n_done / (now - t0)
            eta = (n_frames - n_done) / rate / 60
            print(f"  {n_done}/{n_frames}  ({rate:.0f} frames/s, ETA {eta:.1f} min)")
            last_log = now

    if current_ep >= 0 and ep_buf is not None:
        flush(current_ep, ep_buf)

    # ---- Metadata ----
    metadata = {
        "source_repo": args.src_repo,
        "vjepa2_repo": args.vjepa2_repo,
        "encoder_dim": config.encoder_dim,
        "n_patches_per_side": P,
        "feature_shape_per_frame": [P, P, config.encoder_dim],
        "dtype": args.dtype,
        "source_fps": float(src.fps) if hasattr(src, "fps") else None,
        "num_episodes": len(episode_lengths),
        "num_frames": n_done,
        "camera_key": args.camera_key,
        "state_key": args.state_key,
        "action_key": args.action_key,
        "episode_lengths": {str(k): v for k, v in episode_lengths.items()},
        "wall_time_seconds": round(time.time() - t0, 2),
    }
    with open(dst / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {n_done} frames in {elapsed:.0f}s ({n_done / elapsed:.0f} frames/s).")
    print(f"Cache: {dst}")


if __name__ == "__main__":
    main()
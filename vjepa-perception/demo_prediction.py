"""V-JEPA 2 frame-prediction demo via nearest-neighbor retrieval.

For a held-out validation episode, pick several timesteps t. For each:
  1. Predict mean-pooled features at t+H from frame_t features.
  2. Cosine-similarity search across the training-set feature pool.
  3. Display [input frame_t | true frame_{t+H} | nearest-neighbor frame].

This needs the original RGB frames, so it re-instantiates LeRobotDataset.

Usage:
    python demo_prediction.py \\
        --cache_dir /content/cached_features/libero_object \\
        --checkpoint ./checkpoints/predictor/best.pt \\
        --src_repo lerobot/libero_object_image \\
        --output prediction_demo.png
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from predictor_head import FutureFeaturePredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--src_repo", default="lerobot/libero_object_image",
                   help="Source LeRobot dataset used at precompute time (for RGB lookup).")
    p.add_argument("--camera_key", default="observation.images.image")
    p.add_argument("--output", default="prediction_demo.png")
    p.add_argument("--num_steps", type=int, default=6,
                   help="How many timesteps to visualize (rows in the grid).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def mean_pool(features: torch.Tensor) -> torch.Tensor:
    return features.flatten(-3, -2).mean(dim=-2).float()


def load_episode(cache_dir: Path, ep: int):
    path = cache_dir / f"episode_{ep:06d}.safetensors"
    with safe_open(path, framework="pt") as f:
        return {
            "features": f.get_tensor("features"),
            "frame_indices": f.get_tensor("frame_indices"),
        }


def build_train_pool(cache_dir: Path, train_eps: list[int]):
    """Concatenate mean-pooled features across train episodes and remember
    where each came from (episode + original frame_index in the source dataset)."""
    pool_feats, owners = [], []
    for ep in train_eps:
        data = load_episode(cache_dir, ep)
        pool_feats.append(mean_pool(data["features"]))  # (T, D)
        owners.extend(
            (ep, int(fi.item())) for fi in data["frame_indices"]
        )
    return torch.cat(pool_feats), owners


def get_frame(src, ep_idx: int, frame_idx: int, camera_key: str):
    """Return RGB frame as (H, W, 3) uint8 numpy array."""
    start = int(src.episode_data_index["from"][ep_idx].item())
    sample = src[start + frame_idx]
    frame = sample[camera_key]
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    if hasattr(frame, "numpy"):
        arr = frame.numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)
    all_eps = sorted(int(k) for k in meta["episode_lengths"].keys())

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    horizon = ckpt["horizon"]
    val_eps = ckpt.get("val_episodes", [all_eps[-1]])
    train_eps = [ep for ep in all_eps if ep not in set(val_eps)]
    val_ep = val_eps[0]
    print(f"horizon={horizon}  train episodes={train_eps}  val episode={val_ep}")

    predictor = FutureFeaturePredictor(
        encoder_dim=ckpt["encoder_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(args.device)
    predictor.load_state_dict(ckpt["model_state"])
    predictor.eval()

    # Search pool from train episodes only — model never saw val features here.
    pool_feats, pool_owners = build_train_pool(cache_dir, train_eps)
    pool_feats_n = F.normalize(pool_feats.to(args.device), dim=-1)
    print(f"NN search pool: {pool_feats.shape[0]} train frames")

    val_data = load_episode(cache_dir, val_ep)
    val_features = val_data["features"]                # (T_val, P, P, D)
    val_frame_idxs = val_data["frame_indices"].tolist()
    T_val = val_features.shape[0]

    last_usable = T_val - horizon - 1
    if last_usable <= 0:
        raise RuntimeError(
            f"Val episode only {T_val} frames, need >{horizon + 1} for horizon={horizon}."
        )
    timesteps = np.linspace(0, last_usable, num=min(args.num_steps, last_usable + 1), dtype=int).tolist()
    print(f"Visualizing timesteps {timesteps} (out of {T_val} val frames)")

    print(f"Loading source dataset {args.src_repo} for RGB lookup...")
    src = LeRobotDataset(args.src_repo)

    n_rows = len(timesteps)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, t in enumerate(timesteps):
        with torch.no_grad():
            now = val_features[t : t + 1].to(args.device)  # (1, P, P, D)
            pred = predictor(now)                          # (1, D)
            pred_n = F.normalize(pred, dim=-1)
            sims = (pred_n @ pool_feats_n.T).squeeze(0)    # (N,)
            best = int(sims.argmax().item())
        nn_ep, nn_frame = pool_owners[best]
        sim_score = float(sims[best].item())

        in_frame_idx = val_frame_idxs[t]
        gt_frame_idx = val_frame_idxs[t + horizon]
        in_img = get_frame(src, val_ep, in_frame_idx, args.camera_key)
        gt_img = get_frame(src, val_ep, gt_frame_idx, args.camera_key)
        nn_img = get_frame(src, nn_ep, nn_frame, args.camera_key)

        axes[row, 0].imshow(in_img); axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_img); axes[row, 1].axis("off")
        axes[row, 2].imshow(nn_img); axes[row, 2].axis("off")
        axes[row, 0].set_title(f"input t={t} (ep{val_ep}, frame {in_frame_idx})", fontsize=9)
        axes[row, 1].set_title(f"GT t+{horizon} (frame {gt_frame_idx})", fontsize=9)
        axes[row, 2].set_title(f"predicted (NN): ep{nn_ep}, frame {nn_frame}  cos={sim_score:.2f}",
                                fontsize=9)

    fig.suptitle(
        f"V-JEPA 2 frozen features → learned future predictor → NN retrieval  "
        f"(horizon={horizon} cache steps, val episode {val_ep})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

"""Train a future-feature predictor on cached V-JEPA 2 features.

Builds (frame_t, frame_{t+horizon}) pairs by reading length-(horizon+1) clips
from the cache, training an MLP to predict the future frame's mean-pooled
features from the current frame's patch features.

Can run in:
  1. Active mode (default): Conditioned on the sequence of future actions leading to the future frame.
  2. Passive mode (via --no_actions): Standard baseline predicting future features without action conditioning.

Usage:
    python train_predictor.py \\
        --cache_dir /content/cached_features/libero_object \\
        --output_dir ./checkpoints/predictor \\
        --horizon 5 --epochs 30
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from cached_loader import CachedVJepa2Dataset
from predictor_head import FutureFeaturePredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--horizon", type=int, default=5,
                   help="Predict features this many cache steps ahead.")
    p.add_argument("--val_episodes", type=int, default=1,
                   help="Number of trailing episodes held out for validation.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--no_actions", action="store_true",
                   help="Disable action conditioning (train passive baseline predictor).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def split_indices_by_episode(ds: CachedVJepa2Dataset, n_val: int):
    ep_per_clip = [ep for _, ep, _ in ds._index]
    unique_eps = sorted(set(ep_per_clip))
    if n_val >= len(unique_eps):
        raise ValueError(
            f"--val_episodes={n_val} but cache only has {len(unique_eps)} episodes."
        )
    val_eps = set(unique_eps[-n_val:])
    train_idx = [i for i, ep in enumerate(ep_per_clip) if ep not in val_eps]
    val_idx = [i for i, ep in enumerate(ep_per_clip) if ep in val_eps]
    return train_idx, val_idx, sorted(val_eps)


def mean_pool(features: torch.Tensor) -> torch.Tensor:
    # features: (..., P, P, D) → (..., D)
    return features.flatten(-3, -2).mean(dim=-2).float()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    with open(cache_dir / "metadata.json") as f:
        meta = json.load(f)
    encoder_dim = meta["encoder_dim"]

    # Establish full sliding-window dataset
    full_ds = CachedVJepa2Dataset(cache_dir, clip_length=args.horizon + 1, stride=1)
    
    # Dynamically extract action dimensions
    sample = full_ds[0]
    action_dim = sample["actions"].shape[-1]
    
    train_idx, val_idx, val_eps = split_indices_by_episode(full_ds, args.val_episodes)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    use_actions = not args.no_actions
    pred_action_dim = action_dim if use_actions else None
    pred_horizon = args.horizon if use_actions else None

    print(f"encoder_dim={encoder_dim}  action_dim={pred_action_dim}  horizon={pred_horizon}")
    print(f"total clips={len(full_ds)}  episodes={meta['num_episodes']}")
    print(f"train clips={len(train_ds)}  val clips={len(val_ds)}  (val episodes={val_eps})")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Empty split. Horizon={args.horizon} too large for cache with episodes "
            f"of length {meta['episode_lengths']}."
        )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    predictor = FutureFeaturePredictor(
        encoder_dim=encoder_dim,
        action_dim=pred_action_dim,
        horizon=pred_horizon,
        hidden_dim=args.hidden_dim,
    ).to(device)
    print(f"predictor params: {sum(p.numel() for p in predictor.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # Baseline: predict that the future == the present (mean-pooled identity).
    base_losses = []
    for batch in val_dl:
        feats = batch["features"]                 # (B, H+1, P, P, D)
        now_pool = mean_pool(feats[:, 0])         # (B, D)
        future_pool = mean_pool(feats[:, -1])     # (B, D)
        base_losses.append(loss_fn(now_pool, future_pool).item())
    identity_mse = sum(base_losses) / len(base_losses)
    print(f"baseline (predict future = present) val MSE = {identity_mse:.4f}\n")

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        predictor.train()
        train_losses = []
        for batch in train_dl:
            feats = batch["features"].to(device)        # (B, H+1, P, P, D)
            now = feats[:, 0]                            # (B, P, P, D)
            target = mean_pool(feats[:, -1])             # (B, D)
            
            # Extract action sequence: from t to t+horizon-1
            actions = batch["actions"][:, :args.horizon].to(device) if use_actions else None
            
            pred = predictor(now, actions=actions)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_mse = sum(train_losses) / len(train_losses)

        predictor.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                feats = batch["features"].to(device)
                now = feats[:, 0]
                target = mean_pool(feats[:, -1])
                
                # Extract action sequence
                actions = batch["actions"][:, :args.horizon].to(device) if use_actions else None
                
                val_losses.append(loss_fn(predictor(now, actions=actions), target).item())
        val_mse = sum(val_losses) / len(val_losses)

        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            torch.save({
                "model_state": predictor.state_dict(),
                "args": vars(args),
                "encoder_dim": encoder_dim,
                "hidden_dim": args.hidden_dim,
                "action_dim": pred_action_dim,
                "horizon": pred_horizon,
                "epoch": epoch,
                "val_mse": val_mse,
                "val_episodes": val_eps,
            }, out / "best.pt")

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        marker = " *" if improved else ""
        print(f"epoch {epoch:3d}  train {train_mse:.4f}  val {val_mse:.4f}{marker}")

    with open(out / "history.json", "w") as f:
        json.dump({
            "history": history,
            "baseline_val_mse": identity_mse,
            "best_val_mse": best_val,
        }, f, indent=2)

    print(f"\nbest val MSE: {best_val:.4f}  (identity baseline: {identity_mse:.4f})")
    if best_val < identity_mse * 0.7:
        print("→ predictor beats the identity baseline. Features are predictively useful.")
    else:
        print("→ predictor close to identity baseline. Add more episodes or shorter horizon.")
    print(f"checkpoint: {out / 'best.pt'}")


if __name__ == "__main__":
    main()

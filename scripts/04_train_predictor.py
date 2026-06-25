"""
Entraîne le predictor (world model) sur les données Isaac sim pré-encodées.

Phase A du training :
    - Predictor entraîné from scratch
    - Données : 5000 trajectoires Isaac sim
    - Loss : MSE multi-step
    - Output : checkpoint predictor_simu.pt

Usage:
    python scripts/04_train_predictor.py --config configs/default.yaml
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.predictor import WorldModelPredictor, PredictorConfig
from src.fusion import make_fusion
from src.losses import multi_step_combined
from src.config import load_config, set_seed, log_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data",
                        default="results/encoded/sim_encoded_data.pt",
                        help="Données Isaac sim pré-encodées (depuis 02_encode_dataset)")
    parser.add_argument("--output",
                        default="results/checkpoints/predictor_simu.pt")
    parser.add_argument("--fusion", default="concat_view",
                        help="Stratégie de fusion à utiliser (choisie après M1)")
    parser.add_argument("--rollout-horizon", type=int, default=4,
                        help="Longueur du rollout multi-step pendant le training")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # === Charger les données pré-encodées ===
    encoded_path = ROOT / args.encoded_data
    if not encoded_path.exists():
        print(f"[ERREUR] Pas de données pré-encodées trouvées : {encoded_path}")
        print("Lance d'abord : python scripts/02_encode_dataset.py <dataset_sim_id>")
        sys.exit(1)

    print(f"\nChargement des données : {encoded_path}")
    data = torch.load(encoded_path, weights_only=False)

    z_wrist_t  = data["z_wrist_t"]
    z_global_t = data["z_global_t"]
    z_wrist_t1  = data["z_wrist_t1"]
    z_global_t1 = data["z_global_t1"]
    actions    = data["action"]
    embed_dim  = data["embed_dim"]
    action_dim = actions.shape[-1]

    print(f"  N paires   : {len(actions)}")
    print(f"  Embed dim  : {embed_dim}")
    print(f"  Action dim : {action_dim}")

    # === Build fusion + predictor ===
    fusion = make_fusion(args.fusion, dim=embed_dim)
    pred_cfg = PredictorConfig(
        embed_dim=embed_dim,
        action_dim=action_dim,
        n_layers=6,
        n_heads=12 if embed_dim == 768 else 8,
        ffn_dim=embed_dim * 4,
    )
    predictor = WorldModelPredictor(pred_cfg)
    print(f"\n[Fusion] strategy : {args.fusion}, params : "
          f"{sum(p.numel() for p in fusion.parameters()):,}")
    print(f"[Predictor] params : {predictor.count_parameters():,}")

    fusion.to(device).train()
    predictor.to(device).train()

    # === Split train/val ===
    n = len(actions)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    print(f"\nSplit : {len(train_idx)} train / {len(val_idx)} val")

    # === Optimizer ===
    params = list(fusion.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Training loop ===
    print(f"\n=== Training {args.n_epochs} epochs ===\n")
    history = []
    best_val = float("inf")

    for epoch in range(args.n_epochs):
        # Train
        fusion.train(); predictor.train()
        perm = torch.randperm(len(train_idx))
        train_losses = []

        for i in tqdm(range(0, len(perm), args.batch_size),
                       desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            idx = train_idx[perm[i:i + args.batch_size]]

            zw_t  = z_wrist_t[idx].to(device)
            zg_t  = z_global_t[idx].to(device)
            zw_t1 = z_wrist_t1[idx].to(device)
            zg_t1 = z_global_t1[idx].to(device)
            act   = actions[idx].to(device)

            # Fusion cross-cam
            z_t  = fusion(zw_t, zg_t)
            z_t1 = fusion(zw_t1, zg_t1)

            # Prediction one-step (pour le moment ; multi-step demande des séquences)
            pred = predictor(z_t, act)
            loss = nn.functional.mse_loss(pred, z_t1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # Val
        fusion.eval(); predictor.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(val_idx), args.batch_size):
                idx = val_idx[i:i + args.batch_size]
                zw_t  = z_wrist_t[idx].to(device)
                zg_t  = z_global_t[idx].to(device)
                zw_t1 = z_wrist_t1[idx].to(device)
                zg_t1 = z_global_t1[idx].to(device)
                act   = actions[idx].to(device)

                z_t  = fusion(zw_t, zg_t)
                z_t1 = fusion(zw_t1, zg_t1)
                pred = predictor(z_t, act)
                val_losses.append(nn.functional.mse_loss(pred, z_t1).item())

        val_loss = sum(val_losses) / len(val_losses)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "predictor_state_dict": predictor.state_dict(),
                "fusion_state_dict":    fusion.state_dict(),
                "predictor_config":     pred_cfg.__dict__,
                "fusion_name":          args.fusion,
                "embed_dim":            embed_dim,
                "action_dim":           action_dim,
                "epoch":                epoch,
                "val_loss":             val_loss,
                "history":              history,
            }, out_path)
            print(f"  [Best val, saved to {out_path.name}]")

    print(f"\n=== Training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoint    : {out_path}")


if __name__ == "__main__":
    main()

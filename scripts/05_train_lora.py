"""
Entraîne le LoRA adapter sur les 40 démos réelles.

Phase B du training :
    - Predictor figé (chargé depuis le checkpoint de 04_train_predictor.py)
    - Injection de LoRA dans toutes les couches linéaires du predictor
    - Seuls les paramètres LoRA sont entraînés
    - Données : 40 démos réelles pré-encodées (depuis 02_encode_dataset)
    - Output : checkpoint predictor_real.pt avec LoRA dedans

Usage:
    python scripts/05_train_lora.py --config configs/default.yaml
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.predictor import WorldModelPredictor, PredictorConfig
from src.fusion import make_fusion
from src.lora import LoRAConfig, inject_lora, get_lora_parameters
from src.config import load_config, set_seed, log_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictor-ckpt",
                        default="results/checkpoints/predictor_simu.pt",
                        help="Checkpoint du predictor entraîné sur Isaac (de 04)")
    parser.add_argument("--real-data",
                        default="results/encoded/real_encoded_data.pt",
                        help="40 démos réelles pré-encodées (depuis 02)")
    parser.add_argument("--output",
                        default="results/checkpoints/predictor_real.pt")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # === Charger le predictor entraîné sur sim ===
    ckpt_path = ROOT / args.predictor_ckpt
    if not ckpt_path.exists():
        print(f"[ERREUR] Pas de predictor checkpoint trouvé : {ckpt_path}")
        print("Lance d'abord : python scripts/04_train_predictor.py")
        sys.exit(1)

    print(f"\nChargement du predictor : {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)

    pred_cfg = PredictorConfig(**ckpt["predictor_config"])
    predictor = WorldModelPredictor(pred_cfg)
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    fusion = make_fusion(ckpt["fusion_name"], dim=ckpt["embed_dim"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])

    print(f"  Predictor params : {predictor.count_parameters():,}")
    print(f"  Fusion strategy  : {ckpt['fusion_name']}")
    print(f"  Embed dim        : {ckpt['embed_dim']}")

    # === Injection LoRA ===
    print(f"\n=== Injection LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}) ===")
    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    predictor = inject_lora(predictor, lora_cfg, verbose=False)

    # === Charger les démos réelles pré-encodées ===
    real_path = ROOT / args.real_data
    if not real_path.exists():
        print(f"\n[ERREUR] Données réelles non trouvées : {real_path}")
        print("Pré-encode tes 40 démos avec : python scripts/02_encode_dataset.py")
        sys.exit(1)

    print(f"\nChargement des démos réelles : {real_path}")
    real = torch.load(real_path, weights_only=False)

    z_wrist_t  = real["z_wrist_t"]
    z_global_t = real["z_global_t"]
    z_wrist_t1  = real["z_wrist_t1"]
    z_global_t1 = real["z_global_t1"]
    actions    = real["action"]

    print(f"  N paires : {len(actions)}")

    # Split train/val (gardons quelques démos en val)
    n = len(actions)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    fusion.to(device).eval()       # fusion gelée (entraînée en Phase A)
    predictor.to(device).train()    # predictor en train mode pour LoRA

    # === Optimiser uniquement les params LoRA ===
    lora_params = get_lora_parameters(predictor)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"\n  Params LoRA à entraîner : {n_lora:,}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Training loop ===
    print(f"\n=== Training LoRA sur démos réelles ===\n")
    history = []
    best_val = float("inf")

    for epoch in range(args.n_epochs):
        predictor.train()
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

            with torch.no_grad():
                z_t  = fusion(zw_t, zg_t)
                z_t1 = fusion(zw_t1, zg_t1)

            pred = predictor(z_t, act)
            loss = nn.functional.mse_loss(pred, z_t1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # Val
        predictor.eval()
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
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            # Sauvegarder le predictor complet (predictor + LoRA inside)
            torch.save({
                "predictor_state_dict": predictor.state_dict(),
                "fusion_state_dict":    fusion.state_dict(),
                "predictor_config":     pred_cfg.__dict__,
                "fusion_name":          ckpt["fusion_name"],
                "lora_config":          lora_cfg.__dict__,
                "embed_dim":            ckpt["embed_dim"],
                "action_dim":           ckpt["action_dim"],
                "epoch":                epoch,
                "val_loss":             val_loss,
                "history":              history,
            }, out_path)
            print(f"  [Best val, saved]")

    print(f"\n=== LoRA training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoint    : {out_path}")


if __name__ == "__main__":
    main()

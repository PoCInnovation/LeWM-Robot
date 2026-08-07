"""
Entraîne le predictor (world model) sur des latents pré-encodés (format v2).

Corrections vs l'ancienne version :
    - FUSION GELÉE par défaut : la fusion entraînable était dans le chemin du
      target de la MSE (z_t1) → risque de collapse de représentation avec une
      val loss excellente. (--train-fusion pour l'ancien comportement, avec
      target détaché + warning.)
    - ACTIONS NORMALISÉES (z-score, stats du dataset) : les actions brutes
      SO-101 vont de -104 à +101 — le CEM échantillonne en espace normalisé.
    - SPLIT PAR ÉPISODE : le split par paires fuyait de l'info entre train et
      val (frames voisines quasi identiques).
    - BASELINE IDENTITÉ loggée à chaque val : mse(z_t, z_t1). Si le modèle ne
      bat pas "copier l'entrée", il n'a pas appris de dynamique.
    - delta_timesteps choisi ici (--delta), pas figé à l'encodage.
    - Multi-step autorégressif optionnel : --horizon H (> 1) déroule le
      predictor sur ses propres prédictions, comme le fera le CEM.
    - bf16 autocast (MI250X) + checkpoint/resume (walltime 24 h, requeue).

Usage:
    python scripts/04_train_predictor.py --encoded-data results/encoded/<name>
        [--fusion cross_attn_bd] [--delta 2] [--horizon 1]
        [--n-epochs 30] [--batch-size 32] [--lr 1e-4]
        [--output-dir results/checkpoints/predictor_sim] [--resume]
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.predictor import WorldModelPredictor, PredictorConfig
from src.fusion import make_fusion
from src.encoded_data import EncodedEpisodes, PairsView, SequenceView, split_episodes
from src.config import load_config, set_seed, log_environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data", required=True,
                        help="Dossier des latents pré-encodés (format v2)")
    parser.add_argument("--output-dir",
                        default="results/checkpoints/predictor_sim")
    parser.add_argument("--fusion", default="cross_attn_bd",
                        help="Stratégie de fusion (cross_attn_bd = gagnante M1)")
    parser.add_argument("--fusion-ckpt", default=None,
                        help="Poids de fusion pré-entraînés (depuis 03). "
                             "Sinon : fusion gelée à l'init.")
    parser.add_argument("--train-fusion", action="store_true",
                        help="Entraîner la fusion (target DÉTACHÉ). "
                             "Défaut : fusion gelée.")
    parser.add_argument("--delta", type=int, default=None,
                        help="delta_timesteps pour la paire (t, t+delta). "
                             "Défaut : dataset.delta_timesteps de la config.")
    parser.add_argument("--horizon", type=int, default=1,
                        help="1 = one-step. >1 = multi-step autorégressif.")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true",
                        help="Désactiver l'autocast bf16 (défaut: activé sur GPU)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre depuis <output-dir>/last.pt")
    return parser.parse_args()


def fuse_batch(fusion, batch, device, horizon: int):
    """Applique la fusion cross-cam. Renvoie (z_t, targets, actions)."""
    if horizon == 1:
        z_t = fusion(batch["z_wrist_t"].to(device), batch["z_global_t"].to(device))
        z_t1 = fusion(batch["z_wrist_t1"].to(device), batch["z_global_t1"].to(device))
        return z_t, z_t1.unsqueeze(1), batch["action"].to(device).unsqueeze(1)
    # Multi-step : fusionner chaque frame de la fenêtre (B, H+1, N, D)
    zw, zg = batch["z_wrist"].to(device), batch["z_global"].to(device)
    B, H1 = zw.shape[:2]
    fused = fusion(zw.flatten(0, 1), zg.flatten(0, 1))
    fused = fused.reshape(B, H1, *fused.shape[1:])
    return fused[:, 0], fused[:, 1:], batch["actions"].to(device)


def rollout_loss(predictor, z0, targets, actions):
    """
    Rollout autorégressif + MSE par step.

    Args:
        z0      : (B, N, D)
        targets : (B, H, N, D)
        actions : (B, H, A)
    Returns:
        loss (scalaire), identity_mse (scalaire, baseline "copier z_t")
    """
    H = actions.shape[1]
    z = z0
    losses, id_losses = [], []
    for t in range(H):
        pred = predictor(z, actions[:, t])
        losses.append(nn.functional.mse_loss(pred, targets[:, t]))
        id_losses.append(nn.functional.mse_loss(
            z.detach(), targets[:, t].detach()))
        z = pred
    return torch.stack(losses).mean(), torch.stack(id_losses).mean()


def main():
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and not args.no_bf16
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"

    delta = args.delta if args.delta is not None \
        else int(cfg["dataset"].get("delta_timesteps", 1))
    if delta == 1:
        print("[ATTENTION] delta=1 : à 30 fps, 'copier z_t' est un minimum "
              "local très fort. Surveiller la baseline identité ; "
              "envisager --delta 2/4.")

    # === Données ===
    data = EncodedEpisodes(ROOT / args.encoded_data
                           if not Path(args.encoded_data).is_absolute()
                           else args.encoded_data)
    print(f"\n[Data] {data.summary()}")
    embed_dim = data.embed_dim
    action_dim = data.action_stats.mean.shape[0]

    val_fraction = float(cfg["dataset"].get("val_fraction", 0.2))
    train_eps, val_eps = split_episodes(len(data), val_fraction, cfg["seed"])
    print(f"[Split par épisode] {len(train_eps)} train / {len(val_eps)} val")

    def make_view(eps):
        if args.horizon == 1:
            return PairsView(data, eps, delta=delta)
        return SequenceView(data, eps, horizon=args.horizon, delta=delta)

    train_ds, val_ds = make_view(train_eps), make_view(val_eps)
    print(f"[Views] {len(train_ds)} train items / {len(val_ds)} val items "
          f"(delta={delta}, horizon={args.horizon})")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # === Fusion (gelée par défaut) + predictor ===
    fusion = make_fusion(args.fusion, dim=embed_dim)
    if args.fusion_ckpt:
        fusion.load_state_dict(torch.load(ROOT / args.fusion_ckpt,
                                          weights_only=True))
        print(f"[Fusion] Poids chargés depuis {args.fusion_ckpt}")
    if args.train_fusion:
        print("[Fusion] ENTRAÎNABLE (target détaché pour éviter le collapse).")
    else:
        for p in fusion.parameters():
            p.requires_grad = False
        print("[Fusion] GELÉE (défaut — la fusion est dans le chemin du target).")

    pred_cfg = PredictorConfig(
        embed_dim=embed_dim,
        action_dim=action_dim,
        n_layers=args.n_layers,
        n_heads=12 if embed_dim % 12 == 0 else 8,
        ffn_dim=embed_dim * 4,
    )
    predictor = WorldModelPredictor(pred_cfg)
    print(f"[Fusion] {args.fusion}, params: "
          f"{sum(p.numel() for p in fusion.parameters()):,} "
          f"({'train' if args.train_fusion else 'frozen'})")
    print(f"[Predictor] params: {predictor.count_parameters():,}")
    print(f"[Precision] bf16 autocast: {'ON' if use_bf16 else 'off'}")

    fusion.to(device)
    predictor.to(device)

    params = list(predictor.parameters())
    if args.train_fusion:
        params += [p for p in fusion.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Resume ===
    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume and last_path.exists():
        ck = torch.load(last_path, weights_only=False, map_location=device)
        predictor.load_state_dict(ck["predictor_state_dict"])
        fusion.load_state_dict(ck["fusion_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        history = ck["history"]
        torch.set_rng_state(ck["rng_state"].cpu())
        if device == "cuda" and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng_state"].cpu())
        print(f"[Resume] Reprise à l'epoch {start_epoch} "
              f"(best val: {best_val:.5f})")
    elif args.resume:
        print(f"[Resume] Pas de {last_path}, démarrage from scratch.")

    def checkpoint_payload(epoch, val_loss):
        return {
            "predictor_state_dict": predictor.state_dict(),
            "fusion_state_dict":    fusion.state_dict(),
            "predictor_config":     pred_cfg.__dict__,
            "fusion_name":          args.fusion,
            "fusion_trained":       args.train_fusion,
            "embed_dim":            embed_dim,
            "action_dim":           action_dim,
            "delta":                delta,
            "horizon":              args.horizon,
            "action_stats":         data.action_stats.to_dict(),
            "proprio_stats":        data.proprio_stats.to_dict(),
            "normalized_actions":   True,
            "encoded_data_meta": {k: data.meta[k] for k in
                                  ("encoder_family", "encoder_size",
                                   "image_size", "dataset_id")},
            "epoch":                epoch,
            "val_loss":             val_loss,
            "best_val":             best_val,
            "history":              history,
        }

    # === Training loop ===
    print(f"\n=== Training epochs {start_epoch}..{args.n_epochs - 1} ===\n")
    for epoch in range(start_epoch, args.n_epochs):
        fusion.train(args.train_fusion)
        predictor.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                if args.train_fusion:
                    z_t, targets, actions = fuse_batch(fusion, batch, device,
                                                       args.horizon)
                    targets = targets.detach()   # anti-collapse
                else:
                    with torch.no_grad():
                        z_t, targets, actions = fuse_batch(fusion, batch,
                                                           device, args.horizon)
                loss, _ = rollout_loss(predictor, z_t, targets, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / max(1, len(train_losses))

        # Val (+ baseline identité)
        fusion.eval(); predictor.eval()
        val_losses, id_losses = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    z_t, targets, actions = fuse_batch(fusion, batch, device,
                                                       args.horizon)
                    loss, id_loss = rollout_loss(predictor, z_t, targets, actions)
                val_losses.append(loss.item())
                id_losses.append(id_loss.item())

        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_identity = sum(id_losses) / max(1, len(id_losses))
        scheduler.step()

        ratio = val_loss / val_identity if val_identity > 0 else float("nan")
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "val_identity": val_identity,
                        "lr": scheduler.get_last_lr()[0]})
        flag = "  << ne bat PAS la baseline identité !" if ratio >= 1.0 else ""
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"identité={val_identity:.5f}  (val/id={ratio:.3f}){flag}")

        # Save best + last
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint_payload(epoch, val_loss), best_path)
            print(f"  [best → {best_path.name}]")
        torch.save({**checkpoint_payload(epoch, val_loss),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state()
                                       if device == "cuda" else None)},
                   last_path)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n=== Training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoints   : {best_path} (+ last.pt, history.json)")
    if history and history[-1]["val"] >= history[-1]["val_identity"]:
        print("\n[ATTENTION] Le modèle ne bat pas la baseline identité — "
              "essayer --delta plus grand, --horizon > 1, ou plus d'epochs.")


if __name__ == "__main__":
    main()

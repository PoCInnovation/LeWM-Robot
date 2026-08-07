"""
Entraîne le LoRA adapter sur les démos réelles pré-encodées (format v2).

Phase B du training :
    - Predictor figé (chargé depuis best.pt de 04_train_predictor.py)
    - Injection LoRA (uniquement les modules ciblés par LoRAConfig)
    - Seuls les paramètres LoRA sont entraînés
    - Fusion gelée (chargée du checkpoint)
    - NORMALISATION : réutilise les stats d'actions du checkpoint 04 (pas
      celles des données réelles) pour rester dans le même espace d'entrée.
    - Split par épisode, baseline identité, bf16, checkpoint/resume : comme 04.

Usage:
    python scripts/05_train_lora.py --encoded-data results/encoded/<real>
        [--predictor-ckpt results/checkpoints/predictor_sim/best.pt]
        [--output-dir results/checkpoints/predictor_real]
        [--lora-rank 8] [--n-epochs 20] [--resume]
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
from src.lora import LoRAConfig, inject_lora, get_lora_parameters
from src.encoded_data import EncodedEpisodes, PairsView, NormStats, split_episodes
from src.config import load_config, set_seed, log_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--predictor-ckpt",
                        default="results/checkpoints/predictor_sim/best.pt",
                        help="Checkpoint du predictor entraîné (de 04)")
    parser.add_argument("--encoded-data", required=True,
                        help="Démos réelles pré-encodées (dossier format v2)")
    parser.add_argument("--output-dir",
                        default="results/checkpoints/predictor_real")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--delta", type=int, default=None,
                        help="Défaut : le delta du checkpoint 04")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and not args.no_bf16
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"

    # === Charger le predictor entraîné (Phase A) ===
    ckpt_path = ROOT / args.predictor_ckpt
    if not ckpt_path.exists():
        print(f"[ERREUR] Checkpoint introuvable : {ckpt_path}")
        print("Lance d'abord : python scripts/04_train_predictor.py")
        sys.exit(1)

    print(f"\nChargement du predictor : {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    pred_cfg = PredictorConfig(**ckpt["predictor_config"])
    predictor = WorldModelPredictor(pred_cfg)
    predictor.load_state_dict(ckpt["predictor_state_dict"])

    fusion = make_fusion(ckpt["fusion_name"], dim=ckpt["embed_dim"])
    fusion.load_state_dict(ckpt["fusion_state_dict"])

    # Stats de normalisation DU CHECKPOINT (espace d'entrée du predictor)
    if "action_stats" not in ckpt:
        print("[ERREUR] Checkpoint sans action_stats : ré-entraîner avec le "
              "nouveau 04_train_predictor.py (actions normalisées).")
        sys.exit(1)
    action_stats = NormStats.from_dict(ckpt["action_stats"])
    proprio_stats = NormStats.from_dict(ckpt["proprio_stats"])
    delta = args.delta if args.delta is not None else int(ckpt.get("delta", 1))

    print(f"  Predictor params : {predictor.count_parameters():,}")
    print(f"  Fusion           : {ckpt['fusion_name']} (gelée)")
    print(f"  Delta            : {delta}")

    # === Injection LoRA ===
    print(f"\n=== Injection LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}) ===")
    lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    predictor = inject_lora(predictor, lora_cfg, verbose=False)

    # === Données réelles ===
    data = EncodedEpisodes(ROOT / args.encoded_data
                           if not Path(args.encoded_data).is_absolute()
                           else args.encoded_data)
    print(f"\n[Data] {data.summary()}")
    if data.embed_dim != ckpt["embed_dim"]:
        print(f"[ERREUR] embed_dim des données réelles ({data.embed_dim}) != "
              f"checkpoint ({ckpt['embed_dim']}) — encodeurs différents ?")
        sys.exit(1)

    # Vérif : les stats réelles ne divergent pas trop de celles du sim
    real_stats = data.action_stats
    ratio = (real_stats.std / action_stats.std).abs()
    if (ratio > 3).any() or (ratio < 1 / 3).any():
        print("[ATTENTION] std des actions réelles très différente du sim "
              f"(ratio par dim : {ratio.round(decimals=2).tolist()}) — "
              "vérifier les unités !")

    val_fraction = float(cfg["dataset"].get("val_fraction", 0.2))
    train_eps, val_eps = split_episodes(len(data), val_fraction, cfg["seed"])
    print(f"[Split par épisode] {len(train_eps)} train / {len(val_eps)} val")

    # PairsView normalise avec les stats DU DATASET → désactiver, normaliser
    # manuellement avec les stats du checkpoint.
    train_ds = PairsView(data, train_eps, delta=delta,
                         norm_actions=False, norm_proprio=False)
    val_ds = PairsView(data, val_eps, delta=delta,
                       norm_actions=False, norm_proprio=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    fusion.to(device).eval()          # gelée (entraînée/fixée en Phase A)
    for p in fusion.parameters():
        p.requires_grad = False
    predictor.to(device)

    lora_params = get_lora_parameters(predictor)
    print(f"\n  Params LoRA à entraîner : {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Resume ===
    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume and last_path.exists():
        ck = torch.load(last_path, weights_only=False, map_location=device)
        predictor.load_state_dict(ck["predictor_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        history = ck["history"]
        torch.set_rng_state(ck["rng_state"].cpu())
        print(f"[Resume] Reprise à l'epoch {start_epoch}")

    def run_batch(batch):
        act = action_stats.normalize(batch["action"].to(device))
        with torch.no_grad():
            z_t = fusion(batch["z_wrist_t"].to(device),
                         batch["z_global_t"].to(device))
            z_t1 = fusion(batch["z_wrist_t1"].to(device),
                          batch["z_global_t1"].to(device))
        pred = predictor(z_t, act)
        loss = nn.functional.mse_loss(pred, z_t1)
        id_loss = nn.functional.mse_loss(z_t, z_t1)
        return loss, id_loss

    def checkpoint_payload(epoch, val_loss):
        return {
            "predictor_state_dict": predictor.state_dict(),
            "fusion_state_dict":    fusion.state_dict(),
            "predictor_config":     pred_cfg.__dict__,
            "fusion_name":          ckpt["fusion_name"],
            "lora_config":          lora_cfg.__dict__,
            "embed_dim":            ckpt["embed_dim"],
            "action_dim":           ckpt["action_dim"],
            "delta":                delta,
            "action_stats":         action_stats.to_dict(),
            "proprio_stats":        proprio_stats.to_dict(),
            "normalized_actions":   True,
            "encoded_data_meta":    ckpt.get("encoded_data_meta"),
            "epoch":                epoch,
            "val_loss":             val_loss,
            "best_val":             best_val,
            "history":              history,
        }

    # === Training loop ===
    print(f"\n=== Training LoRA epochs {start_epoch}..{args.n_epochs - 1} ===\n")
    for epoch in range(start_epoch, args.n_epochs):
        predictor.train()
        train_losses = []
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss, _ = run_batch(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / max(1, len(train_losses))

        predictor.eval()
        val_losses, id_losses = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    loss, id_loss = run_batch(batch)
                val_losses.append(loss.item())
                id_losses.append(id_loss.item())
        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_identity = sum(id_losses) / max(1, len(id_losses))
        scheduler.step()

        ratio = val_loss / val_identity if val_identity > 0 else float("nan")
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "val_identity": val_identity})
        flag = "  << ne bat PAS la baseline identité !" if ratio >= 1.0 else ""
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"identité={val_identity:.5f}  (val/id={ratio:.3f}){flag}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint_payload(epoch, val_loss), best_path)
            print(f"  [best → {best_path.name}]")
        torch.save({**checkpoint_payload(epoch, val_loss),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state()},
                   last_path)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n=== LoRA training terminé ===")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Checkpoint    : {best_path}")


if __name__ == "__main__":
    main()

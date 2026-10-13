"""
Entraîne le predictor (world model) sur les données pré-encodées.

Phase A du training :
    - Predictor entraîné from scratch ; fusion cross-cam GELÉE, reprise des
      poids entraînés par 03 (cf. --fusion-ckpt / --train-fusion)
    - Actions normalisées (z-score, stats du train sauvées dans le checkpoint)
    - Loss : MSE one-step z_t → z_t+1 ; baseline « rien ne bouge »
      (z_t+1 = z_t) loggée à chaque epoch : ratio < 1 = dynamique apprise
    - Split train/val par épisode
    - Output : checkpoint predictor_simu.pt

GPU (RTX 5090, 32 GB) :
    - latents hébergés en VRAM (hardware.data_device: auto) → zéro transfert ;
    - autocast bf16 + TF32, AdamW fused, --compile (torch.compile) opt-in ;
    - --batch-size 64 par défaut ; mesurer 32/64/128/256 avec le benchmark
      avant d’augmenter (la mémoire dépend de la fusion et de l’encodeur) ;
    - pic VRAM + durée loggés par epoch (dans le checkpoint, clé history).

Usage:
    python scripts/04_train_predictor.py --config configs/default.yaml
        [--encoded-data results/encoded/encoded_data.pt]
        [--fusion concat_view] [--n-epochs 30] [--batch-size 64] [--lr 1e-4]
        [--precision auto] [--data-device auto] [--compile]
"""

import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.predictor import WorldModelPredictor, PredictorConfig
from src.fusion import make_fusion
from src.config import load_config, set_seed, log_environment, setup_hardware
from src.data import split_train_val, describe_split
from src.latents import LatentPairs, load_encoded, train_action_stats
from src.checkpoint import atomic_save
from src.device import (resolve_amp_dtype, make_adamw,
                        maybe_compile, unwrap, autocast_ctx,
                        peak_vram_gb, reset_peak_vram)
from src.tracking import Stopwatch, add_wandb_args, tracker_from_args


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data",
                        default="results/encoded/sim_encoded_data.pt",
                        help="Données pré-encodées (depuis 02_encode_dataset)")
    parser.add_argument("--output",
                        default="results/checkpoints/predictor_simu.pt")
    parser.add_argument("--fusion", default="concat_view",
                        help="Stratégie de fusion à utiliser (choisie après M1)")
    parser.add_argument("--fusion-ckpt", default=None,
                        help="Poids de fusion entraînés par 03 (défaut : "
                             "paths.fusion_weights/<fusion>.pt s'il existe)")
    parser.add_argument("--train-fusion", action="store_true",
                        help="Entraîne aussi la fusion (cible détachée). "
                             "Déconseillé : voir la note sur l'effondrement.")
    parser.add_argument("--rollout-horizon", type=int, default=4,
                        help="Longueur du rollout multi-step pendant le training "
                             "(non utilisé pour l'instant : one-step)")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW (défaut PyTorch). 1e-4 = quasi aucune "
                             "régularisation, trop peu pour 40 épisodes.")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual", action="store_true",
                        help="Sortie z_t + head(...) initialisée à « rien ne bouge »")
    parser.add_argument("--precision", default=None,
                        help="auto (bf16 sur GPU) / bf16 / fp16 / fp32. "
                             "Défaut : hardware.precision")
    parser.add_argument("--data-device", default=None,
                        help="auto / cuda / cpu — où héberger les latents. "
                             "Défaut : hardware.data_device")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile du predictor (défaut : hardware.compile)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Fréquence (en batches) du log de la loss instantanée.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed de l'initialisation et de l'ordre des batches "
                             "(défaut : seed de la config). Le split train/val garde "
                             "le seed de la config : même validation pour tous les seeds.")
    parser.add_argument("--resume", action="store_true",
                        help="Reprend depuis <output>.last.pt s'il existe "
                             "(état sauvé à chaque epoch : machine coupée).")
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(seed)
    hw = setup_hardware(cfg)
    log_environment()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = resolve_amp_dtype(device, args.precision or hw["precision"])
    use_compile = (args.compile or bool(hw.get("compile", False))) and device == "cuda"
    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # === Charger les données pré-encodées ===
    encoded_path = resolve(args.encoded_data)
    if not encoded_path.exists():
        print(f"[ERREUR] Pas de données pré-encodées trouvées : {encoded_path}")
        print("Lance d'abord : python scripts/02_encode_dataset.py")
        sys.exit(1)

    print(f"\nChargement des données : {encoded_path}")
    data = load_encoded(encoded_path)
    embed_dim  = data["embed_dim"]

    latents = LatentPairs(data, args.data_device or hw["data_device"], device,
                          reserve_gb=float(hw["vram_reserve_gb"]))
    data_device = latents.device
    fetch = latents.fetch
    action_dim = latents.action_dim
    n = len(latents)

    print(f"  N paires   : {n}")
    print(f"  Embed dim  : {embed_dim}")
    print(f"  Action dim : {action_dim}")

    # === Build fusion + predictor ===
    # Fusion GELÉE par défaut. La cible z_t+1 passe elle aussi par la fusion :
    # entraînée avec le predictor, elle apprend à envoyer toutes les images
    # sur le même latent (mesuré : variance inter-exemples ÷170 en 100 pas,
    # loss ÷1000 sans rien apprendre de la dynamique). Elle repart des poids
    # entraînés par 03 (probe d'action), qui portent l'info utile.
    fusion = make_fusion(args.fusion, dim=embed_dim)
    n_fusion_params = sum(p.numel() for p in fusion.parameters())
    fusion_ckpt = (resolve(args.fusion_ckpt) if args.fusion_ckpt else
                   resolve(cfg["paths"].get("fusion_weights", "results/fusion_weights"))
                   / f"{args.fusion}.pt")
    if n_fusion_params and fusion_ckpt.exists():
        fusion.load_state_dict(
            torch.load(fusion_ckpt, map_location="cpu", weights_only=False)["fusion_state_dict"])
        print(f"[Fusion] poids de 03 chargés : {fusion_ckpt}")
    elif n_fusion_params:
        if args.fusion_ckpt:
            sys.exit(f"[ERREUR] --fusion-ckpt introuvable : {fusion_ckpt}")
        print(f"[ATTENTION] Pas de poids de fusion ({fusion_ckpt}) : fusion "
              f"{args.fusion} à son init aléatoire. Lancer 03 avant 04.")
    train_fusion = args.train_fusion and n_fusion_params > 0
    for p in fusion.parameters():
        p.requires_grad = train_fusion
    pred_cfg = PredictorConfig(
        embed_dim=embed_dim,
        action_dim=action_dim,
        n_layers=args.n_layers,
        n_heads=12 if embed_dim % 12 == 0 else 8,
        ffn_dim=embed_dim * 4,
        dropout=args.dropout,
        residual=args.residual,
    )
    predictor = WorldModelPredictor(pred_cfg)
    print(f"\n[Fusion] strategy : {args.fusion}, params : {n_fusion_params:,} "
          f"({'entraînée, cible détachée' if train_fusion else 'gelée'})")
    print(f"[Predictor] params : {predictor.count_parameters():,}")
    print(f"[Precision] autocast: {amp_dtype or 'off (fp32)'}")

    fusion.to(device).train(train_fusion)
    predictor.to(device).train()
    raw_predictor = predictor                       # pour le checkpoint
    predictor = maybe_compile(predictor, use_compile)

    # === Split train/val ===
    train_idx, val_idx = split_train_val(data, seed=cfg["seed"],
                                         val_fraction=float(cfg.get("dataset", {}).get("val_fraction", 0.2)))

    print(f"\nSplit : {describe_split(data, train_idx, val_idx)}")

    # Actions en z-score (stats du train, sauvées dans le checkpoint : 05 les
    # réutilise, 06 en tire les bornes du CEM et dénormalise ses actions).
    action_stats = train_action_stats(data, train_idx)
    latents.normalize_actions(action_stats)

    # === Optimizer ===
    params = list(raw_predictor.parameters())
    if train_fusion:
        params += list(fusion.parameters())
    optimizer = make_adamw(params, args.lr, args.weight_decay, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    # === Suivi W&B ===
    run = tracker_from_args(
        args, default_name=f"predictor-{args.fusion}", cfg=cfg, root=ROOT,
        extra_config={
            "device": device,
            "amp_dtype": str(amp_dtype),
            "data_device": str(data_device),
            "compile": use_compile,
            "embed_dim": embed_dim,
            "action_dim": action_dim,
            "n_pairs": n,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "fusion_params": n_fusion_params,
            "fusion_trained": train_fusion,
            "fusion_ckpt": str(fusion_ckpt) if fusion_ckpt.exists() else None,
            "predictor_params": raw_predictor.count_parameters(),
            "phase": "A_sim",
            "train_seed": seed,
        },
    )

    # === Training loop ===
    print(f"\n=== Training {args.n_epochs} epochs ===\n")
    history = []
    best_val = float("inf")
    best_ratio = float("nan")
    global_step = 0
    start_epoch = 0
    run_clock = Stopwatch()

    # État complet sauvé à chaque epoch (écrasé) pour reprendre après coupure.
    last_path = out_path.with_name(out_path.stem + ".last.pt")
    done_path = out_path.with_name(out_path.stem + ".done.json")
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        raw_predictor.load_state_dict(state["predictor_state_dict"])
        if train_fusion:
            fusion.load_state_dict(state["fusion_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"])
        if device == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_val, best_ratio = state["best_val"], state["best_ratio"]
        history = state["history"]
        print(f"[Resume] reprise à l'epoch {start_epoch + 1}/{args.n_epochs} "
              f"(best val {best_val:.5f}) depuis {last_path.name}")

    for epoch in range(start_epoch, args.n_epochs):
        # Train
        fusion.train(train_fusion); predictor.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        train_losses = []
        grad_norms = []
        reset_peak_vram()
        t_epoch = time.time()

        for i in tqdm(range(0, len(perm), args.batch_size),
                      desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            zw_t, zg_t, zw_t1, zg_t1, act = fetch(perm[i:i + args.batch_size])

            with autocast_ctx(device, amp_dtype):
                # Fusion cross-cam (cible toujours sans gradient)
                with torch.set_grad_enabled(train_fusion):
                    z_t = fusion(zw_t, zg_t)
                with torch.no_grad():
                    z_t1 = fusion(zw_t1, zg_t1)
                # Prediction one-step (pour le moment ; multi-step demande des séquences)
                pred = predictor(z_t, act)
            loss = nn.functional.mse_loss(pred.float(), z_t1.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1.0))
            optimizer.step()
            train_losses.append(loss.item())
            grad_norms.append(grad_norm)
            global_step += 1

            if args.log_every and global_step % args.log_every == 0:
                run.log({
                    "train/batch_loss": loss.item(),
                    "train/grad_norm": grad_norm,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }, step=global_step)

        train_loss = sum(train_losses) / len(train_losses)

        # Val
        fusion.eval(); predictor.eval()
        val_losses, id_losses = [], []
        with torch.no_grad():
            for i in range(0, len(val_idx), args.batch_size):
                zw_t, zg_t, zw_t1, zg_t1, act = fetch(val_idx[i:i + args.batch_size])
                with autocast_ctx(device, amp_dtype):
                    z_t  = fusion(zw_t, zg_t)
                    z_t1 = fusion(zw_t1, zg_t1)
                    pred = predictor(z_t, act)
                val_losses.append(
                    nn.functional.mse_loss(pred.float(), z_t1.float()).item())
                id_losses.append(
                    nn.functional.mse_loss(z_t.float(), z_t1.float()).item())

        val_loss = sum(val_losses) / len(val_losses)
        # Baseline « rien ne bouge » (z_t+1 = z_t) : un predictor utile doit
        # passer nettement dessous (ratio < 1).
        id_loss = sum(id_losses) / len(id_losses)
        id_ratio = val_loss / id_loss if id_loss > 0 else float("nan")
        # Relevé avant scheduler.step(), sinon c'est le lr de l'epoch suivant.
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_s = time.time() - t_epoch
        vram = peak_vram_gb() if device == "cuda" else 0.0
        samples_per_s = len(train_idx) / epoch_s if epoch_s > 0 else 0.0

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "val_identity": id_loss,
                        "epoch_seconds": epoch_s, "peak_vram_gb": vram})
        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"(identité {id_loss:.5f}, ratio {id_ratio:.3f})  "
              f"lr={current_lr:.2e}  "
              f"[{epoch_s:.0f}s, {samples_per_s:.0f} éch/s, VRAM pic {vram:.1f} GB]")

        run.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/identity_mse": id_loss,
            "val/ratio_vs_identity": id_ratio,
            "train/lr": current_lr,
            "train/grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "train/grad_norm_max": max(grad_norms),
            "perf/epoch_seconds": epoch_s,
            "perf/samples_per_s": samples_per_s,
            "perf/gpu_peak_gb": vram if device == "cuda" else None,
        }, step=global_step)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_ratio = id_ratio
            atomic_save({
                "predictor_state_dict": unwrap(raw_predictor).state_dict(),
                "fusion_state_dict":    fusion.state_dict(),
                "predictor_config":     pred_cfg.__dict__,
                "fusion_name":          args.fusion,
                "embed_dim":            embed_dim,
                "action_dim":           action_dim,
                "action_stats":         action_stats.to_dict(),
                "epoch":                epoch,
                "val_loss":             val_loss,
                "val_identity":         id_loss,
                "history":              history,
            }, out_path)
            print(f"  [Best val, saved to {out_path.name}]")
            run.log_artifact(
                out_path, name="predictor_simu", artifact_type="model",
                metadata={"epoch": epoch, "val_loss": val_loss,
                          "fusion": args.fusion, "embed_dim": embed_dim},
            )

        atomic_save({
            "predictor_state_dict": unwrap(raw_predictor).state_dict(),
            "fusion_state_dict":    fusion.state_dict(),
            "optimizer":            optimizer.state_dict(),
            "scheduler":            scheduler.state_dict(),
            "rng_cpu":              torch.get_rng_state(),
            "rng_cuda":             (torch.cuda.get_rng_state_all()
                                     if device == "cuda" else None),
            "epoch":                epoch,
            "global_step":          global_step,
            "best_val":             best_val,
            "best_ratio":           best_ratio,
            "history":              history,
        }, last_path)

    total_seconds = run_clock.elapsed
    epoch_seconds = [h["epoch_seconds"] for h in history]
    vals = [h["val"] for h in history]
    # Résumé lu par scripts/campaign.py (run terminé, temps d'epoch mesuré).
    with open(done_path, "w") as f:
        json.dump({
            "best_val_loss": best_val,
            "best_ratio_vs_identity": best_ratio,
            "val_identity": history[-1]["val_identity"] if history else None,
            "epochs": args.n_epochs,
            "seed": seed,
            "best_epoch": vals.index(min(vals)) if vals else None,
            # la val baisse encore sur les 3 dernières epochs → plus d'epochs utiles
            "still_improving": bool(len(vals) >= 4 and min(vals[-3:]) < min(vals[:-3])),
            "epoch_seconds_mean": (sum(epoch_seconds) / len(epoch_seconds)
                                   if epoch_seconds else None),
            "train_seconds": sum(epoch_seconds),
            "checkpoint": str(out_path),
        }, f, indent=2)
    last_path.unlink(missing_ok=True)          # état d'optimiseur : ~3x le modèle
    run.log_summary({
        "best_val_loss": best_val,
        "best_ratio_vs_identity": best_ratio,
        "total_seconds": total_seconds,
        "total_hours": round(total_seconds / 3600, 3),
        "epochs": args.n_epochs,
    })
    run.finish()

    print(f"\n=== Training terminé ===")
    print(f"Best val loss : {best_val:.5f}  (ratio vs identité : {best_ratio:.3f})")
    if not best_ratio < 1.0:
        print("[ATTENTION] Le predictor ne bat pas la baseline « rien ne bouge » "
              "(z_t+1 = z_t) : il n'a pas appris la dynamique.")
    print(f"Durée         : {total_seconds / 60:.1f} min")
    print(f"Checkpoint    : {out_path}")
    if run.url:
        print(f"W&B           : {run.url}")


if __name__ == "__main__":
    main()

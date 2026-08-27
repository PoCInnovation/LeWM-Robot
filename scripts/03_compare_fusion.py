"""
Compare les stratégies de fusion multi-camera via un probe d'action prediction.

Pour chaque stratégie de fusion :
  1. On charge les patches DINOv3 pré-encodés (depuis 02_encode_dataset.py)
  2. On entraîne conjointement FUSION + probe (MLP qui prédit l'action)
  3. On mesure la MAE sur validation
  4. On compare

La meilleure fusion = celle qui rend l'action le plus prédictible
(c'est-à-dire qui préserve le plus d'info utile dans le latent).

GPU (RTX 4090) : les latents sont hébergés en VRAM (hardware.data_device:
auto) → aucun transfert pendant l'entraînement ; autocast bf16, AdamW fused.
Les 4 stratégies × 30 epochs prennent ~1-3 min.

Usage:
    python scripts/03_compare_fusion.py [--encoded-data results/encoded/encoded_data.pt]
                                         [--n-epochs 30] [--batch-size 32]
                                         [--precision auto] [--data-device auto]
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fusion import make_fusion
from src.probes import ActionProbe
from src.config import load_config, set_seed, log_environment, setup_hardware
from src.device import (resolve_amp_dtype, place_tensors, make_adamw,
                        autocast_ctx, peak_vram_gb, reset_peak_vram)


# Stratégies à comparer (on évite proprio_guided dans la baseline pour rester simple)
STRATEGIES_TO_TEST = [
    "concat",
    "concat_view",
    "cross_attn_bd",
    "late_cls",
]


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data", default=None,
                        help="Fichier .pt de 02 (défaut : paths.encoded_data)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Défaut : probe.n_epochs de la config (30)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Défaut : probe.batch_size de la config (32)")
    parser.add_argument("--precision", default=None,
                        help="auto (bf16 sur GPU) / bf16 / fp16 / fp32. "
                             "Défaut : hardware.precision")
    parser.add_argument("--data-device", default=None,
                        help="auto / cuda / cpu — où héberger les latents. "
                             "Défaut : hardware.data_device")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    hw = setup_hardware(cfg)
    log_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = resolve_amp_dtype(device, args.precision or hw["precision"])

    probe_cfg = cfg.get("probe", {})
    n_epochs = args.n_epochs or int(probe_cfg.get("n_epochs", 30))
    batch_size = args.batch_size or int(probe_cfg.get("batch_size", 32))
    lr = float(probe_cfg.get("lr", 1e-3))
    weight_decay = float(probe_cfg.get("weight_decay", 1e-4))
    hidden = int(probe_cfg.get("hidden_dim", 512))
    strategies = cfg.get("fusion", {}).get("strategies_to_test", STRATEGIES_TO_TEST)

    encoded_path = resolve(args.encoded_data or cfg["paths"]["encoded_data"])
    if not encoded_path.exists():
        print(f"[ERREUR] Pas de données encodées trouvées : {encoded_path}")
        print("Lance d'abord : python scripts/02_encode_dataset.py")
        sys.exit(1)

    print("=" * 60)
    print("TEST 3 : Comparaison des stratégies de fusion multi-camera")
    print("=" * 60)
    print(f"[Precision] autocast: {amp_dtype or 'off (fp32)'}")

    # === Charger les patches pré-encodés ===
    data = torch.load(encoded_path, weights_only=False, map_location="cpu")
    dim = data["embed_dim"]
    actions = data["action"]       # (N, action_dim)
    action_dim = actions.shape[-1]
    tensors, data_device = place_tensors(
        {"z_wrist": data["z_wrist_t"], "z_global": data["z_global_t"],
         "action": actions},
        args.data_device or hw["data_device"], device)
    z_wrist, z_global, actions = tensors["z_wrist"], tensors["z_global"], tensors["action"]

    print(f"\nDonnées chargées :")
    print(f"  N paires       : {z_wrist.shape[0]}")
    print(f"  Num patches    : {z_wrist.shape[1]}")
    print(f"  Embed dim      : {dim}")
    print(f"  Action dim     : {action_dim}")

    # === Split train/val (seedé → reproductible) ===
    n = z_wrist.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    print(f"\nSplit : {len(train_idx)} train / {len(val_idx)} val")

    # === Boucle sur chaque stratégie ===
    results = {}

    for strategy_name in strategies:
        print(f"\n{'─' * 60}")
        print(f"Stratégie : {strategy_name}")
        print(f"{'─' * 60}")
        set_seed(cfg["seed"])   # même init pour chaque stratégie
        reset_peak_vram()

        # Instancier la fusion
        fusion_kwargs = {"dim": dim}
        if strategy_name == "proprio_guided":
            fusion_kwargs["action_dim"] = action_dim
        fusion = make_fusion(strategy_name, **fusion_kwargs)
        n_fusion_params = sum(p.numel() for p in fusion.parameters())
        print(f"Paramètres fusion : {n_fusion_params:,}")

        # Probe d'action
        probe = ActionProbe(dim=dim, action_dim=action_dim, hidden=hidden)

        # Joint train fusion + probe
        result = train_joint(
            fusion=fusion,
            probe=probe,
            z_wrist=z_wrist, z_global=z_global, actions=actions,
            train_idx=train_idx, val_idx=val_idx,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, device=device, amp_dtype=amp_dtype,
        )
        if device == "cuda":
            print(f"  VRAM pic : {peak_vram_gb():.2f} GB")

        results[strategy_name] = {
            "params":           n_fusion_params,
            "output_tokens":    result["output_tokens"],
            "final_train_mae":  result["final_train_loss"],
            "final_val_mae":    result["final_val_loss"],
            "best_val_mae":     result["best_val_loss"],
        }

    # === Affichage du résumé ===
    print("\n" + "=" * 60)
    print("RÉSULTATS COMPARATIFS")
    print("=" * 60)
    print(f"{'Stratégie':<18} {'Params':>10} {'Tokens':>8} {'Train MAE':>12} {'Val MAE':>12}")
    print("-" * 62)
    winner = min(results, key=lambda k: results[k]["best_val_mae"])
    for name, r in results.items():
        mark = "  << gagnante" if name == winner else ""
        print(f"{name:<18} {r['params']:>10,} {r['output_tokens']:>8} "
              f"{r['final_train_mae']:>12.5f} {r['best_val_mae']:>12.5f}{mark}")

    # === Sauver les résultats ===
    out_path = resolve(cfg["paths"].get("fusion_results", "results/fusion_comparison.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvés dans : {out_path}")

    # === Plot ===
    plot_path = resolve(cfg["paths"].get("fusion_plot", "results/fusion_comparison.png"))
    plot_results(results, plot_path)
    print(f"Plot sauvé dans       : {plot_path}")
    print(f"\nPour le predictor : python scripts/04_train_predictor.py "
          f"--encoded-data {encoded_path} --fusion {winner}")


def train_joint(fusion, probe, z_wrist, z_global, actions, train_idx, val_idx,
                n_epochs=30, lr=1e-3, weight_decay=1e-4, batch_size=32,
                device="cpu", amp_dtype=None):
    """
    Entraîne fusion + probe conjointement avec une loss L1 (MAE) sur l'action.

    Les latents sont indexés là où ils sont hébergés (VRAM ou RAM) et
    transférés par batch si besoin (no-op quand ils sont déjà en VRAM).
    """
    fusion.to(device).train()
    probe.to(device).train()

    params = list(fusion.parameters()) + list(probe.parameters())
    optimizer = make_adamw(params, lr, weight_decay, device)

    def fetch(idx):
        return (z_wrist[idx].to(device, non_blocking=True),
                z_global[idx].to(device, non_blocking=True),
                actions[idx].to(device, non_blocking=True))

    n = train_idx.shape[0]
    history = []
    best_val = float("inf")
    output_tokens = None

    for epoch in range(n_epochs):
        # Train
        fusion.train(); probe.train()
        perm = train_idx[torch.randperm(n)]
        losses = []
        for i in range(0, n, batch_size):
            zw, zg, act = fetch(perm[i:i + batch_size])
            with autocast_ctx(device, amp_dtype):
                z_fused = fusion(zw, zg)
                pred = probe(z_fused)
            output_tokens = z_fused.shape[1]
            loss = nn.functional.l1_loss(pred.float(), act)   # loss en fp32
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_mae = sum(losses) / len(losses)

        # Val
        fusion.eval(); probe.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, val_idx.shape[0], batch_size):
                zw, zg, act = fetch(val_idx[i:i + batch_size])
                with autocast_ctx(device, amp_dtype):
                    pred_val = probe(fusion(zw, zg))
                val_losses.append(
                    nn.functional.l1_loss(pred_val.float(), act, reduction="sum").item())
        val_mae = sum(val_losses) / (val_idx.shape[0] * act.shape[-1])

        best_val = min(best_val, val_mae)
        history.append((train_mae, val_mae))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  "
                  f"train MAE={train_mae:.5f}  val MAE={val_mae:.5f}")

    return {
        "final_train_loss": history[-1][0],
        "final_val_loss":   history[-1][1],
        "best_val_loss":    best_val,
        "output_tokens":    output_tokens,
        "history":          history,
    }


def plot_results(results: dict, output_path: Path):
    """Bar chart comparatif des stratégies."""
    names = list(results.keys())
    val_maes = [results[n]["best_val_mae"] for n in names]
    params   = [results[n]["params"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1 : Val MAE (lower is better)
    bars1 = axes[0].bar(names, val_maes, color="#26A69A")
    axes[0].set_ylabel("Best Validation MAE (lower = better)")
    axes[0].set_title("Action prediction quality per fusion strategy")
    axes[0].tick_params(axis="x", rotation=20)
    for bar, mae in zip(bars1, val_maes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{mae:.4f}", ha="center", va="bottom", fontsize=9)

    # Plot 2 : Params count (log scale)
    axes[1].bar(names, params, color="#7E57C2")
    axes[1].set_ylabel("Fusion parameters")
    axes[1].set_title("Parameter cost per fusion strategy")
    axes[1].set_yscale("symlog")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

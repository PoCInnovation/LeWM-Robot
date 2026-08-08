"""
Compare les stratégies de fusion multi-camera via un probe d'action prediction.

ADAPTATION MACHINE UNIQUEMENT — méthodologie d'origine préservée :
    - entraînement conjoint FUSION + probe (MLP action), MAE L1 sur actions
      BRUTES, mêmes stratégies et hyperparamètres qu'avant
    - split train/val aléatoire par paires (seedé)

Ce qui change (machine) : lit le format v2 par épisode (encodage shardé) et
streame par batches au lieu de tout charger en RAM (les latents d'un dataset
complet ne tiennent pas en mémoire d'une tâche).

Usage:
    python scripts/03_compare_fusion.py --encoded-data results/encoded/<name>
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fusion import make_fusion
from src.probes import ActionProbe
from src.encoded_data import EncodedEpisodes, PairsView
from src.config import load_config, set_seed, log_environment


# Stratégies à comparer (on évite proprio_guided dans la baseline pour rester simple)
STRATEGIES_TO_TEST = [
    "concat",
    "concat_view",
    "cross_attn_bd",
    "late_cls",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data", required=True,
                        help="Dossier des latents pré-encodés (format v2)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() \
        else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Comparaison des stratégies de fusion multi-camera")
    print("=" * 60)

    # === Données (format v2, actions brutes comme à l'origine) ===
    data_path = Path(args.encoded_data)
    data = EncodedEpisodes(data_path if data_path.is_absolute()
                           else ROOT / data_path)
    print(f"\n[Data] {data.summary()}")
    dim = data.embed_dim
    action_dim = data.action_stats.mean.shape[0]

    pairs = PairsView(data, list(range(len(data))), delta=1,
                      norm_actions=False, norm_proprio=False)
    n = len(pairs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"]))
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size].tolist(), perm[val_size:].tolist()
    print(f"Split : {len(train_idx)} train / {len(val_idx)} val (paires)")

    train_loader = DataLoader(Subset(pairs, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(Subset(pairs, val_idx),
                            batch_size=args.batch_size, shuffle=False)

    results = {}
    for name in STRATEGIES_TO_TEST:
        print(f"\n{'─' * 60}\nStratégie : {name}\n{'─' * 60}")
        set_seed(cfg["seed"])   # même init pour chaque stratégie

        fusion = make_fusion(name, dim=dim).to(device)
        probe = ActionProbe(dim=dim, action_dim=action_dim,
                            hidden=512).to(device)
        n_fusion_params = sum(p.numel() for p in fusion.parameters())
        print(f"Paramètres fusion : {n_fusion_params:,}")

        params = list(fusion.parameters()) + list(probe.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

        best_val, output_tokens = float("inf"), None
        for epoch in range(args.n_epochs):
            fusion.train(); probe.train()
            train_maes = []
            for batch in train_loader:
                z = fusion(batch["z_wrist_t"].to(device),
                           batch["z_global_t"].to(device))
                output_tokens = z.shape[1]
                loss = nn.functional.l1_loss(probe(z),
                                             batch["action"].to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_maes.append(loss.item())
            train_mae = sum(train_maes) / max(1, len(train_maes))

            fusion.eval(); probe.eval()
            val_maes = []
            with torch.no_grad():
                for batch in val_loader:
                    z = fusion(batch["z_wrist_t"].to(device),
                               batch["z_global_t"].to(device))
                    val_maes.append(nn.functional.l1_loss(
                        probe(z), batch["action"].to(device)).item())
            val_mae = sum(val_maes) / max(1, len(val_maes))
            best_val = min(best_val, val_mae)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1:3d}/{args.n_epochs}  "
                      f"train MAE={train_mae:.5f}  val MAE={val_mae:.5f}")

        results[name] = {
            "params":        n_fusion_params,
            "output_tokens": output_tokens,
            "best_val_mae":  best_val,
        }

    # === Résumé ===
    print("\n" + "=" * 60)
    print("RÉSULTATS COMPARATIFS")
    print("=" * 60)
    print(f"{'Stratégie':<18} {'Params':>10} {'Tokens':>8} {'Val MAE':>12}")
    print("-" * 52)
    winner = min(results, key=lambda k: results[k]["best_val_mae"])
    for name, r in results.items():
        mark = "  << gagnante" if name == winner else ""
        print(f"{name:<18} {r['params']:>10,} {r['output_tokens']:>8} "
              f"{r['best_val_mae']:>12.5f}{mark}")

    with open(out_dir / "fusion_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_results(results, out_dir / "fusion_comparison.png")
    print(f"\nRésultats : {out_dir}/fusion_comparison.json + .png")


def plot_results(results: dict, output_path: Path):
    """Bar chart comparatif des stratégies."""
    names = list(results.keys())
    val_maes = [results[n]["best_val_mae"] for n in names]
    params = [results[n]["params"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars1 = axes[0].bar(names, val_maes, color="#26A69A")
    axes[0].set_ylabel("Best Validation MAE (lower = better)")
    axes[0].set_title("Action prediction quality per fusion strategy")
    axes[0].tick_params(axis="x", rotation=20)
    for bar, mae in zip(bars1, val_maes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{mae:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(names, params, color="#7E57C2")
    axes[1].set_ylabel("Fusion parameters")
    axes[1].set_title("Parameter cost per fusion strategy")
    axes[1].set_yscale("symlog")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

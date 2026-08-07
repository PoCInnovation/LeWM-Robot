"""
Compare les stratégies de fusion multi-camera via un probe d'action prediction.

Pour chaque stratégie (lue depuis fusion.strategies_to_test du YAML) :
  1. Charge les latents pré-encodés (format v2, depuis 02_encode_dataset.py)
  2. Entraîne conjointement FUSION + probe (MLP action) — MAE sur validation
  3. Sauvegarde les poids de fusion entraînés → réutilisables par 04
     (--fusion-ckpt), où la fusion est GELÉE.

Corrections vs l'ancienne version :
    - config + seed chargés (résultats reproductibles) ;
    - split par ÉPISODE (le split par paires fuyait de l'info) ;
    - actions normalisées (MAE comparable entre datasets) ;
    - hyperparamètres du probe lus depuis le YAML (probe.*).

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
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fusion import make_fusion
from src.probes import ActionProbe
from src.encoded_data import EncodedEpisodes, PairsView, split_episodes
from src.config import load_config, set_seed, log_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoded-data", required=True,
                        help="Dossier des latents pré-encodés (format v2)")
    parser.add_argument("--output-dir", default="results/fusion")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Défaut : probe.n_epochs de la config")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    log_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    probe_cfg = cfg.get("probe", {})
    n_epochs = args.n_epochs or int(probe_cfg.get("n_epochs", 30))
    batch_size = int(probe_cfg.get("batch_size", 32))
    lr = float(probe_cfg.get("lr", 1e-3))
    weight_decay = float(probe_cfg.get("weight_decay", 1e-4))
    hidden = int(probe_cfg.get("hidden_dim", 512))
    strategies = cfg.get("fusion", {}).get("strategies_to_test",
                                           ["concat", "concat_view",
                                            "cross_attn_bd", "late_cls"])

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Comparaison des stratégies de fusion multi-camera")
    print("=" * 60)

    # === Données (format v2) ===
    data = EncodedEpisodes(ROOT / args.encoded_data
                           if not Path(args.encoded_data).is_absolute()
                           else args.encoded_data)
    print(f"\n[Data] {data.summary()}")
    dim = data.embed_dim
    action_dim = data.action_stats.mean.shape[0]

    val_fraction = float(cfg["dataset"].get("val_fraction", 0.2))
    train_eps, val_eps = split_episodes(len(data), val_fraction, cfg["seed"])
    print(f"[Split par épisode] {len(train_eps)} train / {len(val_eps)} val")

    # delta=1 suffit ici : le probe prédit action_t depuis la frame t
    train_ds = PairsView(data, train_eps, delta=1)
    val_ds = PairsView(data, val_eps, delta=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = {}
    for name in strategies:
        print(f"\n{'─' * 60}\nStratégie : {name}\n{'─' * 60}")
        set_seed(cfg["seed"])   # même init pour chaque stratégie

        kwargs = {"dim": dim}
        if name == "proprio_guided":
            kwargs["action_dim"] = action_dim
        fusion = make_fusion(name, **kwargs).to(device)
        probe = ActionProbe(dim=dim, action_dim=action_dim,
                            hidden=hidden).to(device)
        n_fusion_params = sum(p.numel() for p in fusion.parameters())
        print(f"Paramètres fusion : {n_fusion_params:,}")

        params = list(fusion.parameters()) + list(probe.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        best_val, output_tokens = float("inf"), None
        for epoch in range(n_epochs):
            fusion.train(); probe.train()
            train_maes = []
            for batch in train_loader:
                zw = batch["z_wrist_t"].to(device)
                zg = batch["z_global_t"].to(device)
                act = batch["action"].to(device)        # normalisées
                proprio = batch["proprio"].to(device)
                z = fusion(zw, zg, proprio) if name == "proprio_guided" \
                    else fusion(zw, zg)
                output_tokens = z.shape[1]
                loss = nn.functional.l1_loss(probe(z), act)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_maes.append(loss.item())
            train_mae = sum(train_maes) / max(1, len(train_maes))

            fusion.eval(); probe.eval()
            val_maes = []
            with torch.no_grad():
                for batch in val_loader:
                    zw = batch["z_wrist_t"].to(device)
                    zg = batch["z_global_t"].to(device)
                    act = batch["action"].to(device)
                    proprio = batch["proprio"].to(device)
                    z = fusion(zw, zg, proprio) if name == "proprio_guided" \
                        else fusion(zw, zg)
                    val_maes.append(nn.functional.l1_loss(probe(z), act).item())
            val_mae = sum(val_maes) / max(1, len(val_maes))

            if val_mae < best_val:
                best_val = val_mae
                torch.save(fusion.state_dict(), out_dir / f"{name}.pt")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1:3d}/{n_epochs}  "
                      f"train MAE={train_mae:.5f}  val MAE={val_mae:.5f}")

        results[name] = {
            "params":        n_fusion_params,
            "output_tokens": output_tokens,
            "best_val_mae":  best_val,
            "fusion_ckpt":   str((out_dir / f"{name}.pt").relative_to(ROOT)),
        }

    # === Résumé ===
    print("\n" + "=" * 60)
    print("RÉSULTATS (MAE sur actions NORMALISÉES — plus bas = mieux)")
    print("=" * 60)
    print(f"{'Stratégie':<18} {'Params':>10} {'Tokens':>8} {'Val MAE':>12}")
    print("-" * 52)
    winner = min(results, key=lambda k: results[k]["best_val_mae"])
    for name, r in results.items():
        mark = "  << gagnante" if name == winner else ""
        print(f"{name:<18} {r['params']:>10,} {r['output_tokens']:>8} "
              f"{r['best_val_mae']:>12.5f}{mark}")

    print(f"\nPour le training du predictor :")
    print(f"  python scripts/04_train_predictor.py --encoded-data {args.encoded_data} \\")
    print(f"      --fusion {winner} --fusion-ckpt {results[winner]['fusion_ckpt']}")

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
    axes[0].set_ylabel("Best Validation MAE (normalized actions)")
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

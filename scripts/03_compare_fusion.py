"""
Compare les stratégies de fusion multi-camera via un probe d'action prediction.

Pour chaque stratégie de fusion :
  1. On charge les patches DINOv3 pré-encodés (depuis 02_encode_dataset.py)
  2. On applique la fusion sur (z_wrist, z_global) → z_fused
  3. On entraîne un MLP qui prédit l'action depuis z_fused
  4. On mesure la MAE sur validation
  5. On compare

La meilleure fusion = celle qui rend l'action le plus prédictible
(c'est-à-dire qui préserve le plus d'info utile dans le latent).

Usage:
    python scripts/03_compare_fusion.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.fusion import FUSION_STRATEGIES, make_fusion
from src.probes import ActionProbe, train_probe, ProbeTrainConfig


# Stratégies à comparer (on évite proprio_guided dans la baseline pour rester simple)
STRATEGIES_TO_TEST = [
    "concat",
    "concat_view",
    "cross_attn_bd",
    "late_cls",
]


def main():
    encoded_path = ROOT / "results/encoded/encoded_data.pt"
    if not encoded_path.exists():
        print(f"[ERREUR] Pas de données encodées trouvées : {encoded_path}")
        print("Lance d'abord : python scripts/02_encode_dataset.py <dataset_id>")
        sys.exit(1)

    print("=" * 60)
    print("TEST 3 : Comparaison des stratégies de fusion multi-camera")
    print("=" * 60)

    # === Charger les patches pré-encodés ===
    data = torch.load(encoded_path, weights_only=False)
    z_wrist  = data["z_wrist_t"]   # (N, num_patches, dim)
    z_global = data["z_global_t"]
    actions  = data["action"]      # (N, action_dim)
    dim = data["embed_dim"]
    action_dim = actions.shape[-1]

    print(f"\nDonnées chargées :")
    print(f"  N paires       : {z_wrist.shape[0]}")
    print(f"  Num patches    : {z_wrist.shape[1]}")
    print(f"  Embed dim      : {dim}")
    print(f"  Action dim     : {action_dim}")

    # === Split train/val ===
    n = z_wrist.shape[0]
    perm = torch.randperm(n)
    val_size = max(1, n // 5)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    print(f"\nSplit : {len(train_idx)} train / {len(val_idx)} val")

    # === Boucle sur chaque stratégie ===
    results = {}

    for strategy_name in STRATEGIES_TO_TEST:
        print(f"\n{'─' * 60}")
        print(f"Stratégie : {strategy_name}")
        print(f"{'─' * 60}")

        # Instancier la fusion
        fusion_kwargs = {"dim": dim}
        if strategy_name == "proprio_guided":
            fusion_kwargs["action_dim"] = action_dim
        fusion = make_fusion(strategy_name, **fusion_kwargs)
        n_fusion_params = sum(p.numel() for p in fusion.parameters())
        print(f"Paramètres fusion : {n_fusion_params:,}")

        # Appliquer la fusion (en mode eval, sans gradient)
        fusion.eval()
        with torch.no_grad():
            z_fused_train = fusion(z_wrist[train_idx], z_global[train_idx])
            z_fused_val   = fusion(z_wrist[val_idx],   z_global[val_idx])

        print(f"Latent fusionné train : {z_fused_train.shape}")

        # Pour les stratégies à fusion non-triviale, on entraînerait la fusion
        # ensemble avec le probe. Mais pour une comparaison "à fusion fixe" simple
        # on garde la fusion eval et seul le probe s'entraîne.
        # → cela favorise les fusions qui marchent dès l'init (concat, concat_view)
        # → désavantage les cross-attention qui ont besoin d'entraînement
        # Pour être juste, on va entraîner FUSION + PROBE ensemble :
        z_fused_train_t = z_fused_train  # placeholder for the joint training below

        # Probe d'action
        probe = ActionProbe(dim=dim, action_dim=action_dim, hidden=512)

        # Joint train fusion + probe
        full_model = nn.Sequential()
        full_model.add_module("fusion", fusion)
        full_model.add_module("probe", probe)

        # Pour entraîner conjointement, on doit donner les RAW wrist+global au train
        # → on re-prend les indices et on entraîne avec une boucle adaptée
        result = train_joint(
            fusion=fusion,
            probe=probe,
            z_wrist_train=z_wrist[train_idx],
            z_global_train=z_global[train_idx],
            actions_train=actions[train_idx],
            z_wrist_val=z_wrist[val_idx],
            z_global_val=z_global[val_idx],
            actions_val=actions[val_idx],
            n_epochs=30,
            lr=1e-3,
            batch_size=32,
        )

        results[strategy_name] = {
            "params":           n_fusion_params,
            "output_tokens":    z_fused_train.shape[1],
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
    for name, r in results.items():
        print(f"{name:<18} {r['params']:>10,} {r['output_tokens']:>8} "
              f"{r['final_train_mae']:>12.5f} {r['best_val_mae']:>12.5f}")

    # === Sauver les résultats ===
    out_path = ROOT / "results" / "fusion_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvés dans : {out_path}")

    # === Plot ===
    plot_path = ROOT / "results" / "fusion_comparison.png"
    plot_results(results, plot_path)
    print(f"Plot sauvé dans       : {plot_path}")


def train_joint(fusion, probe, z_wrist_train, z_global_train, actions_train,
                z_wrist_val, z_global_val, actions_val,
                n_epochs=30, lr=1e-3, batch_size=32):
    """Entraîne fusion + probe conjointement avec une loss MSE sur l'action."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fusion.to(device).train()
    probe.to(device).train()

    z_wrist_train, z_global_train = z_wrist_train.to(device), z_global_train.to(device)
    actions_train = actions_train.to(device)
    z_wrist_val, z_global_val = z_wrist_val.to(device), z_global_val.to(device)
    actions_val = actions_val.to(device)

    params = list(fusion.parameters()) + list(probe.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    n = z_wrist_train.shape[0]
    history = []
    best_val = float("inf")

    for epoch in range(n_epochs):
        # Train
        fusion.train(); probe.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            z_fused = fusion(z_wrist_train[idx], z_global_train[idx])
            pred = probe(z_fused)
            loss = nn.functional.l1_loss(pred, actions_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_mae = sum(losses) / len(losses)

        # Val
        fusion.eval(); probe.eval()
        with torch.no_grad():
            z_fused_val = fusion(z_wrist_val, z_global_val)
            pred_val = probe(z_fused_val)
            val_mae = nn.functional.l1_loss(pred_val, actions_val).item()

        best_val = min(best_val, val_mae)
        history.append((train_mae, val_mae))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{n_epochs}  "
                  f"train MAE={train_mae:.5f}  val MAE={val_mae:.5f}")

    return {
        "final_train_loss": history[-1][0],
        "final_val_loss":   history[-1][1],
        "best_val_loss":    best_val,
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

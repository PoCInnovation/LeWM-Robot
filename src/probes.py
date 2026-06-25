"""
Probes pour évaluer la qualité de la fusion multi-camera.

Un "probe" est un petit modèle entraîné sur les latents fusionnés
pour prédire quelque chose (action, position 3D, ...).
La qualité du probe mesure la qualité du latent.

Probes implémentés :
    - ActionProbe   : prédire l'action depuis z_fused (le plus pertinent pour WM)
    - DynamicsProbe : prédire z_t+1 depuis (z_t, action) — mini-WM jouet
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn


class ActionProbe(nn.Module):
    """
    MLP qui prédit l'action depuis le latent fusionné.

    Mesure : à quel point le latent contient l'info nécessaire pour distinguer les actions.
    Une bonne fusion → MAE basse.
    """
    def __init__(self, dim: int = 768, action_dim: int = 6, hidden: int = 512):
        super().__init__()
        self.pool = lambda x: x.mean(dim=1)  # mean pool sur les tokens

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        # z_fused : (B, N, dim)
        pooled = self.pool(z_fused)  # (B, dim)
        return self.mlp(pooled)


class DynamicsProbe(nn.Module):
    """
    Mini predictor jouet : prédit le z_t+1 depuis (z_t, action).

    Plus fidèle au futur usage (prédire des transitions),
    plus représentatif que ActionProbe.
    """
    def __init__(self, dim: int = 768, action_dim: int = 6,
                 n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 2,
                dropout=0.1,
                batch_first=True,
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t:    (B, N, dim) latent fusionné au temps t
            action: (B, action_dim) action prise au temps t
        Returns:
            z_t1_predicted: (B, N, dim)
        """
        # Concaténer le token d'action en début de séquence
        action_token = self.action_proj(action).unsqueeze(1)  # (B, 1, dim)
        x = torch.cat([action_token, z_t], dim=1)             # (B, 1+N, dim)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x[:, 1:]  # retire le token action, retourne le nouveau z


@dataclass
class ProbeTrainConfig:
    """Configuration d'entraînement d'un probe."""
    n_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    val_fraction: float = 0.2
    log_every: int = 10


def train_probe(probe: nn.Module,
                z_train: torch.Tensor,
                y_train: torch.Tensor,
                z_val: torch.Tensor,
                y_val: torch.Tensor,
                config: ProbeTrainConfig) -> dict:
    """
    Entraîne un probe et renvoie un dict de métriques.

    Args:
        probe   : module à entraîner
        z_train : (N, ...) latents d'entraînement
        y_train : (N, ...) cibles d'entraînement
        z_val, y_val : équivalents de validation
        config  : hyperparamètres

    Returns:
        dict avec 'final_train_loss', 'final_val_loss', 'best_val_loss', 'history'
    """
    device = ("cuda" if torch.cuda.is_available() and config.device == "auto"
              else config.device if config.device != "auto" else "cpu")
    probe.to(device)
    z_train, y_train = z_train.to(device), y_train.to(device)
    z_val, y_val     = z_val.to(device),   y_val.to(device)

    optimizer = torch.optim.AdamW(probe.parameters(),
                                   lr=config.lr,
                                   weight_decay=config.weight_decay)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    n = z_train.shape[0]
    for epoch in range(config.n_epochs):
        # Train
        probe.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, config.batch_size):
            idx = perm[i:i + config.batch_size]
            pred = probe(z_train[idx])
            loss = nn.functional.mse_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        # Val
        probe.eval()
        with torch.no_grad():
            pred_val = probe(z_val)
            val_loss = nn.functional.mse_loss(pred_val, y_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        best_val = min(best_val, val_loss)

        if (epoch + 1) % config.log_every == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{config.n_epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}")

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss":   history["val_loss"][-1],
        "best_val_loss":    best_val,
        "history":          history,
    }

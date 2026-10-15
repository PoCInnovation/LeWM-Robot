"""
Losses pour entraîner le world model predictor.

Loss principale :
    - MSE one-step  : prédiction simple z_t → z_t+1
    - MSE multi-step: rollouts plus longs, accumule des erreurs
    - Cosine        : alternative à MSE, parfois meilleure pour features DINOv3

Loss de régularisation optionnelles :
    - DANN (Domain-Adversarial) : si l'on veut aligner sim et real explicitement
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def one_step_mse(predicted: torch.Tensor, target: torch.Tensor,
                  reduction: str = "mean") -> torch.Tensor:
    """
    Loss MSE simple sur une prédiction one-step.

    Args:
        predicted : (B, N, D) prédiction du predictor
        target    : (B, N, D) ground truth z_t+1
    """
    return F.mse_loss(predicted, target, reduction=reduction)


def one_step_cosine(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss cosine : 1 - cosine_similarity entre prédiction et target.

    Plus robuste à la magnitude des features. Parfois meilleur pour DINOv3.
    """
    # Normaliser sur la dim feature
    pred_norm = F.normalize(predicted, dim=-1)
    tgt_norm  = F.normalize(target, dim=-1)
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (B, N)
    return (1 - cos_sim).mean()


def multi_step_mse(predictor: nn.Module,
                    z_start: torch.Tensor,
                    actions: torch.Tensor,
                    targets: torch.Tensor,
                    teacher_forcing: bool = False,
                    grad_checkpoint: bool = False,
                    per_step: bool = False) -> torch.Tensor:
    """
    Loss MSE multi-step : déroule le predictor sur T steps et compare à la séquence cible.

    Le CEM déroule le predictor sur H pas à partir de ses propres prédictions :
    entraîné en one-step, il n'a jamais vu ses erreurs en entrée, et elles
    s'accumulent. En autorégressif, le gradient traverse tout le rollout
    (z_{t+1} prédit sert d'entrée au pas suivant).

    Args:
        z_start : (B, N, D) latent au temps 0
        actions : (B, T, action_dim) actions à appliquer
        targets : (B, T, N, D) latents cibles aux temps 1..T
        teacher_forcing : si True, utilise les targets comme input pour l'étape suivante
                          (évite l'accumulation d'erreur)
                          si False, autorégressif (plus dur, plus réaliste)
        grad_checkpoint : recalcule les activations de chaque pas au backward
                          au lieu de les garder : mémoire ≈ celle d'un seul pas
                          (au lieu de T fois), pour ~1/3 de calcul en plus
        per_step        : renvoie la loss de chaque pas (T,) au lieu de la moyenne

    Losses calculées en fp32 (prédictions sous autocast bf16).
    """
    T = actions.shape[1]
    losses = []
    z = z_start

    for t in range(T):
        if grad_checkpoint and torch.is_grad_enabled():
            pred = checkpoint(predictor, z, actions[:, t, :], use_reentrant=False)
        else:
            pred = predictor(z, actions[:, t, :])
        losses.append(F.mse_loss(pred.float(), targets[:, t].float()))

        # Autorégressif vs teacher forcing
        if teacher_forcing:
            z = targets[:, t]
        else:
            z = pred

    losses = torch.stack(losses)
    return losses if per_step else losses.mean()


def multi_step_combined(predictor: nn.Module,
                         z_start: torch.Tensor,
                         actions: torch.Tensor,
                         targets: torch.Tensor,
                         alpha_mse: float = 1.0,
                         alpha_cos: float = 0.5,
                         teacher_forcing: bool = False) -> dict:
    """
    Combine MSE + Cosine sur un rollout multi-step.

    Returns:
        dict avec 'total', 'mse', 'cosine', 'per_step_loss'
    """
    T = actions.shape[1]
    mses, coss = [], []
    z = z_start

    for t in range(T):
        pred = predictor(z, actions[:, t, :])
        mses.append(F.mse_loss(pred, targets[:, t]))
        coss.append(one_step_cosine(pred, targets[:, t]))

        z = targets[:, t] if teacher_forcing else pred

    mse_total = torch.stack(mses).mean()
    cos_total = torch.stack(coss).mean()
    total = alpha_mse * mse_total + alpha_cos * cos_total

    return {
        "total":         total,
        "mse":           mse_total.detach(),
        "cosine":        cos_total.detach(),
        "per_step_loss": torch.stack(mses).detach(),
    }


class DomainDiscriminator(nn.Module):
    """
    Discriminateur sim vs real pour DANN.

    Prend des features et essaie de prédire si elles viennent du sim ou du réel.
    Le predictor doit le "tromper" -> aligne les distributions.
    """

    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),  # logit binaire sim(0) / real(1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features : (B, N, D) ou (B, D)
        if features.dim() == 3:
            features = features.mean(dim=1)  # mean pool sur les patches
        return self.net(features).squeeze(-1)  # (B,)


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer pour DANN."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Inverse le gradient. À insérer entre le predictor et le discriminateur."""
    return GradientReversal.apply(x, lambda_)

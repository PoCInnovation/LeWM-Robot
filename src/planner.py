"""
CEM (Cross-Entropy Method) planner pour la planification dans l'espace latent.

Au runtime, on:
    1. Observe l'état actuel via DINOv3 + fusion -> z_current
    2. Encode l'image-goal -> z_goal (calculé 1 seule fois)
    3. À chaque step de contrôle, on demande au CEM:
        "Trouve la séquence d'actions qui mène z_current vers z_goal"
    4. CEM échantillonne des actions, simule via le predictor, score, raffine
    5. On exécute la première action de la meilleure séquence
    6. On replanifie (Model Predictive Control)

Algorithm:
    - Sample N_samples séquences d'actions ~ N(mean, std)
    - Pour chaque, rollout via le predictor sur T steps
    - Score par distance à z_goal (cosine ou L2)
    - Garder les top-K (élites)
    - Refit mean/std sur les élites
    - Répéter N_iter fois
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CEMConfig:
    """Configuration du CEM planner."""
    horizon: int = 10                   # nombre de steps planifiés
    n_samples: int = 200                # candidats par itération
    n_elites: int = 20                  # top-K à garder
    n_iterations: int = 3               # raffinements
    action_dim: int = 6                 # 6 servos SO-101
    action_low: float = -1.0            # bornes des actions (normalisées)
    action_high: float = 1.0
    initial_std: float = 1.0
    cost_type: str = "cosine"           # "cosine" / "mse"
    elite_momentum: float = 0.0         # 0 = full refit, >0 = lisser entre iter
    device: str = "auto"


class CEMPlanner:
    """
    Cross-Entropy Method planner pour world model.

    Usage:
        planner = CEMPlanner(predictor, config)
        actions = planner.plan(z_current, z_goal)
        # actions : (horizon, action_dim) — exécuter la première
    """

    def __init__(self, predictor: nn.Module, config: CEMConfig):
        self.predictor = predictor
        self.config = config
        self.device = self._resolve_device()
        self.predictor.to(self.device).eval()

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    @torch.no_grad()
    def plan(self, z_current: torch.Tensor, z_goal: torch.Tensor,
             return_diagnostics: bool = False) -> torch.Tensor:
        """
        Trouve la séquence d'actions optimale.

        Args:
            z_current : (N, D) latent fusionné de l'état actuel
                        (ou (1, N, D), sera squeezé si batch_size=1)
            z_goal    : (N, D) latent fusionné de l'image-goal
            return_diagnostics : si True, renvoie aussi les coûts et stats

        Returns:
            actions : (horizon, action_dim) séquence d'actions optimales
            diagnostics (optional) : dict avec costs, mean_history, ...
        """
        cfg = self.config
        device = self.device

        # Squeeze si on a un batch dim de 1
        if z_current.dim() == 3:
            z_current = z_current[0]
        if z_goal.dim() == 3:
            z_goal = z_goal[0]

        N, D = z_current.shape

        # Init distribution
        mean = torch.zeros(cfg.horizon, cfg.action_dim, device=device)
        std = torch.ones(cfg.horizon, cfg.action_dim, device=device) * cfg.initial_std

        history = {"mean": [], "std": [], "best_cost": []}

        for iteration in range(cfg.n_iterations):
            # 1. Sample actions
            actions = (mean.unsqueeze(0) + std.unsqueeze(0)
                       * torch.randn(cfg.n_samples, cfg.horizon,
                                      cfg.action_dim, device=device))
            actions = actions.clamp(cfg.action_low, cfg.action_high)

            # 2. Rollout en batch
            z_pred = z_current.unsqueeze(0).expand(cfg.n_samples, -1, -1).clone()
            z_pred = z_pred.to(device)
            for t in range(cfg.horizon):
                z_pred = self.predictor(z_pred, actions[:, t, :])

            # 3. Compute cost
            costs = self._compute_cost(z_pred, z_goal)  # (n_samples,)

            # 4. Select elites
            elite_idx = costs.argsort()[:cfg.n_elites]
            elites = actions[elite_idx]  # (K, horizon, action_dim)

            # 5. Refit distribution
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0).clamp(min=1e-3)

            # Momentum (lissage entre itérations)
            mean = (1 - cfg.elite_momentum) * new_mean + cfg.elite_momentum * mean
            std  = (1 - cfg.elite_momentum) * new_std  + cfg.elite_momentum * std

            history["mean"].append(mean.detach().cpu())
            history["std"].append(std.detach().cpu())
            history["best_cost"].append(costs[elite_idx[0]].item())

        if return_diagnostics:
            return mean, history
        return mean

    def _compute_cost(self, z_pred_final: torch.Tensor,
                       z_goal: torch.Tensor) -> torch.Tensor:
        """
        Calcule le coût (distance au goal).

        Args:
            z_pred_final : (n_samples, N, D)
            z_goal       : (N, D)
        Returns:
            costs : (n_samples,)
        """
        if self.config.cost_type == "cosine":
            # Cosine similarity, on minimise 1 - cos_sim
            pred_flat = z_pred_final.reshape(z_pred_final.shape[0], -1)
            goal_flat = z_goal.reshape(-1).unsqueeze(0)
            cos_sim = F.cosine_similarity(pred_flat, goal_flat, dim=-1)
            return 1 - cos_sim
        elif self.config.cost_type == "mse":
            return ((z_pred_final - z_goal.unsqueeze(0)) ** 2).mean(dim=(-1, -2))
        else:
            raise ValueError(f"Unknown cost_type: {self.config.cost_type}")


class MPCController:
    """
    Wrapper Model Predictive Control autour du CEM planner.

    À chaque step de contrôle :
        1. Observe → encode → fusionne → z_current
        2. plan(z_current, z_goal) → séquence d'actions
        3. Exécute la PREMIÈRE action
        4. Boucle

    L'idée du MPC : on n'exécute jamais une longue séquence aveuglément.
    On replanifie à chaque step pour corriger les écarts entre prédiction et réel.
    """

    def __init__(self, planner: CEMPlanner, replan_every: int = 1):
        self.planner = planner
        self.replan_every = replan_every  # replanifie tous les N steps (1 = chaque step)
        self.step_count = 0
        self._cached_plan: Optional[torch.Tensor] = None

    def get_action(self, z_current: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        Renvoie l'action à exécuter maintenant.

        Args:
            z_current : (N, D) latent courant
            z_goal    : (N, D) latent goal

        Returns:
            action : (action_dim,)
        """
        if self.step_count % self.replan_every == 0:
            # Replanifier
            self._cached_plan = self.planner.plan(z_current, z_goal)
            action_idx = 0
        else:
            action_idx = self.step_count % self.replan_every

        action = self._cached_plan[action_idx]
        self.step_count += 1
        return action

    def reset(self):
        """À appeler entre les épisodes pour resetter l'état du MPC."""
        self.step_count = 0
        self._cached_plan = None


if __name__ == "__main__":
    # Sanity check : planner sur un predictor jouet
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.predictor import WorldModelPredictor, PredictorConfig

    print("=== Sanity check CEM Planner ===\n")

    # Mini predictor
    pred_cfg = PredictorConfig(embed_dim=384, action_dim=6, n_layers=2, n_heads=4)
    predictor = WorldModelPredictor(pred_cfg)

    # CEM config
    cem_cfg = CEMConfig(
        horizon=5,            # 5 steps de planification
        n_samples=64,         # 64 candidats
        n_elites=8,           # top 8
        n_iterations=2,       # 2 raffinements
        action_dim=6,
    )
    planner = CEMPlanner(predictor, cem_cfg)

    # États jouets
    N = 50  # nombre de patches
    z_current = torch.randn(N, pred_cfg.embed_dim)
    z_goal    = torch.randn(N, pred_cfg.embed_dim)

    # Planning
    import time
    t0 = time.time()
    actions = planner.plan(z_current, z_goal)
    dt = time.time() - t0
    print(f"Planning latency : {dt * 1000:.1f} ms")
    print(f"Actions shape    : {tuple(actions.shape)} "
          f"(expected: ({cem_cfg.horizon}, 6))")
    print(f"Actions sample 0 : {actions[0].numpy().round(3)}")

    # Test MPC
    print("\nTest MPC controller :")
    mpc = MPCController(planner)
    for step in range(3):
        action = mpc.get_action(z_current, z_goal)
        print(f"  step {step}  action = {action.numpy().round(3)}")

    print("\n[OK] Planner CEM + MPC fonctionnent.")

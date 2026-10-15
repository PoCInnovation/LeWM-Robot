"""
Boucle MPC du world model sur le SO-101 : observation → latent → CEM → cible.

Séparé du matériel (scripts/09_robot_mpc.py) pour être testé sans robot :
    - `WorldModelAgent` : encode + fusionne les 2 caméras, planifie vers le
      latent goal (warm start), convertit la 1re action du plan en cible
      articulaire (degrés) bornée autour de l'état courant ;
    - `run_mpc` : boucle observation → action → attente d'un pas du modèle
      (frameskip frames) sur tout objet qui a l'API robot de lerobot
      (`get_observation()` / `send_action()`) ;
    - `FakeRobot` : robot simulé (images aléatoires) pour tester la boucle.

Mêmes conventions que le training :
    - images (H, W, 3) uint8 RGB des caméras « wrist » et « front »
      (observation.images.wrist / .front du dataset), redimensionnées en
      224 × 224 par l'encodeur comme en 02 ;
    - mode delta : cible = état courant + déplacement planifié ; un pas du
      modèle = frameskip frames (5 à 30 fps ≈ 0,17 s).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.latents import ActionStats
from src.planner import CEMPlanner, shift_plan

# Ordre des articulations de duck_dataset_v3 (meta/info.json, action et
# observation.state) — celui des actions du predictor.
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_roll", "gripper")
CAMERAS = ("wrist", "front")


def image_to_chw(img) -> torch.Tensor:
    """Image caméra (H, W, 3) uint8 RGB (numpy ou tensor) → (3, H, W) uint8."""
    t = torch.as_tensor(np.asarray(img))
    if t.dim() != 3:
        raise ValueError(f"image attendue (H, W, 3), reçu {tuple(t.shape)}")
    if t.shape[-1] == 3 and t.shape[0] != 3:
        t = t.permute(2, 0, 1)
    if t.dtype != torch.uint8:
        raise ValueError(f"image uint8 attendue, reçu {t.dtype}")
    return t.contiguous()


def state_of(obs: Dict) -> torch.Tensor:
    """Positions articulaires (degrés) d'une observation lerobot, ordre MOTORS."""
    return torch.tensor([float(obs[f"{m}.pos"]) for m in MOTORS])


@dataclass
class Step:
    target: torch.Tensor          # (6,) cible envoyée (degrés)
    raw_target: torch.Tensor      # (6,) avant bornage de sécurité
    action: torch.Tensor          # (6,) 1re action du plan, dénormalisée
    cost: float                   # coût CEM final (distance latente au goal)
    plan_ms: float
    clipped: bool


class WorldModelAgent:
    """
    Args:
        encode       : (images (B, 3, H, W) uint8) → patches (B, P, D)
        fusion       : module de fusion du checkpoint
        planner      : CEMPlanner sur le predictor du checkpoint
        stats        : ActionStats du checkpoint (mode absolute / delta)
        max_step_deg : écart max |cible − état| par articulation et par pas
                       (sécurité, en plus de max_relative_target de lerobot)
        warm_start   : repart du plan précédent décalé d'un pas
    """

    def __init__(self, encode: Callable, fusion: torch.nn.Module,
                 planner: CEMPlanner, stats: ActionStats,
                 max_step_deg: "float | Sequence[float]" = 15.0,
                 warm_start: bool = True):
        self.encode = encode
        self.fusion = fusion
        self.planner = planner
        self.stats = stats
        self.max_step = torch.as_tensor(max_step_deg, dtype=torch.float32
                                        ).expand(len(MOTORS)).clone()
        self.warm_start = warm_start
        self.z_goal: Optional[torch.Tensor] = None
        self._plan: Optional[torch.Tensor] = None
        self._used = 0                     # actions du plan déjà exécutées

    @torch.no_grad()
    def latent(self, wrist: torch.Tensor, front: torch.Tensor) -> torch.Tensor:
        """(3, H, W) uint8 ×2 → latent fusionné (N, D)."""
        z_w = self.encode(wrist.unsqueeze(0))
        z_f = self.encode(front.unsqueeze(0))
        return self.fusion(z_w.float(), z_f.float())[0]

    def set_goal(self, wrist: torch.Tensor, front: torch.Tensor) -> None:
        self.z_goal = self.latent(wrist, front)
        self._plan = None

    def reset(self) -> None:
        self._plan = None

    def act(self, wrist: torch.Tensor, front: torch.Tensor,
            state: torch.Tensor) -> Step:
        if self.z_goal is None:
            raise RuntimeError("set_goal() avant act()")
        z = self.latent(wrist, front)
        init = shift_plan(self._plan, max(1, self._used)) if self.warm_start else None
        t0 = time.time()
        plan, diag = self.planner.plan(z, self.z_goal, return_diagnostics=True,
                                       init_mean=init)
        if self.planner.device.type == "cuda":
            torch.cuda.synchronize()
        plan_ms = (time.time() - t0) * 1000
        self._plan = plan.detach().float().cpu()
        self._used = 0
        return self.follow(state, cost=float(diag["best_cost"][-1]), plan_ms=plan_ms)

    def follow(self, state: torch.Tensor, cost: float = float("nan"),
               plan_ms: float = 0.0) -> Step:
        """Action suivante du plan courant, sans replanifier (cible calculée
        depuis l'état mesuré maintenant)."""
        if self._plan is None or self._used >= len(self._plan):
            raise RuntimeError("plus d'action dans le plan : appeler act()")
        action = self.stats.denormalize(self._plan[self._used])
        self._used += 1
        state = state.float()
        raw = state + action if self.stats.mode == "delta" else action
        target = torch.maximum(torch.minimum(raw, state + self.max_step),
                               state - self.max_step)
        return Step(target=target, raw_target=raw, action=action, cost=cost,
                    plan_ms=plan_ms, clipped=bool((target != raw).any()))


def run_mpc(robot, agent: WorldModelAgent, *, frameskip: int, fps: float = 30.0,
            max_steps: int = 100, execute: bool = False,
            actions_per_plan: int = 1,
            stop_cost: Optional[float] = None,
            log_path: Optional[Path] = None,
            sleep: Callable[[float], None] = time.sleep,
            verbose: bool = True) -> List[dict]:
    """
    Boucle MPC : à chaque pas, observe, planifie, envoie la cible (si
    `execute`), puis laisse frameskip/fps secondes au bras pour l'atteindre
    — la durée d'un pas du modèle. Le temps de planification s'ajoute
    (bras immobile pendant le CEM).

    Sans `execute` : rien n'est envoyé (vérifier les cibles proposées en
    bougeant le bras à la main).

    actions_per_plan k > 1 : exécute les k premières actions du plan avant
    de replanifier (cibles recalculées depuis l'état mesuré) — le bras
    s'arrête k fois moins souvent pendant le CEM (~0,6 s sur 4090 pour
    un pas de 0,17 s), au prix d'une boucle moins réactive.

    Returns:
        une entrée de log par pas (aussi écrite en JSONL dans log_path)
    """
    step_s = frameskip / fps
    log: List[dict] = []
    out = open(log_path, "a") if log_path else None
    try:
        for i in range(max_steps):
            obs = robot.get_observation()
            state = state_of(obs)
            replan = i % max(1, actions_per_plan) == 0
            step = (agent.act(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]), state)
                    if replan else agent.follow(state))
            sent = None
            if execute:
                sent = robot.send_action({f"{m}.pos": float(v)
                                          for m, v in zip(MOTORS, step.target)})
                t_sent = time.time()
            entry = {"step": i, "time": time.time(), "state": state.tolist(),
                     "target": step.target.tolist(), "raw_target": step.raw_target.tolist(),
                     "action": step.action.tolist(), "cost": step.cost,
                     "plan_ms": step.plan_ms, "clipped": step.clipped,
                     "executed": execute, "replanned": replan,
                     "sent": [float(sent[f"{m}.pos"]) for m in MOTORS] if sent else None}
            log.append(entry)
            if out:
                out.write(json.dumps(entry) + "\n")
                out.flush()
            if verbose:
                delta = (step.target - state).numpy().round(1)
                print(f"[MPC] pas {i:3d}  "
                      + (f"coût {step.cost:.4f}  plan {step.plan_ms:.0f} ms  " if replan
                         else "(suite du plan)  ")
                      + f"Δ cible {delta}{'  (borné)' if step.clipped else ''}"
                      + ("" if execute else "  [non envoyé]"))
            if replan and stop_cost is not None and step.cost <= stop_cost:
                if verbose:
                    print(f"[MPC] coût ≤ {stop_cost} : goal atteint (au sens du latent)")
                break
            if execute:
                sleep(max(0.0, step_s - (time.time() - t_sent)))
    finally:
        if out:
            out.close()
    return log


@dataclass
class FakeRobot:
    """Robot simulé (API lerobot) : images aléatoires, la cible devient l'état."""
    height: int = 480
    width: int = 640
    seed: int = 0
    state: List[float] = field(default_factory=lambda: [0.0, -90.0, 90.0, 60.0, 0.0, 5.0])
    sent: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> Dict:
        obs = {f"{m}.pos": v for m, v in zip(MOTORS, self.state)}
        for cam in CAMERAS:
            obs[cam] = self._rng.integers(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        self.sent.append(dict(action))
        self.state = [float(action[f"{m}.pos"]) for m in MOTORS]
        return action

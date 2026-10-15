"""Boucle MPC robot (src/control.py) sans matériel — CPU only."""

import json

import numpy as np
import torch

from src.control import MOTORS, FakeRobot, WorldModelAgent, image_to_chw, run_mpc, state_of
from src.fusion import make_fusion
from src.latents import ActionStats
from src.planner import CEMConfig, CEMPlanner, shift_plan
from src.predictor import PredictorConfig, WorldModelPredictor

P, D = 4, 16


def stub_encode(images):
    """Patches déterministes dérivés des pixels (B, 3, H, W) uint8 → (B, P, D)."""
    x = images.float().mean(dim=(1, 2, 3), keepdim=True) / 255.0
    return x.view(-1, 1, 1).expand(-1, P, D) + torch.linspace(0, 1, D)


def make_agent(mode="delta", max_step=15.0, warm_start=True):
    torch.manual_seed(0)
    pred = WorldModelPredictor(PredictorConfig(embed_dim=D, action_dim=6, n_layers=1,
                                               n_heads=4, ffn_dim=32, residual=True))
    stats = ActionStats(mean=[1.0] * 6, std=[2.0] * 6, low=[-6.0] * 6, high=[6.0] * 6,
                        mode=mode)
    low, high = stats.normalized_bounds()
    planner = CEMPlanner(pred, CEMConfig(horizon=3, n_samples=16, n_elites=4,
                                         n_iterations=2, action_dim=6, action_low=low,
                                         action_high=high, device="cpu", precision="fp32"))
    return WorldModelAgent(stub_encode, make_fusion("concat", dim=D), planner, stats,
                           max_step_deg=max_step, warm_start=warm_start)


def test_image_and_state_conversion():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert image_to_chw(img).shape == (3, 480, 640)
    obs = {f"{m}.pos": float(i) for i, m in enumerate(MOTORS)}
    assert state_of(obs).tolist() == [0, 1, 2, 3, 4, 5]


def test_delta_target_is_state_plus_denormalized_action_and_bounded():
    robot = FakeRobot()
    agent = make_agent(mode="delta", max_step=100.0)
    obs = robot.get_observation()
    agent.set_goal(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]))
    state = state_of(obs)
    step = agent.act(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]), state)
    assert torch.allclose(step.target, state + step.action)
    assert torch.allclose(step.action, agent._plan[0] * 2.0 + 1.0)   # dénormalisation

    agent = make_agent(mode="delta", max_step=0.5)
    agent.set_goal(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]))
    step = agent.act(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]), state)
    assert ((step.target - state).abs() <= 0.5 + 1e-6).all()


def test_absolute_mode_targets_the_action():
    robot = FakeRobot()
    agent = make_agent(mode="absolute", max_step=1e3)
    obs = robot.get_observation()
    agent.set_goal(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]))
    step = agent.act(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]), state_of(obs))
    assert torch.allclose(step.target, step.action)


def test_warm_start_shifts_previous_plan():
    plan = torch.arange(6.0).view(3, 2)
    assert shift_plan(plan).tolist() == [[2, 3], [4, 5], [0, 0]]
    assert shift_plan(None) is None


def test_dry_run_sends_nothing_and_execute_moves_the_fake_arm(tmp_path):
    robot = FakeRobot()
    agent = make_agent()
    obs = robot.get_observation()
    agent.set_goal(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]))
    log = run_mpc(robot, agent, frameskip=5, max_steps=3, execute=False, verbose=False)
    assert len(log) == 3 and robot.sent == []

    waits = []
    path = tmp_path / "run.jsonl"
    log = run_mpc(robot, agent, frameskip=5, max_steps=3, execute=True,
                  log_path=path, sleep=waits.append, verbose=False)
    assert len(robot.sent) == 3 and len(waits) == 3
    assert all(0 <= w <= 5 / 30 for w in waits)
    assert robot.state == log[-1]["target"]
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["step"] for l in lines] == [0, 1, 2] and lines[0]["executed"]


def test_stop_cost_ends_the_episode():
    robot = FakeRobot()
    agent = make_agent()
    obs = robot.get_observation()
    agent.set_goal(image_to_chw(obs["wrist"]), image_to_chw(obs["front"]))
    log = run_mpc(robot, agent, frameskip=5, max_steps=10, stop_cost=10.0, verbose=False)
    assert len(log) == 1

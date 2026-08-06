"""Closed-loop tests for the scripted pick-and-place policy.

These run the real ``ScriptedPickPlace`` against the CPU surrogate, so what is
checked is the behaviour of the shipped controller, not a reimplementation.

The assertions are deliberately about *logic*, not about task success: whether
the arm actually grips a 2.5 cm cube depends on link geometry the surrogate only
approximates. What must hold on any reasonable arm is that the state machine
always terminates, never emits NaN, and never commands a joint past its stop —
because a single violation of those, repeated over a 5000-episode run, silently
poisons the dataset.
"""

import pytest
import torch

from fake_env import FakeEnv, FakeRigidObject, run_controller
from so101_surrogate import RobotSurrogate, randomized_kinematics

from sim_to_real_so101.utils.scripted_policy import ScriptedPickPlace

# Reproduces scenes/cube_to_box.py, including its reset randomisation ranges.
CUBE_HOME = (0.22, -0.09, 0.06)
BOX_HOME = (0.22, 0.10, 0.038)
CUBE_RANGE = {"x": (-0.06, 0.06), "y": (-0.05, 0.05)}
BOX_RANGE = {"x": (-0.05, 0.05), "y": (-0.03, 0.04)}

MAX_STEPS = 1500
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _placements(seed):
    """Cube and box poses as a reset of ``cube_to_box`` would draw them."""
    generator = torch.Generator().manual_seed(seed)

    def offset(ranges):
        return [
            ranges.get(axis, (0.0, 0.0))[0]
            + torch.rand(1, generator=generator).item()
            * (ranges.get(axis, (0.0, 0.0))[1] - ranges.get(axis, (0.0, 0.0))[0])
            for axis in ("x", "y", "z")
        ]

    cube_offset = offset(CUBE_RANGE)
    box_offset = offset(BOX_RANGE)
    cube = tuple(a + b for a, b in zip(CUBE_HOME, cube_offset))
    box = tuple(a + b for a, b in zip(BOX_HOME, box_offset))
    return cube, box


def build_env(seed=0, robot=None):
    cube, box = _placements(seed)
    return FakeEnv(
        robot=robot,
        objects={"Cube": FakeRigidObject(cube), "BoxFloor": FakeRigidObject(box)},
    )


def _assert_command_sanity(env, result):
    """Commands must be finite and inside the joint stops, always."""
    actions = result["actions"]
    assert not torch.isnan(actions).any(), "policy emitted NaN"
    assert not torch.isinf(actions).any(), "policy emitted inf"

    limits = env.robot.data.soft_joint_pos_limits[0]
    below = actions < limits[:, 0] - 1e-5
    above = actions > limits[:, 1] + 1e-5
    assert not below.any(), f"commanded below the lower stop on joints {below.any(0).nonzero().flatten().tolist()}"
    assert not above.any(), f"commanded above the upper stop on joints {above.any(0).nonzero().flatten().tolist()}"


@pytest.mark.parametrize("seed", SEEDS)
def test_state_machine_always_terminates(seed):
    """No placement may leave the policy spinning — that would hang collection."""
    env = build_env(seed)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    assert result["status"] in ("done", "failed"), (
        f"still running after {MAX_STEPS} steps (seed {seed})"
    )
    assert result["steps"] < MAX_STEPS
    _assert_command_sanity(env, result)


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_survives_randomised_arm_geometry(seed):
    """The controller must not be tuned to one particular set of link lengths."""
    robot = RobotSurrogate(kinematics=randomized_kinematics(seed))
    env = build_env(seed, robot=robot)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    assert result["status"] in ("done", "failed")
    _assert_command_sanity(env, result)


@pytest.mark.parametrize("gain,step", [(0.2, 0.05), (0.5, 0.15), (0.9, 0.4)])
def test_survives_a_sluggish_or_twitchy_follower(gain, step):
    """Actuator responsiveness is only approximated; the loop must tolerate it."""
    robot = RobotSurrogate(tracking_gain=gain, max_joint_step=step)
    env = build_env(0, robot=robot)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    assert result["status"] in ("done", "failed")
    _assert_command_sanity(env, result)


def test_runs_when_the_wrist_body_is_absent():
    """``scripted_policy`` has a fallback path for this; exercise it."""
    robot = RobotSurrogate(
        body_names=["base", "shoulder", "upper_arm", "forearm", "link4", "gripper", "jaw"]
    )
    env = build_env(0, robot=robot)
    policy = ScriptedPickPlace(env)
    assert policy._wrist_body_id is None

    result = run_controller(env, policy, max_steps=MAX_STEPS)
    assert result["status"] in ("done", "failed")
    _assert_command_sanity(env, result)


def test_runs_when_the_jaw_body_is_absent():
    """Falls back to the gripper body as the control point."""
    robot = RobotSurrogate(
        body_names=["base", "shoulder", "upper_arm", "forearm", "wrist", "gripper", "tip"]
    )
    env = build_env(0, robot=robot)
    policy = ScriptedPickPlace(env)
    assert policy._jaw_body_id is None

    result = run_controller(env, policy, max_steps=MAX_STEPS)
    assert result["status"] in ("done", "failed")
    _assert_command_sanity(env, result)


def test_reset_restarts_the_state_machine():
    """The collection loop resets between episodes; state must not leak across."""
    env = build_env(0)
    policy = ScriptedPickPlace(env)
    run_controller(env, policy, max_steps=MAX_STEPS)
    assert policy.status != "running"

    env.robot.set_joint_pos(env.robot.data.default_joint_pos[0])
    policy.reset()
    assert policy.status == "running"
    assert policy._state == 0

    result = run_controller(env, policy, max_steps=MAX_STEPS)
    assert result["status"] in ("done", "failed")
    assert result["steps"] > 1, "second episode terminated immediately"


def test_gripper_opens_and_closes():
    """A pick-and-place that never actuates the jaw is not picking anything."""
    env = build_env(0)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    jaw = result["actions"][:, 5]
    assert jaw.max() - jaw.min() > 0.3, "the jaw barely moved over the episode"


def test_live_tuning_keys_adjust_the_grasp_offsets():
    """The O/L/I/K/J/H hot keys are the binome's only in-sim tuning handle."""
    import types

    env = build_env(0)
    policy = ScriptedPickPlace(env)

    def press(key):
        event = types.SimpleNamespace(
            input=types.SimpleNamespace(name=key), type="KEY_PRESS"
        )
        policy._on_key(event)

    before = (policy._grasp_offset, policy._finger_len, policy._grasp_lateral)
    press("O")
    press("I")
    press("J")
    assert policy._grasp_offset > before[0]
    assert policy._finger_len > before[1]
    assert policy._grasp_lateral > before[2]

    press("L")
    press("K")
    press("H")
    assert policy._grasp_offset == pytest.approx(before[0], abs=1e-9)
    assert policy._finger_len == pytest.approx(before[1], abs=1e-9)
    assert policy._grasp_lateral == pytest.approx(before[2], abs=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect #2 (SIMULATION.md): the scripted policy pins Wrist_Roll to its "
        "home value, so that action dimension has std=0 across the whole dataset — "
        "confirmed on the 200 collected episodes. Remove this marker once phase 2 "
        "makes the roll move."
    ),
)
def test_wrist_roll_is_actuated():
    """Every action dimension must carry signal for the world model to learn it."""
    env = build_env(0)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    assert result["actions"][:, 4].std() > 1e-3


def _check_result(env, result):
    from sim_to_real_so101.utils.episode_validation import check_episode

    return check_episode(
        result["actions"].numpy(),
        states=result["joint_pos"].numpy(),
        joint_limits=env.robot.data.soft_joint_pos_limits[0].numpy(),
    )


def test_the_validity_filter_accepts_a_healthy_policy_episode():
    """End-to-end guard: the shipped controller must survive the shipped filter.

    This caught the filter's first version rejecting *every* pick-and-place
    episode: the policy flips the jaw from JAW_OPEN to JAW_CLOSED in one frame
    (a measured 0.77 rad step) and that read as a diverging solver.
    """
    env = build_env(3)  # a placement the policy carries through to "done"
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)
    assert result["status"] == "done"

    report = _check_result(env, result)
    assert report.valid, report.summary()
    # The known dead dimension must still be surfaced, not swallowed.
    assert report.stats["dead_dimensions"] == [4]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_filter_never_rejects_a_policy_episode_for_divergence(seed):
    """No real trajectory may ever be mistaken for a blown-up solver.

    Failed attempts are kept by design, so the only acceptable rejection is
    genuine degeneracy — never the jaw set-point step, never normal IK motion.
    """
    from sim_to_real_so101.utils.episode_validation import DIVERGED

    env = build_env(seed)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    report = _check_result(env, result)
    assert DIVERGED not in {item["code"] for item in report.reasons}, report.summary()


def test_the_filter_rejects_an_episode_where_the_wrist_wedges():
    """The filter earning its keep on a real degenerate case.

    With this placement the wrist servo drives Wrist_Pitch into its stop and the
    IK never recovers: the arm creeps for 425 frames with three action
    dimensions flat and one joint welded to a limit. Nothing usable — and,
    unlike a failed grasp, nothing a world model can learn from.
    """
    from sim_to_real_so101.utils.episode_validation import JOINT_PINNED

    env = build_env(1)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    report = _check_result(env, result)
    assert not report.valid
    assert JOINT_PINNED in {item["code"] for item in report.reasons}
    assert report.stats["saturation_fraction"][3] == pytest.approx(1.0)


def test_the_validity_filter_tolerates_the_jaw_setpoint_step():
    """Same guard, on commands alone — the recorder may only have those."""
    from sim_to_real_so101.utils.episode_validation import check_episode

    env = build_env(0)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    actions = result["actions"].numpy()
    jaw_step = float(abs(actions[1:, 5] - actions[:-1, 5]).max())
    assert jaw_step > 0.5, "fixture no longer exercises the set-point step"

    report = check_episode(actions)
    assert report.valid, report.summary()


def test_all_other_action_dimensions_carry_signal():
    """Guards the five dimensions that do move, so none silently goes flat."""
    env = build_env(0)
    policy = ScriptedPickPlace(env)
    result = run_controller(env, policy, max_steps=MAX_STEPS)

    stds = result["actions"].std(0)
    for joint in (0, 1, 2, 3, 5):
        assert stds[joint] > 1e-3, f"joint {joint} is constant over the episode"

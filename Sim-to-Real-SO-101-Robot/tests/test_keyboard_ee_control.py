"""Closed-loop tests for the cartesian keyboard teleoperation.

This controller is what a human uses to record demonstrations by hand, and it
shares its IK and its wrist servo with the scripted policy — so a regression
here breaks both. Key events are pushed through the real event handler rather
than by poking at internals, so the key bindings themselves are covered.
"""

import types

import pytest
import torch

from fake_env import FakeEnv
from so101_surrogate import RobotSurrogate, randomized_kinematics

from sim_to_real_so101.utils.keyboard_ee_control import (
    JAW_KEYS,
    MOVE_BINDINGS,
    KeyboardEEControl,
)


def press(controller, key):
    controller._on_keyboard_event(
        types.SimpleNamespace(input=types.SimpleNamespace(name=key), type="KEY_PRESS")
    )


def release(controller, key):
    controller._on_keyboard_event(
        types.SimpleNamespace(input=types.SimpleNamespace(name=key), type="KEY_RELEASE")
    )


def ee_position(env):
    body_id = env.robot.find_bodies(["gripper"])[0][0]
    return env.robot.data.body_pos_w[0, body_id] - env.scene.env_origins[0]


def drive(env, controller, steps):
    """Run ``steps`` control steps and return the commands that were issued."""
    commands = []
    for _ in range(steps):
        targets = controller.step()
        commands.append(list(targets))
        env.step(targets)
    return torch.tensor(commands)


def test_idle_controller_does_not_drift():
    """With no key held the arm must hold station — otherwise recordings drift."""
    env = FakeEnv()
    controller = KeyboardEEControl(env)

    start = ee_position(env).clone()
    drive(env, controller, 200)
    assert torch.linalg.norm(ee_position(env) - start) < 1e-4


@pytest.mark.parametrize("key,axis,direction", [
    (key, axis, direction) for key, (axis, direction) in MOVE_BINDINGS.items()
])
def test_movement_keys_move_the_end_effector_the_right_way(key, axis, direction):
    """The damped-least-squares IK must actually track the commanded direction.

    Asserting the sign of the displacement along the commanded axis is
    geometry-independent: it holds for any non-singular arm, which is what makes
    it a meaningful check despite the surrogate's approximate link lengths.
    """
    env = FakeEnv()
    controller = KeyboardEEControl(env)
    drive(env, controller, 5)  # let the controller sync onto the current pose

    start = ee_position(env).clone()
    press(controller, key)
    drive(env, controller, 120)
    release(controller, key)

    displacement = ee_position(env) - start
    moved = float(displacement[axis]) * direction
    assert moved > 0.01, (
        f"key '{key}' should move axis {axis} by {direction:+.0f}, "
        f"got displacement {displacement.tolist()}"
    )


@pytest.mark.parametrize("key,direction", list(JAW_KEYS.items()))
def test_jaw_keys_open_and_close_the_gripper(key, direction):
    env = FakeEnv()
    controller = KeyboardEEControl(env)
    drive(env, controller, 5)

    before = float(env.robot.data.joint_pos[0][5])
    press(controller, key)
    drive(env, controller, 60)
    release(controller, key)
    after = float(env.robot.data.joint_pos[0][5])

    assert (after - before) * direction > 1e-3


def test_roll_keys_actuate_wrist_roll():
    """The dimension the scripted policy leaves dead is reachable by hand."""
    env = FakeEnv()
    controller = KeyboardEEControl(env)
    drive(env, controller, 5)

    before = float(env.robot.data.joint_pos[0][4])
    press(controller, "D")
    drive(env, controller, 60)
    release(controller, "D")
    assert float(env.robot.data.joint_pos[0][4]) - before > 1e-3


def test_home_key_returns_to_the_spawn_pose():
    env = FakeEnv()
    controller = KeyboardEEControl(env)

    press(controller, "UP")
    drive(env, controller, 100)
    release(controller, "UP")

    press(controller, "H")
    release(controller, "H")
    drive(env, controller, 400)

    home = env.robot.data.default_joint_pos[0]
    assert torch.allclose(env.robot.data.joint_pos[0], home, atol=2e-2)


def test_home_motion_is_cancelled_by_manual_input():
    """Documented behaviour: touching a control key aborts the return home."""
    env = FakeEnv()
    controller = KeyboardEEControl(env)
    drive(env, controller, 5)

    press(controller, "H")
    release(controller, "H")
    drive(env, controller, 5)
    assert controller._homing is not None

    press(controller, "UP")
    drive(env, controller, 1)
    assert controller._homing is None


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_commands_stay_inside_the_joint_limits(seed):
    """Hold every key at once and hammer the arm into its stops."""
    robot = RobotSurrogate(kinematics=randomized_kinematics(seed))
    env = FakeEnv(robot=robot)
    controller = KeyboardEEControl(env)

    for key in list(MOVE_BINDINGS) + list(JAW_KEYS) + ["T", "D"]:
        press(controller, key)
    commands = drive(env, controller, 400)

    assert not torch.isnan(commands).any()
    assert not torch.isinf(commands).any()
    limits = robot.data.soft_joint_pos_limits[0]
    assert torch.all(commands >= limits[:, 0] - 1e-5)
    assert torch.all(commands <= limits[:, 1] + 1e-5)


def test_reset_resyncs_after_a_world_reset():
    """After ``env.reset()`` the controller must re-latch onto the new pose."""
    env = FakeEnv()
    controller = KeyboardEEControl(env)
    press(controller, "UP")
    drive(env, controller, 60)
    release(controller, "UP")

    env.robot.set_joint_pos(env.robot.data.default_joint_pos[0])
    controller.reset()
    assert controller._held == set()

    start = ee_position(env).clone()
    drive(env, controller, 50)
    assert torch.linalg.norm(ee_position(env) - start) < 1e-4

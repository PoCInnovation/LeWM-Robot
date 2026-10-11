"""The controllers must behave the same in *simulated time* at any control rate.

Decision D2 moves the environment from 60 Hz to 30 Hz control so the dataset's
30 fps label becomes true. That halves the number of frames per episode, and any
constant expressed per *frame* rather than per *second* silently halves the
motion it produces. These tests pin the property that makes the change safe:
identical behaviour per second, not per frame.

They are also what makes the retuning verifiable without a GPU — the smoke test
on the real machine then only has to confirm the physics, not the arithmetic.
"""

import pytest
import torch

from fake_env import FakeEnv, FakeRigidObject, run_controller
from test_keyboard_ee_control import drive, ee_position, press, release
from test_scripted_policy import _placements

from sim_to_real_so101.utils.keyboard_ee_control import KeyboardEEControl
from sim_to_real_so101.utils.scripted_policy import ScriptedPickPlace

RATES = (60.0, 30.0)


def _run_policy_at(seed, control_hz):
    cube, box = _placements(seed)
    env = FakeEnv(
        objects={"Cube": FakeRigidObject(cube), "BoxFloor": FakeRigidObject(box)},
        control_hz=control_hz,
    )
    result = run_controller(env, ScriptedPickPlace(env), max_steps=3000)
    result["seconds"] = result["steps"] / control_hz
    return result


@pytest.mark.parametrize("seed", [0, 2, 3, 4])
def test_scripted_policy_takes_the_same_time_at_both_rates(seed):
    """Same outcome, same duration in seconds, same place at the end."""
    fast, slow = (_run_policy_at(seed, hz) for hz in RATES)

    assert fast["status"] == slow["status"], (
        f"outcome changed with the control rate: {fast['status']} vs {slow['status']}"
    )
    assert slow["seconds"] == pytest.approx(fast["seconds"], rel=0.08), (
        f"{fast['seconds']:.2f}s at 60 Hz vs {slow['seconds']:.2f}s at 30 Hz"
    )
    drift = float(torch.linalg.norm(fast["ee_pos"][-1] - slow["ee_pos"][-1]))
    assert drift < 0.01, f"end effector ends {drift * 1000:.1f} mm apart"


@pytest.mark.parametrize("seed", [0, 3])
def test_halving_the_rate_halves_the_frame_count(seed):
    """The dataset shrinks by half — the storage and throughput win of D2."""
    fast, slow = (_run_policy_at(seed, hz) for hz in RATES)
    assert slow["steps"] == pytest.approx(fast["steps"] / 2, rel=0.1)


@pytest.mark.parametrize("key,axis,direction", [("UP", 0, +1.0), ("W", 2, +1.0)])
def test_a_key_held_for_one_second_moves_the_same_distance(key, axis, direction):
    """Teleop feel must not change with the control rate either."""
    travelled = {}
    for control_hz in RATES:
        env = FakeEnv(control_hz=control_hz)
        controller = KeyboardEEControl(env)
        drive(env, controller, int(0.2 * control_hz))  # settle

        start = ee_position(env).clone()
        press(controller, key)
        drive(env, controller, int(control_hz))  # exactly one second
        release(controller, key)
        travelled[control_hz] = float(ee_position(env)[axis] - start[axis]) * direction

    assert travelled[60.0] > 0.01
    assert travelled[30.0] == pytest.approx(travelled[60.0], rel=0.15), travelled


def test_the_gripper_takes_the_same_time_to_close_at_both_rates():
    """A frame-counted hold would close the jaw twice as fast at 30 Hz."""
    durations = {}
    for control_hz in RATES:
        env = FakeEnv(
            objects={
                "Cube": FakeRigidObject((0.22, -0.09, 0.06)),
                "BoxFloor": FakeRigidObject((0.22, 0.10, 0.038)),
            },
            control_hz=control_hz,
        )
        policy = ScriptedPickPlace(env)
        durations[control_hz] = policy._gripper_frames / control_hz

    assert durations[30.0] == pytest.approx(durations[60.0], rel=0.05)
    assert durations[60.0] == pytest.approx(25 / 60.0, rel=0.05)


def test_the_state_timeout_is_the_same_duration_at_both_rates():
    """Otherwise 30 Hz would declare failure after half the simulated time."""
    timeouts = {}
    for control_hz in RATES:
        env = FakeEnv(
            objects={
                "Cube": FakeRigidObject((0.22, -0.09, 0.06)),
                "BoxFloor": FakeRigidObject((0.22, 0.10, 0.038)),
            },
            control_hz=control_hz,
        )
        policy = ScriptedPickPlace(env)
        timeouts[control_hz] = policy._state_timeout / control_hz

    assert timeouts[30.0] == pytest.approx(timeouts[60.0], rel=0.05)


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("sim_to_real_so101.tasks.so101_env_cfg", "SO101TeleopEnvCfg"),
        ("sim_to_real_so101.tasks.task_env_cfg", "SO101TaskEnvCfg"),
    ],
)
def test_the_environment_config_runs_at_30_hz(module_name, class_name):
    """Guards decision D2 at its source, so it cannot be reverted by accident.

    The config is instantiated and its real ``__post_init__`` is run, so this
    asserts the value the simulator would actually use — not the presence of a
    line of source.
    """
    import importlib

    cfg = getattr(importlib.import_module(module_name), class_name)()
    cfg.__post_init__()

    control_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    assert control_hz == pytest.approx(30.0), (
        f"{class_name} steps at {control_hz} Hz while the recorder stamps the "
        "dataset at 30 fps — see decision D2"
    )
    # One render per control step: the halved render count is the throughput win.
    assert cfg.sim.render_interval == cfg.decimation


def test_physics_still_runs_at_120_hz():
    """Only the control rate changes; contact fidelity must not be traded away."""
    from sim_to_real_so101.tasks.so101_env_cfg import SO101TeleopEnvCfg

    cfg = SO101TeleopEnvCfg()
    cfg.__post_init__()
    assert 1.0 / cfg.sim.dt == pytest.approx(120.0)

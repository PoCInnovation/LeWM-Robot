"""Tests for the calibration probe's report logic.

These exist because of a real failure: the first probe sent to the GPU machine
crashed on both of its most important sections. ``find_bodies`` was called with
candidate names that the arm does not have, and Isaac *raises* on an unmatched
pattern rather than returning an empty list. The run came back with no joint
limits and no kinematics — a wasted round trip.

The launcher script itself cannot be imported without an Omniverse app, which
is why the logic now lives in ``utils/probe_report.py`` and is driven here
against the surrogate.
"""

import pytest
import torch

from fake_env import FakeEnv
from so101_surrogate import BODY_NAMES, JOINT_NAMES, RobotSurrogate

from sim_to_real_so101.utils import probe_report

# Exactly what the RTX 3080 machine reported on 2026-08-06.
REAL_BODY_NAMES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]


def test_the_surrogate_matches_the_real_articulation_layout():
    """Guards the offline model against the measured ground truth."""
    assert BODY_NAMES == REAL_BODY_NAMES
    assert JOINT_NAMES == [
        "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"
    ]


def test_body_ids_skips_names_the_arm_does_not_have():
    """The regression. ``moving_jaw``/``Jaw``/``Wrist`` are not on this arm."""
    robot = RobotSurrogate()
    resolved = probe_report.body_ids(robot, probe_report.BODY_CANDIDATES)

    assert set(resolved) == {"gripper", "jaw", "wrist", "base"}
    assert resolved["gripper"] == 5
    assert resolved["jaw"] == 6
    assert resolved["wrist"] == 4
    assert resolved["base"] == 0


def test_body_ids_on_an_arm_with_none_of_the_candidates():
    robot = RobotSurrogate(body_names=[f"link{i}" for i in range(7)])
    assert probe_report.body_ids(robot, ["gripper", "jaw"]) == {}


def test_probe_robot_does_not_raise_on_absent_candidates():
    """The exact call that failed on the real machine."""
    robot = RobotSurrogate()
    info = probe_report.probe_robot(robot)

    assert info["joint_names"] == JOINT_NAMES
    assert info["body_names"] == REAL_BODY_NAMES
    assert info["num_joints"] == 6
    assert info["is_fixed_base"] is True
    assert len(info["soft_joint_pos_limits"]) == 6
    assert len(info["default_joint_pos"]) == 6


def test_probe_robot_reports_the_jacobian_indexing_convention():
    """``ee_body_id - 1`` on a fixed base — what both controllers assume."""
    robot = RobotSurrogate()
    info = probe_report.probe_robot(robot)
    assert info["ee_body_id"] == 5
    assert info["jacobian_body_id"] == 4


def test_sample_kinematics_produces_usable_samples():
    env = FakeEnv()
    samples = probe_report.sample_kinematics(
        env, env.robot, n_samples=8, seed=0, step_sim=lambda: None
    )

    assert len(samples) == 8
    for sample in samples:
        assert len(sample["q"]) == 6
        assert set(sample["body_pos"]) == {"gripper", "jaw", "wrist", "base"}
        assert len(sample["body_pos"]["gripper"]) == 3
        assert len(sample["body_quat"]["gripper"]) == 4
        # 6 x num_joints, Isaac's layout.
        assert len(sample["jacobian"]) == 6
        assert len(sample["jacobian"][0]) == 6


def test_the_first_sample_is_the_spawn_pose():
    """So the dump always contains the configuration the robot starts in."""
    env = FakeEnv()
    samples = probe_report.sample_kinematics(
        env, env.robot, n_samples=3, seed=0, step_sim=lambda: None
    )
    expected = env.robot.data.default_joint_pos[0].tolist()
    assert samples[0]["q"] == pytest.approx(expected, abs=1e-5)


def test_sampled_configurations_stay_inside_the_joint_limits():
    env = FakeEnv()
    samples = probe_report.sample_kinematics(
        env, env.robot, n_samples=40, seed=1, step_sim=lambda: None
    )
    limits = env.robot.data.soft_joint_pos_limits[0]
    for sample in samples:
        q = torch.tensor(sample["q"])
        assert torch.all(q >= limits[:, 0] - 1e-5)
        assert torch.all(q <= limits[:, 1] + 1e-5)


def test_sampled_configurations_actually_span_the_joint_space():
    """A dump where every sample sits near the home pose would be useless."""
    env = FakeEnv()
    samples = probe_report.sample_kinematics(
        env, env.robot, n_samples=60, seed=2, step_sim=lambda: None
    )
    q = torch.tensor([sample["q"] for sample in samples])
    limits = env.robot.data.soft_joint_pos_limits[0]
    span = (q.max(0).values - q.min(0).values) / (limits[:, 1] - limits[:, 0])
    assert torch.all(span > 0.5), f"poor coverage per joint: {span.tolist()}"


def test_the_jacobian_recorded_is_the_gripper_one():
    env = FakeEnv()
    samples = probe_report.sample_kinematics(
        env, env.robot, n_samples=1, seed=0, step_sim=lambda: None
    )
    direct = env.robot.kinematics.jacobian(env.robot.data.joint_pos[0], 5)
    assert torch.allclose(torch.tensor(samples[0]["jacobian"]), direct, atol=1e-5)


def test_run_section_records_a_failure_instead_of_propagating_it():
    """One broken section must not cost the whole report."""
    report = probe_report.new_report("Task", 101)

    assert probe_report.run_section(report, "good", lambda: {"value": 1}) is True
    assert probe_report.run_section(
        report, "bad", lambda: (_ for _ in ()).throw(ValueError("boom"))
    ) is False

    assert report["good"] == {"value": 1}
    assert "bad" not in report
    assert "ValueError: boom" in report["errors"]["bad"]


def test_the_summary_survives_a_partial_report():
    """It is printed even when sections failed — that is when it matters most."""
    report = probe_report.new_report("Task", 101)
    report["errors"]["robot"] = "ValueError: nope"
    probe_report.print_summary(report)  # must not raise


def test_the_summary_survives_a_complete_report(capsys):
    robot = RobotSurrogate()
    report = probe_report.new_report("Task", 101)
    report["robot"] = probe_report.probe_robot(robot)
    report["environment"] = probe_report.probe_environment(".")
    report["sim"] = {"physics_hz": 120.0, "decimation": 4, "control_hz": 30.0}

    probe_report.print_summary(report)
    out = capsys.readouterr().out
    assert "ISAAC PROBE" in out
    assert "gripper" in out


def test_the_report_round_trips_through_json(tmp_path):
    """Everything written must be JSON-serialisable — tensors are not."""
    import json

    env = FakeEnv()
    report = probe_report.new_report("Task", 101)
    report["robot"] = probe_report.probe_robot(env.robot)
    report["fk_samples"] = probe_report.sample_kinematics(
        env, env.robot, n_samples=3, seed=0, step_sim=lambda: None
    )

    path = probe_report.write_report(report, str(tmp_path / "out" / "probe.json"))
    reloaded = json.loads(open(path, encoding="utf-8").read())
    assert reloaded["robot"]["joint_names"] == JOINT_NAMES
    assert len(reloaded["fk_samples"]) == 3


def test_probe_environment_reports_the_fields_the_go_no_go_needs():
    info = probe_report.probe_environment(".")
    assert "python" in info and "platform" in info
    assert set(info["packages"]) >= {"torch", "isaaclab", "lerobot", "numpy"}
    assert "cuda_available" in info["gpu"]
    assert "available" in info["ffmpeg"]
    assert "free_gb" in info["disk"] or "error" in info["disk"]

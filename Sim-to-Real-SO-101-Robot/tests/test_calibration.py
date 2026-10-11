"""Checks that come alive once a real machine has produced a calibration dump.

Every test here is skipped until ``outputs/isaac_probe.json`` exists. They are
the assumptions the offline harness makes about the real robot — joint order,
body names, the jacobian indexing convention, the control rate — each stated as
an assertion so that a mismatch is reported rather than silently absorbed.

They are also the fastest way to read the dump when it arrives: run
``pytest tests/test_calibration.py -v`` and whatever the offline model got wrong
shows up by name.
"""

import pytest

from calibration import fk_error_against_probe, has_probe, load_probe, surrogate_from_probe
from so101_surrogate import DEFAULT_JOINT_POS, JOINT_NAMES

pytestmark = pytest.mark.skipif(
    not has_probe(),
    reason="no calibration dump yet — run scripts/isaac_probe.py on the GPU machine",
)


@pytest.fixture(scope="module")
def probe():
    return load_probe()


def section(probe, name):
    """Return a dump section, or skip when the probe failed to capture it.

    A section can be missing because the probe hit an error there. That is
    reported once, by ``test_the_probe_ran_without_errors``; the tests that
    depend on the section then skip instead of failing with a KeyError, so the
    output points at the cause rather than burying it.
    """
    value = probe.get(name)
    if not value:
        reason = (probe.get("errors") or {}).get(name, "not captured")
        pytest.skip(f"probe has no '{name}' section ({reason})")
    return value


def test_the_probe_ran_without_errors(probe):
    """A partial dump still loads; say so loudly rather than testing around it."""
    errors = probe.get("errors") or {}
    assert not errors, "the probe failed on: " + "; ".join(
        f"{key} -> {value}" for key, value in errors.items()
    )


def test_joint_order_is_what_the_code_assumes(probe):
    """Action vectors map positionally onto dataset columns; order is load-bearing."""
    assert section(probe, "robot")["joint_names"] == JOINT_NAMES


def test_the_bodies_the_controllers_look_up_exist(probe):
    """``scripted_policy`` silently changes behaviour when these are missing."""
    body_names = section(probe, "robot")["body_names"]
    assert "gripper" in body_names, "the IK target body is missing"
    for optional in ("jaw", "wrist"):
        if optional not in body_names:
            pytest.fail(
                f"body '{optional}' absent — scripted_policy falls back to a coarser "
                "grasp point; update the surrogate to match"
            )


def test_jacobian_indexing_convention_holds(probe):
    """Both controllers compute ``ee_body_id - 1``; confirm that is right."""
    robot_info = section(probe, "robot")
    assert robot_info["is_fixed_base"] is True
    expected = robot_info["ee_body_id"] - 1
    assert robot_info["jacobian_body_id"] == expected


def test_home_pose_matches_the_configured_spawn(probe):
    measured = section(probe, "robot")["default_joint_pos"]
    assert measured == pytest.approx(DEFAULT_JOINT_POS, abs=1e-3)


def test_joint_limits_cover_the_motor_mapping(probe):
    """The motor scale maps onto these stops; a narrower joint clips commands.

    A mismatch here means part of the -100..100 motor range is unreachable, so
    recorded actions saturate — which is defect #3's mechanism.
    """
    from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface

    interface = LeRobotSO101Interface(
        device="cpu", port="/dev/null", id="test", cameras={}, fps=30, kind="leader"
    )
    limits = section(probe, "robot")["soft_joint_pos_limits"]
    import math

    mismatched = []
    for index, name in enumerate(interface.joint_names):
        mapped_low = math.radians(float(interface.joint_mins[index]))
        mapped_high = math.radians(float(interface.joint_maxs[index]))
        low, high = limits[index]
        if mapped_low < low - 1e-3 or mapped_high > high + 1e-3:
            mismatched.append(
                f"{name}: motor maps to [{mapped_low:.3f}, {mapped_high:.3f}] rad "
                f"but the joint stops at [{low:.3f}, {high:.3f}]"
            )
    assert not mismatched, "\n".join(mismatched)


def test_control_rate_is_the_one_the_dataset_claims(probe):
    """D2: the dataset is written at 30 fps, so control must run at 30 Hz.

    Expected to fail until phase 2 lands ``decimation = 4``; when it does, this
    is the check that confirms it on the real machine.
    """
    control_hz = section(probe, "sim")["control_hz"]
    assert control_hz == pytest.approx(30.0, abs=0.5), (
        f"control runs at {control_hz} Hz while the recorder "
        "stamps the dataset at 30 fps — see decision D2"
    )


def test_cameras_match_the_dataset_feature_shapes(probe):
    """480x640 on both, or the recorder writes a shape the real dataset lacks."""
    cameras = section(probe, "scene")["cameras"]
    assert cameras, "no cameras found on the task"
    for name, camera in cameras.items():
        assert (camera["height"], camera["width"]) == (480, 640), f"{name}: {camera}"


def test_surrogate_can_be_rebuilt_from_the_dump(probe):
    """The harness must actually pick the measured values up."""
    robot_info = section(probe, "robot")
    robot = surrogate_from_probe(probe)
    assert robot.joint_names == robot_info["joint_names"]
    assert robot.body_names == robot_info["body_names"]
    assert robot.data.default_joint_pos[0].tolist() == pytest.approx(
        robot_info["default_joint_pos"], abs=1e-5
    )


def test_report_how_far_the_offline_geometry_is_from_the_real_arm(probe):
    """Not a pass/fail on geometry — a measurement of what the harness can claim.

    Logic tests hold regardless. Distance-based tuning (the grasp offsets) is
    only trustworthy once this error is small, so the number is printed.
    """
    error = fk_error_against_probe(probe)
    if error is None:
        pytest.skip("dump carries no FK samples for the gripper body")

    print(
        f"\nplaceholder chain vs real arm: mean {error['mean_m'] * 1000:.1f} mm, "
        f"median {error['median_m'] * 1000:.1f} mm, max {error['max_m'] * 1000:.1f} mm "
        f"over {error['n_samples']} configurations"
    )
    assert error["n_samples"] > 0

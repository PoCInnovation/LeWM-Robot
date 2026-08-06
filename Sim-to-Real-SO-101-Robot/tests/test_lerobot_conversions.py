"""Unit conversions between sim radians and the LeRobot motor scale.

This is the seam where a silent error would be worst: it is applied to every
frame of both ``action`` and ``observation.state``, so a wrong mapping produces
a dataset that looks perfectly well-formed and is quietly unusable.

Two of the tests below are anchored on values measured in the 200 episodes
already collected (``datasets/cube_dataset/meta/stats.json``). They tie this
code to real recorded data rather than to a re-derivation of the same formula.
"""

import math

import pytest
import torch

from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.scripted_policy import DEFAULT_POSE, JAW_OPEN


@pytest.fixture
def interface():
    return LeRobotSO101Interface(
        device="cpu", port="/dev/null", id="test", cameras={}, fps=30, kind="leader"
    )


def test_motor_to_radians_round_trips(interface):
    """Any motor command must survive a trip through radians unchanged."""
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        motor = torch.empty(6).uniform_(-100.0, 100.0, generator=generator)
        motor[5] = motor[5].abs()  # the gripper channel is 0..100, not -100..100
        recovered = interface.get_raw_actions_from_radians(
            interface.get_mapped_actions_vectorized(motor)
        )
        assert torch.allclose(motor, recovered, atol=1e-4)


def test_radians_to_motor_round_trips(interface):
    """And the other way round, over the whole reachable joint range."""
    lows = interface.joint_mins * math.pi / 180
    highs = interface.joint_maxs * math.pi / 180

    generator = torch.Generator().manual_seed(1)
    for _ in range(50):
        u = torch.rand(6, generator=generator)
        radians = lows + u * (highs - lows)
        recovered = interface.get_mapped_actions_vectorized(
            interface.get_raw_actions_from_radians(radians)
        )
        assert torch.allclose(radians, recovered, atol=1e-6)


def test_motor_extremes_land_on_the_joint_stops(interface):
    """-100/+100 must map to the ends of each joint range, gripper 0..100."""
    low = interface.get_mapped_actions_vectorized(
        torch.tensor([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0])
    )
    high = interface.get_mapped_actions_vectorized(
        torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    )
    assert torch.allclose(low, interface.joint_mins * math.pi / 180, atol=1e-6)
    assert torch.allclose(high, interface.joint_maxs * math.pi / 180, atol=1e-6)


def test_mapping_matches_the_documented_motor_ranges(interface):
    """Guards the table the whole sim-to-real correspondence rests on."""
    expected = {
        "shoulder_pan": (-110, 110),
        "shoulder_lift": (-100, 100),
        "elbow_flex": (-100, 90),
        "wrist_flex": (-95, 95),
        "wrist_roll": (-160, 160),
        "gripper": (-10, 100),
    }
    for index, name in enumerate(interface.joint_names):
        assert name in expected
        assert float(interface.joint_mins[index]) == expected[name][0]
        assert float(interface.joint_maxs[index]) == expected[name][1]


def test_home_wrist_roll_reproduces_the_constant_seen_in_the_dataset():
    """The dataset's dead ``wrist_roll`` channel is exactly the home pose.

    ``meta/stats.json`` of the 200 collected episodes reports wrist_roll with
    min = max = mean = -57.42 and std = 0. Confirming that this is precisely the
    home value pushed through the conversion pins down defect #2: the channel is
    not noisy or clipped, it is simply never commanded.
    """
    interface = LeRobotSO101Interface(
        device="cpu", port="/dev/null", id="test", cameras={}, fps=30, kind="leader"
    )
    radians = torch.tensor(
        [DEFAULT_POSE[name] for name in
         ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]]
    )
    motor = interface.get_raw_actions_from_radians(radians)
    assert float(motor[4]) == pytest.approx(-57.42, abs=0.01)


def test_jaw_open_reproduces_the_gripper_maximum_seen_in_the_dataset(interface):
    """``JAW_OPEN`` maps to 40.34, the exact gripper max in the collected data."""
    radians = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, JAW_OPEN])
    motor = interface.get_raw_actions_from_radians(radians)
    assert float(motor[5]) == pytest.approx(40.34, abs=0.02)


def test_conversion_is_monotonic(interface):
    low = interface.get_mapped_actions_vectorized(torch.full((6,), -50.0).abs_() * -1)
    mid = interface.get_mapped_actions_vectorized(torch.zeros(6))
    high = interface.get_mapped_actions_vectorized(torch.full((6,), 50.0))
    assert torch.all(low <= mid)
    assert torch.all(mid <= high)

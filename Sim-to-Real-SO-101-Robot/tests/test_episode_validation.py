"""Tests for the episode validity filter.

The property that matters most is the negative one: a *failed* episode must
never be rejected. Several tests below therefore build episodes that look bad
in task terms — the gripper closes on nothing, the object is never reached — and
assert they survive.
"""

import numpy as np
import pytest

from sim_to_real_so101.utils.episode_validation import (
    DIVERGED,
    JOINT_PINNED,
    MOTIONLESS,
    NOT_FINITE,
    TOO_SHORT,
    WARN_DEAD_DIMENSION,
    WARN_MAYBE_TRUNCATED,
    WARN_SATURATED,
    EpisodeCheckConfig,
    check_episode,
    summarize_episodes,
)

LIMITS = np.array(
    [
        [-1.92, 1.92],
        [-1.75, 1.75],
        [-1.75, 1.57],
        [-1.66, 1.66],
        [-2.79, 2.79],
        [-0.17, 1.75],
    ]
)


def smooth_episode(n_frames=300, n_joints=6, amplitude=0.4, seed=0):
    """A plausible, well-behaved trajectory: smooth and inside the stops."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames)[:, None]
    phases = rng.uniform(0.0, 2 * np.pi, size=(1, n_joints))
    return amplitude * np.sin(2 * np.pi * t + phases)


def codes(report):
    return {item["code"] for item in report.reasons}


def warning_codes(report):
    return {item["code"] for item in report.warnings}


def test_a_normal_episode_is_kept():
    report = check_episode(smooth_episode(), joint_limits=LIMITS)
    assert report.valid, report.summary()
    assert report.stats["n_frames"] == 300
    assert report.stats["duration_s"] == pytest.approx(10.0, abs=0.01)


# --- the property that matters: failures are data, not rubbish ----------- #

def test_a_failed_grasp_is_kept():
    """Gripper closes on empty air, arm returns home — a normal failed episode."""
    actions = smooth_episode(n_frames=250)
    actions[:, 5] = np.concatenate([np.full(120, 0.6), np.full(130, -0.15)])
    report = check_episode(actions, joint_limits=LIMITS)
    assert report.valid, report.summary()


def test_an_episode_where_the_arm_barely_moves_is_still_kept():
    """Small but real motion is a poor attempt, not a degenerate recording."""
    actions = smooth_episode(amplitude=0.02)
    report = check_episode(actions, joint_limits=LIMITS)
    assert report.valid, report.summary()


def test_a_short_but_real_episode_is_kept_with_a_warning():
    report = check_episode(smooth_episode(n_frames=45), joint_limits=LIMITS)
    assert report.valid
    assert "very_short" in warning_codes(report)


# --- degenerate episodes are rejected ------------------------------------ #

def test_truncated_episode_is_rejected():
    report = check_episode(smooth_episode(n_frames=5))
    assert not report.valid
    assert TOO_SHORT in codes(report)


def test_nan_is_rejected():
    actions = smooth_episode()
    actions[42, 3] = np.nan
    report = check_episode(actions)
    assert not report.valid
    assert NOT_FINITE in codes(report)


def test_inf_is_rejected():
    actions = smooth_episode()
    actions[10, 0] = np.inf
    report = check_episode(actions)
    assert not report.valid
    assert NOT_FINITE in codes(report)


def test_a_completely_motionless_episode_is_rejected():
    report = check_episode(np.zeros((300, 6)))
    assert not report.valid
    assert MOTIONLESS in codes(report)


def test_a_diverging_ik_is_rejected():
    """A metre-per-frame jump is the IK blowing up, never a real motion."""
    actions = smooth_episode()
    actions[150:, 2] += 3.0
    report = check_episode(actions)
    assert not report.valid
    assert DIVERGED in codes(report)


def test_a_joint_welded_to_its_stop_is_rejected():
    actions = smooth_episode()
    actions[:, 4] = LIMITS[4, 1]
    report = check_episode(actions, joint_limits=LIMITS)
    assert not report.valid
    assert JOINT_PINNED in codes(report)


def test_mismatched_action_and_state_shapes_are_rejected():
    report = check_episode(smooth_episode(), states=smooth_episode(n_frames=299))
    assert not report.valid


@pytest.mark.parametrize("bad", [np.zeros((0, 6)), np.zeros((300, 0)), np.zeros(300)])
def test_malformed_arrays_are_rejected_without_raising(bad):
    report = check_episode(bad)
    assert not report.valid


# --- warnings: reported, never fatal ------------------------------------- #

def test_a_constant_dimension_is_warned_about_not_rejected():
    """This is exactly how defect #2 (dead wrist_roll) presents itself."""
    actions = smooth_episode()
    actions[:, 4] = -1.6034
    report = check_episode(actions, joint_limits=LIMITS)
    assert report.valid, report.summary()
    assert WARN_DEAD_DIMENSION in warning_codes(report)
    assert report.stats["dead_dimensions"] == [4]


def test_a_mostly_saturated_joint_is_warned_about_not_rejected():
    """Defect #3: wrist_flex pinned at its stop for much of the trajectory.

    Built the way real saturation happens — a wide swing clipped by the stop —
    rather than by splicing, which would fake a discontinuity.
    """
    actions = smooth_episode()
    actions[:, 3] = np.clip(actions[:, 3] * 8.0, LIMITS[3, 0], LIMITS[3, 1])
    report = check_episode(actions, joint_limits=LIMITS)
    assert report.valid, report.summary()
    assert WARN_SATURATED in warning_codes(report)


def test_an_episode_at_buffer_capacity_is_flagged_as_possibly_truncated():
    """The recorder drops frames past capacity with only a print (defect #10)."""
    config = EpisodeCheckConfig(buffer_capacity=1200)
    report = check_episode(smooth_episode(n_frames=1200), config=config)
    assert report.valid
    assert WARN_MAYBE_TRUNCATED in warning_codes(report)


def test_wrong_limits_shape_does_not_crash_the_check():
    report = check_episode(smooth_episode(), joint_limits=np.zeros((3, 2)))
    assert report.valid
    assert "saturation_fraction" not in report.stats


# --- aggregation ---------------------------------------------------------- #

def test_summary_counts_and_rates():
    reports = [check_episode(smooth_episode(seed=i)) for i in range(8)]
    reports.append(check_episode(np.zeros((300, 6))))
    reports.append(check_episode(smooth_episode(n_frames=3)))

    summary = summarize_episodes(reports)
    assert summary["total"] == 10
    assert summary["kept"] == 8
    assert summary["rejected"] == 2
    assert summary["reasons"][MOTIONLESS] == 1
    assert summary["reasons"][TOO_SHORT] == 1
    assert summary["frames_mean"] == pytest.approx(300.0)


def test_summary_surfaces_a_dimension_that_is_dead_across_the_whole_run():
    """The collection-level check that would have caught defect #2 at episode 1."""
    reports = []
    for seed in range(20):
        actions = smooth_episode(seed=seed)
        actions[:, 4] = -1.6034
        reports.append(check_episode(actions))

    summary = summarize_episodes(reports)
    assert summary["dimensions_always_dead"] == [4]


def test_summary_of_nothing_does_not_crash():
    assert summarize_episodes([])["total"] == 0

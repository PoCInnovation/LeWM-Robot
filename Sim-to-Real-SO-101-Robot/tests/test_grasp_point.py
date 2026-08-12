"""The grasp point must stay between the fingers however the wrist turns.

The old construction rebuilt the point every frame from world-frame terms, one
of which projected onto the **world** horizontal plane. That is only valid at a
fixed gripper orientation — which held by accident, because the scripted
pick-and-place never rotates the wrist.

The exploration and reaching modes rotate it on purpose, so these tests pin the
property that makes them possible: the grasp point is rigidly attached to the
gripper. They also measure how far the old construction drifted, so the reason
for the change stays visible.
"""

import math

import pytest
import torch

from fake_env import FakeEnv
from so101_surrogate import RobotSurrogate

from sim_to_real_so101.utils.arm_control import (
    FINGER_LEN,
    GRASP_LATERAL,
    GraspPoint,
    matrix_from_quat,
)

GRIPPER, JAW, WRIST = 5, 6, 4


def legacy_world_point(env, robot, finger_len=FINGER_LEN, grasp_lateral=GRASP_LATERAL):
    """The construction that shipped before, kept to compare against."""
    origin = env.scene.env_origins[0]
    positions = robot.data.body_pos_w[0]
    gripper = positions[GRIPPER] - origin

    point = (positions[JAW] - origin).clone()
    axis = gripper - (positions[WRIST] - origin)
    point = point + finger_len * axis / torch.linalg.norm(axis)

    lateral = ((positions[JAW] - origin) - gripper).clone()
    lateral[2] = 0.0
    norm = torch.linalg.norm(lateral)
    if norm > 1e-6:
        point = point + grasp_lateral * lateral / norm
    return point


def local_of(env, robot, world_point):
    """Express a world point in the gripper's own frame."""
    rotation = matrix_from_quat(robot.data.body_quat_w[0, GRIPPER])
    gripper = robot.data.body_pos_w[0, GRIPPER] - env.scene.env_origins[0]
    return rotation.transpose(0, 1) @ (world_point - gripper)


def test_it_reproduces_the_old_construction_at_the_reference_pose():
    """Non-regression: the task's tuning was done at the spawn pose."""
    env = FakeEnv()
    grasp = GraspPoint(env, env.robot)

    expected = legacy_world_point(env, env.robot)
    assert torch.allclose(grasp.world(), expected, atol=1e-6)


@pytest.mark.parametrize("roll", [-1.6034, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
def test_the_grasp_point_is_rigid_under_wrist_roll(roll):
    """The defining property. This is what unlocks the rotating modes."""
    env = FakeEnv()
    grasp = GraspPoint(env, env.robot)
    reference = grasp.local_offset.clone()

    q = env.robot.data.default_joint_pos[0].clone()
    q[4] = roll
    env.robot.set_joint_pos(q)

    drift = torch.linalg.norm(local_of(env, env.robot, grasp.world()) - reference)
    assert float(drift) < 1e-6, f"grasp point moved {float(drift) * 1000:.3f} mm in the gripper frame"


@pytest.mark.parametrize("pitch", [1.5148, 1.0, 0.5, 0.0, -0.5, -1.0])
def test_the_grasp_point_is_rigid_under_wrist_pitch(pitch):
    env = FakeEnv()
    grasp = GraspPoint(env, env.robot)
    reference = grasp.local_offset.clone()

    q = env.robot.data.default_joint_pos[0].clone()
    q[3] = pitch
    env.robot.set_joint_pos(q)

    drift = torch.linalg.norm(local_of(env, env.robot, grasp.world()) - reference)
    assert float(drift) < 1e-6


def test_the_grasp_point_is_rigid_over_the_whole_joint_space():
    """Not just one joint at a time — any configuration at all."""
    env = FakeEnv()
    grasp = GraspPoint(env, env.robot)
    reference = grasp.local_offset.clone()
    limits = env.robot.data.soft_joint_pos_limits[0]

    generator = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(100):
        u = torch.rand(6, generator=generator)
        env.robot.set_joint_pos(limits[:, 0] + u * (limits[:, 1] - limits[:, 0]))
        drift = float(torch.linalg.norm(local_of(env, env.robot, grasp.world()) - reference))
        worst = max(worst, drift)
    assert worst < 1e-5, f"worst drift {worst * 1000:.4f} mm"


def test_the_old_construction_really_did_drift():
    """Documents the size of the problem the change fixes.

    Guards against someone reverting to the world-frame construction on the
    grounds that it 'looked equivalent'.
    """
    env = FakeEnv()
    reference = local_of(env, env.robot, legacy_world_point(env, env.robot)).clone()

    worst = 0.0
    for roll in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        q = env.robot.data.default_joint_pos[0].clone()
        q[4] = roll
        env.robot.set_joint_pos(q)
        drift = float(
            torch.linalg.norm(local_of(env, env.robot, legacy_world_point(env, env.robot)) - reference)
        )
        worst = max(worst, drift)

    # Measured at 34 mm on the calibrated arm — wider than the 25 mm cube.
    assert worst > 0.02, f"expected the legacy drift to be large, got {worst * 1000:.1f} mm"


def test_the_grasp_point_sits_between_the_fingers_not_on_the_gripper_body():
    """A zero offset would mean the IK aims at the wrong place entirely."""
    env = FakeEnv()
    grasp = GraspPoint(env, env.robot)
    reach = float(torch.linalg.norm(grasp.local_offset))
    assert 0.05 < reach < 0.15, f"grasp point {reach * 1000:.0f} mm from the gripper origin"


@pytest.mark.parametrize("attribute,delta", [("finger_len", 0.01), ("grasp_lateral", 0.005)])
def test_tuning_recalibrates_the_frozen_offset(attribute, delta):
    """The live O/L/I/K/J/H keys must actually move the point."""
    env = FakeEnv()
    grasp = GraspPoint(env, env.robot)
    before = grasp.local_offset.clone()

    setattr(grasp, attribute, getattr(grasp, attribute) + delta)
    moved = float(torch.linalg.norm(grasp.local_offset - before))
    assert moved == pytest.approx(delta, rel=0.2), f"moved {moved * 1000:.2f} mm for a {delta * 1000:.0f} mm change"


def test_a_longer_finger_reaches_further_along_the_gripper_axis():
    env = FakeEnv()
    short = GraspPoint(env, env.robot, finger_len=0.03)
    long = GraspPoint(env, env.robot, finger_len=0.09)
    assert torch.linalg.norm(long.local_offset) > torch.linalg.norm(short.local_offset)


def test_it_falls_back_to_the_gripper_body_when_there_is_no_jaw():
    robot = RobotSurrogate(
        body_names=["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "tip"]
    )
    env = FakeEnv(robot=robot)
    grasp = GraspPoint(env, robot)

    assert grasp.jaw_body_id is None
    assert torch.allclose(grasp.local_offset, torch.zeros(3), atol=1e-9)
    expected = robot.data.body_pos_w[0, GRIPPER] - env.scene.env_origins[0]
    assert torch.allclose(grasp.world(), expected, atol=1e-6)


def test_it_refuses_an_articulation_without_a_gripper():
    robot = RobotSurrogate(body_names=[f"link{i}" for i in range(7)])
    with pytest.raises(ValueError, match="gripper"):
        GraspPoint(FakeEnv(robot=robot), robot)


def test_matrix_from_quat_matches_the_surrogates_own_rotations():
    """The conversion is on the critical path; check it against the FK."""
    robot = RobotSurrogate()
    limits = robot.data.soft_joint_pos_limits[0]
    generator = torch.Generator().manual_seed(3)

    for _ in range(20):
        u = torch.rand(6, generator=generator)
        robot.set_joint_pos(limits[:, 0] + u * (limits[:, 1] - limits[:, 0]))
        from_quat = matrix_from_quat(robot.data.body_quat_w[0, GRIPPER])
        from_fk = robot.frames()[1][GRIPPER]
        assert torch.allclose(from_quat, from_fk, atol=1e-5)


def test_identity_quaternion_gives_the_identity_matrix():
    assert torch.allclose(
        matrix_from_quat(torch.tensor([1.0, 0.0, 0.0, 0.0])), torch.eye(3), atol=1e-9
    )


def test_a_quarter_turn_about_z_is_recovered():
    half = math.sqrt(0.5)
    rotation = matrix_from_quat(torch.tensor([half, 0.0, 0.0, half]))
    assert torch.allclose(rotation @ torch.tensor([1.0, 0.0, 0.0]),
                          torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

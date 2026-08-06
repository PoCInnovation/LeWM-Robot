"""Validate the surrogate before anything is validated *with* it.

The whole offline harness rests on one property: the jacobian the surrogate
hands to the IK must be the true derivative of the surrogate's own forward
kinematics. If that ever drifts, every controller test built on top becomes
meaningless — so it is checked first, and checked numerically.
"""

import math

import pytest
import torch

from so101_surrogate import (
    DEFAULT_JOINT_POS,
    JOINT_LIMITS_DEG,
    JOINT_NAMES,
    RobotSurrogate,
    SO101Kinematics,
    randomized_kinematics,
)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def _random_config(kinematics, seed):
    generator = torch.Generator().manual_seed(seed)
    limits = torch.tensor(
        [[math.radians(lo), math.radians(hi)] for lo, hi in JOINT_LIMITS_DEG],
        dtype=torch.float64,
    )
    u = torch.rand(kinematics.num_joints, generator=generator, dtype=torch.float64)
    return limits[:, 0] + u * (limits[:, 1] - limits[:, 0])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("body_index", [4, 5, 6])
def test_jacobian_matches_numerical_fk(seed, body_index):
    """Analytic jacobian == central finite difference of the FK, per column."""
    kinematics = SO101Kinematics(dtype=torch.float64)
    q = _random_config(kinematics, seed)

    analytic = kinematics.jacobian(q, body_index)[0:3]

    eps = 1e-6
    for joint in range(kinematics.num_joints):
        q_plus = q.clone()
        q_minus = q.clone()
        q_plus[joint] += eps
        q_minus[joint] -= eps
        numeric = (
            kinematics.body_pos(q_plus, body_index) - kinematics.body_pos(q_minus, body_index)
        ) / (2 * eps)
        assert torch.allclose(analytic[:, joint], numeric, atol=1e-6), (
            f"joint {joint} of body {body_index}: "
            f"analytic {analytic[:, joint].tolist()} vs numeric {numeric.tolist()}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_jacobian_columns_past_the_body_are_zero(seed):
    """A revolute joint cannot move a body that sits before it in the chain."""
    kinematics = SO101Kinematics(dtype=torch.float64)
    q = _random_config(kinematics, seed)
    for body_index in range(1, len(kinematics.chain) + 1):
        jacobian = kinematics.jacobian(q, body_index)
        assert torch.count_nonzero(jacobian[:, body_index:]) == 0


def test_physx_view_indexing_matches_isaac_convention():
    """``get_jacobians()[i]`` must describe body ``i+1`` on a fixed-base arm.

    This is the convention ``keyboard_ee_control`` and ``scripted_policy`` both
    rely on (``jac_body_id = ee_body_id - 1``). Getting it wrong would silently
    drive the IK with the wrong body's jacobian.
    """
    robot = RobotSurrogate()
    jacobians = robot.root_physx_view.get_jacobians()
    assert jacobians.shape == (1, robot.num_bodies - 1, 6, robot.num_joints)

    ee_body_id = robot.find_bodies(["gripper"])[0][0]
    jac_body_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    direct = robot.kinematics.jacobian(robot.data.joint_pos[0], ee_body_id)
    assert torch.allclose(jacobians[0, jac_body_id], direct, atol=1e-6)


def test_body_lookup_matches_the_names_the_code_asks_for():
    """The controllers look these up by name; absence changes their behaviour."""
    robot = RobotSurrogate()
    for name in ["gripper", "jaw", "wrist", "base"]:
        ids, _ = robot.find_bodies([name])
        assert ids, f"body '{name}' missing from the surrogate"
    assert robot.find_bodies(["does_not_exist"])[0] == []


def test_joint_order_matches_the_action_vector():
    """Action ordering is load-bearing: it maps 1:1 onto the dataset columns."""
    robot = RobotSurrogate()
    assert robot.joint_names == JOINT_NAMES
    ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
    assert ids == list(range(6))


def test_home_pose_is_the_configured_spawn_pose():
    robot = RobotSurrogate()
    assert robot.data.default_joint_pos[0].tolist() == pytest.approx(DEFAULT_JOINT_POS, abs=1e-6)


def test_actuator_never_leaves_the_joint_limits():
    """Even commanded far past the stops, the surrogate stays inside them."""
    robot = RobotSurrogate()
    limits = robot.data.soft_joint_pos_limits[0]
    for target in (torch.full((6,), 100.0), torch.full((6,), -100.0)):
        for _ in range(200):
            robot.apply_targets(target)
        q = robot.data.joint_pos[0]
        assert torch.all(q >= limits[:, 0] - 1e-6)
        assert torch.all(q <= limits[:, 1] + 1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_randomized_geometry_stays_self_consistent(seed):
    """The randomised arms used to check for overfitting are themselves valid."""
    kinematics = randomized_kinematics(seed, dtype=torch.float64)
    q = _random_config(kinematics, seed)
    analytic = kinematics.jacobian(q, 5)[0:3]

    eps = 1e-6
    for joint in range(kinematics.num_joints):
        q_plus, q_minus = q.clone(), q.clone()
        q_plus[joint] += eps
        q_minus[joint] -= eps
        numeric = (
            kinematics.body_pos(q_plus, 5) - kinematics.body_pos(q_minus, 5)
        ) / (2 * eps)
        assert torch.allclose(analytic[:, joint], numeric, atol=1e-6)

"""A CPU stand-in for the SO-101 articulation, good enough to close the loop.

The controllers (``KeyboardEEControl``, ``ScriptedPickPlace``) only ever read
four things off the articulation: joint positions, body world positions, the
jacobian, and the joint limits. This module provides all four from a plain
serial-chain model, so those controllers can be driven for hundreds of steps
under pytest without Isaac Sim.

**What this does and does not prove.** The link geometry here is approximate
until the calibration dump from a real machine lands (see
``scripts/isaac_probe.py``). That does not weaken the tests, because the model
is *self-consistent*: its jacobian is the exact derivative of its own forward
kinematics (asserted in ``test_surrogate.py``). Any IK loop that converges here
converges for this geometry — and the test suite deliberately runs the
controllers over **randomised chain geometries** as well, so nothing under test
can be secretly tuned to one particular arm. Geometry-specific numbers (the
grasp offsets) are the one thing this cannot settle; that is what the smoke test
on the real machine is for.

Joint limits and the home pose are *not* guessed: they come from
``assets/so101.py`` and the motor ranges in ``utils/lerobot_interface.py``.
"""

import math

import torch

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Bodies in articulation order, as reported by the calibration probe on a real
# machine. Index 0 is the root, which is why the jacobian array is offset by one
# on a fixed-base arm.
BODY_NAMES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]

# Home pose — exact, copied from SO101_CFG.init_state.joint_pos.
DEFAULT_JOINT_POS = [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]

# Joint ranges, in degrees, from LeRobotSO101Interface.SO101_USD_MAPPING.
JOINT_LIMITS_DEG = [
    (-110.0, 110.0),   # Rotation      / shoulder_pan
    (-100.0, 100.0),   # Pitch         / shoulder_lift
    (-100.0, 90.0),    # Elbow         / elbow_flex
    (-95.0, 95.0),     # Wrist_Pitch   / wrist_flex
    (-160.0, 160.0),   # Wrist_Roll    / wrist_roll
    (-10.0, 100.0),    # Jaw           / gripper
]

# Link geometry **solved from the calibration dump** (200 configurations from a
# real Isaac Sim run, 2026-08-06) by ``tests/fit_chain.py``: the joint axes come
# from the recorded jacobians in closed form, the link offsets from a
# least-squares solve. Residual against the measured body positions is
# 0.0001 mm — the chain is recovered exactly, not approximated.
#
# Re-run ``.venv/bin/python tests/fit_chain.py`` to regenerate these after a new
# dump.
#
# The Jaw axis is the one value not measured: body frame origins sit on their
# own joint axis, so rotating the jaw does not move the jaw body's origin and
# the axis leaves no trace in the position data. It is set parallel to the pitch
# axes — what a hinged jaw is — and nothing the surrogate is used for reads it.
DEFAULT_CHAIN = [
    {"axis": (-3e-06, 0.0, -1.0), "origin": (0.020791, -0.023075, 0.074541)},   # Rotation
    {"axis": (1.0, 6e-06, -3e-06), "origin": (-0.006092, -0.030399, 0.074541)},  # Pitch
    {"axis": (1.0, 7e-06, -3e-06), "origin": (-0.006092, -0.028, 0.11257)},      # Elbow
    {"axis": (1.0, 6e-06, -3e-06), "origin": (-0.006092, -0.1349, 0.0052)},      # Wrist_Pitch
    {"axis": (-1e-05, 1.0, -6e-06), "origin": (0.0181, -0.0611, 0.0)},           # Wrist_Roll
    {"axis": (1.0, 6e-06, -3e-06), "origin": (0.0188, -0.0234, 0.0202)},         # Jaw
]

# Where the base sits in the world. Confirmed by the calibration dump: base
# position (-0.05, 0, 0), orientation quaternion (0.707107, 0, 0, 0.707107).
BASE_POS = (-0.05, 0.0, 0.0)
BASE_YAW = math.pi / 2


def _rotation_about_axis(axis, angle):
    """Rodrigues rotation matrix for a unit ``axis`` and an ``angle`` in radians."""
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_minus_c = 1.0 - c
    return torch.stack(
        [
            torch.stack([c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s]),
            torch.stack([y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s]),
            torch.stack([z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c]),
        ]
    )


def _quat_from_matrix(m):
    """Rotation matrix to quaternion in ``(w, x, y, z)``, Isaac's convention."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return torch.stack([w, x, y, z])


class SO101Kinematics:
    """Forward kinematics and geometric jacobian of a serial revolute chain."""

    def __init__(self, chain=None, base_pos=BASE_POS, base_yaw=BASE_YAW, dtype=torch.float32):
        self.chain = [dict(link) for link in (chain or DEFAULT_CHAIN)]
        self.dtype = dtype
        self.num_joints = len(self.chain)
        self.base_pos = torch.tensor(base_pos, dtype=dtype)
        cos_yaw, sin_yaw = math.cos(base_yaw), math.sin(base_yaw)
        self.base_rot = torch.tensor(
            [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )

    def frames(self, q):
        """World pose of every body.

        Returns ``(positions, rotations)`` with one entry per body, the root
        first, so indices line up with :data:`BODY_NAMES`.
        """
        q = q.to(self.dtype)
        positions = [self.base_pos]
        rotations = [self.base_rot]

        pos = self.base_pos
        rot = self.base_rot
        for i, link in enumerate(self.chain):
            origin = torch.tensor(link["origin"], dtype=self.dtype)
            pos = pos + rot @ origin
            rot = rot @ _rotation_about_axis(link["axis"], q[i])
            positions.append(pos)
            rotations.append(rot)

        return torch.stack(positions), torch.stack(rotations)

    def body_pos(self, q, body_index):
        return self.frames(q)[0][body_index]

    def jacobian(self, q, body_index):
        """Geometric jacobian (6 x num_joints) of ``body_index``."""
        return self.jacobian_from_frames(*self.frames(q), body_index)

    def jacobian_from_frames(self, positions, rotations, body_index):
        """Jacobian from already-computed frames, to avoid recomputing the FK.

        Rows 0-2 are linear, rows 3-5 angular — Isaac's layout. A revolute joint
        only influences bodies further down the chain, so columns past the body
        are zero.
        """
        target = positions[body_index]
        jacobian = torch.zeros((6, self.num_joints), dtype=self.dtype)
        for i, link in enumerate(self.chain):
            # Body k is placed after joint k-1, so joint i moves body k iff i < k.
            if i >= body_index:
                continue
            axis_world = rotations[i + 1] @ torch.tensor(link["axis"], dtype=self.dtype)
            jacobian[0:3, i] = torch.linalg.cross(axis_world, target - positions[i + 1])
            jacobian[3:6, i] = axis_world
        return jacobian


class _ArticulationData:
    """Mirrors the attribute names of ``robot.data`` in Isaac Lab."""

    def __init__(self, robot):
        self._robot = robot

    @property
    def joint_pos(self):
        return self._robot._joint_pos.unsqueeze(0)

    @property
    def joint_vel(self):
        return self._robot._joint_vel.unsqueeze(0)

    @property
    def body_pos_w(self):
        positions, _ = self._robot.frames()
        return (positions + self._robot.env_origin).unsqueeze(0)

    @property
    def body_quat_w(self):
        _, rotations = self._robot.frames()
        return torch.stack([_quat_from_matrix(r) for r in rotations]).unsqueeze(0)

    @property
    def default_joint_pos(self):
        return self._robot._default_joint_pos.unsqueeze(0)

    @property
    def soft_joint_pos_limits(self):
        return self._robot._limits.unsqueeze(0)

    @property
    def joint_pos_limits(self):
        return self._robot._limits.unsqueeze(0)

    @property
    def joint_names(self):
        return list(self._robot.joint_names)

    @property
    def body_names(self):
        return list(self._robot.body_names)


class _PhysxView:
    def __init__(self, robot):
        self._robot = robot

    def get_jacobians(self):
        """Jacobians of every body except the root, as Isaac reports them."""
        positions, rotations = self._robot.frames()
        stacked = torch.stack(
            [
                self._robot.kinematics.jacobian_from_frames(positions, rotations, index)
                for index in range(1, self._robot.num_bodies)
            ]
        )
        return stacked.unsqueeze(0)


class RobotSurrogate:
    """Articulation-shaped object backed by :class:`SO101Kinematics`.

    The actuator model is a rate-limited first-order tracker. It is deliberately
    crude: the point is that the controllers must cope with an imperfect
    follower, and the tests vary its responsiveness to prove they do.
    """

    # Cap on how far a joint can travel in one control step, per second. At a
    # lower control rate the actuator has proportionally longer to move, so the
    # per-step cap is derived from the rate rather than fixed.
    MAX_JOINT_RATE = 9.0  # rad/s (0.15 rad per step at 60 Hz)

    def __init__(
        self,
        kinematics=None,
        joint_names=None,
        body_names=None,
        default_joint_pos=None,
        limits_deg=None,
        tracking_gain=0.5,
        max_joint_step=None,
        control_hz=30.0,
        env_origin=(0.0, 0.0, 0.0),
        device="cpu",
        dtype=torch.float32,
    ):
        self.kinematics = kinematics or SO101Kinematics(dtype=dtype)
        self.joint_names = list(joint_names or JOINT_NAMES)
        self.body_names = list(body_names or BODY_NAMES)
        self.device = device
        self.dtype = dtype
        self.is_fixed_base = True
        self.tracking_gain = tracking_gain
        self.control_hz = control_hz
        self.max_joint_step = (
            max_joint_step if max_joint_step is not None else self.MAX_JOINT_RATE / control_hz
        )
        self.env_origin = torch.tensor(env_origin, dtype=dtype)

        limits_deg = limits_deg or JOINT_LIMITS_DEG
        self._limits = torch.tensor(
            [[math.radians(lo), math.radians(hi)] for lo, hi in limits_deg], dtype=dtype
        )
        self._default_joint_pos = torch.tensor(
            default_joint_pos or DEFAULT_JOINT_POS, dtype=dtype
        )
        self._joint_pos = self._default_joint_pos.clone()
        self._joint_vel = torch.zeros_like(self._joint_pos)
        self._frames_cache = None
        self.data = _ArticulationData(self)
        self.root_physx_view = _PhysxView(self)

    def frames(self):
        """Cached FK for the current configuration.

        A control step reads body positions and the jacobian several times; the
        FK is identical across all of them, so it is computed once.
        """
        if self._frames_cache is None:
            self._frames_cache = self.kinematics.frames(self._joint_pos)
        return self._frames_cache

    # -- Isaac Lab lookup API ------------------------------------------- #
    @property
    def num_joints(self):
        return len(self.joint_names)

    @property
    def num_bodies(self):
        return len(self.body_names)

    def find_joints(self, names, preserve_order=False):
        return self._find(names, self.joint_names, "joint")

    def find_bodies(self, names, preserve_order=False):
        return self._find(names, self.body_names, "body")

    @staticmethod
    def _find(names, available, kind):
        """Name lookup with Isaac Lab's semantics.

        Isaac treats the patterns as regexes and **raises** when one of them
        matches nothing — it does not return an empty list. Reproducing that is
        what makes the ``is None`` fallbacks in the controllers testable: with a
        forgiving lookup they were unreachable code.
        """
        ids, found, missing = [], [], []
        for name in names:
            if name in available:
                ids.append(available.index(name))
                found.append(name)
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                "Not all regular expressions are matched! Please check that the "
                f"regular expressions are correct: \n\t{missing[0]}: []\n"
                f"Available strings: {available}"
            )
        return ids, found

    # -- Driving the surrogate ------------------------------------------ #
    def set_joint_pos(self, q):
        self._joint_pos = torch.as_tensor(q, dtype=self.dtype).clone()
        self._frames_cache = None

    # -- Isaac Lab write API, enough to drive the calibration probe ------ #
    def write_joint_state_to_sim(self, positions, velocities=None, env_ids=None):
        values = torch.as_tensor(positions, dtype=self.dtype)
        self.set_joint_pos(values[0] if values.ndim == 2 else values)
        if velocities is not None:
            velocities = torch.as_tensor(velocities, dtype=self.dtype)
            self._joint_vel = velocities[0] if velocities.ndim == 2 else velocities

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        values = torch.as_tensor(target, dtype=self.dtype)
        self._joint_target = values[0] if values.ndim == 2 else values

    def write_data_to_sim(self):
        return None

    def update(self, dt):
        return None

    def apply_targets(self, targets):
        """Advance one control step toward ``targets`` and clamp to the limits."""
        targets = torch.as_tensor(list(targets), dtype=self.dtype)
        delta = (targets - self._joint_pos) * self.tracking_gain
        delta = torch.clamp(delta, -self.max_joint_step, self.max_joint_step)
        new_pos = torch.clamp(
            self._joint_pos + delta, self._limits[:, 0], self._limits[:, 1]
        )
        self._joint_vel = new_pos - self._joint_pos
        self._joint_pos = new_pos
        self._frames_cache = None
        return self._joint_pos


def randomized_kinematics(seed, dtype=torch.float32):
    """A plausible but different arm, to prove the controllers are not overfit."""
    generator = torch.Generator().manual_seed(seed)

    def jitter(value, scale):
        return value * float(1.0 + scale * (torch.rand(1, generator=generator).item() - 0.5))

    chain = []
    for link in DEFAULT_CHAIN:
        origin = tuple(
            jitter(component, 0.6) if abs(component) > 1e-6 else component
            for component in link["origin"]
        )
        chain.append({"axis": link["axis"], "origin": origin})
    return SO101Kinematics(chain=chain, dtype=dtype)

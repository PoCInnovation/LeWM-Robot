"""Building blocks shared by every SO-101 trajectory generator.

Each collection mode drives the arm differently, but they all need the same
three things: a grasp point, a differential IK, and a wrist servo. Keeping them
here means a fix lands once instead of in every mode.
"""

import torch

# Geometry of the grasp point, calibrated on the real gripper. These are the
# values the pick-and-place task was tuned with.
FINGER_LEN = 0.060      # reach beyond the jaw body toward the very fingertip
GRASP_LATERAL = 0.005   # sideways shift so the object sits between the two jaws

JAW_BODY_CANDIDATES = ["jaw", "moving_jaw", "Jaw"]
WRIST_BODY_CANDIDATES = ["wrist", "Wrist"]

JOINT_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Damping of the least-squares inverse. Large enough to stay well behaved near
# singularities, small enough to track accurately elsewhere.
IK_DAMPING = 0.05


def matrix_from_quat(quat):
    """Rotation matrix from a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ]
    )


def find_first_body(robot, candidates):
    """Index of the first candidate the articulation actually has, else None.

    ``find_bodies`` treats its argument as a regex and raises when nothing
    matches, so candidates are checked against the body list instead.
    """
    body_names = list(robot.data.body_names)
    return next(
        (body_names.index(name) for name in candidates if name in body_names), None
    )


class GraspPoint:
    """Where the fingers actually close, as a fixed point of the gripper body.

    The control point used to be rebuilt every frame from world-frame terms: the
    jaw body, plus a reach along the wrist-to-gripper direction, plus a sideways
    shift taken in the **world** horizontal plane. That construction is only
    valid while the gripper keeps one orientation. Measured on the calibrated
    arm, the point it produces drifts by up to **34 mm** as ``Wrist_Roll`` turns
    and **4.7 mm** as the tilt changes — more than the width of the 25 mm cube
    the task picks up.

    It survived because the scripted pick-and-place never rotates the wrist. The
    exploration and reaching modes do, by design, so the grasp point is instead
    expressed as a constant offset in the gripper's own frame: rigidly attached
    to the fingers, therefore correct under any rotation.

    The offset is calibrated once, at construction, so that it reproduces the
    original construction exactly at the pose the task was tuned at.
    """

    def __init__(self, env, robot, finger_len=FINGER_LEN, grasp_lateral=GRASP_LATERAL):
        self._env = env
        self._robot = robot
        self._finger_len = finger_len
        self._grasp_lateral = grasp_lateral

        self.gripper_body_id = find_first_body(robot, ["gripper"])
        if self.gripper_body_id is None:
            raise ValueError("no 'gripper' body on this articulation")
        self.jaw_body_id = find_first_body(robot, JAW_BODY_CANDIDATES)
        self.wrist_body_id = find_first_body(robot, WRIST_BODY_CANDIDATES)
        if self.jaw_body_id is None:
            print("[WARNING]: no jaw body found - grasp point falls back to gripper body")

        self.local_offset = None
        self.recalibrate()

    # -- tuning ---------------------------------------------------------- #
    @property
    def finger_len(self):
        return self._finger_len

    @finger_len.setter
    def finger_len(self, value):
        self._finger_len = value
        self.recalibrate()

    @property
    def grasp_lateral(self):
        return self._grasp_lateral

    @grasp_lateral.setter
    def grasp_lateral(self, value):
        self._grasp_lateral = value
        self.recalibrate()

    # -- geometry -------------------------------------------------------- #
    def _body_pos(self, body_id):
        return self._robot.data.body_pos_w[0, body_id] - self._env.scene.env_origins[0]

    def _gripper_rotation(self):
        return matrix_from_quat(self._robot.data.body_quat_w[0, self.gripper_body_id])

    def _reference_world_point(self):
        """The original world-frame construction, used only to calibrate."""
        gripper = self._body_pos(self.gripper_body_id)
        if self.jaw_body_id is None:
            return gripper

        point = self._body_pos(self.jaw_body_id)
        if self.wrist_body_id is not None and abs(self._finger_len) > 1e-6:
            axis = gripper - self._body_pos(self.wrist_body_id)
            norm = float(torch.linalg.norm(axis))
            if norm > 1e-6:
                point = point + self._finger_len * (axis / norm)

        if abs(self._grasp_lateral) > 1e-6:
            lateral = (self._body_pos(self.jaw_body_id) - gripper).clone()
            lateral[2] = 0.0
            norm = float(torch.linalg.norm(lateral))
            if norm > 1e-6:
                point = point + self._grasp_lateral * (lateral / norm)
        return point

    def recalibrate(self):
        """Freeze the current grasp point into the gripper frame.

        Call after changing ``finger_len`` or ``grasp_lateral``; the robot should
        be at its spawn pose, which is where the task's offsets were tuned.
        """
        rotation = self._gripper_rotation()
        offset = self._reference_world_point() - self._body_pos(self.gripper_body_id)
        self.local_offset = rotation.transpose(0, 1) @ offset
        return self.local_offset

    def world(self):
        """Current grasp point, in the environment frame."""
        return (
            self._body_pos(self.gripper_body_id)
            + self._gripper_rotation() @ self.local_offset
        )


class ArmController:
    """Joint bookkeeping, differential IK and wrist servo, shared by every mode.

    Both the keyboard teleoperation and the scripted policy resolve the same
    joints, clamp against the same limits, and steer the arm with the same
    damped least-squares inverse. Holding that in one place means a fix reaches
    every trajectory generator at once instead of one copy at a time.

    Subclasses drive the arm by producing a cartesian target for the grasp point
    and letting :meth:`solve_arm` turn it into joint commands.
    """

    def __init__(self, env, damping=IK_DAMPING):
        self._env = env
        robot = env.scene["robot"]
        self._robot = robot
        self._device = robot.device
        self._damping = damping

        self._arm_ids = robot.find_joints(
            ["Rotation", "Pitch", "Elbow"], preserve_order=True
        )[0]
        self._pitch_id = self._arm_ids[1]
        self._elbow_id = self._arm_ids[2]
        self._wrist_id = robot.find_joints(["Wrist_Pitch"])[0][0]
        self._roll_id = robot.find_joints(["Wrist_Roll"])[0][0]
        self._jaw_id = robot.find_joints(["Jaw"])[0][0]
        self._out_ids = [
            self._arm_ids[0], self._pitch_id, self._elbow_id,
            self._wrist_id, self._roll_id, self._jaw_id,
        ]

        ee_body_id = robot.find_bodies(["gripper"])[0][0]
        self._ee_body_id = ee_body_id
        # The jacobian array skips the root link on a fixed-base articulation.
        self._jac_body_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = robot.data.joint_pos_limits
        self._limits = limits[0]
        self._eye3 = torch.eye(3, device=self._device)

        default_jp = robot.data.default_joint_pos[0]
        self._home = [float(default_jp[i]) for i in self._out_ids]

    # -- state ----------------------------------------------------------- #
    def joint_pos(self):
        return self._robot.data.joint_pos[0]

    def jacobian(self):
        return self._robot.root_physx_view.get_jacobians()[0, self._jac_body_id]

    def clamp(self, value, joint_id):
        lo = float(self._limits[joint_id, 0])
        hi = float(self._limits[joint_id, 1])
        return max(lo, min(hi, value))

    # -- control --------------------------------------------------------- #
    def wrist_signs(self, jac):
        """How Pitch and Elbow rotate the gripper relative to Wrist_Pitch.

        The three axes are parallel on the SO-101, so holding their signed sum
        holds the gripper's angle to the ground while the arm moves. The signs
        are read from the jacobian rather than hardcoded, so the servo survives
        a change of joint convention in the USD.
        """
        axis_w = jac[3:6, self._wrist_id]
        s_p = float(torch.sign(torch.dot(jac[3:6, self._pitch_id], axis_w)))
        s_e = float(torch.sign(torch.dot(jac[3:6, self._elbow_id], axis_w)))
        return (s_p or 1.0), (s_e or 1.0)

    def solve_arm(self, delta, jac, joint_pos, dq_max):
        """Damped least-squares step of (Rotation, Pitch, Elbow) toward ``delta``.

        Returns the three clamped joint targets.
        """
        j_pos = jac[0:3][:, self._arm_ids]
        jjt = j_pos @ j_pos.T + (self._damping ** 2) * self._eye3
        dq = j_pos.T @ torch.linalg.solve(jjt, delta)
        dq = torch.clamp(dq, -dq_max, dq_max)
        q_arm = joint_pos[self._arm_ids] + dq
        return (
            self.clamp(float(q_arm[0]), self._arm_ids[0]),
            self.clamp(float(q_arm[1]), self._pitch_id),
            self.clamp(float(q_arm[2]), self._elbow_id),
        )

import math

import carb
import omni.appwindow
import torch

JOINT_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

REFERENCE_HZ = 60.0

POS_RATE = 0.002 * REFERENCE_HZ
ANG_RATE = 0.015 * REFERENCE_HZ
JAW_RATE = 0.015 * REFERENCE_HZ
DQ_MAX_RATE = 0.05 * REFERENCE_HZ
HOME_RATE = 0.01 * REFERENCE_HZ
HOME_MIN_S = 30 / REFERENCE_HZ

LEASH = 0.06
IK_DAMPING = 0.05

HOME_KEY = "H"

MOVE_BINDINGS = {
    "UP": (0, +1.0),
    "DOWN": (0, -1.0),
    "LEFT": (1, +1.0),
    "RIGHT": (1, -1.0),
    "W": (2, +1.0),
    "PAGE_UP": (2, +1.0),
    "S": (2, -1.0),
    "PAGE_DOWN": (2, -1.0),
}

PITCH_KEYS = {"T": +1.0, "G": -1.0}
ROLL_KEYS = {"D": +1.0, "A": -1.0}
JAW_KEYS = {"Q": +1.0, "E": -1.0}


class KeyboardEEControl:
    """Cartesian keyboard teleoperation of the SO-101 end-effector.

    Movement keys displace a virtual target point in the robot base frame; a
    damped-least-squares differential IK on the (Rotation, Pitch, Elbow)
    joints tracks it so several joints move together, like human teleop.
    The wrist pitch is servoed to keep the gripper tilt constant while the
    arm moves (Pitch/Elbow/Wrist_Pitch axes are parallel on the SO-101, so
    holding their signed sum holds the gripper angle w.r.t. the ground).
    """

    def __init__(self, env):
        self._env = env
        robot = env.scene["robot"]
        self._robot = robot
        self._device = robot.device

        self._dt = float(getattr(env, "step_dt", None) or 1.0 / REFERENCE_HZ)
        self._pos_step = POS_RATE * self._dt
        self._ang_step = ANG_RATE * self._dt
        self._jaw_step = JAW_RATE * self._dt
        self._dq_max = DQ_MAX_RATE * self._dt
        self._home_step = HOME_RATE * self._dt
        self._home_min_frames = max(1, round(HOME_MIN_S / self._dt))

        self._arm_ids = robot.find_joints(
            ["Rotation", "Pitch", "Elbow"], preserve_order=True
        )[0]
        self._pitch_id = self._arm_ids[1]
        self._elbow_id = self._arm_ids[2]
        self._wrist_id = robot.find_joints(["Wrist_Pitch"])[0][0]
        self._roll_id = robot.find_joints(["Wrist_Roll"])[0][0]
        self._jaw_id = robot.find_joints(["Jaw"])[0][0]

        ee_body_id = robot.find_bodies(["gripper"])[0][0]
        self._ee_body_id = ee_body_id
        self._jac_body_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = robot.data.joint_pos_limits
        self._limits = limits[0]

        self._eye3 = torch.eye(3, device=self._device)

        self._held = set()
        self._needs_sync = True
        self._target_pos = None
        self._pitch_sum = 0.0
        self._roll_target = 0.0
        self._jaw_target = 0.0

        self._out_ids = [
            self._arm_ids[0],
            self._pitch_id,
            self._elbow_id,
            self._wrist_id,
            self._roll_id,
            self._jaw_id,
        ]
        default_jp = robot.data.default_joint_pos[0]
        self._home_targets = [float(default_jp[i]) for i in self._out_ids]
        self._control_keys = (
            set(MOVE_BINDINGS) | set(PITCH_KEYS) | set(ROLL_KEYS) | set(JAW_KEYS)
        )
        self._home_requested = False
        self._homing = None
        self._last_targets = None

        self._window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

        print("[INFO]: Cartesian keyboard control active (labels AZERTY):")
        print("        Fleche haut / bas   : avancer / reculer")
        print("        Fleche gauche/droite: gauche / droite")
        print("        Z / S               : monter / descendre (alt: PgUp / PgDn)")
        print("        T / G               : incliner la pince (haut / bas)")
        print("        Q / D               : rotation de la pince (roll)")
        print("        A / E               : ouvrir / fermer la pince")
        print("        H                   : retour auto a la position de depart")
        print("        R : reset  |  P : start/stop recording  |  C : annuler")

    def _on_keyboard_event(self, event, *args, **kwargs):
        inp = getattr(event, "input", None)
        name = getattr(inp, "name", inp)
        if not isinstance(name, str):
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == HOME_KEY:
                self._home_requested = True
            self._held.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
        return True

    def _wrist_signs(self, jac):
        axis_w = jac[3:6, self._wrist_id]
        s_p = float(torch.sign(torch.dot(jac[3:6, self._pitch_id], axis_w)))
        s_e = float(torch.sign(torch.dot(jac[3:6, self._elbow_id], axis_w)))
        return (s_p or 1.0), (s_e or 1.0)

    def _clamp(self, value, joint_id):
        lo = float(self._limits[joint_id, 0])
        hi = float(self._limits[joint_id, 1])
        return max(lo, min(hi, value))

    def step(self):
        """Call once per simulation frame. Returns the 6 joint targets in JOINT_ORDER."""
        robot = self._robot
        env_origin = self._env.scene.env_origins[0]
        joint_pos = robot.data.joint_pos[0]

        if self._home_requested:
            self._home_requested = False
            if self._homing is None:
                start = self._last_targets or [
                    float(joint_pos[i]) for i in self._out_ids
                ]
                max_delta = max(
                    abs(h - s) for h, s in zip(self._home_targets, start)
                )
                frames = max(self._home_min_frames, int(max_delta / self._home_step))
                self._homing = {"start": start, "frame": 0, "frames": frames}
                print("[INFO]: Returning to home position...")

        if self._homing is not None:
            if self._held & self._control_keys:
                self._homing = None
                self._needs_sync = True
                print("[INFO]: Return to home cancelled by manual input.")
            else:
                hom = self._homing
                hom["frame"] += 1
                s = min(1.0, hom["frame"] / hom["frames"])
                s = 0.5 - 0.5 * math.cos(math.pi * s)
                targets = [
                    a + (b - a) * s
                    for a, b in zip(hom["start"], self._home_targets)
                ]
                if hom["frame"] >= hom["frames"]:
                    self._homing = None
                    self._needs_sync = True
                    print("[INFO]: Home position reached.")
                self._last_targets = targets
                return targets

        ee_pos = robot.data.body_pos_w[0, self._ee_body_id] - env_origin
        jac = robot.root_physx_view.get_jacobians()[0, self._jac_body_id]
        s_p, s_e = self._wrist_signs(jac)

        if self._needs_sync:
            self._needs_sync = False
            self._target_pos = ee_pos.clone()
            self._pitch_sum = float(
                s_p * joint_pos[self._pitch_id]
                + s_e * joint_pos[self._elbow_id]
                + joint_pos[self._wrist_id]
            )
            self._roll_target = float(joint_pos[self._roll_id])
            self._jaw_target = float(joint_pos[self._jaw_id])

        for key, (axis, direction) in MOVE_BINDINGS.items():
            if key in self._held:
                self._target_pos[axis] += direction * self._pos_step
        self._target_pos[2] = torch.clamp(self._target_pos[2], min=0.0)

        delta = self._target_pos - ee_pos
        dist = float(torch.linalg.norm(delta))
        if dist > LEASH:
            self._target_pos = ee_pos + delta * (LEASH / dist)
            delta = self._target_pos - ee_pos

        j_pos = jac[0:3][:, self._arm_ids]
        jjt = j_pos @ j_pos.T + (IK_DAMPING**2) * self._eye3
        dq = j_pos.T @ torch.linalg.solve(jjt, delta)
        dq = torch.clamp(dq, -self._dq_max, self._dq_max)
        q_arm = joint_pos[self._arm_ids] + dq

        q_rotation = self._clamp(float(q_arm[0]), self._arm_ids[0])
        q_pitch = self._clamp(float(q_arm[1]), self._pitch_id)
        q_elbow = self._clamp(float(q_arm[2]), self._elbow_id)

        for key, direction in PITCH_KEYS.items():
            if key in self._held:
                self._pitch_sum += direction * self._ang_step
        q_wrist = self._pitch_sum - (s_p * q_pitch + s_e * q_elbow)
        q_wrist = self._clamp(q_wrist, self._wrist_id)
        self._pitch_sum = q_wrist + (s_p * q_pitch + s_e * q_elbow)

        for key, direction in ROLL_KEYS.items():
            if key in self._held:
                self._roll_target += direction * self._ang_step
        self._roll_target = self._clamp(self._roll_target, self._roll_id)

        for key, direction in JAW_KEYS.items():
            if key in self._held:
                self._jaw_target += direction * self._jaw_step
        self._jaw_target = self._clamp(self._jaw_target, self._jaw_id)

        targets = [
            q_rotation,
            q_pitch,
            q_elbow,
            q_wrist,
            self._roll_target,
            self._jaw_target,
        ]
        self._last_targets = targets
        return targets

    def reset(self):
        """Re-sync targets on the robot state after a world reset."""
        self._needs_sync = True
        self._homing = None
        self._home_requested = False
        self._last_targets = None
        self._held.clear()

    def cleanup(self):
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None

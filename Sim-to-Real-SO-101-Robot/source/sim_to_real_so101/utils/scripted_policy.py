"""Scripted pick-and-place policy for automatic dataset collection.

Reuses the same damped-least-squares differential IK as the cartesian keyboard
controller, but drives the end-effector through a sequence of motion primitives
(move_above / descend / grasp / lift / move_to / release / home) whose targets
are read from the true object poses in simulation. Produces joint actions frame
by frame, exactly like the keyboard controller, so it plugs into the same
env.step() + recorder path.

All distances are in meters, in the robot base frame.
"""

import carb
import omni.appwindow
import torch

JOINT_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

DEFAULT_POSE = {
    "Rotation": -0.2736,
    "Pitch": -0.6109,
    "Elbow": -0.0745,
    "Wrist_Pitch": 1.5148,
    "Wrist_Roll": -1.6034,
    "Jaw": -0.1465,
}

# --- IK tuning (mirrors keyboard_ee_control) ---------------------------
REFERENCE_HZ = 60.0

IK_DAMPING = 0.05
DQ_MAX_RATE = 0.05 * REFERENCE_HZ       # rad/s of joint motion per IK step
POS_SPEED_RATE = 0.004 * REFERENCE_HZ   # m/s the target advances toward a waypoint
ANG_SPEED_RATE = 0.04 * REFERENCE_HZ    # rad/s for the return-home motion
HOME_MAX_S = 200 / REFERENCE_HZ         # force the home state to finish after this

# --- Task tuning (EXPECT to fine-tune these visually) ------------------
# The control point is the GRASP POINT (between the fingers), computed as the
# gripper body shifted by FINGER_LEN along the gripper's pointing axis. So the
# offsets below are of the actual fingertips above the target point.
FINGER_LEN = 0.060        # reach beyond the jaw body toward the very fingertip
GRASP_OFFSET = 0.015      # jaw height above the cube center at grasp (tune live O/L)
GRASP_LATERAL = 0.005     # sideways shift so the cube sits between the two jaws
APPROACH_OFFSET = 0.08    # jaw height above the cube for the pre-grasp pose
CARRY_OFFSET = 0.14       # high transport height, to clear the box walls
PLACE_OFFSET = 0.09       # fingertip height above the box floor when arriving
PLACE_DROP_OFFSET = 0.04  # fingertip height above the box floor when releasing
PLACE_Z_LIFT = 0.022      # place reference height above the box-floor body origin
Z_MIN = 0.03              # hard floor for the target height (never drive into the table)

JAW_OPEN = 0.6            # jaw target (rad) when open
JAW_CLOSED = -0.35        # jaw target (rad) when closed on the cube
GRASP_TILT = 1.55         # wrist-pitch "sum" held during the task

POS_TOL = 0.012           # position tolerance to consider a waypoint reached (m)
GRIPPER_HOLD_S = 25 / REFERENCE_HZ   # time to hold while the gripper opens/closes
STATE_TIMEOUT_S = 400 / REFERENCE_HZ  # max time per motion state before failure


class ScriptedPickPlace:
    """State-machine pick-and-place controller for one rigid object into a target."""

    def __init__(self, env, pick_object="Cube", place_object="BoxFloor",
                 place_at=(0.22, 0.10, 0.06)):
        self._env = env
        robot = env.scene["robot"]
        self._robot = robot
        self._device = robot.device
        self._pick_object = pick_object
        self._place_object = place_object  # read live so placing follows the box
        self._place_at = tuple(place_at)   # fallback if the box object is absent

        self._dt = float(getattr(env, "step_dt", None) or 1.0 / REFERENCE_HZ)
        self._pos_speed = POS_SPEED_RATE * self._dt
        self._ang_speed = ANG_SPEED_RATE * self._dt
        self._dq_max = DQ_MAX_RATE * self._dt
        self._gripper_frames = max(1, round(GRIPPER_HOLD_S / self._dt))
        self._state_timeout = max(1, round(STATE_TIMEOUT_S / self._dt))
        self._home_max_frames = max(1, round(HOME_MAX_S / self._dt))

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
        self._jac_body_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

        # The moving-jaw body sits at the business end of the gripper, so its
        # world position is a good, measured proxy for the grasp point (no need
        # to extrapolate an unknown finger length along a tilted axis).
        self._jaw_body_id = None
        for cand in ["jaw", "moving_jaw", "Jaw"]:
            ids = robot.find_bodies([cand])[0]
            if ids:
                self._jaw_body_id = ids[0]
                break
        if self._jaw_body_id is None:
            print("[WARNING]: no jaw body found - grasp point falls back to gripper body")

        # Wrist body: the wrist->gripper vector is the gripper's pointing axis,
        # used to push the control point out to the real fingertips.
        self._wrist_body_id = None
        for cand in ["wrist", "Wrist"]:
            ids = robot.find_bodies([cand])[0]
            if ids:
                self._wrist_body_id = ids[0]
                break

        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = robot.data.joint_pos_limits
        self._limits = limits[0]
        self._eye3 = torch.eye(3, device=self._device)

        # Home = the robot's actual spawn pose, so it returns exactly there.
        default_jp = robot.data.default_joint_pos[0]
        self._home = [float(default_jp[i]) for i in self._out_ids]

        # Live-tunable grasp height (O / L) and fingertip reach (I / K), so both
        # can be dialed in without restarting the sim. Kept out of reset() so
        # they persist across retries.
        self._grasp_offset = GRASP_OFFSET
        self._finger_len = FINGER_LEN
        self._grasp_lateral = GRASP_LATERAL
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_kb = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )
        print("[TUNE] O/L=height(5mm)  I/K=reach(10mm)  J/H=lateral(5mm)  then R to retry.")

        self.reset()

    def _on_key(self, event, *args, **kwargs):
        inp = getattr(event, "input", None)
        name = getattr(inp, "name", inp)
        if isinstance(name, str) and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == "O":
                self._grasp_offset += 0.005
                print(f"[TUNE] GRASP_OFFSET = {self._grasp_offset:.3f}")
            elif name == "L":
                self._grasp_offset -= 0.005
                print(f"[TUNE] GRASP_OFFSET = {self._grasp_offset:.3f}")
            elif name == "I":
                self._finger_len += 0.01
                print(f"[TUNE] FINGER_LEN = {self._finger_len:.3f}")
            elif name == "K":
                self._finger_len -= 0.01
                print(f"[TUNE] FINGER_LEN = {self._finger_len:.3f}")
            elif name == "J":
                self._grasp_lateral += 0.005
                print(f"[TUNE] GRASP_LATERAL = {self._grasp_lateral:.3f}")
            elif name == "H":
                self._grasp_lateral -= 0.005
                print(f"[TUNE] GRASP_LATERAL = {self._grasp_lateral:.3f}")
        return True

    # ------------------------------------------------------------------ #
    def reset(self):
        """Restart the state machine (call after each env.reset())."""
        self._target_pos = None
        self._roll_target = float(DEFAULT_POSE["Wrist_Roll"])
        self._jaw_target = JAW_OPEN
        self._pitch_sum = GRASP_TILT
        self._state = 0
        self._timer = 0
        self._needs_sync = True
        self.status = "running"   # running | done | failed
        # Ordered plan: each entry is (name, kind, payload).
        # For "reach", payload = (target_kind, z_offset_above_target).
        self._plan = [
            ("open_gripper",    "gripper", JAW_OPEN),
            ("move_above_pick", "reach",  ("pick", APPROACH_OFFSET)),
            ("descend",         "reach",  ("pick", GRASP_OFFSET)),
            ("grasp",           "gripper", JAW_CLOSED),
            ("lift",            "reach",  ("pick_frozen", CARRY_OFFSET)),
            ("move_above_place", "reach", ("place", CARRY_OFFSET)),
            ("lower_place",     "reach",  ("place", PLACE_DROP_OFFSET)),
            ("release",         "gripper", JAW_OPEN),
            ("home",            "home",    None),
        ]
        self._frozen_xy = None  # cube xy latched at grasp time
        self._frozen_z = None   # cube z latched at grasp time

    # ------------------------------------------------------------------ #
    def _obj_pos(self, name):
        return (self._env.scene[name].data.root_pos_w[0]
                - self._env.scene.env_origins[0])

    def _place_ref(self):
        """Live placement reference (follows the box), falling back to place_at."""
        try:
            p = self._obj_pos(self._place_object)
            return float(p[0]), float(p[1]), float(p[2]) + PLACE_Z_LIFT
        except Exception:
            return self._place_at

    def is_success(self, xy_tol=0.06, z_max=0.09):
        """True if the cube ended up inside the box footprint (not still held)."""
        cube = self._obj_pos(self._pick_object)
        px, py, _ = self._place_ref()
        return (
            abs(float(cube[0]) - px) < xy_tol
            and abs(float(cube[1]) - py) < xy_tol
            and float(cube[2]) < z_max
        )

    def _clamp(self, value, joint_id):
        lo = float(self._limits[joint_id, 0])
        hi = float(self._limits[joint_id, 1])
        return max(lo, min(hi, value))

    def _wrist_signs(self, jac):
        axis_w = jac[3:6, self._wrist_id]
        s_p = float(torch.sign(torch.dot(jac[3:6, self._pitch_id], axis_w)))
        s_e = float(torch.sign(torch.dot(jac[3:6, self._elbow_id], axis_w)))
        return (s_p or 1.0), (s_e or 1.0)

    def _waypoint_target(self, kind, z_offset):
        """World-frame goal position for a 'reach' primitive.

        z_offset is added above the target reference height (cube center for a
        pick, box floor for a place), then floored at Z_MIN for safety.
        """
        if kind == "place":
            ref = self._place_ref()
            xy = torch.tensor(ref[:2], device=self._device)
            z0 = ref[2]
        elif kind == "pick_frozen":
            xy = self._frozen_xy
            z0 = self._frozen_z
        else:  # "pick"
            p = self._obj_pos(self._pick_object)
            xy = p[:2]
            z0 = float(p[2])
        goal = torch.zeros(3, device=self._device)
        goal[:2] = xy
        goal[2] = max(Z_MIN, z0 + z_offset)
        return goal

    # ------------------------------------------------------------------ #
    def step(self):
        """Advance the state machine one frame. Returns the 6 joint targets."""
        robot = self._robot
        joint_pos = robot.data.joint_pos[0]
        body_pos = robot.data.body_pos_w[0, self._ee_body_id] - self._env.scene.env_origins[0]
        jac = robot.root_physx_view.get_jacobians()[0, self._jac_body_id]
        s_p, s_e = self._wrist_signs(jac)

        # Control point = the grasp point: the moving-jaw body, pushed out by
        # FINGER_LEN along the gripper's pointing axis (wrist->gripper) to reach
        # the real fingertips. Aiming the fingertips (not the jaw origin) at the
        # target stops the gripper from overshooting the cube.
        env_origin = self._env.scene.env_origins[0]
        if self._jaw_body_id is not None:
            jaw_pos = robot.data.body_pos_w[0, self._jaw_body_id] - env_origin
            ee_pos = jaw_pos.clone()
            if self._wrist_body_id is not None and abs(self._finger_len) > 1e-6:
                wrist_pos = robot.data.body_pos_w[0, self._wrist_body_id] - env_origin
                u = body_pos - wrist_pos
                n = float(torch.linalg.norm(u))
                if n > 1e-6:
                    ee_pos = ee_pos + self._finger_len * (u / n)
            # Sideways shift along the gripper opening axis (horizontal part of
            # gripper-body -> moving-jaw), so the cube ends up between the fixed
            # and moving jaw rather than under the fixed one.
            if abs(self._grasp_lateral) > 1e-6:
                o = (jaw_pos - body_pos).clone()
                o[2] = 0.0
                on = float(torch.linalg.norm(o))
                if on > 1e-6:
                    ee_pos = ee_pos + self._grasp_lateral * (o / on)
        else:
            ee_pos = body_pos

        if self._needs_sync:
            self._needs_sync = False
            self._target_pos = ee_pos.clone()

        if self.status != "running":
            return [self._clamp(v, self._out_ids[i]) for i, v in enumerate(self._home_hold())]

        name, kind, payload = self._plan[self._state]
        self._timer += 1
        advance = False

        if kind == "gripper":
            self._jaw_target = float(payload)
            if self._timer >= self._gripper_frames:
                advance = True

        elif kind == "home":
            # Ease every joint toward the spawn pose.
            done = True
            targets = []
            cur = [float(joint_pos[i]) for i in self._out_ids]
            for i, (c, h) in enumerate(zip(cur, self._home)):
                d = h - c
                if abs(d) > self._ang_speed:
                    done = False
                    c = c + self._ang_speed * (1.0 if d > 0 else -1.0)
                else:
                    c = h
                targets.append(self._clamp(c, self._out_ids[i]))
            if done or self._timer >= self._home_max_frames:
                self.status = "done"
            return targets

        else:  # "reach"
            kindname, height = payload
            if name == "descend":
                height = self._grasp_offset  # live-tunable grasp height
            goal = self._waypoint_target(kindname, height)
            # Move the virtual target toward the goal at bounded speed.
            delta = goal - self._target_pos
            dist = float(torch.linalg.norm(delta))
            if dist > self._pos_speed:
                self._target_pos = self._target_pos + delta * (self._pos_speed / dist)
            else:
                self._target_pos = goal
            # Safety floor: never command the target into the table.
            if float(self._target_pos[2]) < Z_MIN:
                self._target_pos[2] = Z_MIN
            # Reached when the actual EE is within tolerance of the goal.
            if float(torch.linalg.norm(goal - ee_pos)) < POS_TOL:
                if kindname == "pick":
                    p = self._obj_pos(self._pick_object)
                    self._frozen_xy = p[:2].clone()
                    self._frozen_z = float(p[2])
                advance = True

        # Timeout safety on motion states.
        if self._timer >= self._state_timeout and kind != "gripper":
            self.status = "failed"

        # --- IK toward self._target_pos (arm) --------------------------
        delta = self._target_pos - ee_pos
        j_pos = jac[0:3][:, self._arm_ids]
        jjt = j_pos @ j_pos.T + (IK_DAMPING ** 2) * self._eye3
        dq = j_pos.T @ torch.linalg.solve(jjt, delta)
        dq = torch.clamp(dq, -self._dq_max, self._dq_max)
        q_arm = joint_pos[self._arm_ids] + dq

        q_rotation = self._clamp(float(q_arm[0]), self._arm_ids[0])
        q_pitch = self._clamp(float(q_arm[1]), self._pitch_id)
        q_elbow = self._clamp(float(q_arm[2]), self._elbow_id)

        # Wrist servo holds the gripper tilt (pointing down) while the arm moves.
        q_wrist = self._clamp(self._pitch_sum - (s_p * q_pitch + s_e * q_elbow), self._wrist_id)
        self._pitch_sum = q_wrist + (s_p * q_pitch + s_e * q_elbow)

        self._roll_target = self._clamp(self._roll_target, self._roll_id)
        self._jaw_target = self._clamp(self._jaw_target, self._jaw_id)

        if advance:
            self._state += 1
            self._timer = 0
            if self._state >= len(self._plan):
                self.status = "done"

        return [q_rotation, q_pitch, q_elbow, q_wrist, self._roll_target, self._jaw_target]

    def _home_hold(self):
        """Hold a fixed pose once done/failed (constant command = no jitter)."""
        if self.status == "done":
            return list(self._home)  # settle exactly at the spawn pose
        jp = self._robot.data.joint_pos[0]
        return [float(jp[i]) for i in self._out_ids]

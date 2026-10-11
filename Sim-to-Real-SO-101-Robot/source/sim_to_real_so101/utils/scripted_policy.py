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

from sim_to_real_so101.utils.arm_control import ArmController, GraspPoint

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


class ScriptedPickPlace(ArmController):
    """State-machine pick-and-place controller for one rigid object into a target."""

    #: Task geometry the scene may override, with the values the cube task was
    #: tuned at. Anything not listed here is not meant to vary per scene.
    DEFAULT_TASK = {
        "pick": "Cube",
        "place": "BoxFloor",
        "place_at": (0.22, 0.10, 0.06),
        "grasp_offset": GRASP_OFFSET,
        "approach_offset": APPROACH_OFFSET,
        "carry_offset": CARRY_OFFSET,
        "place_drop_offset": PLACE_DROP_OFFSET,
        "place_z_lift": PLACE_Z_LIFT,
        "finger_len": FINGER_LEN,
        "grasp_lateral": GRASP_LATERAL,
        "jaw_open": JAW_OPEN,
        "jaw_closed": JAW_CLOSED,
        "grasp_tilt": GRASP_TILT,
        "z_min": Z_MIN,
        "success_xy_tol": 0.06,
        "success_z_max": 0.09,
    }

    def __init__(self, env, task=None, **overrides):
        """Drive one object into one target.

        ``task`` carries the object names and the grasp geometry, so a new scene
        is a data change rather than a code change. Unspecified keys fall back to
        :attr:`DEFAULT_TASK`, which is the tuning the cube task ships with.
        """
        super().__init__(env)
        robot = self._robot

        self.task = {**self.DEFAULT_TASK, **(task or {}), **overrides}
        unknown = set(self.task) - set(self.DEFAULT_TASK)
        if unknown:
            raise ValueError(
                f"unknown task key(s) {sorted(unknown)}; "
                f"expected any of {sorted(self.DEFAULT_TASK)}"
            )

        self._pick_object = self.task["pick"]
        self._place_object = self.task["place"]  # read live so placing follows the box
        self._place_at = tuple(self.task["place_at"])  # fallback if the target is absent

        self._dt = float(getattr(env, "step_dt", None) or 1.0 / REFERENCE_HZ)
        self._pos_speed = POS_SPEED_RATE * self._dt
        self._ang_speed = ANG_SPEED_RATE * self._dt
        self._dq_max = DQ_MAX_RATE * self._dt
        self._gripper_frames = max(1, round(GRIPPER_HOLD_S / self._dt))
        self._state_timeout = max(1, round(STATE_TIMEOUT_S / self._dt))
        self._home_max_frames = max(1, round(HOME_MAX_S / self._dt))

        # Grasp point, held as a fixed offset in the gripper frame so it stays
        # valid when the wrist rotates. See utils/arm_control.GraspPoint.
        self._grasp_point = GraspPoint(
            env, robot, self.task["finger_len"], self.task["grasp_lateral"]
        )
        self._jaw_body_id = self._grasp_point.jaw_body_id
        self._wrist_body_id = self._grasp_point.wrist_body_id

        # Live-tunable grasp height (O / L) and fingertip reach (I / K), so both
        # can be dialed in without restarting the sim. Kept out of reset() so
        # they persist across retries.
        self._grasp_offset = self.task["grasp_offset"]
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

    # The live-tuning keys change the grasp geometry, so the frozen offset has
    # to be recomputed; the properties keep that from being forgotten.
    @property
    def _finger_len(self):
        return self._grasp_point.finger_len

    @_finger_len.setter
    def _finger_len(self, value):
        self._grasp_point.finger_len = value

    @property
    def _grasp_lateral(self):
        return self._grasp_point.grasp_lateral

    @_grasp_lateral.setter
    def _grasp_lateral(self, value):
        self._grasp_point.grasp_lateral = value

    # ------------------------------------------------------------------ #
    def reset(self):
        """Restart the state machine (call after each env.reset())."""
        self._target_pos = None
        self._roll_target = float(DEFAULT_POSE["Wrist_Roll"])
        self._jaw_target = self.task["jaw_open"]
        self._pitch_sum = self.task["grasp_tilt"]
        self._state = 0
        self._timer = 0
        self._needs_sync = True
        self.status = "running"   # running | done | failed
        # Ordered plan: each entry is (name, kind, payload).
        # For "reach", payload = (target_kind, z_offset_above_target).
        self._plan = [
            ("open_gripper",    "gripper", self.task["jaw_open"]),
            ("move_above_pick", "reach",  ("pick", self.task["approach_offset"])),
            ("descend",         "reach",  ("pick", self._grasp_offset)),
            ("grasp",           "gripper", self.task["jaw_closed"]),
            ("lift",            "reach",  ("pick_frozen", self.task["carry_offset"])),
            ("move_above_place", "reach", ("place", self.task["carry_offset"])),
            ("lower_place",     "reach",  ("place", self.task["place_drop_offset"])),
            ("release",         "gripper", self.task["jaw_open"]),
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
            return float(p[0]), float(p[1]), float(p[2]) + self.task["place_z_lift"]
        except Exception:
            return self._place_at

    def is_success(self, xy_tol=None, z_max=None):
        """True if the cube ended up inside the box footprint (not still held)."""
        xy_tol = self.task["success_xy_tol"] if xy_tol is None else xy_tol
        z_max = self.task["success_z_max"] if z_max is None else z_max
        cube = self._obj_pos(self._pick_object)
        px, py, _ = self._place_ref()
        return (
            abs(float(cube[0]) - px) < xy_tol
            and abs(float(cube[1]) - py) < xy_tol
            and float(cube[2]) < z_max
        )

    def _waypoint_target(self, kind, z_offset):
        """World-frame goal position for a 'reach' primitive.

        z_offset is added above the target reference height (cube center for a
        pick, box floor for a place), then floored at the task's z_min.
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
        goal[2] = max(self.task["z_min"], z0 + z_offset)
        return goal

    # ------------------------------------------------------------------ #
    def step(self):
        """Advance the state machine one frame. Returns the 6 joint targets."""
        robot = self._robot
        joint_pos = robot.data.joint_pos[0]
        jac = self.jacobian()
        s_p, s_e = self.wrist_signs(jac)

        # Control point = the grasp point, between the fingers. Rigidly attached
        # to the gripper, so it follows the wrist however it turns.
        ee_pos = self._grasp_point.world()

        if self._needs_sync:
            self._needs_sync = False
            self._target_pos = ee_pos.clone()

        if self.status != "running":
            return [self.clamp(v, self._out_ids[i]) for i, v in enumerate(self._home_hold())]

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
                targets.append(self.clamp(c, self._out_ids[i]))
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
            if float(self._target_pos[2]) < self.task["z_min"]:
                self._target_pos[2] = self.task["z_min"]
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
        q_rotation, q_pitch, q_elbow = self.solve_arm(
            self._target_pos - ee_pos, jac, joint_pos, self._dq_max
        )

        # Wrist servo holds the gripper tilt (pointing down) while the arm moves.
        q_wrist = self.clamp(self._pitch_sum - (s_p * q_pitch + s_e * q_elbow), self._wrist_id)
        self._pitch_sum = q_wrist + (s_p * q_pitch + s_e * q_elbow)

        self._roll_target = self.clamp(self._roll_target, self._roll_id)
        self._jaw_target = self.clamp(self._jaw_target, self._jaw_id)

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

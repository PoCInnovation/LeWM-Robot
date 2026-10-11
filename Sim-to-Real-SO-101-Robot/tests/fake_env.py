"""Minimal stand-in for a ``ManagerBasedRLEnv``, enough to drive the controllers.

The controllers reach into the environment for exactly three things: the robot
articulation, the world position of scene objects, and the environment origin.
Nothing else of Isaac's environment is needed to run them in closed loop.
"""

import torch

from so101_surrogate import RobotSurrogate


class FakeRigidObject:
    """A scene object with a world pose, and nothing more."""

    def __init__(self, pos, device="cpu", dtype=torch.float32):
        self._pos = torch.tensor(pos, dtype=dtype)
        self.device = device
        self.data = self

    @property
    def root_pos_w(self):
        return self._pos.unsqueeze(0)

    def set_pos(self, pos):
        self._pos = torch.as_tensor(pos, dtype=self._pos.dtype).clone()


class FakeScene:
    def __init__(self, robot, objects=None, env_origin=(0.0, 0.0, 0.0), dtype=torch.float32):
        self._entities = {"robot": robot}
        self._entities.update(objects or {})
        self.env_origins = torch.tensor(env_origin, dtype=dtype).unsqueeze(0)

    def __getitem__(self, name):
        return self._entities[name]

    def __contains__(self, name):
        return name in self._entities

    def keys(self):
        return self._entities.keys()

    def add(self, name, entity):
        self._entities[name] = entity


class FakeEnv:
    """Environment shim wrapping a :class:`RobotSurrogate` and a few objects."""

    def __init__(
        self,
        robot=None,
        objects=None,
        env_origin=(0.0, 0.0, 0.0),
        device="cpu",
        control_hz=30.0,
    ):
        self.device = device
        if robot is None:
            robot = RobotSurrogate(
                env_origin=env_origin, device=device, control_hz=control_hz
            )
        self.robot = robot
        self.scene = FakeScene(robot, objects=objects, env_origin=env_origin)
        self.num_envs = 1
        # Isaac Lab exposes the control period as ``step_dt``; the controllers
        # read it to size their per-frame motion, so the harness must too.
        self.step_dt = 1.0 / control_hz

    def step(self, targets):
        """Apply one control step, the way ``env.step`` would."""
        return self.robot.apply_targets(targets)


def run_controller(env, controller, max_steps=2000, stop_when_done=True):
    """Drive a controller to completion and collect what it commanded.

    Returns a dict with the command history, the joint trajectory, the end
    effector path and the final status — everything the assertions need.
    """
    actions = []
    joint_positions = []
    ee_positions = []
    robot = env.robot
    ee_body_id = robot.find_bodies(["gripper"])[0][0]

    steps = 0
    for _ in range(max_steps):
        targets = controller.step()
        actions.append(list(targets))
        env.step(targets)
        joint_positions.append(robot.data.joint_pos[0].tolist())
        ee_positions.append(
            (robot.data.body_pos_w[0, ee_body_id] - env.scene.env_origins[0]).tolist()
        )
        steps += 1
        if stop_when_done and getattr(controller, "status", "running") != "running":
            break

    return {
        "actions": torch.tensor(actions),
        "joint_pos": torch.tensor(joint_positions),
        "ee_pos": torch.tensor(ee_positions),
        "steps": steps,
        "status": getattr(controller, "status", None),
    }

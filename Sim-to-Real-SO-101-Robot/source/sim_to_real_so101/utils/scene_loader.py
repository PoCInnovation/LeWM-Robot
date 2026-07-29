"""Loader for user-defined scenes.

Each scene is a python file in ``source/sim_to_real_so101/scenes/`` defining a
``SCENE`` dict (see the README in that directory). Objects listed there are
dynamically injected into the environment config before ``gym.make``, so any
teleop task of this repo can be enriched with a custom scene via ``--scene``.
"""

import importlib.util
import os

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg

SCENES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenes"
)


def randomize_group_offset(env, env_ids, asset_names, pos_range):
    """Move a set of objects together by one shared random offset on reset.

    Used to jitter a multi-part object (e.g. a box made of separate walls) as a
    single rigid unit, instead of scattering each part independently.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)

    def _draw(key):
        lo, hi = pos_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=env.device).uniform_(lo, hi)

    dx, dy, dz = _draw("x"), _draw("y"), _draw("z")
    for name in asset_names:
        asset = env.scene[name]
        root = asset.data.default_root_state[env_ids].clone()
        root[:, 0] += dx
        root[:, 1] += dy
        root[:, 2] += dz
        root[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(root[:, :7], env_ids)
        asset.write_root_velocity_to_sim(root[:, 7:], env_ids)


def list_scenes():
    """Names of the scenes available in the scenes directory."""
    if not os.path.isdir(SCENES_DIR):
        return []
    return sorted(
        f[:-3]
        for f in os.listdir(SCENES_DIR)
        if f.endswith(".py") and not f.startswith("_")
    )


def load_scene_spec(name):
    """Import ``scenes/<name>.py`` and return its ``SCENE`` dict."""
    path = os.path.join(SCENES_DIR, f"{name}.py")
    if not os.path.isfile(path):
        available = ", ".join(list_scenes()) or "(none)"
        raise FileNotFoundError(
            f"Scene '{name}' not found in {SCENES_DIR}. Available scenes: {available}"
        )
    spec = importlib.util.spec_from_file_location(f"so101_scene_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "SCENE"):
        raise ValueError(f"Scene file {path} must define a SCENE dict.")
    return module.SCENE


def _spawn_cfg(obj):
    """Build the spawner config for one object entry of the SCENE dict."""
    kind = obj.get("type", "cuboid")

    if kind == "usd":
        usd_path = obj["usd_path"]
        if not os.path.isabs(usd_path):
            usd_path = os.path.join(SCENES_DIR, usd_path)
        return sim_utils.UsdFileCfg(usd_path=usd_path)

    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=obj.get("static", False)
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=obj.get("mass", 0.05)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(obj.get("color", (0.5, 0.5, 0.5)))
        ),
    )
    if kind == "cuboid":
        return sim_utils.CuboidCfg(size=tuple(obj["size"]), **common)
    if kind == "sphere":
        return sim_utils.SphereCfg(radius=obj["radius"], **common)
    if kind == "cylinder":
        return sim_utils.CylinderCfg(
            radius=obj["radius"], height=obj["height"], **common
        )
    raise ValueError(f"Unknown object type '{kind}' (use cuboid/sphere/cylinder/usd)")


def apply_scene(env_cfg, name):
    """Inject the objects of scene ``name`` into an environment config."""
    scene = load_scene_spec(name)

    env_cfg.events.reset_scene_objects = EventTermCfg(
        func=mdp.reset_scene_to_default, mode="reset"
    )

    names = []
    for obj in scene["objects"]:
        obj_name = obj["name"]
        cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + obj_name,
            spawn=_spawn_cfg(obj),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(obj.get("pos", (0.25, 0.0, 0.1))),
                rot=tuple(obj.get("rot", (1.0, 0.0, 0.0, 0.0))),
            ),
        )
        setattr(env_cfg.scene, obj_name, cfg)
        names.append(obj_name)

        pos_range = obj.get("pos_range")
        if pos_range:
            setattr(
                env_cfg.events,
                f"randomize_{obj_name}",
                EventTermCfg(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": pos_range,
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg(obj_name),
                    },
                ),
            )

    for group_name, group_cfg in scene.get("groups", {}).items():
        members = [o["name"] for o in scene["objects"]
                   if o.get("group") == group_name]
        pos_range = group_cfg.get("pos_range")
        if members and pos_range:
            setattr(
                env_cfg.events,
                f"randomize_group_{group_name}",
                EventTermCfg(
                    func=randomize_group_offset,
                    mode="reset",
                    params={"asset_names": members, "pos_range": pos_range},
                ),
            )

    print(f"[INFO]: Loaded custom scene '{name}' with objects: {', '.join(names)}")
    return env_cfg

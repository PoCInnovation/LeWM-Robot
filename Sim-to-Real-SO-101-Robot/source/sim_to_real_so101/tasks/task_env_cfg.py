# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import (
    randomize_light_exposure,
    randomize_sky_light,
    randomize_mat_rotation,
    randomize_camera_focal_length,
    randomize_camera_pose,
    image,
    image_raw,
)

from .so101_env_cfg import (
    SO101TeleopEnvCfg,
    LerobotSo101BaseSceneCfg,
    EventCfg,
    ObservationsCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

camera_object = TiledCameraCfg(
    prim_path="",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,  # x10 of real
        focal_length=13.5,  # 10th of real
        focus_distance=0.05,  # 5cm in front of the camera
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
        convention="opengl",
    ),
)


def _look_at_quat_opengl(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) orienting an OpenGL camera at ``eye`` to look at ``target``.

    OpenGL convention: the camera looks down its local -Z axis, +Y is up, +X is right.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.array(
        [
            [r[0], u[0], -f[0]],
            [r[1], u[1], -f[1]],
            [r[2], u[2], -f[2]],
        ]
    )
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return tuple(q / np.linalg.norm(q))


EXTERNAL_CAM_EYE = (0.22, -0.50, 0.18)
EXTERNAL_CAM_TARGET = (0.22, 0.0, 0.07)
EXTERNAL_CAM_FOCUS = float(
    np.linalg.norm(np.array(EXTERNAL_CAM_TARGET) - np.array(EXTERNAL_CAM_EYE))
)


@configclass
class SO101TaskSceneCfg(LerobotSo101BaseSceneCfg):
    floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, 0.0, 0.0257 - 0.05)),
    )
    wall_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallBack",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 6.0, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.88)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.35, 0.0, 1.5)),
    )
    wall_side = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallSide",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 0.1, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.88)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.45, 1.5)),
    )
    room_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoomLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )

    # sky_light = AssetBaseCfg(
    #     prim_path="/World/sky_light",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=1000.0,
    #         texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
    #         visible_in_primary_ray=False,
    #         enable_color_temperature=True,
    #         color_temperature=6500.0
    #     )
    # )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/mat.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0, 0.032),
            rot=euler_angles_to_quat(np.array([0, 0, 90]), degrees=True),
        ),
    )

    # Camera
    camera_ego = camera_object.replace()
    camera_ego.prim_path = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_ego.offset.pos = (-0.005, 0.06, -0.062)
    camera_ego.offset.rot = euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True)

    camera_external_D455 = camera_object.replace()
    camera_external_D455.prim_path = "{ENV_REGEX_NS}/external_cam"
    camera_external_D455.spawn = sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,
        focal_length=13.5,
        focus_distance=EXTERNAL_CAM_FOCUS,
    )
    camera_external_D455.offset = TiledCameraCfg.OffsetCfg(
        pos=EXTERNAL_CAM_EYE,
        rot=_look_at_quat_opengl(EXTERNAL_CAM_EYE, EXTERNAL_CAM_TARGET),
        convention="opengl",
    )


@configclass
class TaskEventCfg(EventCfg):
    """Configuration for events."""

    reset_lightbox_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("room_light"),
        },
    )

    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={
            "yaw_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("mat"),
        },
    )

    reset_camera_ego_fov = EventTerm(
        func=randomize_camera_focal_length,
        mode="reset",
        params={
            "focal_length_range": (12.0, 15.0),  # ~±10% around 13.5mm
            "asset_cfg": SceneEntityCfg("camera_ego"),
        },
    )

    reset_camera_external_pose = EventTerm(
        func=randomize_camera_pose,
        mode="reset",
        params={
            "prim_path_pattern": "{ENV_REGEX_NS}/external_cam",
            "pos_range": {
                "x": (-0.02, 0.02),  # ±2cm
                "y": (-0.02, 0.02),
                "z": (-0.01, 0.01),
            },
            "rot_range": {
                "roll": (-0.05, 0.05),  # ±3°
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )


@configclass
class TaskObservationsCfg(ObservationsCfg):

    @configclass
    class VisualCfg(ObsGroup):

        rgb_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        rgb_external_D455 = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_external_D455"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


@configclass
class SO101TaskEnvCfg(SO101TeleopEnvCfg):
    """Configuration for the task environment."""

    scene: SO101TaskSceneCfg = SO101TaskSceneCfg()
    events: TaskEventCfg = TaskEventCfg()
    observations: TaskObservationsCfg = TaskObservationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # self.sim.render.enable_translucency = True
        # carb_settings = {
        # "rtx.reflections.enabled": True,
        # "rtx.translucency.reflectAtAllBounce": True,
        # "rtx.translucency.sampleRoughness": True,
        # "rtx.translucency.reflectionThroughputThreshold": 0.05,
        # "rtx.translucency.maxRefractionBounces": 5,
        # "rtx.raytracing.fractionalCutoutOpacity": True,
        # }
        # self.sim.render.carb_settings = carb_settings

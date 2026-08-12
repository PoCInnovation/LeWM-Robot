"""A scene declares its own task, so adding one is a data change.

``--auto`` used to be wired to ``cube_to_box`` by hard-coded object names, which
meant every new scene needed an edit to the policy. The scene now carries a
``task`` section, and the offline validator resolves the names against the
objects it declares — a typo is caught in a second instead of minutes into an
Isaac Sim launch.
"""

import pytest

from fake_env import FakeEnv, FakeRigidObject

from sim_to_real_so101.utils import scene_validation as sv
from sim_to_real_so101.utils.scene_loader import load_task_spec
from sim_to_real_so101.utils.scripted_policy import ScriptedPickPlace

BASE_OBJECT = {
    "name": "Cube",
    "type": "cuboid",
    "size": (0.025, 0.025, 0.025),
    "mass": 0.02,
    "pos": (0.22, -0.09, 0.06),
}
TARGET_OBJECT = {
    "name": "BoxFloor",
    "type": "cuboid",
    "size": (0.10, 0.10, 0.006),
    "static": True,
    "pos": (0.22, 0.10, 0.038),
}


def scene(task):
    return {"objects": [dict(BASE_OBJECT), dict(TARGET_OBJECT)], "task": task}


def messages(spec):
    return " | ".join(issue.message for issue in sv.validate_scene(spec).issues)


def build_env(objects=None):
    return FakeEnv(
        objects=objects
        or {
            "Cube": FakeRigidObject((0.22, -0.09, 0.06)),
            "BoxFloor": FakeRigidObject((0.22, 0.10, 0.038)),
        }
    )


# --- the policy side ------------------------------------------------------ #

def test_defaults_reproduce_the_shipped_tuning():
    policy = ScriptedPickPlace(build_env())
    assert policy.task["pick"] == "Cube"
    assert policy.task["place"] == "BoxFloor"
    assert policy.task == ScriptedPickPlace.DEFAULT_TASK


def test_a_scene_can_rename_the_objects():
    """The point of the change: no code edit for a new scene."""
    env = build_env(
        {"Mug": FakeRigidObject((0.22, -0.09, 0.06)), "Tray": FakeRigidObject((0.22, 0.10, 0.038))}
    )
    policy = ScriptedPickPlace(env, task={"pick": "Mug", "place": "Tray"})

    assert policy._pick_object == "Mug"
    assert policy._place_object == "Tray"
    assert policy.step()  # resolves both objects without raising


def test_a_scene_can_retune_the_grasp_geometry():
    env = build_env()
    policy = ScriptedPickPlace(env, task={"finger_len": 0.09, "carry_offset": 0.2})

    assert policy._grasp_point.finger_len == pytest.approx(0.09)
    assert any(
        payload == ("pick_frozen", 0.2)
        for _, kind, payload in policy._plan
        if kind == "reach"
    ), "carry_offset did not reach the motion plan"


def test_unspecified_keys_keep_their_default():
    policy = ScriptedPickPlace(build_env(), task={"pick": "Cube"})
    assert policy.task["carry_offset"] == ScriptedPickPlace.DEFAULT_TASK["carry_offset"]


def test_a_misspelled_task_key_is_refused_loudly():
    """Silently ignoring it would mean collecting with the wrong geometry."""
    with pytest.raises(ValueError, match="unknown task key"):
        ScriptedPickPlace(build_env(), task={"carry_ofset": 0.2})


def test_success_tolerances_come_from_the_task():
    env = build_env()
    strict = ScriptedPickPlace(env, task={"success_xy_tol": 0.0001})
    lenient = ScriptedPickPlace(env, task={"success_xy_tol": 1.0})
    assert strict.is_success() is False
    assert lenient.is_success() is True


# --- the validator side --------------------------------------------------- #

def test_the_shipped_scene_declares_a_valid_task():
    reports = {r.scene: r for r in sv.validate_all()}
    assert reports["cube_to_box"].ok, reports["cube_to_box"].format()
    assert load_task_spec("cube_to_box")["pick"] == "Cube"


def test_a_task_naming_a_missing_object_is_rejected():
    assert "not an object of this scene" in messages(scene({"pick": "Duck"}))


def test_picking_a_static_object_is_rejected():
    """Scenery cannot be picked up; this would fail silently at runtime."""
    assert "cannot pick up scenery" in messages(scene({"pick": "BoxFloor"}))


def test_an_unknown_task_key_is_rejected():
    assert "unknown 'task' key" in messages(scene({"carry_ofset": 0.2}))


def test_a_malformed_place_at_is_rejected():
    assert "must be 3 numbers" in messages(scene({"place_at": (0.2, 0.1)}))


def test_a_non_numeric_offset_is_rejected():
    assert "must be a number" in messages(scene({"carry_offset": "high"}))


def test_a_task_that_is_not_a_dict_is_rejected():
    assert "must be a dict" in messages(scene(["Cube", "BoxFloor"]))


def test_a_scene_without_a_task_stays_valid():
    """Keyboard-only scenes do not need one."""
    assert sv.validate_scene({"objects": [dict(BASE_OBJECT)]}).ok


def test_the_validator_covers_every_key_the_policy_accepts():
    """A key the policy takes but the validator rejects would be maddening."""
    spec = scene({key: value for key, value in ScriptedPickPlace.DEFAULT_TASK.items()})
    spec["task"]["pick"] = "Cube"
    spec["task"]["place"] = "BoxFloor"
    assert sv.validate_scene(spec).ok, sv.validate_scene(spec).format()

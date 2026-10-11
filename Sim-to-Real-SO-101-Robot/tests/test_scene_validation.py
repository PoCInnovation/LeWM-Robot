"""Tests for the offline scene validator.

A validator nobody has seen fire is worthless, so every rule is exercised with a
scene that deliberately breaks it. The shipped scene is also checked, to make
sure the rules do not fire on known-good input.
"""

import copy

import pytest

from sim_to_real_so101.utils import scene_validation as sv


BASE_OBJECT = {
    "name": "Cube",
    "type": "cuboid",
    "size": (0.025, 0.025, 0.025),
    "color": (0.9, 0.15, 0.15),
    "mass": 0.02,
    "pos": (0.22, -0.09, 0.06),
}


def scene(*objects, **extra):
    spec = {"objects": [copy.deepcopy(obj) for obj in objects]}
    spec.update(extra)
    return spec


def messages(spec, level=None):
    report = sv.validate_scene(spec)
    issues = report.issues if level is None else [i for i in report.issues if i.level == level]
    return " | ".join(issue.message for issue in issues)


def test_the_shipped_scene_is_clean():
    """cube_to_box must stay valid — it is the reference for every new scene."""
    reports = sv.validate_all()
    assert reports, "no scenes found"
    for report in reports:
        assert report.ok, report.format()
        assert not report.warnings, report.format()


def test_a_minimal_valid_scene_passes():
    assert sv.validate_scene(scene(BASE_OBJECT)).ok


def test_empty_scene_is_rejected():
    assert not sv.validate_scene({"objects": []}).ok
    assert not sv.validate_scene({}).ok


def test_duplicate_names_are_rejected():
    """The loader setattr's by name, so a duplicate silently drops an object."""
    assert "duplicate" in messages(scene(BASE_OBJECT, BASE_OBJECT))


def test_invalid_usd_identifier_is_rejected():
    for bad in ["my cube", "cube-1", "1cube", "a/b"]:
        obj = {**BASE_OBJECT, "name": bad}
        assert not sv.validate_scene(scene(obj)).ok, bad


def test_unknown_type_is_rejected():
    obj = {**BASE_OBJECT, "type": "pyramid"}
    assert "unknown type" in messages(scene(obj))


def test_missing_geometry_field_is_rejected():
    sphere = {"name": "Ball", "type": "sphere", "pos": (0.22, 0.0, 0.06)}
    assert "requires 'radius'" in messages(scene(sphere))

    cylinder = {"name": "Can", "type": "cylinder", "radius": 0.02, "pos": (0.22, 0.0, 0.06)}
    assert "requires 'height'" in messages(scene(cylinder))


def test_non_positive_dimensions_are_rejected():
    assert "strictly positive" in messages(scene({**BASE_OBJECT, "size": (0.02, 0.0, 0.02)}))
    assert "strictly positive" in messages(
        scene({"name": "Ball", "type": "sphere", "radius": -0.01, "pos": (0.22, 0, 0.06)})
    )


def test_bad_mass_is_rejected():
    assert "strictly positive" in messages(scene({**BASE_OBJECT, "mass": 0.0}))
    assert "beyond what the SO-101 can move" in messages(scene({**BASE_OBJECT, "mass": 12.0}))


def test_out_of_gamut_colour_is_rejected():
    assert "in [0, 1]" in messages(scene({**BASE_OBJECT, "color": (255, 0, 0)}))


def test_non_unit_quaternion_is_rejected():
    assert "unit quaternion" in messages(scene({**BASE_OBJECT, "rot": (1.0, 1.0, 0.0, 0.0)}))
    assert sv.validate_scene(scene({**BASE_OBJECT, "rot": (1.0, 0.0, 0.0, 0.0)})).ok


def test_object_buried_in_the_floor_is_an_error():
    """Below the floor the solver ejects it violently on the first step."""
    assert "inside the floor" in messages(scene({**BASE_OBJECT, "pos": (0.22, -0.09, 0.0)}))


def test_object_below_the_mat_is_a_warning():
    assert "below the mat surface" in messages(
        scene({**BASE_OBJECT, "pos": (0.22, -0.09, 0.04)}), sv.WARNING
    )


def test_object_dropped_from_high_up_is_a_warning():
    assert "above the mat" in messages(scene({**BASE_OBJECT, "pos": (0.22, -0.09, 0.5)}), sv.WARNING)


def test_pos_range_that_leaves_the_workspace_is_a_warning():
    """Catches a reset that puts the target where the arm cannot go."""
    obj = {**BASE_OBJECT, "pos_range": {"x": (-0.20, 0.20)}}
    assert "outside the comfortable reach" in messages(scene(obj), sv.WARNING)


def test_static_scenery_is_exempt_from_the_reach_check():
    """Box walls legitimately sit at the edge of the workspace."""
    wall = {
        "name": "Wall", "type": "cuboid", "size": (0.006, 0.10, 0.05),
        "static": True, "pos": (0.40, 0.10, 0.066),
    }
    assert not sv.validate_scene(scene(wall)).warnings


def test_bad_pos_range_key_is_rejected():
    assert "not one of" in messages(scene({**BASE_OBJECT, "pos_range": {"yaww": (0, 1)}}))


def test_inverted_pos_range_is_rejected():
    assert "min > max" in messages(scene({**BASE_OBJECT, "pos_range": {"x": (0.1, -0.1)}}))


def test_typo_in_an_object_key_is_flagged():
    """Silently ignored by the loader, so the setting would be lost."""
    assert "unrecognised key" in messages(scene({**BASE_OBJECT, "mas": 0.02}), sv.WARNING)


def test_undeclared_group_is_an_error():
    obj = {**BASE_OBJECT, "group": "box"}
    assert "not declared in 'groups'" in messages(scene(obj))


def test_declared_group_without_pos_range_is_a_warning():
    obj = {**BASE_OBJECT, "group": "box"}
    assert "does nothing" in messages(scene(obj, groups={"box": {}}), sv.WARNING)


def test_group_with_no_members_is_a_warning():
    assert "no members" in messages(
        scene(BASE_OBJECT, groups={"box": {"pos_range": {"x": (-0.05, 0.05)}}}), sv.WARNING
    )


def test_overlapping_objects_are_flagged():
    other = {**BASE_OBJECT, "name": "Cube2"}
    assert "overlap at spawn" in messages(scene(BASE_OBJECT, other), sv.WARNING)


def test_objects_in_the_same_group_may_touch():
    """A box built from several walls is one object; its parts share faces."""
    wall_a = {
        "name": "WallA", "type": "cuboid", "size": (0.10, 0.006, 0.05),
        "static": True, "group": "box", "pos": (0.22, 0.147, 0.066),
    }
    wall_b = {**wall_a, "name": "WallB", "pos": (0.22, 0.150, 0.066)}
    report = sv.validate_scene(
        scene(wall_a, wall_b, groups={"box": {"pos_range": {"x": (-0.05, 0.05)}}})
    )
    assert not any("overlap" in issue.message for issue in report.issues)


def test_missing_usd_asset_is_an_error():
    obj = {"name": "Prop", "type": "usd", "usd_path": "nope.usd", "pos": (0.22, 0, 0.06)}
    assert "usd_path not found" in messages(scene(obj))


def test_report_formatting_mentions_counts():
    report = sv.validate_scene(scene({**BASE_OBJECT, "size": (0.0, 0.0, 0.0)}))
    assert "error" in report.format()
    assert not report.ok


@pytest.mark.parametrize("bad", [None, [], "objects", 42])
def test_malformed_specs_do_not_crash(bad):
    """The validator must report, never raise — it runs before everything else."""
    report = sv.validate_scene(bad)
    assert not report.ok

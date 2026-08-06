"""Offline validation of scene specs, with no Isaac Sim involved.

A malformed scene only shows up minutes into an Isaac Sim launch, on the machine
that has the GPU. Since scenes are authored on a machine that has none, they are
checked here instead: this module imports nothing from isaaclab, so it runs
anywhere in under a second.

What it catches: structural mistakes (missing or misspelled fields, duplicate
names, unknown groups), objects buried in the table or floating above it,
objects that reset outside the arm's reach, and initial interpenetration.

Errors are things that will break or silently corrupt the run. Warnings are
things that are probably a mistake but may be deliberate.

Use it from the command line::

    validate_scenes                 # every scene in scenes/
    validate_scenes cube_to_box     # just one
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass, field

SCENES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenes"
)

# Repo spatial conventions, from USAGE.md and the Teleop-Task floor/mat configs.
SUPPORT_Z = 0.035          # top surface of the mat, where objects come to rest
FLOOR_Z = 0.0257           # top surface of the room floor
REACH_X = (0.15, 0.30)     # comfortable workspace of the arm
REACH_Y = (-0.15, 0.15)
SPAWN_CLEARANCE_MAX = 0.15  # dropping an object from higher than this is suspect

KNOWN_TYPES = {"cuboid", "sphere", "cylinder", "usd"}
POSE_RANGE_KEYS = {"x", "y", "z", "roll", "pitch", "yaw"}
KNOWN_OBJECT_KEYS = {
    "name", "type", "size", "radius", "height", "usd_path", "color", "mass",
    "static", "pos", "rot", "pos_range", "group",
}

ERROR = "ERROR"
WARNING = "WARNING"


@dataclass
class Issue:
    level: str
    scope: str
    message: str

    def __str__(self):
        return f"[{self.level}] {self.scope}: {self.message}"


@dataclass
class ValidationReport:
    scene: str
    issues: list = field(default_factory=list)

    def add(self, level, scope, message):
        self.issues.append(Issue(level, scope, message))

    @property
    def errors(self):
        return [issue for issue in self.issues if issue.level == ERROR]

    @property
    def warnings(self):
        return [issue for issue in self.issues if issue.level == WARNING]

    @property
    def ok(self):
        return not self.errors

    def format(self):
        if not self.issues:
            return f"{self.scene}: OK"
        lines = [f"{self.scene}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        lines.extend(f"  {issue}" for issue in self.issues)
        return "\n".join(lines)


def load_scene_spec_raw(path):
    """Import a scene file and return its ``SCENE`` dict, without isaaclab."""
    spec = importlib.util.spec_from_file_location(
        f"so101_scene_check_{os.path.basename(path)[:-3]}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "SCENE"):
        raise ValueError(f"{path} does not define a SCENE dict")
    return module.SCENE


def list_scene_files(scenes_dir=SCENES_DIR):
    if not os.path.isdir(scenes_dir):
        return []
    return sorted(
        os.path.join(scenes_dir, name)
        for name in os.listdir(scenes_dir)
        if name.endswith(".py") and not name.startswith("_")
    )


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _half_extents(obj):
    """Half-size of the object's axis-aligned box, or None if unknown."""
    kind = obj.get("type", "cuboid")
    if kind == "cuboid":
        size = obj.get("size")
        if not size or len(size) != 3:
            return None
        return tuple(float(component) / 2.0 for component in size)
    if kind == "sphere":
        radius = obj.get("radius")
        return None if radius is None else (float(radius),) * 3
    if kind == "cylinder":
        radius, height = obj.get("radius"), obj.get("height")
        if radius is None or height is None:
            return None
        return (float(radius), float(radius), float(height) / 2.0)
    return None  # usd: extents are inside the asset


def _check_geometry_fields(obj, name, report):
    """Every primitive type needs its own dimension fields; verify they are sane."""
    kind = obj.get("type", "cuboid")
    if kind not in KNOWN_TYPES:
        report.add(ERROR, name, f"unknown type '{kind}' (expected one of {sorted(KNOWN_TYPES)})")
        return

    required = {
        "cuboid": ["size"],
        "sphere": ["radius"],
        "cylinder": ["radius", "height"],
        "usd": ["usd_path"],
    }[kind]
    for key in required:
        if key not in obj:
            report.add(ERROR, name, f"type '{kind}' requires '{key}'")

    if kind == "cuboid" and "size" in obj:
        size = obj["size"]
        if len(size) != 3 or not all(_is_number(v) for v in size):
            report.add(ERROR, name, "'size' must be 3 numbers (x, y, z)")
        elif any(float(v) <= 0 for v in size):
            report.add(ERROR, name, f"'size' must be strictly positive, got {tuple(size)}")

    for key in ("radius", "height"):
        if key in obj:
            if not _is_number(obj[key]):
                report.add(ERROR, name, f"'{key}' must be a number")
            elif float(obj[key]) <= 0:
                report.add(ERROR, name, f"'{key}' must be strictly positive")


def _check_common_fields(obj, name, report):
    unknown = set(obj) - KNOWN_OBJECT_KEYS
    if unknown:
        # Silently ignored by the loader, so a typo here loses the setting.
        report.add(WARNING, name, f"unrecognised key(s) {sorted(unknown)} — ignored by scene_loader")

    color = obj.get("color")
    if color is not None:
        if len(color) != 3 or not all(_is_number(v) for v in color):
            report.add(ERROR, name, "'color' must be 3 numbers")
        elif not all(0.0 <= float(v) <= 1.0 for v in color):
            report.add(ERROR, name, f"'color' components must be in [0, 1], got {tuple(color)}")

    mass = obj.get("mass")
    if mass is not None:
        if not _is_number(mass):
            report.add(ERROR, name, "'mass' must be a number")
        elif float(mass) <= 0:
            report.add(ERROR, name, f"'mass' must be strictly positive, got {mass}")
        elif float(mass) > 5.0 and not obj.get("static", False):
            report.add(WARNING, name, f"mass {mass} kg is far beyond what the SO-101 can move")

    pos = obj.get("pos")
    if pos is not None and (len(pos) != 3 or not all(_is_number(v) for v in pos)):
        report.add(ERROR, name, "'pos' must be 3 numbers")

    rot = obj.get("rot")
    if rot is not None:
        if len(rot) != 4 or not all(_is_number(v) for v in rot):
            report.add(ERROR, name, "'rot' must be a quaternion of 4 numbers (w, x, y, z)")
        else:
            norm = sum(float(v) ** 2 for v in rot) ** 0.5
            if abs(norm - 1.0) > 1e-3:
                report.add(ERROR, name, f"'rot' is not a unit quaternion (norm {norm:.4f})")

    pos_range = obj.get("pos_range")
    if pos_range is not None:
        if not isinstance(pos_range, dict):
            report.add(ERROR, name, "'pos_range' must be a dict")
        else:
            for key, bounds in pos_range.items():
                if key not in POSE_RANGE_KEYS:
                    report.add(
                        ERROR, name,
                        f"'pos_range' key '{key}' is not one of {sorted(POSE_RANGE_KEYS)}",
                    )
                elif len(bounds) != 2 or not all(_is_number(v) for v in bounds):
                    report.add(ERROR, name, f"'pos_range[{key}]' must be (min, max)")
                elif float(bounds[0]) > float(bounds[1]):
                    report.add(ERROR, name, f"'pos_range[{key}]' has min > max: {tuple(bounds)}")


def _check_placement(obj, name, report):
    """Objects must rest on the mat, not inside it and not far above it."""
    pos = obj.get("pos")
    half = _half_extents(obj)
    if pos is None or half is None:
        return

    bottom = float(pos[2]) - half[2]
    if bottom < FLOOR_Z - 1e-6:
        report.add(
            ERROR, name,
            f"spawns inside the floor (bottom z = {bottom:.4f} < {FLOOR_Z})",
        )
    elif bottom < SUPPORT_Z - 1e-6:
        report.add(
            WARNING, name,
            f"spawns below the mat surface (bottom z = {bottom:.4f} < {SUPPORT_Z}) — "
            "it will be pushed out by the solver",
        )
    elif bottom > SUPPORT_Z + SPAWN_CLEARANCE_MAX:
        report.add(
            WARNING, name,
            f"spawns {bottom - SUPPORT_Z:.3f} m above the mat — it will fall a long way",
        )


def _check_reachability(obj, name, report):
    """Manipulable objects must stay reachable across their whole reset range."""
    if obj.get("static", False):
        return  # scenery does not need to be within reach
    pos = obj.get("pos")
    if pos is None:
        return

    pos_range = obj.get("pos_range") or {}
    for axis, index, bounds in (("x", 0, REACH_X), ("y", 1, REACH_Y)):
        low, high = pos_range.get(axis, (0.0, 0.0))
        extreme_low = float(pos[index]) + float(low)
        extreme_high = float(pos[index]) + float(high)
        if extreme_low < bounds[0] or extreme_high > bounds[1]:
            report.add(
                WARNING, name,
                f"resets to {axis} in [{extreme_low:.3f}, {extreme_high:.3f}], "
                f"outside the comfortable reach {bounds}",
            )


def _check_overlaps(objects, report):
    """Interpenetration at spawn makes the solver eject objects on the first step."""
    boxes = []
    for obj in objects:
        pos, half = obj.get("pos"), _half_extents(obj)
        if pos is None or half is None:
            continue
        boxes.append((obj["name"], obj.get("group"), pos, half))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            name_a, group_a, pos_a, half_a = boxes[i]
            name_b, group_b, pos_b, half_b = boxes[j]
            # Parts of the same multi-piece object are allowed to touch.
            if group_a is not None and group_a == group_b:
                continue
            overlap = all(
                abs(float(pos_a[axis]) - float(pos_b[axis])) < half_a[axis] + half_b[axis] - 1e-6
                for axis in range(3)
            )
            if overlap:
                report.add(
                    WARNING, f"{name_a}/{name_b}",
                    "bounding boxes overlap at spawn — the solver will push them apart",
                )


def validate_scene(spec, scene_name="<scene>", scenes_dir=SCENES_DIR):
    """Check one ``SCENE`` dict and return a :class:`ValidationReport`."""
    report = ValidationReport(scene_name)

    if not isinstance(spec, dict):
        report.add(ERROR, scene_name, "SCENE must be a dict")
        return report

    objects = spec.get("objects")
    if not objects:
        report.add(ERROR, scene_name, "SCENE has no 'objects' entry, or it is empty")
        return report
    if not isinstance(objects, (list, tuple)):
        report.add(ERROR, scene_name, "'objects' must be a list")
        return report

    unknown_top = set(spec) - {"objects", "groups"}
    if unknown_top:
        report.add(WARNING, scene_name, f"unrecognised top-level key(s) {sorted(unknown_top)}")

    seen = set()
    valid_objects = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            report.add(ERROR, f"objects[{index}]", "each object must be a dict")
            continue
        name = obj.get("name")
        if not name:
            report.add(ERROR, f"objects[{index}]", "missing 'name'")
            continue
        if not isinstance(name, str) or not name.replace("_", "").isalnum() or name[0].isdigit():
            report.add(
                ERROR, str(name),
                "'name' must be a valid USD identifier (letters, digits, underscore; "
                "not starting with a digit)",
            )
            continue
        if name in seen:
            report.add(ERROR, name, "duplicate object name — the second one overwrites the first")
            continue
        seen.add(name)

        _check_geometry_fields(obj, name, report)
        _check_common_fields(obj, name, report)
        _check_placement(obj, name, report)
        _check_reachability(obj, name, report)

        if obj.get("type") == "usd":
            usd_path = obj.get("usd_path")
            if usd_path:
                resolved = usd_path if os.path.isabs(usd_path) else os.path.join(scenes_dir, usd_path)
                if not os.path.isfile(resolved):
                    report.add(ERROR, name, f"usd_path not found: {resolved}")

        valid_objects.append(obj)

    _check_overlaps(valid_objects, report)

    groups = spec.get("groups") or {}
    if not isinstance(groups, dict):
        report.add(ERROR, scene_name, "'groups' must be a dict")
    else:
        declared = set(groups)
        used = {obj.get("group") for obj in valid_objects if obj.get("group")}
        for missing in sorted(used - declared):
            report.add(
                ERROR, scene_name,
                f"objects reference group '{missing}' which is not declared in 'groups' — "
                "they will never be jittered together",
            )
        for unused in sorted(declared - used):
            report.add(WARNING, scene_name, f"group '{unused}' has no members")
        for group_name, group_cfg in groups.items():
            pos_range = (group_cfg or {}).get("pos_range")
            if pos_range is None:
                report.add(WARNING, scene_name, f"group '{group_name}' has no 'pos_range' — it does nothing")
            elif not isinstance(pos_range, dict):
                report.add(ERROR, scene_name, f"group '{group_name}': 'pos_range' must be a dict")
            else:
                for key in pos_range:
                    if key not in POSE_RANGE_KEYS:
                        report.add(
                            ERROR, scene_name,
                            f"group '{group_name}': pos_range key '{key}' is not one of "
                            f"{sorted(POSE_RANGE_KEYS)}",
                        )

    return report


def validate_scene_file(path, scenes_dir=None):
    name = os.path.basename(path)[:-3]
    scenes_dir = scenes_dir or os.path.dirname(path)
    try:
        spec = load_scene_spec_raw(path)
    except Exception as exc:
        report = ValidationReport(name)
        report.add(ERROR, name, f"could not import the scene: {type(exc).__name__}: {exc}")
        return report
    return validate_scene(spec, scene_name=name, scenes_dir=scenes_dir)


def validate_all(scenes_dir=SCENES_DIR):
    return [validate_scene_file(path, scenes_dir) for path in list_scene_files(scenes_dir)]


def main():
    parser = argparse.ArgumentParser(
        description="Validate scene specs without launching Isaac Sim."
    )
    parser.add_argument("scenes", nargs="*", help="Scene names to check (default: all).")
    parser.add_argument("--scenes_dir", default=SCENES_DIR)
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as failures."
    )
    args = parser.parse_args()

    if args.scenes:
        paths = [os.path.join(args.scenes_dir, f"{name}.py") for name in args.scenes]
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            print(f"[ERROR]: no such scene file: {', '.join(missing)}")
            return 2
        reports = [validate_scene_file(path, args.scenes_dir) for path in paths]
    else:
        reports = validate_all(args.scenes_dir)

    if not reports:
        print(f"[WARNING]: no scenes found in {args.scenes_dir}")
        return 0

    failed = 0
    for report in reports:
        print(report.format())
        if not report.ok or (args.strict and report.warnings):
            failed += 1

    print(f"\n{len(reports) - failed}/{len(reports)} scene(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

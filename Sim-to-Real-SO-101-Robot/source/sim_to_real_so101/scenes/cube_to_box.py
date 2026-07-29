BOX_COLOR = (0.55, 0.35, 0.15)

SCENE = {
    "objects": [
        {
            "name": "Cube",
            "type": "cuboid",
            "size": (0.025, 0.025, 0.025),
            "color": (0.9, 0.15, 0.15),
            "mass": 0.02,
            "pos": (0.22, -0.09, 0.06),
            "pos_range": {"x": (-0.06, 0.06), "y": (-0.05, 0.05)},
        },
        {
            "name": "BoxFloor",
            "type": "cuboid",
            "size": (0.10, 0.10, 0.006),
            "color": BOX_COLOR,
            "static": True,
            "group": "box",
            "pos": (0.22, 0.10, 0.038),
        },
        {
            "name": "BoxWallN",
            "type": "cuboid",
            "size": (0.10, 0.006, 0.05),
            "color": BOX_COLOR,
            "static": True,
            "group": "box",
            "pos": (0.22, 0.147, 0.066),
        },
        {
            "name": "BoxWallS",
            "type": "cuboid",
            "size": (0.10, 0.006, 0.05),
            "color": BOX_COLOR,
            "static": True,
            "group": "box",
            "pos": (0.22, 0.053, 0.066),
        },
        {
            "name": "BoxWallE",
            "type": "cuboid",
            "size": (0.006, 0.10, 0.05),
            "color": BOX_COLOR,
            "static": True,
            "group": "box",
            "pos": (0.267, 0.10, 0.066),
        },
        {
            "name": "BoxWallW",
            "type": "cuboid",
            "size": (0.006, 0.10, 0.05),
            "color": BOX_COLOR,
            "static": True,
            "group": "box",
            "pos": (0.173, 0.10, 0.066),
        },
    ],
    "groups": {
        "box": {"pos_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.04)}},
    },
}

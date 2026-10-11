import carb
import omni.appwindow

JOINT_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

DEFAULT_POSE = {
    "Rotation": -0.2736,
    "Pitch": -0.6109,
    "Elbow": -0.0745,
    "Wrist_Pitch": 1.5148,
    "Wrist_Roll": -1.6034,
    "Jaw": -0.1465,
}

STEP = 0.015

BINDINGS = [
    ("Rotation",    "LEFT", "RIGHT", -1.2, 1.2),
    ("Pitch",       "DOWN", "UP",    -1.2, 1.2),
    ("Elbow",       "D",    "E",     -1.2, 1.2),
    ("Wrist_Pitch", "G",    "T",     -1.2, 1.2),
    ("Wrist_Roll",  "F",    "H",     -1.5, 1.5),
    ("Jaw",         "N",    "B",     -1.0, 1.5),
]


class KeyboardArmControl:
    def __init__(self):
        self.targets = dict(DEFAULT_POSE)
        self._held = set()

        self._window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

        print("[INFO]: Keyboard arm control active. Bindings:")
        for joint, dec, inc, _, _ in BINDINGS:
            print(f"        {joint:12s}: {dec} (-)  /  {inc} (+)")

    def _on_keyboard_event(self, event, *args, **kwargs):
        inp = getattr(event, "input", None)
        name = getattr(inp, "name", inp)
        if not isinstance(name, str):
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._held.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
        return True

    def step(self):
        """Call once per simulation frame. Returns the 6 joint targets in JOINT_ORDER."""
        for joint, dec_key, inc_key, lo, hi in BINDINGS:
            default = DEFAULT_POSE[joint]
            if dec_key in self._held:
                self.targets[joint] -= STEP
            if inc_key in self._held:
                self.targets[joint] += STEP
            self.targets[joint] = max(default + lo, min(default + hi, self.targets[joint]))
        return [self.targets[j] for j in JOINT_ORDER]

    def reset(self):
        self.targets = dict(DEFAULT_POSE)
        self._held.clear()

    def cleanup(self):
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
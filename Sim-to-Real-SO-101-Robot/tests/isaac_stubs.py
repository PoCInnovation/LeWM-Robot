"""Fake ``isaaclab`` / ``omni`` / ``carb`` / ``lerobot`` modules for CPU testing.

The simulation code cannot be imported outside Isaac Sim: it pulls in isaaclab,
omni and carb at module level. Since development happens on a machine with no
NVIDIA GPU, these stubs stand in for those packages so the *logic* — scene
injection, IK, state machines, unit conversions — can run under pytest.

What is faked and what is not:

* ``torch`` is **real**. Every numeric result the tests assert on is computed by
  the same library the simulator uses.
* Isaac config classes are replaced by :class:`StubCfg`, which simply records
  its keyword arguments as attributes. That is enough to assert on what the code
  *built* (sizes, masses, positions, event terms) without a simulator.
* Anything the tests do not look at resolves to a placeholder created on demand.

Attribute naming drives the behaviour: an attribute starting with an uppercase
letter becomes a class, a lowercase one becomes a function. That matches the
Isaac convention (``CuboidCfg`` vs ``reset_scene_to_default``) and means new
imports usually work without touching this file.
"""

import sys
import types


class StubNamespace:
    """Plain attribute bag whose sub-namespaces appear on first access."""

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        child = StubNamespace()
        setattr(self, item, child)
        return child

    def __repr__(self):
        return f"StubNamespace({self.__dict__!r})"


class StubCfgMeta(type):
    """Resolves nested config classes on demand.

    Isaac nests declarations inside their owner — ``RigidObjectCfg.InitialStateCfg``,
    ``TiledCameraCfg.OffsetCfg``, ``FrameTransformerCfg.FrameCfg``. Creating them
    lazily means new ones never need to be declared here.
    """

    def __getattr__(cls, item):
        if item.startswith("_"):
            raise AttributeError(item)
        nested = type(item, (StubCfg,), {"_stub_name": f"{cls.__name__}.{item}"})
        setattr(cls, item, nested)
        return nested


class StubCfg(metaclass=StubCfgMeta):
    """Config object that remembers how it was built.

    Isaac's ``configclass`` types are plain data holders, so recording the
    keyword arguments as attributes reproduces everything the code under test
    relies on.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, item):
        """Config sections appear on first use.

        Isaac's base configs ship sub-objects (``viewer``, ``sim``, ``sim.render``)
        that ``__post_init__`` assigns into. Creating them on demand lets the real
        ``__post_init__`` run unchanged, so its values can be asserted on.
        """
        if item.startswith("_"):
            raise AttributeError(item)
        child = StubNamespace()
        object.__setattr__(self, item, child)
        return child

    def replace(self, **kwargs):
        merged = {**self._kwargs, **kwargs}
        clone = type(self)(*self._args, **merged)
        # Attributes assigned after construction must survive a replace(), the
        # way the task configs mutate cfg objects in place.
        for key, value in self.__dict__.items():
            if key not in ("_args", "_kwargs") and key not in kwargs:
                setattr(clone, key, value)
        return clone

    def copy(self):
        return self.replace()

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"{type(self).__name__}({inner})"


class StubObject:
    """Runtime placeholder whose attributes appear on demand.

    Used for the handful of live Isaac objects the controllers touch — the carb
    input interface, the app window, the keyboard handle. Attributes are cached
    so identity comparisons (``event.type == KeyboardEventType.KEY_PRESS``)
    behave consistently.
    """

    def __init__(self, name="stub"):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        cache = object.__getattribute__(self, "_cache")
        if item not in cache:
            name = f"{object.__getattribute__(self, '_name')}.{item}"
            cache[item] = StubObject(name) if item[:1].isupper() else _stub_callable(name)
        return cache[item]

    def __call__(self, *args, **kwargs):
        return StubObject(f"{object.__getattribute__(self, '_name')}()")

    def __repr__(self):
        return f"<StubObject {object.__getattribute__(self, '_name')}>"


def _stub_callable(name):
    """A callable placeholder that also carries attributes on demand."""
    holder = StubObject(name)

    def _call(*args, **kwargs):
        return holder

    _call.__name__ = name.rsplit(".", 1)[-1]
    _call._stub_name = name
    # Let ``mdp.reset_root_state_uniform`` be both callable and comparable.
    _call.__getattr__ = holder.__getattr__
    return _call


class StubModule(types.ModuleType):
    """Module whose members are created the first time they are imported."""

    def __init__(self, name):
        super().__init__(name)
        self.__dict__["_created"] = {}

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        created = self.__dict__["_created"]
        if item not in created:
            full = f"{self.__name__}.{item}"
            if item[:1].isupper():
                # Distinct subclass per name, so tests can assert on the type.
                created[item] = type(item, (StubCfg,), {"_stub_name": full})
            else:
                created[item] = _stub_callable(full)
        return created[item]


# Modules the simulation code imports at module level. Registering the dotted
# names explicitly keeps ``import a.b.c as x`` working.
STUBBED_MODULES = [
    "carb",
    "carb.eventdispatcher",
    "carb.input",
    "gymnasium",
    "isaaclab",
    "isaaclab.actuators",
    "isaaclab.app",
    "isaaclab.assets",
    "isaaclab.assets.articulation",
    "isaaclab.envs",
    "isaaclab.envs.mdp",
    "isaaclab.managers",
    "isaaclab.scene",
    "isaaclab.sensors",
    "isaaclab.sim",
    "isaaclab.utils",
    "isaaclab.utils.math",
    "isaaclab_tasks",
    "isaaclab_tasks.utils",
    "isaacsim",
    "isaacsim.core",
    "isaacsim.core.prims",
    "isaacsim.core.utils",
    "isaacsim.core.utils.rotations",
    "omni",
    "omni.appwindow",
    "omni.kit",
    "omni.kit.app",
    "pxr",
    # Imported by lerobot_recorder but unused there; stubbed rather than
    # installed, to keep the test venv to torch + numpy + pytest.
    "tqdm",
    # lerobot is imported wholesale by lerobot_interface; none of its behaviour
    # is under test here, only the unit conversions that sit alongside it.
    "lerobot",
    "lerobot.cameras",
    "lerobot.cameras.opencv",
    "lerobot.configs",
    "lerobot.configs.policies",
    "lerobot.datasets",
    "lerobot.datasets.lerobot_dataset",
    "lerobot.datasets.pipeline_features",
    "lerobot.datasets.utils",
    "lerobot.policies",
    "lerobot.policies.factory",
    "lerobot.policies.utils",
    "lerobot.processor",
    "lerobot.robots",
    "lerobot.robots.so101_follower",
    "lerobot.teleoperators",
    "lerobot.teleoperators.so101_leader",
    "lerobot.utils",
    "lerobot.utils.constants",
    "lerobot.utils.control_utils",
    "lerobot.utils.utils",
    "lerobot.utils.visualization_utils",
]


def install():
    """Register the stub modules in ``sys.modules``. Idempotent."""
    for name in STUBBED_MODULES:
        if name in sys.modules and isinstance(sys.modules[name], StubModule):
            continue
        module = StubModule(name)
        sys.modules[name] = module
        # Attach to the parent package so ``import a.b`` then ``a.b`` resolves.
        if "." in name:
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None:
                parent.__dict__["_created"][child] = module
                setattr(parent, child, module)

    _patch_semantics()


def _patch_semantics():
    """Give the few members whose real behaviour matters something faithful."""
    # OBS_STR is concatenated into strings by lerobot_interface.
    sys.modules["lerobot.utils.constants"].OBS_STR = "observation"

    # ``@configclass`` decorates classes; the default stub would replace the
    # class with a placeholder. Identity keeps the declaration usable.
    utils = sys.modules["isaaclab.utils"]
    utils.__dict__["_created"]["configclass"] = lambda cls: cls
    utils.configclass = lambda cls: cls

    # ``sim_to_real_so101.mdp`` re-exports isaaclab's mdp with a star import, and
    # a star import can only see names that already exist — lazy creation is not
    # enough. These are the ones the task configs pull through that way.
    isaac_mdp = sys.modules["isaaclab.envs.mdp"]
    for name in [
        "JointPositionActionCfg",
        "joint_pos",
        "joint_pos_rel",
        "reset_joints_by_offset",
        "reset_scene_to_default",
        "reset_root_state_uniform",
        "image",
        "time_out",
    ]:
        setattr(isaac_mdp, name, getattr(isaac_mdp, name))

    # Keyboard event kinds must be distinct, stable, comparable values.
    carb_input = sys.modules["carb.input"]
    event_type = types.SimpleNamespace(KEY_PRESS="KEY_PRESS", KEY_RELEASE="KEY_RELEASE")
    carb_input.__dict__["_created"]["KeyboardEventType"] = event_type
    carb_input.KeyboardEventType = event_type

    # The controllers subscribe to keyboard events on construction; the
    # subscription handle only needs to exist.
    def _acquire_input_interface():
        return _KeyboardInterface()

    carb_input.__dict__["_created"]["acquire_input_interface"] = _acquire_input_interface
    carb_input.acquire_input_interface = _acquire_input_interface

    carb = sys.modules["carb"]
    carb.__dict__["_created"]["input"] = carb_input
    carb.input = carb_input

    appwindow = sys.modules["omni.appwindow"]

    def _get_default_app_window():
        return _AppWindow()

    appwindow.__dict__["_created"]["get_default_app_window"] = _get_default_app_window
    appwindow.get_default_app_window = _get_default_app_window


class _KeyboardInterface:
    """Records subscriptions so tests can drive key events by hand."""

    def __init__(self):
        self.callbacks = []

    def subscribe_to_keyboard_events(self, keyboard, callback):
        self.callbacks.append(callback)
        return len(self.callbacks)

    def unsubscribe_to_keyboard_events(self, keyboard, handle):
        return None


class _AppWindow:
    def get_keyboard(self):
        return "keyboard"

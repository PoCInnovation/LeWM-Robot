"""Tests for the asynchronous dataset writer.

This is the component where a bug is most expensive: it runs on a background
thread, it wraps everything in a bare ``except``, and a failure there means an
episode that was recorded never reaches the dataset. On a 5000-episode run that
shows up as "the collection came up short" with no explanation.

The regression these tests pin down: ``--save_mp4`` without ``--depth`` used to
raise ``KeyError`` *before* the dataset commit, so every episode was silently
discarded.
"""

import types

import pytest
import torch

import sim_to_real_so101.utils.lerobot_recorder as recorder_module
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder

CAMERAS = {"wrist": {"height": 4, "width": 4}}


class FakeDataset:
    """Records what the writer commits, without touching the disk."""

    instances = []

    def __init__(self, repo_id=None, root=None, **kwargs):
        self.repo_id = repo_id
        self.root = root
        self.frames = []
        self.saved_episodes = 0
        self.finalized = 0
        self.meta = types.SimpleNamespace(total_episodes=0)
        FakeDataset.instances.append(self)

    @classmethod
    def create(cls, repo_id, fps=None, features=None, root=None, robot_type=None):
        return cls(repo_id=repo_id, root=root)

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1
        self.meta.total_episodes += 1

    def finalize(self):
        self.finalized += 1


@pytest.fixture
def make_recorder(tmp_path, monkeypatch):
    """Build a recorder wired to :class:`FakeDataset`, with ffmpeg stubbed out."""
    FakeDataset.instances.clear()
    monkeypatch.setattr(recorder_module, "LeRobotDataset", FakeDataset)

    created = []

    def _make(**kwargs):
        options = dict(
            task_name="test task",
            repo_id="user/test",
            dataset_root=str(tmp_path / "ds"),
            fps=30,
            device="cpu",
            cameras=CAMERAS,
        )
        options.update(kwargs)
        recorder = LeRobotRecorder(**options)
        recorder.init_dataset()
        recorder.video_calls = []
        monkeypatch.setattr(
            recorder,
            "_save_video",
            lambda frames, camera, kind, index: recorder.video_calls.append((camera, kind, index)),
        )
        created.append(recorder)
        return recorder

    yield _make

    for recorder in created:
        recorder.episode_processor_stop_event.set()


def push_episode(recorder, n_frames=5):
    """Fill the buffers the way ``lerobot_agent`` does, then queue the episode."""
    for _ in range(n_frames):
        recorder.push_frame_to_buffer(
            torch.zeros(6),
            torch.zeros(6),
            {name: torch.zeros(4, 4, 3, dtype=torch.uint8) for name in CAMERAS},
            {name: torch.zeros(4, 4, 1) for name in CAMERAS},
            {name: torch.zeros(4, 4, 3, dtype=torch.uint8) for name in CAMERAS},
        )
    recorder.save_episode(
        types.SimpleNamespace(event_name=LeRobotRecorder.STOP_RECORDING_EVENT)
    )
    recorder.episode_queue.join()


def latest_dataset():
    return FakeDataset.instances[-1]


def test_an_episode_reaches_the_dataset(make_recorder):
    recorder = make_recorder()
    push_episode(recorder, n_frames=7)

    assert recorder.num_recorded_episodes == 1
    assert recorder.num_failed_episodes == 0
    assert FakeDataset.instances[0].saved_episodes == 1
    assert len(FakeDataset.instances[0].frames) == 7


def test_save_mp4_without_depth_no_longer_discards_the_episode(make_recorder):
    """The regression: ``--save_mp4`` alone used to lose every episode.

    ``episode_data`` only carries depth and segmentation when those flags are
    on, and the export read them unconditionally — before the dataset commit.
    """
    recorder = make_recorder(save_mp4=True, depth=False, instance_id_seg=False)
    push_episode(recorder)

    assert recorder.num_recorded_episodes == 1, "the episode was lost again"
    assert recorder.num_failed_episodes == 0
    assert FakeDataset.instances[0].saved_episodes == 1
    # RGB is always available, so it is still exported.
    assert ("wrist", "rgb", 1) in recorder.video_calls
    assert not any(kind == "depth" for _, kind, _ in recorder.video_calls)


def test_save_mp4_with_depth_and_segmentation_exports_all_three(make_recorder):
    recorder = make_recorder(save_mp4=True, depth=True, instance_id_seg=True)
    push_episode(recorder)

    assert recorder.num_recorded_episodes == 1
    kinds = {kind for _, kind, _ in recorder.video_calls}
    assert kinds == {"rgb", "depth", "instance_id_segmentation"}


def test_a_failing_video_export_never_costs_the_episode(make_recorder, monkeypatch):
    """ffmpeg missing or erroring must not cost recorded data."""
    recorder = make_recorder(save_mp4=True)

    def explode(*args, **kwargs):
        raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr(recorder, "_save_video", explode)
    push_episode(recorder)

    assert recorder.num_recorded_episodes == 1
    assert recorder.num_failed_episodes == 0
    assert FakeDataset.instances[0].saved_episodes == 1


def test_a_writer_failure_is_counted_not_swallowed(make_recorder, monkeypatch):
    """A lost episode must be visible, so a short run explains itself."""
    recorder = make_recorder()
    monkeypatch.setattr(
        FakeDataset.instances[0],
        "save_episode",
        lambda: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    push_episode(recorder)

    assert recorder.num_recorded_episodes == 0
    assert recorder.num_failed_episodes == 1


def test_cancelling_clears_the_buffers_without_queueing(make_recorder):
    recorder = make_recorder()
    for _ in range(4):
        recorder.push_frame_to_buffer(
            torch.zeros(6),
            torch.zeros(6),
            {name: torch.zeros(4, 4, 3, dtype=torch.uint8) for name in CAMERAS},
            {},
            {},
        )
    assert recorder.current_frame == 4

    recorder.cancel_recording(
        types.SimpleNamespace(event_name=LeRobotRecorder.CANCEL_RECORDING_EVENT)
    )
    assert recorder.current_frame == 0
    assert recorder.episode_queue.empty()


def test_dataset_features_match_the_real_dataset_schema(make_recorder):
    """Sim and real must stay mergeable: same keys, same shapes, same names."""
    recorder = make_recorder(robot_type="so_follower")
    features = recorder.dataset_features

    assert set(features) == {"observation.state", "observation.images.wrist", "action"}
    assert features["action"]["shape"] == (6,)
    assert features["observation.state"]["shape"] == (6,)
    assert features["action"]["names"] == [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    assert features["observation.images.wrist"]["shape"] == (4, 4, 3)


def test_buffer_capacity_follows_the_frame_rate(make_recorder):
    """Capacity is 40 s of recording; at 30 Hz that is 1200 frames."""
    recorder = make_recorder(fps=30)
    assert recorder.capcity == 1200

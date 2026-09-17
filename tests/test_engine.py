import numpy as np
from dataclasses import dataclass

from world_models.observation import Observation
from world_models.engine import RolloutEngine


@dataclass
class _FakeSpec:
    repo_id: str = "fake/model"
    revision: str = "main"
    height: int = 4
    width: int = 4
    fps: int = 16


class _FakeBackend:
    spec = _FakeSpec()

    def predict(self, observation, *, prompt, seed, num_frames,
                guidance_scale, num_inference_steps) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randint(0, 256, (num_frames, 4, 4, 3)).astype(np.uint8)


def test_engine_records_a_faithful_manifest() -> None:
    obs = Observation(np.zeros((5, 4, 4, 3), dtype=np.uint8))
    result = RolloutEngine(_FakeBackend()).generate(
        obs, prompt="x", seed=7, num_frames=9)
    assert result.frames.shape == (9, 4, 4, 3)
    assert result.manifest.seed == 7
    assert result.manifest.checkpoint == "fake/model"
    assert result.manifest.conditioning_num_frames == 5
    assert len(result.manifest.conditioning_frames_sha256) == 64
    assert result.manifest.output_fps == 16


def test_engine_is_reproducible_by_seed() -> None:
    obs = Observation(np.zeros((5, 4, 4, 3), dtype=np.uint8))
    engine = RolloutEngine(_FakeBackend())
    a = engine.generate(obs, prompt="x", seed=1).manifest.run_id
    b = engine.generate(obs, prompt="x", seed=1).manifest.run_id
    assert a == b


def test_distinct_context_windows_do_not_share_an_identifier() -> None:
    engine = RolloutEngine(_FakeBackend())
    frames = np.zeros((6, 4, 4, 3), dtype=np.uint8)
    frames[-1] = 255
    first = Observation(frames[:5], fps=16)
    second = Observation(frames[1:], fps=16)
    a = engine.generate(first, prompt="x", seed=1).manifest
    b = engine.generate(second, prompt="x", seed=1).manifest
    assert a.run_id != b.run_id
    assert a.conditioning_frames_sha256 != b.conditioning_frames_sha256
    assert a.conditioning_fps == 16


def test_conditioning_hash_includes_array_shape() -> None:
    engine = RolloutEngine(_FakeBackend())
    a = Observation(np.zeros((1, 8, 4, 3), dtype=np.uint8))
    b = Observation(np.zeros((2, 4, 4, 3), dtype=np.uint8))
    first = engine.generate(a, prompt="x", seed=1).manifest
    second = engine.generate(b, prompt="x", seed=1).manifest
    assert first.conditioning_frames_sha256 != second.conditioning_frames_sha256

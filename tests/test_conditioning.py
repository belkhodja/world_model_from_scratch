import numpy as np
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from world_models.observation import Observation
from world_models.backends import cosmos
from world_models.backends.cosmos import CHECKPOINTS, CosmosBackend, to_conditioning


def test_guardrail_nltk_data_replaces_hf_symlinks(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    source = snapshot / "blocklist" / "nltk_data" / "tokenizers"
    source.mkdir(parents=True)
    blob = tmp_path / "blob"
    blob.write_text("tokenizer data")
    (source / "punkt.txt").symlink_to(blob)
    staging = tmp_path / "staging"
    staging.mkdir()
    nltk = SimpleNamespace(data=SimpleNamespace(path=[]))

    monkeypatch.delenv("NLTK_DATA", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda *_args, **_kwargs: str(snapshot)
    ))
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setattr(cosmos.tempfile, "mkdtemp", lambda **_kwargs: str(staging))

    cosmos._stage_guardrail_nltk_data("guardrail-checkpoint")

    copied = staging / "nltk_data" / "tokenizers" / "punkt.txt"
    assert copied.read_text() == "tokenizer data"
    assert not copied.is_symlink()
    assert nltk.data.path[0] == str(staging / "nltk_data")


def test_conditioning_returns_one_pil_frame_per_input() -> None:
    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    conditioning = to_conditioning(Observation(frames))

    assert len(conditioning) == 8
    assert all(isinstance(f, Image.Image) for f in conditioning)
    assert conditioning[0].size == (64, 64)  # unmodified; the pipeline resizes


@pytest.mark.parametrize("context_length", [1, 5])
def test_backend_routes_image_and_video_conditioning(monkeypatch, context_length):
    class FakeGenerator:
        def __init__(self, device):
            assert device == "cuda"

        def manual_seed(self, seed):
            assert seed == 3
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext, Generator=FakeGenerator))
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=[[Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))]])

    backend = CosmosBackend(pipeline, CHECKPOINTS["cosmos-predict2.5-2b-720p"])
    observation = Observation(np.zeros((context_length, 4, 4, 3), dtype=np.uint8))
    frames = backend.predict(observation, prompt="test", seed=3, num_frames=1,
                             guidance_scale=7.0, num_inference_steps=15)
    assert frames.shape == (1, 4, 4, 3)
    if context_length == 1:
        assert isinstance(calls[0]["image"], Image.Image)
        assert calls[0]["video"] is None
    else:
        assert calls[0]["image"] is None
        assert len(calls[0]["video"]) == 5

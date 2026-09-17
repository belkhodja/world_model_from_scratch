"""Verify model revision provenance without downloading or running a model."""
from contextlib import nullcontext
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest

from world_models.backends import cosmos
from world_models.experiment import RunManifest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/chapter_03_generate_pai.py"
SPEC = spec_from_file_location("chapter03_generate_pai", SCRIPT)
generation = module_from_spec(SPEC)
SPEC.loader.exec_module(generation)


@pytest.fixture
def subset(tmp_path):
    directory = tmp_path / "subset"
    (directory / "condition_image").mkdir(parents=True)
    (directory / "vqa").mkdir()
    image = directory / "condition_image" / "input.png"
    iio.imwrite(image, np.zeros((16, 16, 3), dtype=np.uint8))
    question = directory / "vqa" / "physics_test.json"
    question.write_text("[]\n")
    records = [{"video_id": "physics_test", "image_name": "input.png",
                "prompt_en": "A ball rolls across the floor."}]
    generation.write_json(directory / "full_info.json", records)
    generation.write_json(directory / "subset.json", {
        "dataset": generation.DATASET,
        "revision": generation.DATA_REVISION,
        "video_ids": ["physics_test"],
        "sha256": {
            str(path.relative_to(directory)): generation.sha256(path)
            for path in (image, question, directory / "full_info.json")
        },
    })
    return directory


@pytest.fixture
def model_loads(monkeypatch):
    loads = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            loads.append({"checkpoint": checkpoint, **kwargs})
            return cls()

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            frames = np.zeros((kwargs["num_frames"], 16, 16, 3), dtype=np.uint8)
            return SimpleNamespace(frames=[frames])

    # The real backend and engine still create and save the run manifest.
    # Only GPU/model operations are replaced with deterministic CPU fixtures.
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch.bfloat16 = "test-bfloat16"
    fake_torch.inference_mode = nullcontext
    fake_torch.Generator = lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed)
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.Cosmos2_5_PredictBasePipeline = Pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    # A revision must be passed directly, without querying a moving Hub ref.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(cosmos, "apply_guardrail_compatibility_patch", lambda: None)
    spec = cosmos.CHECKPOINTS["cosmos-predict2.5-2b-720p"]
    monkeypatch.setitem(cosmos.CHECKPOINTS, "cosmos-predict2.5-2b-720p",
                        replace(spec, height=16, width=16))
    return loads


@pytest.mark.parametrize("revision_args,expected", [
    ([], "0d37c7498f54cee3c599d438d895a0a4a8608064"),
    (["--checkpoint-revision", "abcdef0123456789abcdef0123456789abcdef01"],
     "abcdef0123456789abcdef0123456789abcdef01"),
])
def test_generation_uses_same_revision_for_model_and_saved_records(
    tmp_path, monkeypatch, subset, model_loads, revision_args, expected,
):
    output = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "generate", "--subset", str(subset), "--output", str(output),
        "--seeds", "0", "--num-frames", "5", "--steps", "1", *revision_args,
    ])

    generation.main()

    assert len(model_loads) == 1
    assert model_loads[0]["revision"] == expected
    record = generation.read_json(output / "generation.json")
    assert record["checkpoint_revision"] == expected
    index = json.loads((output / "run_index.jsonl").read_text())
    manifest = RunManifest.load(index["manifest"])
    assert manifest.revision == expected
    assert manifest.checkpoint == model_loads[0]["checkpoint"]
    assert manifest.run_id == index["run_id"]
    assert Path(index["video"]).is_file()


@pytest.mark.parametrize("revision", ["main", "diffusers/base/post-trained", "0d37c74",
                                      "a" * 39, "a" * 41, "g" * 40])
def test_cli_rejects_mutable_or_invalid_revision_before_loading_inputs(
    tmp_path, monkeypatch, capsys, revision,
):
    monkeypatch.setattr(generation, "load_prepared",
                        lambda *args: pytest.fail("Invalid revision reached generation"))
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "generate", "--subset", str(tmp_path / "missing"),
        "--output", str(tmp_path / "generated"), "--checkpoint-revision", revision,
    ])
    with pytest.raises(SystemExit) as error:
        generation.main()
    assert error.value.code == 2
    assert "40-character hexadecimal commit SHA" in capsys.readouterr().err
    assert not (tmp_path / "generated").exists()


def test_direct_generation_rejects_mutable_revision_before_loading_inputs(monkeypatch):
    monkeypatch.setattr(generation, "load_prepared",
                        lambda *args: pytest.fail("Invalid revision reached generation"))
    with pytest.raises(generation.argparse.ArgumentTypeError, match="commit SHA"):
        generation.generate(SimpleNamespace(checkpoint_revision="main"))

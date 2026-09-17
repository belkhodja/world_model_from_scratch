"""CPU tests use tiny videos and synthetic scores solely as parser fixtures."""
from dataclasses import replace
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from world_models.evaluation import collect_vbench, prepare_vbench, sha256_file
from world_models.experiment import RunManifest


def _run(directory: Path, name: str, seed: int = 0) -> tuple[RunManifest, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / f"{name}.mp4"
    frames = np.full((5, 16, 16, 3), 40 + seed, dtype=np.uint8)
    iio.imwrite(video, frames, fps=8)
    manifest = RunManifest(
        checkpoint="test/model", revision="test", seed=seed, num_frames=5,
        guidance_scale=7.0, num_inference_steps=15, height=16, width=16,
        prompt="test", observation="test.png", outputs=[str(video)], versions={},
    )
    manifest.save(directory, f"{name}.manifest.json")
    return manifest, video


@pytest.fixture
def prepared(tmp_path):
    assets = tmp_path / "assets"
    _run(assets, "first", 0)
    _run(assets, "second", 1)
    path = prepare_vbench(assets, tmp_path / "prepared")
    return path, json.loads(path.read_text())


def _results(tmp_path, prepared, dimension="subject_consistency"):
    path, index = prepared
    raw = {dimension: [0.7, [
        {"video_path": f"/gpu/machine/{entry['video']}", "video_results": score}
        for entry, score in zip(index["entries"], [0.6, 0.8])
    ]]}
    results = tmp_path / "official_eval_results.json"
    results.write_text(json.dumps(raw))
    return results, raw


def _collect(tmp_path, prepared, results, dimension="subject_consistency"):
    return collect_vbench(prepared[0], results, dimensions=[dimension],
                          evaluator_revision="test-revision",
                          output=tmp_path / "report")


def test_prepare_deduplicates_identical_artifacts(tmp_path):
    assets = tmp_path / "assets"
    manifest, video = _run(assets, "first")
    alias = assets / "alias.mp4"
    alias.write_bytes(video.read_bytes())
    replace(manifest, outputs=[str(alias)]).save(assets, "alias.manifest.json")
    # An unmanifested source clip must not become a benchmark case.
    (assets / "source.mp4").write_bytes(video.read_bytes())
    path = prepare_vbench(assets, tmp_path / "prepared")
    record = json.loads(path.read_text())
    assert record["scope"] == "custom-clips"
    assert len(record["entries"]) == 1
    entry = record["entries"][0]
    assert len(entry["sources"]) == 2
    assert entry["media"] == {"fps": 8.0, "num_frames": 5, "height": 16,
                              "width": 16, "duration_seconds": 5 / 8}
    assert sha256_file(path.parent / entry["video"]) == sha256_file(video)
    assert prepare_vbench(assets, path.parent) == path  # unchanged rerun is safe


def test_prepare_rejects_legacy_id_collision(tmp_path):
    assets = tmp_path / "assets"
    manifest, _ = _run(assets, "first")
    _, different_video = _run(assets, "second", seed=1)
    replace(manifest, outputs=[str(different_video)]).save(assets, "second.manifest.json")
    with pytest.raises(ValueError, match="Different video bytes share run_id"):
        prepare_vbench(assets, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_prepare_rejects_missing_video(tmp_path):
    manifest, video = _run(tmp_path / "assets", "first")
    video.unlink()
    with pytest.raises(FileNotFoundError, match="No paired rollout"):
        prepare_vbench(tmp_path / "assets", tmp_path / "prepared")


def test_prepare_rejects_stale_staged_video(tmp_path, prepared):
    path, _ = prepared
    (path.parent / "videos" / "stale.mp4").write_bytes(b"stale")
    with pytest.raises(ValueError, match="Unexpected files"):
        prepare_vbench(tmp_path / "assets", path.parent)


def test_collect_joins_scores_across_machine_paths(tmp_path, prepared):
    results, raw = _results(tmp_path, prepared)
    # VBench's aggregate is preserved even if it differs from the simple mean.
    raw["subject_consistency"][0] = 0.71
    results.write_text(json.dumps(raw))
    path = _collect(tmp_path, prepared, results)
    report = json.loads(path.read_text())
    assert report["scope"] == "custom-clips"
    assert report["evaluator"]["revision"] == "test-revision"
    assert report["index_sha256"] == sha256_file(prepared[0])
    aggregate = report["aggregates"]["subject_consistency"]
    assert aggregate == {"official_score": 0.71, "mean_per_video": 0.7,
                         "num_videos": 2}
    assert {row["run_id"] for row in report["rows"]} == {
        entry["run_id"] for entry in prepared[1]["entries"]
    }
    assert (path.parent / "scores.csv").read_text().count("\n") == 3


def test_collect_accepts_dynamic_degree_booleans(tmp_path, prepared):
    results, raw = _results(tmp_path, prepared, "dynamic_degree")
    for item, value in zip(raw["dynamic_degree"][1], [True, False]):
        item["video_results"] = value
    results.write_text(json.dumps(raw))
    report = json.loads(_collect(tmp_path, prepared, results, "dynamic_degree").read_text())
    assert [row["score"] for row in report["rows"]] == [1.0, 0.0]


@pytest.mark.parametrize("problem, expected", [
    ("missing", "Missing videos"),
    ("duplicate", "Duplicate video result"),
    ("unknown", "Unknown video"),
    ("nan", "finite numeric"),
    ("infinity", "finite numeric"),
    ("string", "finite numeric"),
    ("bool", "got a boolean"),
    ("aggregate_nan", "finite numeric"),
    ("dimension", "Missing or malformed"),
])
def test_collect_rejects_invalid_result_sets(tmp_path, prepared, problem, expected):
    results, raw = _results(tmp_path, prepared)
    values = raw["subject_consistency"][1]
    if problem == "missing":
        values.pop()
    elif problem == "duplicate":
        values.append(values[0])
    elif problem == "unknown":
        values[0]["video_path"] = "/elsewhere/unknown.mp4"
    elif problem in ("nan", "infinity", "string", "bool"):
        values[0]["video_results"] = {
            "nan": float("nan"), "infinity": float("inf"), "string": "0.5", "bool": True,
        }[problem]
    elif problem == "aggregate_nan":
        raw["subject_consistency"][0] = float("nan")
    else:
        del raw["subject_consistency"]
    results.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=expected):
        _collect(tmp_path, prepared, results)
    assert not (tmp_path / "report").exists()


def test_collect_rejects_changed_staged_bytes(tmp_path, prepared):
    results, _ = _results(tmp_path, prepared)
    index, record = prepared
    (index.parent / record["entries"][0]["video"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _collect(tmp_path, prepared, results)


def test_collect_rejects_changed_manifest(tmp_path, prepared):
    results, _ = _results(tmp_path, prepared)
    index, record = prepared
    record["entries"][0]["manifest"]["seed"] = 991
    index.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest run_id mismatch"):
        _collect(tmp_path, prepared, results)


def test_collect_rejects_duplicate_json_dimensions(tmp_path, prepared):
    results, _ = _results(tmp_path, prepared)
    results.write_text('{"subject_consistency": [], "subject_consistency": []}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        _collect(tmp_path, prepared, results)


def test_collect_requires_evaluator_revision(tmp_path, prepared):
    results, _ = _results(tmp_path, prepared)
    with pytest.raises(ValueError, match="evaluator_revision"):
        collect_vbench(prepared[0], results, evaluator_revision="  ",
                       output=tmp_path / "report")

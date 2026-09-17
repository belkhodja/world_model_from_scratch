"""Prepare saved rollouts for VBench and join its official per-video results.

No model scores are computed here. VBench owns the metric implementations;
these helpers preserve the link between a score and the artifact it evaluates.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from statistics import fmean
from typing import Any

from world_models.experiment import RunManifest


VBENCH_DIMENSIONS = ("subject_consistency", "motion_smoothness", "dynamic_degree")


def sha256_file(path: str | Path) -> str:
    """Hash artifact bytes without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def _write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def video_metadata(path: str | Path) -> dict[str, int | float]:
    """Inspect decoded frame count and encoded timing using imageio/FFmpeg."""
    import imageio.v2 as iio

    with iio.get_reader(str(path)) as reader:
        metadata = reader.get_meta_data()
        fps = float(metadata["fps"])
        num_frames = int(reader.count_frames())
        frame = reader.get_data(0)
    if not math.isfinite(fps) or fps <= 0 or num_frames < 1:
        raise ValueError(f"Invalid video timing: {path}")
    return {
        "num_frames": num_frames,
        "fps": fps,
        "height": int(frame.shape[0]),
        "width": int(frame.shape[1]),
        "duration_seconds": num_frames / fps,
    }


def _paired_video(manifest_path: Path, manifest: RunManifest) -> Path:
    if manifest_path.name.endswith(".manifest.json"):
        adjacent = manifest_path.with_name(
            manifest_path.name.removesuffix(".manifest.json") + ".mp4"
        )
    else:
        adjacent = manifest_path.parent / "rollout.mp4"
    if adjacent.is_file():
        return adjacent.resolve()
    if len(manifest.outputs) == 1:
        recorded = Path(manifest.outputs[0])
        for candidate in (manifest_path.parent / recorded, recorded):
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"No paired rollout for {manifest_path}")


def prepare_vbench(assets: str | Path, output: str | Path) -> Path:
    """Copy unique manifest-backed videos into an immutable evaluation input set.

    Accepts ``name.manifest.json`` next to ``name.mp4`` and run directories
    containing ``manifest.json``/``rollout.mp4``. Duplicate run IDs must have
    identical video bytes. Unmanifested source clips are not evaluated.
    """
    assets, output = Path(assets).resolve(), Path(output).resolve()
    if not assets.is_dir():
        raise FileNotFoundError(f"No asset directory: {assets}")
    manifest_paths = sorted(set(assets.rglob("*.manifest.json"))
                            | set(assets.rglob("manifest.json")))
    if not manifest_paths:
        raise ValueError(f"No run manifests found under {assets}")
    entries: dict[str, dict] = {}
    for path in manifest_paths:
        manifest = RunManifest.load(path)
        video = _paired_video(path, manifest)
        checksum = sha256_file(video)
        source = {"manifest": str(path), "manifest_sha256": sha256_file(path),
                  "video": str(video)}
        if manifest.run_id in entries:
            if entries[manifest.run_id]["video_sha256"] != checksum:
                raise ValueError(f"Different video bytes share run_id {manifest.run_id}")
            entries[manifest.run_id]["sources"].append(source)
            continue
        entries[manifest.run_id] = {
            "run_id": manifest.run_id,
            "video": f"videos/{manifest.run_id}.mp4",
            "video_sha256": checksum,
            "media": video_metadata(video),
            "manifest": asdict(manifest) | {"run_id": manifest.run_id},
            "sources": [source],
        }

    # Validate the whole source set before writing. Refuse to mix stale videos
    # into an existing evaluator directory (VBench scans every MP4 in it).
    videos_dir = output / "videos"
    expected_names = {f"{run_id}.mp4" for run_id in entries}
    if videos_dir.exists():
        extra = {p.name for p in videos_dir.iterdir()} - expected_names
        if extra:
            raise ValueError(f"Unexpected files in {videos_dir}: {sorted(extra)}")
        for entry in entries.values():
            target = output / entry["video"]
            if target.exists() and sha256_file(target) != entry["video_sha256"]:
                raise ValueError(f"Staged video differs from source: {target}")
    videos_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries.values():
        target = output / entry["video"]
        if not target.exists():
            shutil.copy2(entry["sources"][0]["video"], target)
        if sha256_file(target) != entry["video_sha256"]:
            raise ValueError(f"Copy checksum mismatch: {target}")
    index = {
        "schema_version": 1,
        "benchmark": "VBench",
        "scope": "custom-clips",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing": "byte-for-byte copy; no trimming, resizing, or frame-rate conversion",
        "entries": list(entries.values()),
    }
    path = output / "index.json"
    _write_json(path, index)
    return path


def _score(value: Any, *, label: str, boolean_allowed: bool = False) -> float:
    if isinstance(value, bool) and not boolean_allowed:
        raise ValueError(f"Expected a numeric score for {label}, got a boolean")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric score for {label}")
    return float(value)


def collect_vbench(
    index: str | Path,
    results: str | Path,
    *,
    dimensions: list[str] | tuple[str, ...] = VBENCH_DIMENSIONS,
    evaluator_revision: str,
    output: str | Path,
) -> Path:
    """Validate VBench's ``{dimension: [aggregate, per_video]}`` result JSON.

    Each per-video object contains ``video_path`` and ``video_results``.
    Join by the staged filename so result files can move between machines.
    A missing, duplicate, unknown, or nonfinite result fails the collection.
    The official aggregate and an explicitly named per-video mean are retained
    separately; the latter need not match VBench's frame-weighted aggregation.
    """
    index, results, output = Path(index).resolve(), Path(results).resolve(), Path(output)
    if not evaluator_revision.strip():
        raise ValueError("evaluator_revision must identify the VBench checkout")
    dimensions = list(dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("Specify each requested dimension exactly once")
    unknown = set(dimensions) - set(VBENCH_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unsupported chapter dimensions: {sorted(unknown)}")
    prepared = _read_json(index)
    if (prepared.get("schema_version") != 1 or prepared.get("benchmark") != "VBench"
            or prepared.get("scope") != "custom-clips"):
        raise ValueError("Expected a Chapter 3 VBench custom-clips index")
    entries = prepared.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Evaluation index must contain entries")
    by_name: dict[str, dict] = {}
    for entry in entries:
        run_id = entry["run_id"]
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise ValueError(f"Invalid run_id in evaluation index: {run_id!r}")
        name = f"{run_id}.mp4"
        if entry["video"] != f"videos/{name}" or name in by_name:
            raise ValueError(f"Invalid or duplicate indexed video: {entry['video']}")
        if sha256_file(index.parent / entry["video"]) != entry["video_sha256"]:
            raise ValueError(f"Staged video checksum mismatch: {name}")
        manifest_record = dict(entry["manifest"])
        recorded_id = manifest_record.pop("run_id", None)
        if recorded_id != run_id or RunManifest(**manifest_record).run_id != run_id:
            raise ValueError(f"Indexed manifest run_id mismatch: {run_id}")
        by_name[name] = entry
    raw = _read_json(results)
    if not isinstance(raw, dict):
        raise ValueError("VBench result JSON must be an object keyed by dimension")
    rows, aggregates = [], {}
    for dimension in dimensions:
        value = raw.get(dimension)
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], list):
            raise ValueError(f"Missing or malformed VBench dimension: {dimension}")
        official_score = _score(value[0], label=f"{dimension} aggregate")
        seen: set[str] = set()
        dimension_scores = []
        for item in value[1]:
            if not isinstance(item, dict) or not isinstance(item.get("video_path"), str):
                raise ValueError(f"Malformed per-video result for {dimension}")
            name = Path(item["video_path"]).name
            if name not in by_name:
                raise ValueError(f"Unknown video in {dimension}: {name}")
            if name in seen:
                raise ValueError(f"Duplicate video result in {dimension}: {name}")
            seen.add(name)
            score = _score(item.get("video_results"), label=f"{dimension}/{name}",
                           boolean_allowed=dimension == "dynamic_degree")
            entry = by_name[name]
            manifest = entry["manifest"]
            rows.append({
                "run_id": entry["run_id"], "dimension": dimension, "score": score,
                "video": entry["video"], "video_sha256": entry["video_sha256"],
                "seed": manifest["seed"], "guidance_scale": manifest["guidance_scale"],
                "num_inference_steps": manifest["num_inference_steps"],
            })
            dimension_scores.append(score)
        missing = set(by_name) - seen
        if missing:
            raise ValueError(f"Missing videos in {dimension}: {sorted(missing)}")
        aggregates[dimension] = {
            "official_score": official_score,
            "mean_per_video": fmean(dimension_scores),
            "num_videos": len(dimension_scores),
        }
    report = {
        "schema_version": 1,
        "benchmark": "VBench", "scope": "custom-clips",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluator": {"revision": evaluator_revision, "results": str(results),
                      "results_sha256": sha256_file(results)},
        "index": str(index), "index_sha256": sha256_file(index),
        "dimensions": dimensions, "aggregates": aggregates, "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    _write_json(report_path, report)
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report_path

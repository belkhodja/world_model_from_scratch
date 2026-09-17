#!/usr/bin/env python3
"""Copy the recorded PAI pilot into review assets after verifying its identity.

The saved Chapter 3 result indexes define the expected video hashes and runs.
Videos, original manifests, and evaluator records are copied byte for byte.
This command does not generate videos or run an evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from world_models.evaluation import sha256_file
from world_models.experiment import RunManifest


RESULTS = REPO_ROOT / "assets/chapter_03/results"
PROVENANCE_FILES = (
    "evaluator_environment.txt", "evaluator_revision.txt", "judge_revision.txt",
    "uv.lock", "vqa_vllm_compat.patch",
)


def prepare_review(runs: Path, output: Path) -> Path:
    """Validate the complete source set before copying any review artifact."""
    cases = {case["video_id"]: case for case in json.loads(
        (RESULTS / "pai_subset_full_info.json").read_text()
    )}
    copies: list[tuple[Path, Path, str]] = []
    entries = []
    provenance = {}
    source_records = {}
    matched_settings = None

    for guidance in (3, 7):
        name = f"guidance{guidance}"
        source = runs / name
        generation_path = RESULTS / f"pai_{name}_generation.json"
        index_path = RESULTS / f"pai_{name}_run_index.jsonl"
        plan = json.loads(generation_path.read_text())
        rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
        if plan["guidance_scale"] != guidance or plan["conditioning"] != "image":
            raise ValueError(f"Unexpected generation settings in {generation_path}")
        common = {key: plan[key] for key in (
            "dataset", "dataset_revision", "checkpoint", "checkpoint_revision", "fps",
            "num_frames", "num_inference_steps", "seeds", "video_ids",
        )}
        if matched_settings is not None and common != matched_settings:
            raise ValueError("The saved pilot settings differ beyond guidance")
        matched_settings = common
        expected = {(case, seed) for case in plan["video_ids"] for seed in plan["seeds"]}
        actual = {(row["video_id"], row["seed"]) for row in rows}
        if (len(rows) != plan["expected_videos"] or len(actual) != len(rows)
                or actual != expected or len({row["run_id"] for row in rows}) != len(rows)):
            raise ValueError(f"Incomplete or duplicate pilot entries in {index_path}")

        source_records[name] = {
            path.name: sha256_file(path) for path in (generation_path, index_path)
        }
        for row in rows:
            video_id, seed, run_id = row["video_id"], row["seed"], row["run_id"]
            if video_id not in cases or not re.fullmatch(r"[0-9a-f]{12}", run_id):
                raise ValueError(f"Invalid case or run ID in {index_path}: {row}")
            filename = f"{video_id}__{seed}.mp4"
            if Path(row["video"]).name != filename:
                raise ValueError(f"Unexpected indexed video filename: {row['video']}")
            video = source / "videos" / filename
            video_hash = sha256_file(video)
            if video_hash != row["video_sha256"]:
                raise ValueError(f"Video differs from the recorded pilot: {video}")
            manifest_path = source / "runs" / run_id / "manifest.json"
            manifest = RunManifest.load(manifest_path)
            required = {
                "run_id": run_id, "seed": seed, "guidance_scale": guidance,
                "checkpoint": plan["checkpoint"], "revision": plan["checkpoint_revision"],
                "num_frames": plan["num_frames"],
                "num_inference_steps": plan["num_inference_steps"],
                "output_fps": plan["fps"], "conditioning_num_frames": 1,
                "prompt": cases[video_id]["prompt_en"],
            }
            for field, expected_value in required.items():
                if getattr(manifest, field) != expected_value:
                    raise ValueError(f"Manifest {field} disagrees with the saved pilot: {manifest_path}")
            if Path(manifest.observation).name != cases[video_id]["image_name"]:
                raise ValueError(f"Manifest uses a different conditioning image: {manifest_path}")
            manifest_hash = sha256_file(manifest_path)
            video_relative = Path(name) / "videos" / filename
            manifest_relative = Path(name) / "manifests" / f"{run_id}.json"
            copies.extend(((video, video_relative, video_hash),
                           (manifest_path, manifest_relative, manifest_hash)))
            entries.append({
                "video_id": video_id, "seed": seed, "guidance": guidance, "run_id": run_id,
                "video": video_relative.as_posix(), "video_sha256": video_hash,
                "manifest": manifest_relative.as_posix(), "manifest_sha256": manifest_hash,
            })

        provenance[name] = {}
        for filename in PROVENANCE_FILES:
            source_path = source / "evaluation" / filename
            checksum = sha256_file(source_path)
            relative = Path(name) / "evaluation" / filename
            copies.append((source_path, relative, checksum))
            provenance[name][filename] = {"path": relative.as_posix(), "sha256": checksum}

    # Reject changed destinations before copying; identical reruns are safe.
    for _, relative, checksum in copies:
        target = output / relative
        if target.exists() and sha256_file(target) != checksum:
            raise ValueError(f"Existing review artifact differs from its source: {target}")
    for source_path, relative, checksum in copies:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source_path, target)
        if sha256_file(target) != checksum:
            raise ValueError(f"Review copy checksum mismatch: {target}")

    review = {
        "schema_version": 1,
        "path_base": "Paths in entries and provenance are relative to this index.",
        "preservation": "Byte-for-byte copies; no video transcoding or manifest rewriting.",
        "provenance_note": (
            "Historical evaluator records are preserved as recorded. The saved judge SHA "
            "does not establish that the original evaluator loaded that pinned revision."
        ),
        "source_record_sha256": source_records,
        "num_videos": len(entries),
        "entries": entries,
        "provenance": provenance,
    }
    path = output / "review_index.json"
    path.write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="Parent of the downloaded guidance3 and guidance7 runs")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "assets/chapter_03/pai")
    args = parser.parse_args()
    path = prepare_review(args.runs, args.output)
    print(f"Verified and prepared 56 recorded videos, 56 manifests, and 10 evaluator records: {path}")


if __name__ == "__main__":
    main()

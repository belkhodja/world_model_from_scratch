#!/usr/bin/env python3
"""Prepare fixed PAI-Bench-G cases and generate manifest-backed Cosmos videos.

Preparation downloads only the selected images and annotations. Generation
requires the Chapter 2 GPU environment. This script does not run an evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASET = "shi-labs/physical-ai-bench-generation"
DATA_REVISION = "c36b7beb69bd7c4adcedc1431407fce98f9dfe44"
DEFAULT_CHECKPOINT_REVISION = "0d37c7498f54cee3c599d438d895a0a4a8608064"
PROMPT_FILE = "cosmos_predict2_bench_full_info.json"
CATEGORIES = ("av", "common_sense", "human", "industry", "misc", "physics", "robot")


def immutable_checkpoint_revision(value: str) -> str:
    """Require an immutable model commit so separate runs use the same weights."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError(
            "checkpoint revision must be a full 40-character hexadecimal commit SHA"
        )
    return value.lower()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category_for(video_id: str) -> str:
    for category in CATEGORIES:
        if video_id.startswith(category + "_"):
            return category
    raise ValueError(f"Unknown PAI category: {video_id}")


def select_cases(records: list[dict], per_category: int) -> list[dict]:
    """Select the first IDs in each category, independent of source row order."""
    if per_category < 1:
        raise ValueError("per_category must be positive")
    seen = set()
    groups = {category: [] for category in CATEGORIES}
    for row in records:
        video_id = row["video_id"]
        if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise ValueError(f"Invalid video_id: {video_id!r}")
        if video_id in seen:
            raise ValueError(f"Duplicate video_id: {video_id}")
        seen.add(video_id)
        image_name = row["image_name"]
        if not isinstance(image_name, str) or Path(image_name).name != image_name:
            raise ValueError(f"image_name must be a filename: {image_name!r}")
        if not isinstance(row["prompt_en"], str) or not row["prompt_en"].strip():
            raise ValueError(f"Missing prompt: {video_id}")
        groups[category_for(video_id)].append(row)
    selected = []
    for category in CATEGORIES:
        group = sorted(groups[category], key=lambda row: row["video_id"])
        if len(group) < per_category:
            raise ValueError(f"Not enough cases in {category}: {len(group)}")
        selected.extend(group[:per_category])
    return selected


def prepare(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    if args.output.exists():
        raise FileExistsError(f"Use a new subset directory: {args.output}")

    def download(filename: str) -> Path:
        return Path(hf_hub_download(
            repo_id=DATASET, repo_type="dataset", revision=DATA_REVISION,
            filename=filename,
        ))

    source = download(PROMPT_FILE)
    selected = select_cases(read_json(source), args.per_category)
    args.output.mkdir(parents=True)
    hashes = {}
    for row in selected:
        for filename in (f"condition_image/{row['image_name']}", f"vqa/{row['video_id']}.json"):
            target = args.output / filename
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(download(filename), target)
            hashes[filename] = sha256(target)
    write_json(args.output / "full_info.json", selected)
    hashes["full_info.json"] = sha256(args.output / "full_info.json")
    write_json(args.output / "subset.json", {
        "dataset": DATASET,
        "revision": DATA_REVISION,
        "source_metadata_sha256": sha256(source),
        "selection": "first video_ids in lexical order within each category",
        "per_category": args.per_category,
        "video_ids": [row["video_id"] for row in selected],
        "sha256": hashes,
    })
    print(f"Prepared {len(selected)} cases in {args.output}")


def load_prepared(subset: Path, limit: int | None = None) -> list[dict]:
    """Validate the downloaded inputs before any model is loaded."""
    metadata = read_json(subset / "subset.json")
    if metadata["dataset"] != DATASET or metadata["revision"] != DATA_REVISION:
        raise ValueError("Subset was prepared from a different dataset revision")
    for filename, expected in metadata["sha256"].items():
        path = (subset / filename).resolve()
        if not path.is_relative_to(subset.resolve()):
            raise ValueError(f"Subset path escapes its directory: {filename}")
        if sha256(path) != expected:
            raise ValueError(f"Changed subset file: {filename}")
    records = read_json(subset / "full_info.json")
    if [row["video_id"] for row in records] != metadata["video_ids"]:
        raise ValueError("Subset metadata does not match selected cases")
    if limit is not None and not 1 <= limit <= len(records):
        raise ValueError(f"limit must be between 1 and {len(records)}")
    return records[:limit] if limit is not None else records


def generate(args: argparse.Namespace) -> None:
    revision = immutable_checkpoint_revision(args.checkpoint_revision)
    records = load_prepared(args.subset, args.limit)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be between 0 and num-shards - 1")
    records = records[args.shard_index::args.num_shards]
    if not records:
        raise ValueError("This shard contains no cases")
    if args.output.exists():
        raise FileExistsError(f"Use a new generation directory: {args.output}")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if args.num_frames < 1 or (args.num_frames - 1) % 4 or args.steps < 1:
        raise ValueError("Use a positive 4k+1 frame count and positive inference steps")
    if args.guidance < 0:
        raise ValueError("guidance must be nonnegative")

    import torch
    from diffusers import Cosmos2_5_PredictBasePipeline
    from world_models.backends.cosmos import CHECKPOINTS, CosmosBackend, apply_guardrail_compatibility_patch
    from world_models.engine import RolloutEngine
    from world_models.observation import load_image

    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires the Chapter 2 CUDA environment")
    spec = CHECKPOINTS["cosmos-predict2.5-2b-720p"]
    spec = replace(spec, revision=revision)
    apply_guardrail_compatibility_patch()
    pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
        spec.repo_id, revision=spec.revision, torch_dtype=torch.bfloat16,
    ).to("cuda")
    engine = RolloutEngine(CosmosBackend(pipeline, spec))

    args.output.mkdir(parents=True)
    videos_dir = args.output / "videos"
    videos_dir.mkdir()
    write_json(args.output / "full_info.json", records)
    write_json(args.output / "generation.json", {
        "dataset": DATASET, "dataset_revision": DATA_REVISION,
        "subset_manifest": str((args.subset / "subset.json").resolve()),
        "subset_manifest_sha256": sha256(args.subset / "subset.json"),
        "conditioning": "image", "checkpoint": spec.repo_id,
        "checkpoint_revision": spec.revision, "fps": spec.fps,
        "num_frames": args.num_frames, "num_inference_steps": args.steps,
        "guidance_scale": args.guidance, "seeds": args.seeds,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "video_ids": [row["video_id"] for row in records],
        "category_counts": dict(Counter(category_for(row["video_id"]) for row in records)),
        "expected_videos": len(records) * len(args.seeds),
    })
    index_path = args.output / "run_index.jsonl"
    for row in records:
        observation = load_image(args.subset / "condition_image" / row["image_name"])
        for seed in args.seeds:
            result = engine.generate(
                observation, prompt=row["prompt_en"], seed=seed,
                num_frames=args.num_frames, num_inference_steps=args.steps,
                guidance_scale=args.guidance, save_to=args.output / "runs",
            )
            target = videos_dir / f"{row['video_id']}__{seed}.mp4"
            shutil.copyfile(result.manifest.outputs[0], target)
            manifest = args.output / "runs" / result.manifest.run_id / "manifest.json"
            with index_path.open("a") as handle:
                handle.write(json.dumps({
                    "video_id": row["video_id"], "seed": seed,
                    "run_id": result.manifest.run_id,
                    "manifest": str(manifest.resolve()),
                    "video": str(target.resolve()), "video_sha256": sha256(target),
                }) + "\n")
            print(target, flush=True)


def merge(args: argparse.Namespace) -> None:
    """Merge disjoint generation shards into one evaluator input."""
    if args.output.exists():
        raise FileExistsError(f"Use a new generation directory: {args.output}")
    records = load_prepared(args.subset)
    record_by_id = {row["video_id"]: row for row in records}
    configs = []
    rows = []
    seen = set()
    for input_dir in args.inputs:
        config = read_json(input_dir / "generation.json")
        configs.append(config)
        for line in (input_dir / "run_index.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["video_id"], row["seed"])
            if key in seen:
                raise ValueError(f"Duplicate generated video: {key}")
            if row["video_id"] not in record_by_id:
                raise ValueError(f"Unknown generated video ID: {row['video_id']}")
            seen.add(key)
            rows.append((input_dir, row))

    comparable = {
        key: value for key, value in configs[0].items()
        if key not in {"video_ids", "category_counts", "expected_videos", "shard_index"}
    }
    for config in configs[1:]:
        candidate = {
            key: value for key, value in config.items()
            if key not in {"video_ids", "category_counts", "expected_videos", "shard_index"}
        }
        if candidate != comparable:
            raise ValueError("Generation settings differ between shards")

    seeds = configs[0]["seeds"]
    expected = {(row["video_id"], seed) for row in records for seed in seeds}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"Incomplete shards; missing={missing}, extra={extra}")

    args.output.mkdir(parents=True)
    videos_dir = args.output / "videos"
    runs_dir = args.output / "runs"
    videos_dir.mkdir()
    runs_dir.mkdir()
    merged_rows = []
    order = {row["video_id"]: index for index, row in enumerate(records)}
    rows.sort(key=lambda item: (order[item[1]["video_id"]], item[1]["seed"]))
    for input_dir, row in rows:
        source_video = Path(row["video"])
        target_video = videos_dir / source_video.name
        shutil.copyfile(source_video, target_video)
        if sha256(target_video) != row["video_sha256"]:
            raise ValueError(f"Changed generated video: {source_video}")
        source_run = input_dir / "runs" / row["run_id"]
        target_run = runs_dir / row["run_id"]
        shutil.copytree(source_run, target_run)
        target_manifest = target_run / "manifest.json"
        manifest_data = read_json(target_manifest)
        manifest_data["outputs"] = [str((target_run / "rollout.mp4").resolve())]
        write_json(target_manifest, manifest_data)
        merged_rows.append({
            **row,
            "manifest": str(target_manifest.resolve()),
            "video": str(target_video.resolve()),
        })

    write_json(args.output / "full_info.json", records)
    merged_config = dict(configs[0])
    merged_config.pop("shard_index", None)
    merged_config["num_shards"] = len(args.inputs)
    merged_config["video_ids"] = [row["video_id"] for row in records]
    merged_config["category_counts"] = dict(Counter(
        category_for(row["video_id"]) for row in records
    ))
    merged_config["expected_videos"] = len(expected)
    merged_config["source_shards"] = [str(path.resolve()) for path in args.inputs]
    write_json(args.output / "generation.json", merged_config)
    with (args.output / "run_index.jsonl").open("w") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Merged {len(merged_rows)} videos into {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download a fixed subset; no GPU needed")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--per-category", type=int, default=2)
    prep.set_defaults(run=prepare)
    gen = commands.add_parser("generate", help="Generate Cosmos I2V rollouts on CUDA")
    gen.add_argument("--subset", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--checkpoint-revision", type=immutable_checkpoint_revision,
                     default=DEFAULT_CHECKPOINT_REVISION,
                     help="Immutable model commit SHA (default: the book's recorded PAI checkpoint)")
    gen.add_argument("--limit", type=int)
    gen.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    gen.add_argument("--num-frames", type=int, default=93)
    gen.add_argument("--steps", type=int, default=36)
    gen.add_argument("--guidance", type=float, default=7.0)
    gen.add_argument("--num-shards", type=int, default=1)
    gen.add_argument("--shard-index", type=int, default=0)
    gen.set_defaults(run=generate)
    combine = commands.add_parser("merge", help="Merge disjoint generation shards")
    combine.add_argument("--subset", type=Path, required=True)
    combine.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(run=merge)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

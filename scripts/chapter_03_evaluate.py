#!/usr/bin/env python3
"""Prepare Chapter 2 videos for VBench and collect its official results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from world_models.evaluation import VBENCH_DIMENSIONS, collect_vbench, prepare_vbench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Stage unique manifest-backed videos")
    prepare.add_argument("--assets", type=Path, default=Path("assets/chapter_02"))
    prepare.add_argument("--output", type=Path, default=Path("outputs/chapter_03/vbench"))
    collect = subparsers.add_parser("collect", help="Validate and join official VBench scores")
    collect.add_argument("--index", type=Path, required=True)
    collect.add_argument("--results", type=Path, required=True)
    collect.add_argument("--dimensions", nargs="+", choices=VBENCH_DIMENSIONS,
                         default=list(VBENCH_DIMENSIONS))
    collect.add_argument("--evaluator-revision", required=True,
                         help="Record the exact VBench checkout, e.g. git rev-parse HEAD")
    collect.add_argument("--output", type=Path, default=Path("outputs/chapter_03/report"))
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_vbench(args.assets, args.output)
    else:
        path = collect_vbench(args.index, args.results, dimensions=args.dimensions,
                              evaluator_revision=args.evaluator_revision, output=args.output)
    print(path)


if __name__ == "__main__":
    main()

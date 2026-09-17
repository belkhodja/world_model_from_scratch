"""Cosmos-specific backend: checkpoint metadata, boundary adapters, and the driver.

This is the only module that drives the diffusers Cosmos pipeline. Everything
above it depends on the book's model-agnostic interfaces, not on Cosmos.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CheckpointSpec:
    """A checkpoint and the input contract it expects.

    Values are transcribed from the official model card and should be
    re-verified against the loaded pipeline configuration. ``num_context_frames``
    applies to video conditioning; conditioning on a still image uses one frame.
    """

    repo_id: str
    revision: str
    num_context_frames: int
    height: int
    width: int
    fps: int


CHECKPOINTS: dict[str, CheckpointSpec] = {
    "cosmos-predict2.5-2b-720p": CheckpointSpec(
        repo_id="nvidia/Cosmos-Predict2.5-2B",
        revision="diffusers/base/post-trained",
        num_context_frames=5,
        height=704,
        width=1280,
        fps=16,
    ),
}


def _stage_guardrail_nltk_data(checkpoint_id: str) -> None:
    """Copy HF-linked NLTK data to regular files when NLTK has no override."""
    if os.environ.get("NLTK_DATA"):
        return

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        checkpoint_id, allow_patterns=["blocklist/nltk_data/*"]
    ))
    source = snapshot / "blocklist" / "nltk_data"
    if not any(path.is_symlink() for path in source.rglob("*")):
        return
    target = Path(tempfile.mkdtemp(prefix="cosmos-guardrail-nltk-")) / "nltk_data"
    shutil.copytree(source, target, symlinks=False)
    os.environ["NLTK_DATA"] = str(target)

    import nltk

    nltk.data.path.insert(0, str(target))


def apply_guardrail_compatibility_patch() -> None:
    """Keep cosmos_guardrail 0.3.1 active with current Diffusers and NLTK.

    Version 0.3.1 exposes a read-only property that fails when Diffusers checks
    the pipeline device. Its NLTK data also uses Hugging Face cache symlinks
    rejected by NLTK's hardened file reader. Other versions are left unchanged.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        guardrail_version = version("cosmos_guardrail")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "cosmos_guardrail is required for Cosmos inference. Install "
            "requirements-chapter-02-gpu.txt."
        ) from exc

    if guardrail_version == "0.3.1":
        import torch
        from cosmos_guardrail.cosmos_guardrail import (
            COSMOS_GUARDRAIL_CHECKPOINT,
            CosmosSafetyChecker,
        )

        _stage_guardrail_nltk_data(COSMOS_GUARDRAIL_CHECKPOINT)
        CosmosSafetyChecker.device = property(lambda self: torch.device("cuda"))


def to_conditioning(observation) -> list:
    """Convert a canonical Observation into pipeline conditioning frames.

    Returns the frames as PIL images. The pipeline resizes them to the
    checkpoint resolution and selects the conditioning window itself, so no
    trimming or resizing is done here.
    """
    from PIL import Image

    return [Image.fromarray(frame) for frame in observation.frames]


def to_frames(pipeline_output) -> np.ndarray:
    """Convert a Cosmos pipeline result into (T, H, W, 3) uint8 frames."""
    frames = pipeline_output.frames[0]
    return np.stack([np.asarray(f) for f in frames])


class CosmosBackend:
    """Turns an Observation into generated frames using the Cosmos pipeline.

    The only class that drives Diffusers. Construct it from a loaded pipeline
    and its CheckpointSpec.
    """

    def __init__(self, pipeline, spec: CheckpointSpec):
        self.pipeline = pipeline
        self.spec = spec

    def predict(self, observation, *, prompt: str, seed: int, num_frames: int,
                guidance_scale: float, num_inference_steps: int) -> np.ndarray:
        import torch

        conditioning = to_conditioning(observation)
        is_image = observation.num_frames == 1
        with torch.inference_mode():
            result = self.pipeline(
                image=conditioning[0] if is_image else None,
                video=None if is_image else conditioning,
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )
        return to_frames(result)

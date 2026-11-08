"""Memory-conscious, framework-independent views over episode archives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WindowSpec:
    """Temporal layout of a sample, in saved simulator timesteps."""

    context_steps: int
    prediction_gap: int
    target_steps: int
    sample_stride: int = 1

    @property
    def required_steps(self) -> int:
        return self.context_steps + self.prediction_gap + self.target_steps

    def validate(self) -> None:
        if self.context_steps <= 0 or self.target_steps <= 0:
            raise ValueError("context_steps and target_steps must be positive")
        if self.prediction_gap < 0 or self.sample_stride <= 0:
            raise ValueError("prediction_gap must be non-negative and sample_stride positive")


@dataclass(frozen=True)
class EpisodeData:
    """Canonical episode arrays. Windows are derived as views, never copied."""

    state: FloatArray
    inputs: FloatArray
    initial_relative_energy: FloatArray
    time: FloatArray
    split: str

    def validate(self) -> None:
        if self.state.ndim != 3 or self.state.shape[-1] != 4:
            raise ValueError("state must have shape (episodes, timesteps, 4)")
        if self.inputs.shape != (*self.state.shape[:2], 2):
            raise ValueError("inputs must have shape (episodes, timesteps, 2)")
        if self.initial_relative_energy.shape != (self.state.shape[0],):
            raise ValueError("initial_relative_energy must have one value per episode")
        if self.time.shape != (self.state.shape[1],):
            raise ValueError("time must have one value per timestep")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_split(manifest_path: Path, split: str, *, verify_checksum: bool = True) -> EpisodeData:
    """Load one manifest-selected archive and optionally verify its checksum."""
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        entry = manifest["splits"][split]
    except KeyError as error:
        raise ValueError(f"unknown split {split!r}") from error
    archive_path = manifest_path.parent / entry["file"]
    if verify_checksum and _sha256(archive_path) != entry["sha256"]:
        raise ValueError(f"checksum mismatch for {archive_path}")
    with np.load(archive_path) as archive:
        episode_data = EpisodeData(
            state=archive["state"], inputs=archive["inputs"],
            initial_relative_energy=archive["initial_relative_energy"], time=archive["time"], split=split,
        )
    episode_data.validate()
    return episode_data


class WindowedEpisodes:
    """Arithmetic index over windows without materializing copied samples."""

    def __init__(self, episodes: EpisodeData, spec: WindowSpec) -> None:
        episodes.validate()
        spec.validate()
        if episodes.state.shape[1] < spec.required_steps:
            raise ValueError("episodes are shorter than one requested sample")
        self.episodes, self.spec = episodes, spec
        self.windows_per_episode = 1 + (episodes.state.shape[1] - spec.required_steps) // spec.sample_stride

    def __len__(self) -> int:
        return self.episodes.state.shape[0] * self.windows_per_episode

    def __getitem__(self, index: int) -> dict[str, FloatArray | float | int | str]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode, offset = divmod(index, self.windows_per_episode)
        start = offset * self.spec.sample_stride
        context_end = start + self.spec.context_steps
        target_start = context_end + self.spec.prediction_gap
        target_end = target_start + self.spec.target_steps
        return {
            "context_state": self.episodes.state[episode, start:context_end],
            "target_state": self.episodes.state[episode, target_start:target_end],
            "actions": self.episodes.inputs[episode, context_end:target_end],
            "episode_index": episode,
            "initial_relative_energy": float(self.episodes.initial_relative_energy[episode]),
            "split": self.episodes.split,
        }

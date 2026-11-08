"""Reproducible Stage-0 split generation and provenance recording."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

from data.stage0 import Stage0DatasetConfig, generate_stage0_dataset
from sim.double_pendulum import DoublePendulumParameters

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DATASET_SCHEMA_VERSION = "stage0-v1"
SPLIT_NAMES = ("train", "validation", "test", "hard_validation", "hard_test")


@dataclass(frozen=True)
class Stage0SplitConfig:
    """Dataset sizes and energy support for a complete Stage-0 benchmark.

    The ordinary splits are independently sampled from the training energy
    distribution. Hard splits are independently sampled from a disjoint,
    higher-energy band and are never used for model or hyperparameter choice.
    """

    seed: int = 0
    train_episodes: int = 1_000
    validation_episodes: int = 200
    test_episodes: int = 200
    hard_validation_episodes: int = 200
    hard_test_episodes: int = 200
    hard_energy_min: float = 1.75
    hard_energy_max: float = 1.95

    def validate(self, base_config: Stage0DatasetConfig) -> None:
        if any(getattr(self, f"{name}_episodes") <= 0 for name in SPLIT_NAMES):
            raise ValueError("every split must contain at least one episode")
        if not 0.0 <= self.hard_energy_min < self.hard_energy_max <= 2.0:
            raise ValueError("hard energy band must lie within [0, 2]")
        if self.hard_energy_min <= base_config.energy_max:
            raise ValueError("hard energy band must be disjoint above the ordinary energy band")


def _archive_payload(
    dataset: dict[str, np.ndarray],
    config: Stage0DatasetConfig,
    parameters: DoublePendulumParameters,
) -> dict[str, np.ndarray | float | int]:
    return {
        **dataset,
        "mass_1": parameters.mass_1,
        "mass_2": parameters.mass_2,
        "length_1": parameters.length_1,
        "length_2": parameters.length_2,
        "gravity": parameters.gravity,
        "damping_1": parameters.damping_1,
        "damping_2": parameters.damping_2,
        "seed": config.seed,
    }


def write_stage0_archive(
    output: Path,
    dataset: dict[str, np.ndarray],
    config: Stage0DatasetConfig,
    parameters: DoublePendulumParameters,
) -> None:
    """Write one self-describing Stage-0 archive."""
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **_archive_payload(dataset, config, parameters))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_energy_histogram(energies: dict[str, np.ndarray], output: Path) -> None:
    """Plot one comparable initial-energy distribution per split."""
    figure, axis = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.0, 2.0, 41)
    for name in SPLIT_NAMES:
        axis.hist(energies[name], bins=bins, histtype="step", linewidth=1.8, label=name)
    axis.set(
        title="Stage-0 initial relative-energy distributions",
        xlabel="Initial relative energy",
        ylabel="Episode count",
        xlim=(0.0, 2.0),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def generate_stage0_splits(
    base_config: Stage0DatasetConfig,
    split_config: Stage0SplitConfig,
    output_dir: Path,
    parameters: DoublePendulumParameters | None = None,
) -> dict[str, dict[str, object]]:
    """Generate independent ordinary and high-energy challenge splits.

    A ``SeedSequence`` derives an independent, recorded seed for every split;
    a split can therefore be reproduced without relying on generation order.
    """
    split_config.validate(base_config)
    parameters = parameters or DoublePendulumParameters(damping_1=0.0, damping_2=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    child_sequences = np.random.SeedSequence(split_config.seed).spawn(len(SPLIT_NAMES))
    episodes = {
        "train": split_config.train_episodes,
        "validation": split_config.validation_episodes,
        "test": split_config.test_episodes,
        "hard_validation": split_config.hard_validation_episodes,
        "hard_test": split_config.hard_test_episodes,
    }
    manifest_splits: dict[str, dict[str, object]] = {}
    energies: dict[str, np.ndarray] = {}
    for name, sequence in zip(SPLIT_NAMES, child_sequences, strict=True):
        split_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        is_hard = name.startswith("hard_")
        config = replace(
            base_config,
            seed=split_seed,
            episodes=episodes[name],
            energy_min=split_config.hard_energy_min if is_hard else base_config.energy_min,
            energy_max=split_config.hard_energy_max if is_hard else base_config.energy_max,
            energy_mean=(split_config.hard_energy_min + split_config.hard_energy_max) / 2
            if is_hard
            else base_config.energy_mean,
            energy_std=(split_config.hard_energy_max - split_config.hard_energy_min) / 4
            if is_hard
            else base_config.energy_std,
        )
        dataset = generate_stage0_dataset(config, parameters)
        archive = output_dir / f"{name}.npz"
        write_stage0_archive(archive, dataset, config, parameters)
        energy = dataset["initial_relative_energy"]
        energies[name] = energy
        manifest_splits[name] = {
            "file": archive.name,
            "sha256": _sha256(archive),
            "generation": asdict(config),
            "episodes": int(dataset["state"].shape[0]),
            "timesteps": int(dataset["state"].shape[1]),
            "initial_relative_energy": {
                "min": float(energy.min()),
                "max": float(energy.max()),
                "mean": float(energy.mean()),
            },
        }

    histogram = output_dir / "initial_energy_histograms.png"
    _plot_energy_histogram(energies, histogram)
    manifest: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "root_seed": split_config.seed,
        "parameters": asdict(parameters),
        "ordinary_energy_band": [base_config.energy_min, base_config.energy_max],
        "hard_energy_band": [split_config.hard_energy_min, split_config.hard_energy_max],
        "histogram": histogram.name,
        "splits": manifest_splits,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_splits

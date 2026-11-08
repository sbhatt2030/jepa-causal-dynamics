"""Generate a reproducible Stage-0 train/validation/test benchmark bundle.

Example:
    .venv\\Scripts\\python -m data.generate_stage0_splits --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.stage0 import Stage0DatasetConfig
from data.stage0_splits import Stage0SplitConfig, generate_stage0_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "stage0")
    parser.add_argument("--train-episodes", type=int, default=1_000)
    parser.add_argument("--validation-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--hard-validation-episodes", type=int, default=200)
    parser.add_argument("--hard-test-episodes", type=int, default=200)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--output-stride", type=int, default=10)
    parser.add_argument("--energy-mean", type=float, default=1.2)
    parser.add_argument("--energy-std", type=float, default=0.30)
    parser.add_argument("--energy-min", type=float, default=0.7)
    parser.add_argument("--energy-max", type=float, default=1.7)
    parser.add_argument("--hard-energy-min", type=float, default=1.75)
    parser.add_argument("--hard-energy-max", type=float, default=1.95)
    args = parser.parse_args()

    base_config = Stage0DatasetConfig(
        duration=args.duration,
        dt=args.dt,
        output_stride=args.output_stride,
        energy_mean=args.energy_mean,
        energy_std=args.energy_std,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
    )
    split_config = Stage0SplitConfig(
        seed=args.seed,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes,
        hard_validation_episodes=args.hard_validation_episodes,
        hard_test_episodes=args.hard_test_episodes,
        hard_energy_min=args.hard_energy_min,
        hard_energy_max=args.hard_energy_max,
    )
    manifest = generate_stage0_splits(base_config, split_config, args.output_dir)
    print(f"Wrote Stage-0 bundle to {args.output_dir}")
    for name, split in manifest.items():
        energy = split["initial_relative_energy"]
        print(f"{name}: {split['episodes']} episodes, energy [{energy['min']:.3f}, {energy['max']:.3f}]")


if __name__ == "__main__":
    main()

"""Generate and save a conservative, unforced Stage-0 dataset.

Example:
    .venv\\Scripts\\python -m data.generate_stage0 --episodes 100 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np

from data.stage0 import Stage0DatasetConfig, generate_stage0_dataset
from sim.double_pendulum import DoublePendulumParameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log_dataset_run(
    config: Stage0DatasetConfig,
    parameters: DoublePendulumParameters,
    dataset: dict[str, np.ndarray],
    output: Path,
    experiment_name: str,
    run_name: str | None,
    tracking_uri: str | None,
) -> None:
    """Record the dataset provenance and generated archive in MLflow."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").resolve().as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "dataset.stage": "0",
                "dataset.seed": config.seed,
                "dataset.episodes": config.episodes,
                "simulation.duration": config.duration,
                "simulation.dt": config.dt,
                "simulation.output_stride": config.output_stride,
                "energy.mean": config.energy_mean,
                "energy.std": config.energy_std,
                "energy.min": config.energy_min,
                "energy.max": config.energy_max,
                "pendulum.mass_1": parameters.mass_1,
                "pendulum.mass_2": parameters.mass_2,
                "pendulum.length_1": parameters.length_1,
                "pendulum.length_2": parameters.length_2,
                "pendulum.gravity": parameters.gravity,
                "pendulum.damping_1": parameters.damping_1,
                "pendulum.damping_2": parameters.damping_2,
            }
        )
        initial_energy = dataset["initial_relative_energy"]
        mlflow.log_metrics(
            {
                "dataset.episodes": float(dataset["state"].shape[0]),
                "dataset.timesteps": float(dataset["state"].shape[1]),
                "energy.initial.min": float(initial_energy.min()),
                "energy.initial.max": float(initial_energy.max()),
                "energy.initial.mean": float(initial_energy.mean()),
            }
        )
        mlflow.log_artifact(str(output), artifact_path="datasets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--energy-mean", type=float, default=1.2)
    parser.add_argument("--energy-std", type=float, default=0.30)
    parser.add_argument("--energy-min", type=float, default=0.7)
    parser.add_argument("--energy-max", type=float, default=1.7)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "stage0.npz")
    parser.add_argument("--experiment-name", default="jepa-causal-dynamics")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (defaults to the project's local mlruns directory).",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    config = Stage0DatasetConfig(
        seed=args.seed,
        episodes=args.episodes,
        energy_mean=args.energy_mean,
        energy_std=args.energy_std,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
    )
    parameters = DoublePendulumParameters(damping_1=0.0, damping_2=0.0)
    dataset = generate_stage0_dataset(config, parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **dataset,
        mass_1=parameters.mass_1,
        mass_2=parameters.mass_2,
        length_1=parameters.length_1,
        length_2=parameters.length_2,
        gravity=parameters.gravity,
        damping_1=parameters.damping_1,
        damping_2=parameters.damping_2,
        seed=config.seed,
    )
    if not args.no_mlflow:
        log_dataset_run(
            config,
            parameters,
            dataset,
            args.output,
            args.experiment_name,
            args.run_name,
            args.tracking_uri,
        )
    print(f"Wrote {args.output}")
    print(f"state shape: {dataset['state'].shape}; energy range: ({config.energy_min}, {config.energy_max})")


if __name__ == "__main__":
    main()

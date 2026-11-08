import numpy as np

from data.dataset import WindowSpec, WindowedEpisodes, load_manifest_split
from data.stage0 import Stage0DatasetConfig, generate_stage0_dataset, sample_initial_states
from data.stage0_splits import Stage0SplitConfig, generate_stage0_splits
from sim.double_pendulum import DoublePendulumParameters, relative_energy


def test_energy_filtered_initial_states_are_reproducible() -> None:
    config = Stage0DatasetConfig(episodes=4, seed=12)
    parameters = DoublePendulumParameters()
    first = sample_initial_states(config, parameters)
    second = sample_initial_states(config, parameters)
    np.testing.assert_array_equal(first, second)
    energy = relative_energy(first, parameters)
    assert np.all((energy >= 0.7) & (energy <= 1.7))


def test_stage0_dataset_is_unforced_and_episode_shaped() -> None:
    dataset = generate_stage0_dataset(
        Stage0DatasetConfig(episodes=3, duration=0.1, dt=0.005, output_stride=1, seed=3)
    )
    assert dataset["state"].shape == (3, 21, 4)
    assert dataset["inputs"].shape == (3, 21, 2)
    np.testing.assert_allclose(dataset["inputs"], 0.0)


def test_stage0_split_bundle_records_independent_seeds_and_energy_bands(tmp_path) -> None:
    base = Stage0DatasetConfig(duration=0.02, dt=0.01, output_stride=1, energy_min=0.4, energy_max=0.8)
    splits = Stage0SplitConfig(
        seed=9,
        train_episodes=2,
        validation_episodes=2,
        test_episodes=2,
        hard_validation_episodes=2,
        hard_test_episodes=2,
        hard_energy_min=1.1,
        hard_energy_max=1.3,
    )
    manifest = generate_stage0_splits(base, splits, tmp_path)

    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "initial_energy_histograms.png").is_file()
    seeds = {split["generation"]["seed"] for split in manifest.values()}
    assert len(seeds) == 5
    for name, split in manifest.items():
        energy = split["initial_relative_energy"]
        if name.startswith("hard_"):
            assert 1.1 <= energy["min"] <= energy["max"] <= 1.3
        else:
            assert 0.4 <= energy["min"] <= energy["max"] <= 0.8


def test_windowed_episodes_uses_source_array_views(tmp_path) -> None:
    base = Stage0DatasetConfig(duration=0.04, dt=0.01, output_stride=1, energy_min=0.4, energy_max=0.8)
    splits = Stage0SplitConfig(
        seed=4, train_episodes=2, validation_episodes=1, test_episodes=1,
        hard_validation_episodes=1, hard_test_episodes=1, hard_energy_min=1.1, hard_energy_max=1.3,
    )
    generate_stage0_splits(base, splits, tmp_path)
    episodes = load_manifest_split(tmp_path / "manifest.json", "train")
    windows = WindowedEpisodes(episodes, WindowSpec(context_steps=2, prediction_gap=1, target_steps=2))

    assert len(windows) == 2
    sample = windows[0]
    assert np.shares_memory(sample["context_state"], episodes.state)
    assert np.shares_memory(sample["target_state"], episodes.state)
    assert np.shares_memory(sample["actions"], episodes.inputs)

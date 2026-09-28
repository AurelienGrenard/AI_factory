"""Tests for dataset-independent conditional terminal GAN plumbing."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from learning.common.data import (
    make_preassigned_selection,
    prepare_terminal_dataset,
)
from learning.common.data.terminal_transforms import (
    fit_terminal_transform,
)
from learning.gan.config import load_config
from learning.gan.engine import train_experiment
from learning.gan.networks import build_critic, build_generator
from learning.gan.noise import sample_noise
from learning.gan.objectives import (
    critic_logistic,
    critic_wasserstein_gp,
    generator_logistic_regularized,
    generator_wasserstein_regularized,
)


def write_terminal_dataset(
    path: Path, *, group_count: int = 40, paths_per_group: int = 3
) -> None:
    row_count = group_count * paths_per_group
    header = {
        "database_id": "synthetic_samples",
        "model_family": "synthetic",
        "row_count": row_count,
        "time_convention": {"days_per_year": 252},
        "construction": {
            "parameter_count": group_count,
            "paths_per_parameter": paths_per_group,
            "row_order": "parameter-major, then path-major",
        },
    }
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(header)[:-1] + ',\n"samples": [\n'
        )
        for index in range(row_count):
            group = index // paths_per_group
            path_index = index % paths_per_group
            maturity_days = 63 + path_index * 63
            parameter = group / group_count
            row = {
                "id": str(index),
                "parameters": {"parameter": parameter},
                "maturity_days": maturity_days,
                "T": maturity_days / 252,
                "values": {
                    "positive": (
                        0.1
                        + parameter * parameter
                        + 0.01 * path_index
                    ),
                    "state": (
                        parameter
                        + 0.05 * (path_index - 1)
                    ),
                },
            }
            stream.write(json.dumps(row))
            stream.write(
                ",\n" if index + 1 < row_count else "\n"
            )
        stream.write("]\n}\n")


class TerminalGanDataTest(unittest.TestCase):
    def test_cache_observable_selection_and_group_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "samples.json"
            write_terminal_dataset(dataset)
            prepared = prepare_terminal_dataset(
                dataset,
                cache_root=root / "cache",
                split_seed=123,
                observables=["state"],
                chunk_size=7,
            )
            reopened = prepare_terminal_dataset(
                dataset,
                cache_root=root / "cache",
                split_seed=123,
                observables=["state"],
            )
            self.assertEqual(prepared.fingerprint, reopened.fingerprint)
            self.assertEqual(prepared.conditions.shape, (120, 2))
            self.assertEqual(prepared.observations.shape, (120, 1))
            self.assertEqual(
                prepared.schema.observable_names, ("state",)
            )
            self.assertEqual(
                prepared.schema.available_observable_names,
                ("positive", "state"),
            )
            self.assertTrue(np.isfinite(prepared.conditions).all())
            self.assertTrue(np.isfinite(prepared.observations).all())

            selection = make_preassigned_selection(
                prepared.split_codes,
                train_size=30,
                split_seed=123,
                selection_seed=7,
                group_ids=prepared.parameter_groups,
            )
            train_groups = set(
                prepared.parameter_groups[
                    selection.train_indices
                ].tolist()
            )
            validation_groups = set(
                prepared.parameter_groups[
                    selection.validation_indices
                ].tolist()
            )
            test_groups = set(
                prepared.parameter_groups[
                    selection.test_indices
                ].tolist()
            )
            self.assertFalse(train_groups & validation_groups)
            self.assertFalse(train_groups & test_groups)
            self.assertFalse(validation_groups & test_groups)

    def test_log_transform_round_trip(self) -> None:
        conditions = np.asarray(
            [[0.0], [1.0], [2.0]], dtype=np.float32
        )
        observations = np.asarray(
            [[0.0], [0.5], [2.0]], dtype=np.float32
        )
        transform = fit_terminal_transform(
            conditions,
            observations,
            ("positive",),
            np.arange(3),
            {
                "positive": {
                    "name": "log_shift",
                    "offset": 1e-5,
                    "minimum": 0.0,
                }
            },
        )
        tensor = torch.as_tensor(observations)
        reconstructed = transform.denormalize_observations(
            transform.normalize_observations(tensor)
        )
        torch.testing.assert_close(
            reconstructed, tensor, atol=1e-5, rtol=1e-5
        )


class TerminalGanNetworkTest(unittest.TestCase):
    def test_centered_uniform_noise_has_requested_shape_and_support(self) -> None:
        values = sample_noise(
            4096,
            5,
            "uniform",
            device=torch.device("cpu"),
            random=torch.Generator().manual_seed(1234),
        )
        self.assertEqual(values.shape, (4096, 5))
        self.assertGreaterEqual(float(values.min()), -1.0)
        self.assertLessEqual(float(values.max()), 1.0)
        self.assertLess(abs(float(values.mean())), 0.02)

    def test_unit_uniform_noise_has_requested_shape_and_support(self) -> None:
        values = sample_noise(
            4096,
            5,
            "uniform",
            device=torch.device("cpu"),
            random=torch.Generator().manual_seed(4321),
            uniform_minimum=0.0,
            uniform_maximum=1.0,
        )
        self.assertEqual(values.shape, (4096, 5))
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)
        self.assertLess(abs(float(values.mean()) - 0.5), 0.02)

    def test_wasserstein_gp_is_finite_and_differentiable(self) -> None:
        generator = build_generator(
            {
                "name": "conditional_residual_mlp",
                "latent_dimension": 4,
                "hidden_size": 16,
                "depth": 2,
                "activation": "silu",
                "layer_norm": True,
            },
            condition_dimension=3,
            observation_dimension=2,
        )
        critic = build_critic(
            {
                "name": "conditional_projection_mlp",
                "hidden_size": 16,
                "depth": 2,
                "activation": "leaky_relu",
                "spectral_norm": False,
            },
            condition_dimension=3,
            observation_dimension=2,
        )
        concat_critic = build_critic(
            {
                "name": "conditional_concat_mlp",
                "hidden_size": 16,
                "depth": 2,
                "activation": "leaky_relu",
                "spectral_norm": False,
            },
            condition_dimension=3,
            observation_dimension=2,
        )
        conditions = torch.randn(8, 3)
        real = torch.randn(8, 2)
        self.assertEqual(concat_critic(conditions, real).shape, (8,))
        first_noise = torch.randn(8, 4)
        generated = generator(conditions, first_noise)
        loss = critic_wasserstein_gp(
            critic,
            conditions,
            real,
            generated,
            gradient_penalty_weight=10.0,
            gradient_penalty_target=1.0,
            drift_weight=0.001,
        )
        self.assertTrue(bool(torch.isfinite(loss.total)))
        loss.total.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in critic.parameters()
            )
        )

        second_noise = torch.randn(8, 4)
        second_generated = generator(conditions, second_noise)
        generator_loss = generator_wasserstein_regularized(
            critic,
            conditions,
            real,
            generated,
            first_noise,
            second_observations=second_generated,
            second_noise=second_noise,
            mode_seeking_weight=0.1,
            moment_matching_weight=0.1,
        )
        self.assertTrue(bool(torch.isfinite(generator_loss.total)))
        self.assertGreaterEqual(
            float(generator_loss.mode_seeking_ratio.detach()), 0.0
        )
        self.assertGreaterEqual(
            float(generator_loss.moment_matching.detach()), 0.0
        )

    def test_embedded_logistic_gan_is_finite_and_differentiable(self) -> None:
        generator = build_generator(
            {
                "name": "conditional_embedded_mlp",
                "latent_dimension": 5,
                "noise_distribution": "uniform",
                "noise_minimum": 0.0,
                "noise_maximum": 1.0,
                "parameter_embedding_size": 7,
                "noise_embedding_size": 6,
                "hidden_size": 12,
                "depth": 4,
                "activation": "softplus",
            },
            condition_dimension=3,
            observation_dimension=1,
        )
        critic = build_critic(
            {
                "name": "conditional_embedded_mlp",
                "parameter_embedding_size": 7,
                "observation_embedding_size": 6,
                "hidden_size": 12,
                "depth": 4,
                "activation": "softplus",
                "spectral_norm": False,
            },
            condition_dimension=3,
            observation_dimension=1,
        )
        conditions = torch.randn(16, 3)
        real = torch.randn(16, 1)
        first_noise = sample_noise(
            16,
            5,
            "uniform",
            device=torch.device("cpu"),
            uniform_minimum=0.0,
            uniform_maximum=1.0,
        )
        fake = generator(conditions, first_noise)
        critic_loss = critic_logistic(
            critic, conditions, real, fake
        )
        self.assertTrue(bool(torch.isfinite(critic_loss.total)))
        critic_loss.total.backward()
        critic.zero_grad(set_to_none=True)
        generator.zero_grad(set_to_none=True)
        generator_loss = generator_logistic_regularized(
            critic, conditions, real, fake, first_noise
        )
        self.assertTrue(bool(torch.isfinite(generator_loss.total)))
        generator_loss.total.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in generator.parameters()
            )
        )


class TerminalGanEndToEndTest(unittest.TestCase):
    def test_tiny_cpu_training_writes_finite_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "samples.json"
            write_terminal_dataset(dataset)
            config = load_config(
                None,
                {
                    "dataset": str(dataset),
                    "data": {
                        "observables": ["state"],
                        "train_size": 30,
                        "split_seed": 123,
                        "selection_seed": 7,
                        "cache_root": str(root / "cache"),
                    },
                    "generator": {
                        "name": "conditional_embedded_mlp",
                        "hidden_size": 8,
                        "depth": 2,
                        "latent_dimension": 3,
                        "noise_distribution": "uniform",
                        "noise_minimum": 0.0,
                        "noise_maximum": 1.0,
                        "parameter_embedding_size": 4,
                        "noise_embedding_size": 4,
                        "activation": "softplus",
                    },
                    "critic": {
                        "name": "conditional_embedded_mlp",
                        "hidden_size": 8,
                        "depth": 2,
                        "parameter_embedding_size": 4,
                        "observation_embedding_size": 4,
                        "activation": "softplus",
                    },
                    "objective": {
                        "name": "logistic",
                        "mode_seeking_weight": 0.1,
                        "moment_matching_weight": 0.1,
                    },
                    "training": {
                        "epochs": 1,
                        "batch_size": 10,
                        "critic_steps": 2,
                        "generator_learning_rate": 0.0003,
                        "critic_learning_rate": 0.0007,
                        "validation_interval": 1,
                        "log_interval": 100,
                        "scheduler": "exponential",
                        "scheduler_gamma": 0.99,
                        "ema_decay": 0.0,
                        "device": "cpu",
                    },
                    "evaluation": {
                        "max_samples": 12,
                        "mmd_samples": 8,
                        "projection_count": 4,
                        "condition_projection_count": 2,
                        "condition_bin_count": 2,
                    },
                },
            )
            prepared = prepare_terminal_dataset(
                dataset,
                cache_root=config["data"]["cache_root"],
                split_seed=config["data"]["split_seed"],
                observables=config["data"]["observables"],
            )
            selection = make_preassigned_selection(
                prepared.split_codes,
                train_size=config["data"]["train_size"],
                split_seed=config["data"]["split_seed"],
                selection_seed=config["data"][
                    "selection_seed"
                ],
                group_ids=prepared.parameter_groups,
            )
            run_directory = root / "run"
            result = train_experiment(
                prepared,
                selection,
                config,
                run_directory,
            )
            state = json.loads(
                (run_directory / "state.json").read_text()
            )
            checkpoint = torch.load(
                run_directory / "checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(checkpoint["epoch"], 1)
            runtime = result.manifest["runtime"]
            self.assertEqual(
                runtime["critic_steps"],
                2 * runtime["generator_steps"],
            )
            history = json.loads(
                (run_directory / "history.json").read_text()
            )
            self.assertAlmostEqual(
                history[0]["generator_learning_rate"], 0.000297
            )
            self.assertAlmostEqual(
                history[0]["critic_learning_rate"], 0.000693
            )
            self.assertIn(
                "mode_seeking_ratio", history[0]["train"]
            )
            self.assertIn(
                "moment_matching", history[0]["train"]
            )
            self.assertIn(
                "state_std_ratio", history[0]["validation"]
            )
            self.assertTrue(
                math.isfinite(
                    result.manifest["metrics"]["validation"][
                        "selection_score"
                    ]
                )
            )
            best = torch.load(
                run_directory / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                result.manifest["evaluated_checkpoint"],
                "best_model.pt",
            )
            self.assertEqual(
                result.manifest["evaluated_epoch"],
                result.manifest["best_epoch"],
            )
            self.assertEqual(
                result.manifest["metrics"]["validation"],
                best["validation"],
            )
            for name in (
                "best_model.pt",
                "last_model.pt",
                "run.json",
                "history.json",
            ):
                self.assertTrue((run_directory / name).is_file())


if __name__ == "__main__":
    unittest.main()

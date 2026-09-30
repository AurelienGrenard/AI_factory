"""Resumable conditional WGAN-GP training for terminal distributions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from learning.common.data.selection import DatasetSelection
from learning.common.data.terminal_cache import PreparedTerminalDataset
from learning.common.data.terminal_torch import TerminalBatchLoader
from learning.common.data.terminal_transforms import (
    TerminalTransform,
    fit_terminal_transform,
)

from .metrics import evaluate_generator
from .networks import (
    ConditionalCritic,
    ConditionalGenerator,
    build_critic,
    build_generator,
)
from .noise import sample_noise
from .objectives import (
    critic_logistic,
    critic_wasserstein_gp,
    generator_logistic_regularized,
    generator_wasserstein_regularized,
)


@dataclass(frozen=True)
class ExperimentResult:
    run_directory: Path
    manifest: dict[str, Any]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    return device


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        )
    torch.use_deterministic_algorithms(deterministic)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        canonical = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(canonical.dtype).encode())
        digest.update(str(tuple(canonical.shape)).encode())
        digest.update(canonical.numpy().tobytes())
    return digest.hexdigest()


def _optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    training: dict[str, Any],
):
    betas = tuple(float(value) for value in training["adam_betas"])
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def _scheduler(optimizer, training: dict[str, Any], epochs: int):
    name = training["scheduler"]
    if name == "constant":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training["scheduler_gamma"])
    )


def _loader(
    prepared: PreparedTerminalDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> TerminalBatchLoader:
    return TerminalBatchLoader(
        prepared,
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def _evaluation_indices(
    indices: np.ndarray, maximum: int, seed: int
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= maximum:
        return indices
    random_generator = np.random.default_rng(seed)
    return np.sort(
        random_generator.choice(
            indices, size=maximum, replace=False
        )
    )


def _set_requires_grad(model: torch.nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def _update_ema(
    average: ConditionalGenerator,
    current: ConditionalGenerator,
    decay: float,
) -> None:
    average_parameters = dict(average.named_parameters())
    for name, parameter in current.named_parameters():
        average_parameters[name].mul_(decay).add_(
            parameter, alpha=1.0 - decay
        )
    average_buffers = dict(average.named_buffers())
    for name, buffer in current.named_buffers():
        average_buffers[name].copy_(buffer)


def _assert_finite_loss(loss: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"{name} became non-finite")


def _assert_finite_gradients(
    model: torch.nn.Module, name: str
) -> None:
    for parameter_name, parameter in model.named_parameters():
        if (
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ):
            raise FloatingPointError(
                f"{name} gradient became non-finite at "
                f"{parameter_name}"
            )


def _random_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_random_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [item.detach().cpu() for item in state["cuda"]]
        )


def _checkpoint_payload(
    *,
    generator: ConditionalGenerator,
    critic: ConditionalCritic,
    ema_generator: ConditionalGenerator,
    generator_optimizer,
    critic_optimizer,
    generator_scheduler,
    critic_scheduler,
    epoch: int,
    generator_steps: int,
    critic_steps: int,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
    best_validation_score: float | None,
    best_epoch: int | None,
    config_hash: str,
    dataset_fingerprint: str,
    selection: dict[str, object],
    transform: TerminalTransform,
    initial_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "epoch": epoch,
        "generator_steps": generator_steps,
        "critic_steps": critic_steps,
        "generator_state_dict": generator.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "ema_generator_state_dict": ema_generator.state_dict(),
        "generator_optimizer_state_dict": (
            generator_optimizer.state_dict()
        ),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "generator_scheduler_state_dict": (
            None
            if generator_scheduler is None
            else generator_scheduler.state_dict()
        ),
        "critic_scheduler_state_dict": (
            None
            if critic_scheduler is None
            else critic_scheduler.state_dict()
        ),
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "best_validation_score": best_validation_score,
        "best_epoch": best_epoch,
        "config_hash": config_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "selection": selection,
        "transform": transform.to_dict(),
        "initial_hashes": initial_hashes,
        "random_state": _random_state(),
    }


def train_experiment(
    prepared: PreparedTerminalDataset,
    selection: DatasetSelection,
    config: dict[str, Any],
    run_directory: str | Path,
    *,
    resume: bool = False,
) -> ExperimentResult:
    training = config["training"]
    objective = config["objective"]
    objective_name = str(objective["name"])
    evaluation = config["evaluation"]
    seed = int(training["seed"])
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    critic_updates_per_batch = int(training["critic_steps"])
    validation_interval = int(training["validation_interval"])
    log_interval = int(training["log_interval"])
    deterministic = bool(training.get("deterministic", True))
    _seed_everything(seed, deterministic)
    device = _device(str(training.get("device", "auto")))

    for module_name in config.get("extensions", []):
        importlib.import_module(module_name)
    transform = fit_terminal_transform(
        prepared.conditions,
        prepared.observations,
        prepared.schema.observable_names,
        selection.train_indices,
        config["data"].get("observable_transforms"),
    )
    condition_dimension = len(
        prepared.schema.conditioning_names
    )
    observation_dimension = len(
        prepared.schema.observable_names
    )
    generator = build_generator(
        config["generator"],
        condition_dimension=condition_dimension,
        observation_dimension=observation_dimension,
    ).to(device)
    critic = build_critic(
        config["critic"],
        condition_dimension=condition_dimension,
        observation_dimension=observation_dimension,
    ).to(device)
    ema_generator = copy.deepcopy(generator).to(device).eval()
    _set_requires_grad(ema_generator, False)
    initial_hashes = {
        "generator": _model_hash(generator),
        "critic": _model_hash(critic),
    }

    generator_optimizer = _optimizer(
        generator,
        learning_rate=float(
            training["generator_learning_rate"]
        ),
        training=training,
    )
    critic_optimizer = _optimizer(
        critic,
        learning_rate=float(training["critic_learning_rate"]),
        training=training,
    )
    generator_scheduler = _scheduler(
        generator_optimizer, training, epochs
    )
    critic_scheduler = _scheduler(
        critic_optimizer, training, epochs
    )

    run_directory = Path(run_directory)
    checkpoint_path = run_directory / "checkpoint.pt"
    selection_manifest = selection.manifest()
    config_hash = _canonical_hash(config)
    if resume:
        if not run_directory.is_dir() or not checkpoint_path.is_file():
            raise ValueError(
                f"No resumable GAN checkpoint in {run_directory}"
            )
    else:
        run_directory.mkdir(parents=True, exist_ok=True)
        if (
            checkpoint_path.exists()
            or (run_directory / "initial_models.pt").exists()
        ):
            raise ValueError(
                f"Run directory already contains GAN state: "
                f"{run_directory}"
            )
        _atomic_torch_save(
            run_directory / "initial_models.pt",
            {
                "generator_state_dict": generator.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "initial_hashes": initial_hashes,
                "generator": config["generator"],
                "critic": config["critic"],
                "seed": seed,
            },
        )

    history: list[dict[str, Any]] = []
    completed_epoch = 0
    generator_steps = 0
    critic_steps = 0
    previous_elapsed = 0.0
    best_validation_score: float | None = None
    best_epoch: int | None = None
    if resume:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint["config_hash"] != config_hash:
            raise ValueError(
                "Resume GAN configuration differs from checkpoint"
            )
        if (
            checkpoint["dataset_fingerprint"]
            != prepared.fingerprint
        ):
            raise ValueError(
                "Resume GAN dataset differs from checkpoint"
            )
        if checkpoint["selection"] != selection_manifest:
            raise ValueError(
                "Resume GAN split or selected rows differ"
            )
        if checkpoint["transform"] != transform.to_dict():
            raise ValueError(
                "Resume GAN transforms differ from checkpoint"
            )
        if checkpoint["initial_hashes"] != initial_hashes:
            raise ValueError(
                "Resume GAN initial model identity differs"
            )
        generator.load_state_dict(
            checkpoint["generator_state_dict"]
        )
        critic.load_state_dict(checkpoint["critic_state_dict"])
        ema_generator.load_state_dict(
            checkpoint["ema_generator_state_dict"]
        )
        generator_optimizer.load_state_dict(
            checkpoint["generator_optimizer_state_dict"]
        )
        critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )
        if generator_scheduler is not None:
            state = checkpoint["generator_scheduler_state_dict"]
            if state is None:
                raise ValueError(
                    "Resume checkpoint has no generator scheduler"
                )
            generator_scheduler.load_state_dict(state)
        if critic_scheduler is not None:
            state = checkpoint["critic_scheduler_state_dict"]
            if state is None:
                raise ValueError(
                    "Resume checkpoint has no critic scheduler"
                )
            critic_scheduler.load_state_dict(state)
        history = list(checkpoint["history"])
        completed_epoch = int(checkpoint["epoch"])
        generator_steps = int(checkpoint["generator_steps"])
        critic_steps = int(checkpoint["critic_steps"])
        previous_elapsed = float(
            checkpoint["elapsed_seconds"]
        )
        best_validation_score = checkpoint[
            "best_validation_score"
        ]
        best_epoch = checkpoint["best_epoch"]
        _restore_random_state(checkpoint["random_state"])
    else:
        _atomic_torch_save(
            checkpoint_path,
            _checkpoint_payload(
                generator=generator,
                critic=critic,
                ema_generator=ema_generator,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer,
                generator_scheduler=generator_scheduler,
                critic_scheduler=critic_scheduler,
                epoch=0,
                generator_steps=0,
                critic_steps=0,
                history=[],
                elapsed_seconds=0.0,
                best_validation_score=None,
                best_epoch=None,
                config_hash=config_hash,
                dataset_fingerprint=prepared.fingerprint,
                selection=selection_manifest,
                transform=transform,
                initial_hashes=initial_hashes,
            ),
        )

    maximum_evaluation = int(evaluation["max_samples"])
    validation_indices = _evaluation_indices(
        selection.validation_indices,
        maximum_evaluation,
        int(evaluation["seed"]),
    )
    test_indices = _evaluation_indices(
        selection.test_indices,
        maximum_evaluation,
        int(evaluation["seed"]) + 1,
    )
    validation_loader = _loader(
        prepared,
        validation_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = _loader(
        prepared,
        test_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    state: dict[str, object] = {
        "status": "running",
        "completed_epochs": completed_epoch,
        "total_epochs": epochs,
        "generator_steps": generator_steps,
        "critic_steps": critic_steps,
        "initial_hashes": initial_hashes,
        "dataset_fingerprint": prepared.fingerprint,
        "selection": selection_manifest,
    }
    _atomic_json(run_directory / "state.json", state)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    try:
        for epoch in range(completed_epoch + 1, epochs + 1):
            train_loader = _loader(
                prepared,
                selection.train_indices,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + 1_000_003 * epoch,
            )
            generator.train()
            critic.train()
            critic_totals = {
                "total": 0.0,
                "wasserstein": 0.0,
                "gradient_penalty": 0.0,
                "gradient_norm_mean": 0.0,
                "drift": 0.0,
            }
            generator_totals = {
                "generator_loss": 0.0,
                "generator_adversarial": 0.0,
                "mode_seeking_ratio": 0.0,
                "moment_matching": 0.0,
            }
            critic_update_count = 0
            generator_update_count = 0
            examples = 0
            epoch_started = time.time()
            for batch_position, batch in enumerate(
                train_loader, start=1
            ):
                raw_conditions = batch["conditions"].to(device)
                raw_observations = batch["observations"].to(device)
                conditions = transform.normalize_conditions(
                    raw_conditions
                )
                real_observations = (
                    transform.normalize_observations(
                        raw_observations
                    )
                )
                count = int(conditions.shape[0])
                examples += count

                _set_requires_grad(critic, True)
                for _ in range(critic_updates_per_batch):
                    critic_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        noise = sample_noise(
                            count,
                            generator.latent_dimension,
                            generator.noise_distribution,
                            device=device,
                            uniform_minimum=generator.noise_minimum,
                            uniform_maximum=generator.noise_maximum,
                        )
                        fake_observations = generator(
                            conditions, noise
                        )
                    if objective_name == "wasserstein_gp":
                        critic_loss = critic_wasserstein_gp(
                            critic,
                            conditions,
                            real_observations,
                            fake_observations,
                            gradient_penalty_weight=float(
                                objective[
                                    "gradient_penalty_weight"
                                ]
                            ),
                            gradient_penalty_target=float(
                                objective[
                                    "gradient_penalty_target"
                                ]
                            ),
                            drift_weight=float(
                                objective["drift_weight"]
                            ),
                        )
                    else:
                        critic_loss = critic_logistic(
                            critic,
                            conditions,
                            real_observations,
                            fake_observations,
                        )
                    _assert_finite_loss(
                        critic_loss.total, "critic loss"
                    )
                    critic_loss.total.backward()
                    _assert_finite_gradients(
                        critic, "critic"
                    )
                    critic_optimizer.step()
                    critic_steps += 1
                    critic_update_count += 1
                    for name in critic_totals:
                        critic_totals[name] += float(
                            getattr(critic_loss, name)
                            .detach()
                            .cpu()
                        )

                _set_requires_grad(critic, False)
                generator_optimizer.zero_grad(set_to_none=True)
                noise = sample_noise(
                    count,
                    generator.latent_dimension,
                    generator.noise_distribution,
                    device=device,
                    uniform_minimum=generator.noise_minimum,
                    uniform_maximum=generator.noise_maximum,
                )
                fake_observations = generator(conditions, noise)
                second_noise = None
                second_observations = None
                if float(objective["mode_seeking_weight"]) > 0.0:
                    second_noise = sample_noise(
                        count,
                        generator.latent_dimension,
                        generator.noise_distribution,
                        device=device,
                        uniform_minimum=generator.noise_minimum,
                        uniform_maximum=generator.noise_maximum,
                    )
                    second_observations = generator(
                        conditions, second_noise
                    )
                generator_objective = (
                    generator_wasserstein_regularized
                    if objective_name == "wasserstein_gp"
                    else generator_logistic_regularized
                )
                generator_loss = generator_objective(
                    critic,
                    conditions,
                    real_observations,
                    fake_observations,
                    noise,
                    second_observations=second_observations,
                    second_noise=second_noise,
                    mode_seeking_weight=float(
                        objective["mode_seeking_weight"]
                    ),
                    mode_seeking_epsilon=float(
                        objective["mode_seeking_epsilon"]
                    ),
                    moment_matching_weight=float(
                        objective["moment_matching_weight"]
                    ),
                )
                _assert_finite_loss(
                    generator_loss.total, "generator loss"
                )
                generator_loss.total.backward()
                _assert_finite_gradients(
                    generator, "generator"
                )
                generator_optimizer.step()
                _set_requires_grad(critic, True)
                _update_ema(
                    ema_generator,
                    generator,
                    float(training["ema_decay"]),
                )
                generator_steps += 1
                generator_update_count += 1
                for name, value in (
                    ("generator_loss", generator_loss.total),
                    (
                        "generator_adversarial",
                        generator_loss.adversarial,
                    ),
                    (
                        "mode_seeking_ratio",
                        generator_loss.mode_seeking_ratio,
                    ),
                    (
                        "moment_matching",
                        generator_loss.moment_matching,
                    ),
                ):
                    generator_totals[name] += float(
                        value.detach().cpu()
                    )

                if (
                    batch_position % log_interval == 0
                    or batch_position == len(train_loader)
                ):
                    elapsed = (
                        previous_elapsed
                        + time.time()
                        - started
                    )
                    progress = (
                        epoch - 1
                        + batch_position / len(train_loader)
                    ) / epochs
                    remaining = (
                        elapsed * (1.0 - progress) / progress
                        if progress > 0.0
                        else None
                    )
                    state.update(
                        {
                            "current_epoch": epoch,
                            "completed_batches": batch_position,
                            "total_batches": len(train_loader),
                            "generator_steps": generator_steps,
                            "critic_steps": critic_steps,
                            "elapsed_seconds": elapsed,
                            "estimated_seconds_remaining": (
                                remaining
                            ),
                            "recent": {
                                "generator_loss": float(
                                    generator_loss.total.detach().cpu()
                                ),
                                "generator_adversarial": float(
                                    generator_loss.adversarial
                                    .detach()
                                    .cpu()
                                ),
                                "mode_seeking_ratio": float(
                                    generator_loss.mode_seeking_ratio
                                    .detach()
                                    .cpu()
                                ),
                                "moment_matching": float(
                                    generator_loss.moment_matching
                                    .detach()
                                    .cpu()
                                ),
                                "critic_loss": float(
                                    critic_loss.total.detach().cpu()
                                ),
                                "wasserstein": float(
                                    critic_loss.wasserstein
                                    .detach()
                                    .cpu()
                                ),
                                "gradient_penalty": float(
                                    critic_loss.gradient_penalty
                                    .detach()
                                    .cpu()
                                ),
                                "gradient_norm_mean": float(
                                    critic_loss.gradient_norm_mean
                                    .detach()
                                    .cpu()
                                ),
                            },
                        }
                    )
                    _atomic_json(
                        run_directory / "state.json", state
                    )

            if generator_scheduler is not None:
                generator_scheduler.step()
            if critic_scheduler is not None:
                critic_scheduler.step()
            train_seconds = time.time() - epoch_started
            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "train_seconds": train_seconds,
                "train_examples_per_second": (
                    examples / train_seconds
                ),
                "generator_learning_rate": (
                    generator_optimizer.param_groups[0]["lr"]
                ),
                "critic_learning_rate": (
                    critic_optimizer.param_groups[0]["lr"]
                ),
                "train": {
                    **{
                        name: value / generator_update_count
                        for name, value in generator_totals.items()
                    },
                    **{
                        name: value / critic_update_count
                        for name, value in critic_totals.items()
                    },
                },
            }
            if (
                epoch % validation_interval == 0
                or epoch == epochs
            ):
                validation_metrics = evaluate_generator(
                    ema_generator,
                    validation_loader,
                    prepared.schema,
                    transform,
                    device,
                    evaluation,
                )
                epoch_record["validation"] = (
                    validation_metrics
                )
                score = float(
                    validation_metrics["selection_score"]
                )
                if (
                    best_validation_score is None
                    or score < best_validation_score
                ):
                    best_validation_score = score
                    best_epoch = epoch
                    _atomic_torch_save(
                        run_directory / "best_model.pt",
                        {
                            "epoch": epoch,
                            "generator_state_dict": (
                                generator.state_dict()
                            ),
                            "ema_generator_state_dict": (
                                ema_generator.state_dict()
                            ),
                            "validation": validation_metrics,
                            "generator": config["generator"],
                            "tensor_schema": (
                                prepared.schema.to_dict()
                            ),
                            "transform": transform.to_dict(),
                        },
                    )
            history.append(epoch_record)
            elapsed = (
                previous_elapsed + time.time() - started
            )
            _atomic_torch_save(
                checkpoint_path,
                _checkpoint_payload(
                    generator=generator,
                    critic=critic,
                    ema_generator=ema_generator,
                    generator_optimizer=generator_optimizer,
                    critic_optimizer=critic_optimizer,
                    generator_scheduler=generator_scheduler,
                    critic_scheduler=critic_scheduler,
                    epoch=epoch,
                    generator_steps=generator_steps,
                    critic_steps=critic_steps,
                    history=history,
                    elapsed_seconds=elapsed,
                    best_validation_score=best_validation_score,
                    best_epoch=best_epoch,
                    config_hash=config_hash,
                    dataset_fingerprint=prepared.fingerprint,
                    selection=selection_manifest,
                    transform=transform,
                    initial_hashes=initial_hashes,
                ),
            )
            _atomic_json(
                run_directory / "history.json", history
            )
            state.update(
                {
                    "completed_epochs": epoch,
                    "generator_steps": generator_steps,
                    "critic_steps": critic_steps,
                    "best_validation_score": (
                        best_validation_score
                    ),
                    "best_epoch": best_epoch,
                }
            )
            _atomic_json(
                run_directory / "state.json", state
            )
            validation_text = ""
            if "validation" in epoch_record:
                validation_text = (
                    "  validation_score="
                    f"{epoch_record['validation']['selection_score']:.6g}"
                )
            print(
                f"epoch {epoch:04d}/{epochs:04d}  "
                f"generator={epoch_record['train']['generator_loss']:.6g}  "
                f"critic={epoch_record['train']['total']:.6g}  "
                f"gp={epoch_record['train']['gradient_penalty']:.6g}"
                f"{validation_text}",
                flush=True,
            )

        last_ema_state_dict = copy.deepcopy(
            ema_generator.state_dict()
        )
        best_checkpoint = torch.load(
            run_directory / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        ema_generator.load_state_dict(
            best_checkpoint["ema_generator_state_dict"]
        )
        validation_metrics = evaluate_generator(
            ema_generator,
            validation_loader,
            prepared.schema,
            transform,
            device,
            evaluation,
        )
        test_metrics = evaluate_generator(
            ema_generator,
            test_loader,
            prepared.schema,
            transform,
            device,
            {**evaluation, "seed": int(evaluation["seed"]) + 1},
        )
        elapsed = previous_elapsed + time.time() - started
        training_seconds = sum(
            float(record["train_seconds"])
            for record in history
        )
        _atomic_torch_save(
            run_directory / "last_model.pt",
            {
                "epoch": epochs,
                "generator_state_dict": generator.state_dict(),
                "ema_generator_state_dict": last_ema_state_dict,
                "critic_state_dict": critic.state_dict(),
                "generator": config["generator"],
                "critic": config["critic"],
                "tensor_schema": prepared.schema.to_dict(),
                "transform": transform.to_dict(),
            },
        )
        parameter_counts = {
            "generator": sum(
                parameter.numel()
                for parameter in generator.parameters()
            ),
            "critic": sum(
                parameter.numel()
                for parameter in critic.parameters()
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "dataset_fingerprint": prepared.fingerprint,
            "dataset": prepared.manifest["identity"],
            "tensor_schema": prepared.schema.to_dict(),
            "selection": selection_manifest,
            "transform": transform.to_dict(),
            "config": config,
            "initial_hashes": initial_hashes,
            "runtime": {
                "device": str(device),
                "torch_version": torch.__version__,
                "elapsed_seconds": elapsed,
                "training_seconds": training_seconds,
                "generator_steps": generator_steps,
                "critic_steps": critic_steps,
                "parameter_counts": parameter_counts,
                "peak_gpu_memory_bytes": (
                    int(
                        torch.cuda.max_memory_allocated(
                            device
                        )
                    )
                    if device.type == "cuda"
                    else 0
                ),
            },
            "best_epoch": best_epoch,
            "evaluated_checkpoint": "best_model.pt",
            "evaluated_epoch": int(best_checkpoint["epoch"]),
            "metrics": {
                "validation": validation_metrics,
                "test": test_metrics,
            },
        }
        _atomic_json(run_directory / "run.json", manifest)
        state.update(
            {
                "status": "completed",
                "completed_epochs": epochs,
                "elapsed_seconds": elapsed,
            }
        )
        state.pop("current_epoch", None)
        state.pop("completed_batches", None)
        state.pop("total_batches", None)
        state.pop("estimated_seconds_remaining", None)
        _atomic_json(run_directory / "state.json", state)
        return ExperimentResult(run_directory, manifest)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        _atomic_json(run_directory / "state.json", state)
        raise
    except BaseException as error:
        state.update(
            {
                "status": "failed",
                "error": (
                    f"{type(error).__name__}: {error}"
                ),
            }
        )
        _atomic_json(run_directory / "state.json", state)
        raise

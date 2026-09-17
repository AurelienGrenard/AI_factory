"""Dataset-blind, resumable PyTorch training for composed pricing objectives."""

from __future__ import annotations

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

from learning.common.data.cache import PreparedPricingDataset
from learning.common.data.selection import DatasetSelection
from learning.common.data.torch_dataset import PricingBatchLoader
from learning.common.data.transforms import Standardization, fit_standardization
from learning.common.training import LossContext, build_composite_loss

from .metrics import evaluate
from .representations import build_pricing_model


@dataclass(frozen=True)
class ExperimentResult:
    run_directory: Path
    manifest: dict[str, Any]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def _loader(
    prepared: PreparedPricingDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> PricingBatchLoader:
    return PricingBatchLoader(
        prepared,
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
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


def _optimizer(model: torch.nn.Module, training: dict[str, Any]):
    name = str(training.get("optimizer", "adam")).lower()
    arguments = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **arguments)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **arguments)
    raise ValueError("training.optimizer must be adam or adamw")


def _scheduler(optimizer, training: dict[str, Any], epochs: int):
    name = str(training.get("scheduler", "constant")).lower()
    if name == "constant":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError("training.scheduler must be constant or cosine")


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
    best_validation_rmse: float | None,
    config_hash: str,
    dataset_fingerprint: str,
    selection: dict[str, object],
    transform: Standardization,
    initial_model_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "best_validation_rmse": best_validation_rmse,
        "config_hash": config_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "selection": selection,
        "transform": transform.to_dict(),
        "initial_model_hash": initial_model_hash,
    }


def train_experiment(
    prepared: PreparedPricingDataset,
    selection: DatasetSelection,
    config: dict[str, Any],
    run_directory: str | Path,
    *,
    resume: bool = False,
) -> ExperimentResult:
    training = config["training"]
    seed = int(training["seed"])
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    num_workers = int(training.get("num_workers", 0))
    validation_interval = int(training.get("validation_interval", 10))
    deterministic = bool(training.get("deterministic", True))
    if (
        epochs <= 0
        or batch_size <= 0
        or learning_rate <= 0.0
        or validation_interval <= 0
    ):
        raise ValueError(
            "Invalid positive training epochs, batch size, rate or interval"
        )
    if num_workers != 0:
        raise ValueError(
            "training.num_workers must be 0: the vectorized mmap batch loader does "
            "not use PyTorch DataLoader workers"
        )
    _seed_everything(seed, deterministic)
    device = _device(str(training.get("device", "auto")))
    transform = fit_standardization(
        prepared.features, prepared.values, selection.train_indices
    )
    for module_name in config.get("extensions", []):
        importlib.import_module(module_name)
    model = build_pricing_model(
        config["network"],
        config.get("representation"),
        prepared=prepared,
        train_indices=selection.train_indices,
        transform=transform,
    ).to(device)
    initial_model_hash = _model_hash(model)
    loss_function = build_composite_loss(config["loss"]["terms"])
    if loss_function.requires_input_gradients and not prepared.schema.gradients:
        raise ValueError("The configured objective requires gradients absent from this dataset")
    optimizer = _optimizer(model, training)
    scheduler = _scheduler(optimizer, training, epochs)
    pin_memory = False
    validation_loader = _loader(
        prepared,
        selection.validation_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = _loader(
        prepared,
        selection.test_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    feature_mean = torch.as_tensor(transform.feature_mean, device=device)
    feature_scale = torch.as_tensor(transform.feature_scale, device=device)
    value_mean = torch.as_tensor(transform.value_mean, device=device)
    value_scale = torch.as_tensor(transform.value_scale, device=device)
    gradient_indices_np = np.asarray(
        [gradient.wrt_index for gradient in prepared.schema.gradients], dtype=np.int64
    )
    gradient_indices = torch.as_tensor(
        gradient_indices_np, dtype=torch.long, device=device
    )
    gradient_scales = torch.as_tensor(
        transform.normalized_gradient_scales(gradient_indices_np), device=device
    )

    run_directory = Path(run_directory)
    checkpoint_path = run_directory / "checkpoint.pt"
    selection_manifest = selection.manifest()
    config_hash = _canonical_hash(config)
    if resume:
        if not run_directory.is_dir() or not checkpoint_path.is_file():
            raise ValueError(f"No resumable checkpoint in {run_directory}")
    else:
        run_directory.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists() or (run_directory / "initial_model.pt").exists():
            raise ValueError(f"Run directory already contains training state: {run_directory}")
        _atomic_torch_save(
            run_directory / "initial_model.pt",
            {
                "model_state_dict": model.state_dict(),
                "initial_model_hash": initial_model_hash,
                "network": config["network"],
                "representation": config.get(
                    "representation", {"name": "identity"}
                ),
                "seed": seed,
            },
        )

    history: list[dict[str, Any]] = []
    completed_epoch = 0
    global_step = 0
    previous_elapsed = 0.0
    best_validation_rmse: float | None = None
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["config_hash"] != config_hash:
            raise ValueError("Resume configuration differs from checkpoint")
        if checkpoint["dataset_fingerprint"] != prepared.fingerprint:
            raise ValueError("Resume dataset differs from checkpoint")
        if checkpoint["selection"] != selection_manifest:
            raise ValueError("Resume split or selected rows differ from checkpoint")
        if checkpoint["initial_model_hash"] != initial_model_hash:
            raise ValueError("Resume initial model identity differs from checkpoint")
        if checkpoint["transform"] != transform.to_dict():
            raise ValueError("Resume normalization differs from checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            if checkpoint["scheduler_state_dict"] is None:
                raise ValueError("Resume checkpoint has no configured scheduler")
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = list(checkpoint["history"])
        completed_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        previous_elapsed = float(checkpoint["elapsed_seconds"])
        best_validation_rmse = checkpoint["best_validation_rmse"]

    state = {
        "status": "running",
        "completed_epochs": completed_epoch,
        "total_epochs": epochs,
        "initial_model_hash": initial_model_hash,
        "dataset_fingerprint": prepared.fingerprint,
        "selection": selection_manifest,
    }
    _atomic_json(run_directory / "state.json", state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
            model.train()
            totals: dict[str, float] = {}
            seen = 0
            train_started = time.time()
            for batch in train_loader:
                raw_inputs = batch["features"].to(device, non_blocking=pin_memory)
                inputs = ((raw_inputs - feature_mean) / feature_scale).requires_grad_(
                    loss_function.requires_input_gradients
                )
                raw_targets = batch["values"].to(device, non_blocking=pin_memory)
                targets = (raw_targets - value_mean) / value_scale
                optimizer.zero_grad(set_to_none=True)
                predictions = model(inputs)
                predicted_gradients = None
                target_gradients = None
                raw_predicted_gradients = None
                raw_target_gradients = None
                if loss_function.requires_input_gradients:
                    predicted_gradients = torch.autograd.grad(
                        predictions.sum(), inputs, create_graph=True
                    )[0].index_select(1, gradient_indices)
                    raw_target_gradients = batch["gradients"].to(
                        device, non_blocking=pin_memory
                    )
                    target_gradients = raw_target_gradients * gradient_scales
                    raw_predicted_gradients = predicted_gradients / gradient_scales
                context = LossContext(
                    model=model,
                    normalized_inputs=inputs,
                    normalized_predictions=predictions,
                    normalized_targets=targets,
                    normalized_predicted_gradients=predicted_gradients,
                    normalized_target_gradients=target_gradients,
                    raw_inputs=raw_inputs,
                    raw_predictions=predictions * value_scale + value_mean,
                    raw_targets=raw_targets,
                    raw_predicted_gradients=raw_predicted_gradients,
                    raw_target_gradients=raw_target_gradients,
                    batch={name: value.to(device) for name, value in batch.items()},
                    epoch=epoch,
                    global_step=global_step,
                )
                loss, components = loss_function(context)
                loss.backward()
                optimizer.step()
                global_step += 1
                count = raw_inputs.shape[0]
                seen += count
                for name, value in components.items():
                    totals[name] = totals.get(name, 0.0) + value * count
            train_seconds = time.time() - train_started
            if scheduler is not None:
                scheduler.step()
            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_seconds": train_seconds,
                "train_examples_per_second": seen / train_seconds,
                "train": {name: value / seen for name, value in totals.items()},
            }
            if epoch % validation_interval == 0 or epoch == epochs:
                validation_metrics = evaluate(
                    model, validation_loader, prepared.schema, transform, device
                )
                epoch_record["validation"] = validation_metrics
                validation_rmse = float(validation_metrics["price_rmse"])
                if best_validation_rmse is None or validation_rmse < best_validation_rmse:
                    best_validation_rmse = validation_rmse
                    _atomic_torch_save(
                        run_directory / "best_model.pt",
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "validation": validation_metrics,
                        },
                    )
            history.append(epoch_record)
            elapsed = previous_elapsed + time.time() - started
            _atomic_torch_save(
                checkpoint_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    history=history,
                    elapsed_seconds=elapsed,
                    best_validation_rmse=best_validation_rmse,
                    config_hash=config_hash,
                    dataset_fingerprint=prepared.fingerprint,
                    selection=selection_manifest,
                    transform=transform,
                    initial_model_hash=initial_model_hash,
                ),
            )
            _atomic_json(run_directory / "history.json", history)
            state["completed_epochs"] = epoch
            _atomic_json(run_directory / "state.json", state)
            validation_text = ""
            if "validation" in epoch_record:
                validation_text = (
                    "  validation_price_rmse="
                    f"{epoch_record['validation']['price_rmse']:.6g}"
                )
            print(
                f"epoch {epoch:04d}/{epochs:04d}  "
                f"train_loss={epoch_record['train']['total']:.6g}"
                f"{validation_text}",
                flush=True,
            )

        validation_metrics = evaluate(
            model, validation_loader, prepared.schema, transform, device
        )
        test_metrics = evaluate(model, test_loader, prepared.schema, transform, device)
        elapsed = previous_elapsed + time.time() - started
        training_seconds = sum(
            float(epoch.get("train_seconds", 0.0)) for epoch in history
        )
        _atomic_torch_save(
            run_directory / "last_model.pt",
            {
                "model_state_dict": model.state_dict(),
                "network": config["network"],
                "representation": config.get(
                    "representation", {"name": "identity"}
                ),
                "tensor_schema": prepared.schema.to_dict(),
                "transform": transform.to_dict(),
                "epoch": epochs,
            },
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "dataset_fingerprint": prepared.fingerprint,
            "dataset": prepared.manifest["identity"],
            "tensor_schema": prepared.schema.to_dict(),
            "selection": selection_manifest,
            "transform": transform.to_dict(),
            "config": config,
            "initial_model_hash": initial_model_hash,
            "runtime": {
                "device": str(device),
                "torch_version": torch.__version__,
                "elapsed_seconds": elapsed,
                "global_steps": global_step,
                "parameter_count": parameter_count,
                "training_seconds": training_seconds,
                "training_examples_per_second": (
                    len(selection.train_indices) * len(history) / training_seconds
                ),
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
            },
            "metrics": {"validation": validation_metrics, "test": test_metrics},
        }
        _atomic_json(run_directory / "run.json", manifest)
        state.update({"status": "completed", "completed_epochs": epochs})
        _atomic_json(run_directory / "state.json", state)
        return ExperimentResult(run_directory, manifest)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        _atomic_json(run_directory / "state.json", state)
        raise
    except BaseException as error:
        state.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        _atomic_json(run_directory / "state.json", state)
        raise

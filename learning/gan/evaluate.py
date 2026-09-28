"""Evaluate a completed conditional GAN on another terminal dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from learning.common.data import prepare_terminal_dataset
from learning.common.data.terminal_torch import TerminalBatchLoader
from learning.common.data.terminal_transforms import TerminalTransform

from .metrics import evaluate_generator
from .networks import build_generator


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    manifest = json.loads(
        (arguments.run_directory / "run.json").read_text(
            encoding="utf-8"
        )
    )
    config = manifest["config"]
    evaluation = dict(config["evaluation"])
    if arguments.max_samples is not None:
        if arguments.max_samples <= 1:
            parser.error("--max-samples must exceed one")
        evaluation["max_samples"] = arguments.max_samples
    if arguments.seed is not None:
        evaluation["seed"] = arguments.seed
    batch_size = (
        int(arguments.batch_size)
        if arguments.batch_size is not None
        else int(config["training"]["batch_size"])
    )
    if batch_size <= 0:
        parser.error("--batch-size must be positive")

    source_schema = manifest["tensor_schema"]
    observable_names = list(source_schema["observable_names"])
    split_seed = int(config["data"]["split_seed"])
    prepared = prepare_terminal_dataset(
        arguments.dataset,
        cache_root=(
            arguments.cache_root
            if arguments.cache_root is not None
            else config["data"].get("cache_root")
        ),
        split_seed=split_seed,
        observables=observable_names,
    )
    if (
        prepared.schema.conditioning_names
        != tuple(source_schema["conditioning_names"])
        or prepared.schema.observable_names
        != tuple(observable_names)
    ):
        raise ValueError(
            "External terminal dataset tensor schema differs from the run"
        )

    if arguments.split == "all":
        indices = np.arange(
            prepared.schema.row_count, dtype=np.int64
        )
    else:
        code = {
            "train": 0,
            "validation": 1,
            "test": 2,
        }[arguments.split]
        indices = np.flatnonzero(
            prepared.split_codes == code
        ).astype(np.int64)
    maximum = int(evaluation["max_samples"])
    if indices.size > maximum:
        random = np.random.default_rng(int(evaluation["seed"]))
        indices = np.sort(
            random.choice(
                indices, size=maximum, replace=False
            )
        )
    loader = TerminalBatchLoader(
        prepared,
        indices,
        batch_size=batch_size,
        shuffle=False,
        seed=int(evaluation["seed"]),
    )

    device = _device(arguments.device)
    checkpoint = torch.load(
        arguments.run_directory / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    transform = TerminalTransform.from_dict(
        checkpoint["transform"]
    )
    generator = build_generator(
        checkpoint["generator"],
        condition_dimension=len(
            prepared.schema.conditioning_names
        ),
        observation_dimension=len(
            prepared.schema.observable_names
        ),
    ).to(device)
    generator.load_state_dict(
        checkpoint["ema_generator_state_dict"]
    )
    metrics = evaluate_generator(
        generator,
        loader,
        prepared.schema,
        transform,
        device,
        evaluation,
    )
    report = {
        "schema_version": 1,
        "run_directory": str(arguments.run_directory.resolve()),
        "dataset": str(arguments.dataset.resolve()),
        "dataset_fingerprint": prepared.fingerprint,
        "split": arguments.split,
        "metrics": metrics,
    }
    encoded = json.dumps(
        report, indent=2, sort_keys=True
    ) + "\n"
    output = (
        arguments.output
        if arguments.output is not None
        else arguments.run_directory / "external_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)
    print(f"evaluation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

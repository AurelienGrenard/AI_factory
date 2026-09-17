"""Prepare a published pricing dataset and train one generic deep pricer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learning.common.data import make_dataset_selection, prepare_pricing_dataset

from .config import load_config
from .engine import train_experiment


def _train_size(value: str) -> int | None:
    if value.lower() == "all":
        return None
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("train size must be positive or 'all'")
    return number


def _nested_override(arguments: argparse.Namespace) -> dict[str, Any]:
    override: dict[str, Any] = {}
    for source, section, target in (
        ("dataset", None, "dataset"),
        ("method", None, "method"),
        ("train_size", "data", "train_size"),
        ("split_seed", "data", "split_seed"),
        ("selection_seed", "data", "selection_seed"),
        ("epochs", "training", "epochs"),
        ("batch_size", "training", "batch_size"),
        ("learning_rate", "training", "learning_rate"),
        ("device", "training", "device"),
    ):
        value = getattr(arguments, source)
        if value is None:
            continue
        if section is None:
            override[target] = value
        else:
            override.setdefault(section, {})[target] = value
    return override


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--method", choices=("supervised", "sobolev", "custom"))
    parser.add_argument(
        "--train-size", "--sample-size", dest="train_size", type=_train_size,
        help="random training rows, or 'all'; validation and test remain fixed",
    )
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    config = load_config(arguments.config, _nested_override(arguments))
    data = config["data"]
    prepared = prepare_pricing_dataset(
        config["dataset"], cache_root=data.get("cache_root")
    )
    split_role = data.get("split_role")
    group_ids = None
    if split_role is not None and str(split_role).lower() != "row":
        try:
            role_index = prepared.schema.entity_roles.index(str(split_role))
        except ValueError as error:
            raise ValueError(
                f"split_role={split_role!r} is absent from dataset roles "
                f"{prepared.schema.entity_roles}"
            ) from error
        group_ids = prepared.entity_ordinals[:, role_index]
    selection = make_dataset_selection(
        prepared.schema.row_count,
        train_size=data.get("train_size"),
        split_seed=int(data["split_seed"]),
        selection_seed=int(data["selection_seed"]),
        validation_fraction=float(data["validation_fraction"]),
        test_fraction=float(data["test_fraction"]),
        group_ids=group_ids,
        validation_group_count=(
            None if group_ids is None else int(data["validation_groups"])
        ),
        test_group_count=None if group_ids is None else int(data["test_groups"]),
        core_group_count=(
            None
            if group_ids is None or data.get("core_groups") is None
            else int(data["core_groups"])
        ),
    )
    if arguments.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_directory = Path("artifacts/learning/runs") / f"{prepared.schema.database_id}-{config['method']}-{stamp}"
    else:
        run_directory = arguments.run_dir
    result = train_experiment(
        prepared, selection, config, run_directory, resume=arguments.resume
    )
    print(f"run: {result.run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

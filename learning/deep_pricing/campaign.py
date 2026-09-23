"""Run a resumable, paired deep-pricing experiment matrix."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, TextIO

import yaml

from learning.common.data import make_dataset_selection, prepare_pricing_dataset

from .config import load_config
from .engine import train_experiment


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    pair_id: str
    representation_id: str
    architecture_id: str
    train_size: int
    seed: int
    method: str
    gradient_weight: float | None
    config: dict[str, Any]


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _token(value: float) -> str:
    return format(value, "g").replace(".", "p").replace("-", "m")


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for name, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _deep_merge(result[name], value)
        else:
            result[name] = value
    return result


def expand_runs(specification: dict[str, Any]) -> list[RunSpec]:
    dataset = specification.get("dataset")
    architectures = specification.get("architectures")
    sizes = specification.get("train_sizes")
    seeds = specification.get("seeds")
    gradient_weights = specification.get("gradient_weights", [0.1, 1.0])
    if not dataset or not isinstance(architectures, list) or not architectures:
        raise ValueError("Campaign requires dataset and non-empty architectures")
    if not isinstance(sizes, list) or not sizes or not isinstance(seeds, list) or not seeds:
        raise ValueError("Campaign requires non-empty train_sizes and seeds")
    common_data = specification.get("data", {})
    common_training = specification.get("training", {})
    common_terms = specification.get("common_loss_terms", [])
    configured_representations = specification.get("representations")
    explicit_representations = configured_representations is not None
    representations = (
        [{"id": "raw", "name": "identity"}]
        if configured_representations is None
        else configured_representations
    )
    if not isinstance(representations, list) or not representations:
        raise ValueError("representations must be a non-empty list")
    runs: list[RunSpec] = []
    representation_ids: set[str] = set()
    for representation_entry in representations:
        if not isinstance(representation_entry, dict) or "id" not in representation_entry:
            raise ValueError("Each representation requires a unique id")
        representation_id = str(representation_entry["id"])
        if representation_id in representation_ids:
            raise ValueError(f"Duplicate representation id {representation_id!r}")
        representation_ids.add(representation_id)
        representation = {
            name: value for name, value in representation_entry.items() if name != "id"
        }
        prefix = f"{representation_id}__" if explicit_representations else ""
        for architecture in architectures:
            architecture_id = str(architecture["id"])
            network = architecture["network"]
            training = _deep_merge(common_training, architecture.get("training", {}))
            for size_value in sizes:
                train_size = int(size_value)
                for seed_value in seeds:
                    seed = int(seed_value)
                    pair_id = (
                        f"{prefix}{architecture_id}__n{train_size}__seed{seed}"
                    )
                    modes: list[tuple[str, float | None]] = [("price", None)]
                    modes.extend(
                        ("price_delta", float(weight)) for weight in gradient_weights
                    )
                    for method, gradient_weight in modes:
                        suffix = (
                            "price"
                            if gradient_weight is None
                            else f"price_delta_lambda{_token(gradient_weight)}"
                        )
                        run_id = f"{pair_id}__{suffix}"
                        terms: list[dict[str, Any]] = [
                            {"name": "value_mse", "weight": 1.0}
                        ]
                        if gradient_weight is not None:
                            terms.append(
                                {"name": "gradient_mse", "weight": gradient_weight}
                            )
                        terms.extend(common_terms)
                        override = {
                            "dataset": str(dataset),
                            "method": (
                                "supervised" if method == "price" else "sobolev"
                            ),
                            "data": {**common_data, "train_size": train_size},
                            "network": network,
                            "representation": representation,
                            "training": {**training, "seed": seed},
                            "loss": {"terms": terms},
                            "campaign": {
                                "run_id": run_id,
                                "pair_id": pair_id,
                                "representation_id": representation_id,
                                "architecture_id": architecture_id,
                                "method": method,
                                "gradient_weight": gradient_weight,
                            },
                        }
                        config = load_config(None, override)
                        runs.append(
                            RunSpec(
                                run_id=run_id,
                                pair_id=pair_id,
                                representation_id=representation_id,
                                architecture_id=architecture_id,
                                train_size=train_size,
                                seed=seed,
                                method=method,
                                gradient_weight=gradient_weight,
                                config=config,
                            )
                        )
    return runs


def _selection(prepared, config: dict[str, Any]):
    data = config["data"]
    split_role = str(data.get("split_role", "model"))
    try:
        role_index = prepared.schema.entity_roles.index(split_role)
    except ValueError as error:
        raise ValueError(
            f"split_role={split_role!r} is absent from {prepared.schema.entity_roles}"
        ) from error
    return make_dataset_selection(
        prepared.schema.row_count,
        train_size=int(data["train_size"]),
        split_seed=int(data["split_seed"]),
        selection_seed=int(data["selection_seed"]),
        group_ids=prepared.entity_ordinals[:, role_index],
        validation_group_count=int(data["validation_groups"]),
        test_group_count=int(data["test_groups"]),
        core_group_count=(
            None if data.get("core_groups") is None else int(data["core_groups"])
        ),
    )


def _collect_results(root: Path, runs: list[RunSpec]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in runs:
        result_path = root / "runs" / run.run_id / "run.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record: dict[str, object] = {
            "run_id": run.run_id,
            "pair_id": run.pair_id,
            "representation": run.representation_id,
            "architecture": run.architecture_id,
            "train_size": run.train_size,
            "seed": run.seed,
            "method": run.method,
            "gradient_weight": run.gradient_weight,
            "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            "training_seconds": result["runtime"].get("training_seconds"),
            "training_examples_per_second": result["runtime"].get(
                "training_examples_per_second"
            ),
            "parameter_count": result["runtime"].get("parameter_count"),
            "peak_gpu_memory_bytes": result["runtime"].get("peak_gpu_memory_bytes"),
            "initial_model_hash": result["initial_model_hash"],
            "train_digest": result["selection"]["digests"]["train"],
            "validation_digest": result["selection"]["digests"]["validation"],
            "test_digest": result["selection"]["digests"]["test"],
        }
        for split in ("validation", "test"):
            for name, value in result["metrics"][split].items():
                record[f"{split}_{name}"] = value
        records.append(record)
    if records:
        columns = list(records[0])
        with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    return records


def _verify_pairs(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["pair_id"]), []).append(record)
    failures: list[str] = []
    for pair_id, members in grouped.items():
        for field in (
            "initial_model_hash",
            "train_digest",
            "validation_digest",
            "test_digest",
        ):
            if len({str(member[field]) for member in members}) != 1:
                failures.append(f"{pair_id}: different {field}")
    return {"verified": not failures, "failures": failures, "pair_count": len(grouped)}


def run_campaign(config_path: Path) -> int:
    specification = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(specification, dict):
        raise ValueError("Campaign YAML must be a mapping")
    root = Path(specification["output_directory"])
    root.mkdir(parents=True, exist_ok=True)
    frozen_path = root / "campaign.yaml"
    if frozen_path.exists():
        frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
        if frozen != specification:
            raise ValueError("Campaign output already contains a different specification")
    else:
        frozen_path.write_text(yaml.safe_dump(specification, sort_keys=False), encoding="utf-8")
    runs = expand_runs(specification)
    prepared = prepare_pricing_dataset(
        specification["dataset"], cache_root=specification.get("cache_root")
    )
    state = {
        "status": "running",
        "total_runs": len(runs),
        "runs": {},
    }
    state_path = root / "campaign_state.json"
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["runs"] = previous.get("runs", {})
    _atomic_json(state_path, state)

    for position, run in enumerate(runs, start=1):
        run_directory = root / "runs" / run.run_id
        if (run_directory / "run.json").is_file():
            state["runs"][run.run_id] = "completed"
            continue
        selection = _selection(prepared, run.config)
        resume = (run_directory / "checkpoint.pt").is_file()
        state["current_run"] = run.run_id
        state["completed_runs"] = sum(
            status == "completed" for status in state["runs"].values()
        )
        state["runs"][run.run_id] = "running"
        _atomic_json(state_path, state)
        print(f"\n[{position}/{len(runs)}] {run.run_id}", flush=True)
        run_directory.mkdir(parents=True, exist_ok=True)
        log_mode = "a" if resume else "w"
        with (
            (run_directory / "stdout.log").open(log_mode, encoding="utf-8") as stdout_log,
            (run_directory / "stderr.log").open(log_mode, encoding="utf-8") as stderr_log,
        ):
            try:
                with redirect_stdout(_Tee(sys.stdout, stdout_log)), redirect_stderr(
                    _Tee(sys.stderr, stderr_log)
                ):
                    train_experiment(
                        prepared,
                        selection,
                        run.config,
                        run_directory,
                        resume=resume,
                    )
            except KeyboardInterrupt:
                state["runs"][run.run_id] = "interrupted"
                state["status"] = "interrupted"
                _atomic_json(state_path, state)
                raise
            except BaseException:
                state["runs"][run.run_id] = "failed"
                state["status"] = "failed"
                _atomic_json(state_path, state)
                raise
        state["runs"][run.run_id] = "completed"
        records = _collect_results(root, runs)
        state["pair_verification"] = _verify_pairs(records)
        _atomic_json(state_path, state)

    records = _collect_results(root, runs)
    verification = _verify_pairs(records)
    state.update(
        {
            "status": "completed" if verification["verified"] else "invalid",
            "completed_runs": len(records),
            "pair_verification": verification,
        }
    )
    state.pop("current_run", None)
    _atomic_json(state_path, state)
    print(f"campaign: {root}")
    print(f"completed runs: {len(records)}/{len(runs)}")
    print(f"paired invariants: {'OK' if verification['verified'] else 'FAILED'}")
    return 0 if verification["verified"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    return run_campaign(arguments.config)


if __name__ == "__main__":
    raise SystemExit(main())

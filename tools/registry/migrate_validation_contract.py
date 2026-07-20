"""Normalize persisted validation timing metadata to the current contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.schema import RESULT_YAML_KEYS


def _normalize(timing: dict[str, Any], row_count: int) -> dict[str, Any]:
    normalized = dict(timing)
    normalized.setdefault("benchmark_repetitions", 1)
    normalized.setdefault("benchmark_statistic", "median")
    normalized.setdefault("warmup_calls", 1)
    normalized.setdefault("benchmark_seconds", normalized["wall_seconds"])
    if "kernel_seconds" in normalized:
        normalized.setdefault("benchmark_kernel_seconds", normalized["kernel_seconds"])
    normalized.setdefault("benchmark_row_count", row_count)
    normalized.setdefault("benchmark_workload", "validation slice")
    preferred = (
        "wall_seconds", "kernel_seconds", "simulation_seconds", "payoff_seconds",
        "lsm_seconds", "benchmark_repetitions", "benchmark_statistic",
        "warmup_calls", "benchmark_seconds", "benchmark_kernel_seconds",
        "benchmark_row_count", "benchmark_workload",
    )
    return {
        key: normalized[key] for key in preferred if key in normalized
    }


def main() -> None:
    root = PROJECT_ROOT / "registry/validation/results"
    changed = 0
    for yaml_path in sorted(root.glob("*/*/specifications/*.yaml")):
        specification = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        specification["timing"] = _normalize(
            specification["timing"], int(specification["summary"]["row_count"])
        )
        specification = {
            key: specification[key]
            for key in RESULT_YAML_KEYS
            if key in specification
        }
        yaml_path.write_text(
            yaml.safe_dump(specification, sort_keys=False), encoding="utf-8"
        )
        json_path = yaml_path.parent.parent / "data" / f"{yaml_path.stem}.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if "timing" in data:
            data["timing"] = specification["timing"]
            json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        changed += 1
    print(f"Normalized validation timing metadata in {changed} result databases.")


if __name__ == "__main__":
    main()

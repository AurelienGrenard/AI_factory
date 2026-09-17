"""Compare two runs only after proving a shared dataset and held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    run = path / "run.json" if path.is_dir() else path
    return json.loads(run.read_text(encoding="utf-8"))


def compare(left: dict, right: dict) -> dict[str, object]:
    if left["dataset_fingerprint"] != right["dataset_fingerprint"]:
        raise ValueError("Runs use different prepared datasets")
    for split in ("validation", "test"):
        if left["selection"]["digests"][split] != right["selection"]["digests"][split]:
            raise ValueError(f"Runs use different {split} rows")
    differences: dict[str, float] = {}
    for name, left_value in left["metrics"]["test"].items():
        right_value = right["metrics"]["test"].get(name)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            differences[name] = float(right_value) - float(left_value)
    return {
        "dataset_fingerprint": left["dataset_fingerprint"],
        "left_train_count": left["selection"]["counts"]["train"],
        "right_train_count": right["selection"]["counts"]["train"],
        "test_metric_right_minus_left": differences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(compare(_load(arguments.left), _load(arguments.right)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

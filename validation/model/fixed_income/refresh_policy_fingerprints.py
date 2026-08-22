"""Revalidate fixed-income caches under the current comparison policy."""

from __future__ import annotations

from pathlib import Path

from validation.reference_price_dataset import (
    refresh_validation_policy_fingerprint,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "datasets/model/fixed_income"
REFERENCE_ROOT = ROOT / "validation/datasets/price/fixed_income"


def reference_path(source_path: Path) -> Path:
    """Map one model price dataset to its persistent validation cache."""

    relative = source_path.relative_to(SOURCE_ROOT)
    model_name, marker, *price_path = relative.parts
    if marker != "prices":
        raise ValueError(f"Unexpected fixed-income price path: {source_path}")
    return REFERENCE_ROOT / model_name / Path(*price_path)


def main() -> int:
    """Refresh every existing fixed-income policy fingerprint cache-only."""

    sources = tuple(
        sorted(
            path
            for path in SOURCE_ROOT.rglob("*.json")
            if "prices" in path.parts
        )
    )
    if not sources:
        raise RuntimeError("No fixed-income price dataset was found.")
    for source in sources:
        reference = reference_path(source)
        refresh_validation_policy_fingerprint(source, reference)
    print(f"Refreshed {len(sources)} fixed-income validation policies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

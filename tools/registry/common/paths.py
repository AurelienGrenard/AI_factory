"""Path helpers for the family-organized registry layout."""

from __future__ import annotations

from pathlib import Path

MODEL_FAMILIES = (
    "black_76",
    "rough_bergomi",
    "rough_heston",
    "black_scholes",
    "hull_white",
    "cir_plus_plus",
    "g2_plus_plus",
    "heston",
    "cir",
)

CURVE_FAMILIES = ("nelson_siegel",)

PRODUCT_FAMILIES = (
    "european_calls",
    "digital_calls",
    "caplets",
    "autocalls",
    "asian_arithmetic_calls",
    "asset_or_nothing_calls",
    "cash_or_nothing_calls",
    "lookback_floating_calls",
    "lookback_fixed_calls",
    "quadratic_power_calls",
    "volatility_swaps",
    "variance_swaps",
    "american_puts",
    "log_contracts",
    "down_and_out_calls",
    "down_and_in_calls",
    "up_and_out_calls",
    "up_and_in_calls",
    "vanilla_calls",
    "vanilla_puts",
    "interest_rate_swaps",
    "swaptions",
    "bermudan_swaptions",
    "zero_coupon_bonds",
)


def project_root_from(path: Path) -> Path:
    """Find the repository root from a file path."""
    resolved = path.resolve()
    for parent in resolved.parents:
        if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir():
            return parent
    raise ValueError(f"Could not locate project root from {path}")


def model_family(database_id: str) -> str:
    for family in MODEL_FAMILIES:
        if database_id == family or database_id.startswith(f"{family}_"):
            return family
    raise ValueError(f"Unknown model family for database id: {database_id}")


def product_family(database_id: str) -> str:
    for family in PRODUCT_FAMILIES:
        if database_id == family or database_id.startswith(f"{family}_"):
            return family
    raise ValueError(f"Unknown product family for database id: {database_id}")


def curve_family(database_id: str) -> str:
    for family in CURVE_FAMILIES:
        if database_id == family or database_id.startswith(f"{family}_"):
            return family
    raise ValueError(f"Unknown curve family for database id: {database_id}")


def result_families(database_id: str) -> tuple[str, str]:
    parts = database_id.split("__")
    if len(parts) < 2:
        raise ValueError(f"Result id must contain model and product ids: {database_id}")
    model = model_family(parts[0])
    for part in parts[1:]:
        try:
            return model, product_family(part)
        except ValueError:
            continue
    raise ValueError(f"Unknown product family for result id: {database_id}")


def registry_database_path(
    project_root: Path,
    tier: str,
    kind: str,
    section: str,
    database_id: str,
    suffix: str,
) -> Path:
    root = project_root / "registry" / tier
    if kind == "models":
        return root / kind / model_family(database_id) / section / f"{database_id}.{suffix}"
    if kind == "products":
        return root / kind / product_family(database_id) / section / f"{database_id}.{suffix}"
    if kind == "curves":
        return root / kind / curve_family(database_id) / section / f"{database_id}.{suffix}"
    if kind == "results":
        model, product = result_families(database_id)
        return root / kind / model / product / section / f"{database_id}.{suffix}"
    raise ValueError(f"Unsupported registry kind: {kind}")


def registry_relative_path(
    project_root: Path,
    tier: str,
    kind: str,
    section: str,
    database_id: str,
    suffix: str,
) -> str:
    return registry_database_path(
        project_root,
        tier,
        kind,
        section,
        database_id,
        suffix,
    ).relative_to(project_root).as_posix()

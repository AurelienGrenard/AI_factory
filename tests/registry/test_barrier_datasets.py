from __future__ import annotations

import json
from pathlib import Path

from tools.validation.audit import MIN_RELATIVE_PRICE, comparison_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = (
    "down_and_out_calls",
    "down_and_in_calls",
    "up_and_out_calls",
    "up_and_in_calls",
)
MODELS = (
    ("black_scholes", "black_scholes_01"),
    ("heston", "heston_03"),
    ("rough_bergomi", "rough_bergomi_02"),
    ("rough_heston", "rough_heston_01"),
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _product_rows(family: str) -> list[dict]:
    data = _json(
        PROJECT_ROOT
        / f"registry/production/products/{family}/data/{family}_01.json"
    )
    return [row["parameters"] for row in data["products"]]


def _result_rows(tier: str, model: str, product: str, database_id: str) -> list[dict]:
    return _json(
        PROJECT_ROOT
        / f"registry/{tier}/results/{model}/{product}/data/{database_id}.json"
    )["results"]


def test_barrier_product_pairs_share_contract_terms() -> None:
    rows = {family: _product_rows(family) for family in PRODUCTS}
    for index in range(1_000):
        terms = {
            (rows[family][index]["strike"], rows[family][index]["maturity"])
            for family in PRODUCTS
        }
        assert len(terms) == 1
        assert rows["down_and_out_calls"][index] == rows["down_and_in_calls"][index]
        assert rows["up_and_out_calls"][index] == rows["up_and_in_calls"][index]
        assert rows["down_and_out_calls"][index]["barrier"] < 1.0
        assert rows["up_and_out_calls"][index]["barrier"] > 1.0


def test_barrier_cpp_reproducibility_and_in_out_parity() -> None:
    for model, model_id in MODELS:
        production: dict[str, list[dict]] = {}
        for product in PRODUCTS:
            production_id = f"{model_id}__{product}_01__cpp_gpu_philox_01"
            production[product] = _result_rows(
                "production", model, product, production_id
            )[:100]
            prefix = f"{model_id}__first_100__{product}_01__first_100"
            gpu = _result_rows(
                "validation", model, product, f"{prefix}__cpp_gpu_philox_01"
            )
            cpu = _result_rows(
                "validation", model, product, f"{prefix}__cpp_cpu_philox_01"
            )
            for stored, regenerated, reference in zip(
                production[product], gpu, cpu, strict=True
            ):
                assert stored["outputs"]["price"] == regenerated["outputs"]["price"]
                tolerance = 2.0e-13 if model == "rough_heston" else 2.0e-14
                assert abs(
                    regenerated["outputs"]["price"] - reference["outputs"]["price"]
                ) < tolerance

        for index in range(100):
            down = (
                production["down_and_out_calls"][index]["outputs"]["price"]
                + production["down_and_in_calls"][index]["outputs"]["price"]
            )
            up = (
                production["up_and_out_calls"][index]["outputs"]["price"]
                + production["up_and_in_calls"][index]["outputs"]["price"]
            )
            assert abs(down - up) < 2.0e-14


def test_barrier_audit_notebooks_share_one_structure() -> None:
    signatures = set()
    for model, _ in MODELS:
        for product in PRODUCTS:
            notebook = _json(
                PROJECT_ROOT
                / f"notebooks/validation/{model}/2026-07-14_{model}_{product}_production_audit_01.ipynb"
            )
            signatures.add(
                tuple(
                    (
                        cell["cell_type"],
                        "".join(cell["source"]).splitlines()[0]
                        if cell["cell_type"] == "markdown"
                        else "code",
                    )
                    for cell in notebook["cells"][1:]
                )
            )
    assert len(signatures) == 1


def test_barrier_cuda_sources_have_explicit_headers() -> None:
    for model, _ in MODELS:
        for product in PRODUCTS:
            root = PROJECT_ROOT / f"src_cpp/ai_factory/cuda/{model}"
            assert (root / f"{product}.cuh").is_file()
            assert (root / f"{product}.cu").is_file()


def test_relative_errors_ignore_near_zero_prices() -> None:
    def row(identifier: str, price: float) -> dict:
        return {"id": identifier, "outputs": {"price": price}}

    ignored = comparison_metrics(
        "near zero", [row("1", 0.0)], [row("1", MIN_RELATIVE_PRICE / 10)],
        output="price",
    )
    retained = comparison_metrics(
        "material", [row("1", 1.0e-3)], [row("1", 1.001e-3)],
        output="price",
    )
    assert ignored["max rel error (%)"] is None
    assert retained["max rel error (%)"] is not None

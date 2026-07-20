from __future__ import annotations

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    ("black_scholes", "black_scholes_01"),
    ("heston", "heston_03"),
    ("rough_bergomi", "rough_bergomi_02"),
    ("rough_heston", "rough_heston_01"),
)
PRODUCTS = ("european_calls", "digital_calls")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_terminal_call_products_share_the_same_contract_grid() -> None:
    rows = []
    for product in PRODUCTS:
        data = _json(
            PROJECT_ROOT
            / f"registry/production/products/{product}/data/{product}_01.json"
        )
        assert data["row_count"] == 1_000
        rows.append([row["parameters"] for row in data["products"]])
    assert rows[0] == rows[1]
    assert all(1.0 / 12.0 <= row["maturity"] <= 3.0 for row in rows[0])
    assert all(
        math.exp(-0.2 * row["maturity"]) <= row["strike"]
        <= math.exp(0.2 * row["maturity"])
        for row in rows[0]
    )


def test_terminal_calls_have_explicit_sources_for_every_model() -> None:
    for model, _ in MODELS:
        for product in PRODUCTS:
            assert (PROJECT_ROOT / f"src_cpp/ai_factory/cpu/{model}/{product}.cpp").is_file()
            assert (PROJECT_ROOT / f"src_cpp/ai_factory/cuda/{model}/{product}.cu").is_file()
            assert (PROJECT_ROOT / f"src_python/ai_factory/pytorch/{model}/{product}.py").is_file()


def test_terminal_call_native_reproducibility_and_payoff_bounds() -> None:
    for model, model_id in MODELS:
        for product in PRODUCTS:
            prefix = f"{model_id}__first_100__{product}_01__first_100"
            root = PROJECT_ROOT / f"registry/validation/results/{model}/{product}/data"
            gpu = _json(root / f"{prefix}__cpp_gpu_philox_01.json")["results"]
            cpu = _json(root / f"{prefix}__cpp_cpu_philox_01.json")["results"]
            for gpu_row, cpu_row in zip(gpu, cpu, strict=True):
                assert abs(gpu_row["outputs"]["price"] - cpu_row["outputs"]["price"]) <= 1.0e-12
                assert gpu_row["outputs"]["price"] >= 0.0
                if product == "digital_calls":
                    assert gpu_row["outputs"]["price"] <= 1.0


def test_black_scholes_prices_are_consistent_with_closed_forms() -> None:
    model_data = _json(
        PROJECT_ROOT / "registry/production/models/black_scholes/data/black_scholes_01.json"
    )["models"]
    normal_cdf = lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
    for product in PRODUCTS:
        product_data = _json(
            PROJECT_ROOT / f"registry/production/products/{product}/data/{product}_01.json"
        )["products"]
        results = _json(next(
            (PROJECT_ROOT / f"registry/production/results/black_scholes/{product}/data").glob("*.json")
        ))["results"]
        absolute_errors = []
        call_z_scores = []
        for model_row, product_row, result in zip(model_data, product_data, results, strict=True):
            model = model_row["parameters"]
            terms = product_row["parameters"]
            spot, strike, maturity = model["spot"], terms["strike"], terms["maturity"]
            rate, dividend, volatility = (
                model["risk_free_rate"], model["dividend_yield"], model["volatility"]
            )
            root_t = math.sqrt(maturity)
            d1 = (
                math.log(spot / strike)
                + (rate - dividend + 0.5 * volatility * volatility) * maturity
            ) / (volatility * root_t)
            d2 = d1 - volatility * root_t
            if product == "european_calls":
                expected = (
                    spot * math.exp(-dividend * maturity) * normal_cdf(d1)
                    - strike * math.exp(-rate * maturity) * normal_cdf(d2)
                )
            else:
                expected = math.exp(-rate * maturity) * normal_cdf(d2)
            error = abs(result["outputs"]["price"] - expected)
            absolute_errors.append(error)
            standard_error = result["outputs"]["standard_error"]
            if product == "european_calls" and standard_error > 0.0:
                call_z_scores.append(error / standard_error)
        assert max(absolute_errors) < 0.02
        if call_z_scores:
            assert max(call_z_scores) < 4.0

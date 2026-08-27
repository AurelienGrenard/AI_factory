"""Command-line entry points for isolated Volterra price validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from validation.volterra.common import certify_price
from validation.volterra.dataset import validate_rough_bergomi_dataset
from validation.volterra.fukasawa_gatheral import (
    analyze_fukasawa_gatheral_probe,
)
from validation.volterra.rough_bergomi import (
    RoughBergomiParameters,
    exact_gaussian_european_option_price,
    hybrid_european_option_price,
)
from validation.volterra.rough_heston import (
    ExponentialKernel,
    RoughHestonParameters,
    lifted_heston_european_option_price,
    rough_heston_european_option_price,
)


def _option_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spot", type=float, default=1.0)
    parser.add_argument("--strike", type=float, default=1.0)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--dividend-yield", type=float, default=0.01)
    parser.add_argument("--side", choices=("call", "put"), default="call")
    parser.add_argument("--generated-price", type=float)
    parser.add_argument("--generated-standard-error", type=float, default=0.0)


def _certification_document(value: Any) -> dict[str, Any]:
    document = asdict(value)
    document["z_score"] = (
        document["z_score"] if math.isfinite(document["z_score"]) else None
    )
    return document


def _generated_estimate(arguments: argparse.Namespace) -> tuple[float | None, float]:
    if arguments.generated_json is None:
        return arguments.generated_price, arguments.generated_standard_error
    if arguments.generated_price is not None:
        raise ValueError(
            "Use either --generated-json or --generated-price, not both."
        )
    try:
        document = json.loads(arguments.generated_json.read_text(encoding="utf-8"))
        estimate = document[arguments.side]
        price = float(estimate["price"])
        standard_error = float(estimate["standard_error"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid rough-Heston CUDA probe JSON: {arguments.generated_json}"
        ) from error
    if not all(math.isfinite(value) for value in (price, standard_error)):
        raise ValueError("CUDA probe price and standard error must be finite.")
    if price < 0.0 or standard_error < 0.0:
        raise ValueError("CUDA probe moments must be non-negative.")
    return price, standard_error


def _rough_heston(arguments: argparse.Namespace) -> dict[str, Any]:
    generated_price, generated_standard_error = _generated_estimate(arguments)
    parameters = RoughHestonParameters(
        spot=arguments.spot,
        risk_free_rate=arguments.risk_free_rate,
        dividend_yield=arguments.dividend_yield,
        initial_variance=arguments.initial_variance,
        mean_reversion=arguments.mean_reversion,
        variance_drift=arguments.variance_drift,
        volatility_of_variance=arguments.volatility_of_variance,
        hurst_exponent=arguments.hurst,
        rho=arguments.rho,
    )
    coarse_time_steps = max(2, arguments.riccati_time_steps // 2)
    rough_coarse = rough_heston_european_option_price(
        parameters,
        arguments.strike,
        arguments.maturity,
        arguments.side,
        coarse_time_steps,
        arguments.fourier_cutoff,
        arguments.fourier_points,
    )
    rough_fine = rough_heston_european_option_price(
        parameters,
        arguments.strike,
        arguments.maturity,
        arguments.side,
        arguments.riccati_time_steps,
        arguments.fourier_cutoff,
        arguments.fourier_points,
    )
    output: dict[str, Any] = {
        "model": "rough_heston",
        "reference": "fractional_riccati_lewis",
        "fractional": {
            "coarse_time_steps": coarse_time_steps,
            "coarse_price": rough_coarse,
            "fine_time_steps": arguments.riccati_time_steps,
            "fine_price": rough_fine,
            "time_discretization_indicator": abs(rough_fine - rough_coarse),
        },
    }
    if arguments.kernel_json is not None:
        kernel = ExponentialKernel.from_json(arguments.kernel_json)
        coarse_fourier_points = max(3, (arguments.fourier_points + 1) // 2)
        if coarse_fourier_points % 2 == 0:
            coarse_fourier_points += 1
        lift_coarse = lifted_heston_european_option_price(
            parameters,
            kernel,
            arguments.strike,
            arguments.maturity,
            arguments.side,
            arguments.fourier_cutoff,
            coarse_fourier_points,
        )
        lift_fine = lifted_heston_european_option_price(
            parameters,
            kernel,
            arguments.strike,
            arguments.maturity,
            arguments.side,
            arguments.fourier_cutoff,
            arguments.fourier_points,
        )
        fourier_indicator = abs(lift_fine - lift_coarse)
        output["lift"] = {
            "factor_count": len(kernel.nodes),
            "coarse_fourier_points": coarse_fourier_points,
            "coarse_price": lift_coarse,
            "fine_fourier_points": arguments.fourier_points,
            "fine_price": lift_fine,
            "fourier_discretization_indicator": fourier_indicator,
            "kernel_price_bias": lift_fine - rough_fine,
        }
        if generated_price is not None:
            output["generated"] = {
                "price": generated_price,
                "standard_error": generated_standard_error,
            }
            output["cuda_vs_lift"] = _certification_document(
                certify_price(
                    generated_price,
                    generated_standard_error,
                    lift_fine,
                    numerical_allowance=fourier_indicator,
                )
            )
            output["cuda_vs_continuous_rough"] = _certification_document(
                certify_price(
                    generated_price,
                    generated_standard_error,
                    rough_fine,
                    numerical_allowance=abs(rough_fine - rough_coarse),
                )
            )
    elif generated_price is not None:
        output["generated"] = {
            "price": generated_price,
            "standard_error": generated_standard_error,
        }
        output["cuda_vs_continuous_rough"] = _certification_document(
            certify_price(
                generated_price,
                generated_standard_error,
                rough_fine,
                numerical_allowance=abs(rough_fine - rough_coarse),
            )
        )
    return output


def _parse_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "steps must be comma-separated integers"
        ) from error
    if (
        not steps
        or any(step < 1 for step in steps)
        or tuple(sorted(set(steps))) != steps
    ):
        raise argparse.ArgumentTypeError(
            "steps must be positive, unique and increasing"
        )
    return steps


def _rough_bergomi(arguments: argparse.Namespace) -> dict[str, Any]:
    parameters = RoughBergomiParameters(
        spot=arguments.spot,
        risk_free_rate=arguments.risk_free_rate,
        dividend_yield=arguments.dividend_yield,
        xi_0=arguments.xi_0,
        eta=arguments.eta,
        hurst_exponent=arguments.hurst,
        rho=arguments.rho,
    )
    rows: list[dict[str, Any]] = []
    for step_count in arguments.steps:
        hybrid = hybrid_european_option_price(
            parameters,
            arguments.strike,
            arguments.maturity,
            arguments.side,
            step_count,
            arguments.antithetic_pairs,
            arguments.seed,
            arguments.batch_pairs,
            "direct",
        )
        fft = hybrid_european_option_price(
            parameters,
            arguments.strike,
            arguments.maturity,
            arguments.side,
            step_count,
            arguments.antithetic_pairs,
            arguments.seed,
            arguments.batch_pairs,
            "fft",
        )
        exact = exact_gaussian_european_option_price(
            parameters,
            arguments.strike,
            arguments.maturity,
            arguments.side,
            step_count,
            arguments.antithetic_pairs,
            arguments.seed + 1,
            arguments.batch_pairs,
        )
        rows.append(
            {
                "time_steps": step_count,
                "hybrid_direct": asdict(hybrid),
                "hybrid_numpy_fft": asdict(fft),
                "exact_gaussian_grid": asdict(exact),
                "direct_vs_numpy_fft": _certification_document(
                    certify_price(
                        hybrid.price,
                        0.0,
                        fft.price,
                        0.0,
                        numerical_allowance=2.0e-12,
                    )
                ),
                "hybrid_vs_exact_gaussian": _certification_document(
                    certify_price(
                        hybrid.price,
                        hybrid.standard_error,
                        exact.price,
                        exact.standard_error,
                    )
                ),
            }
        )
        if len(rows) > 1:
            previous = rows[-2]
            rows[-1]["hybrid_refinement_difference"] = (
                hybrid.price - previous["hybrid_direct"]["price"]
            )
            rows[-1]["exact_gaussian_refinement_difference"] = (
                exact.price - previous["exact_gaussian_grid"]["price"]
            )
    output: dict[str, Any] = {
        "model": "rough_bergomi",
        "reference": "joint_gaussian_cholesky_conditional_black",
        "rows": rows,
    }
    if arguments.generated_price is not None:
        finest = rows[-1]["hybrid_direct"]
        output["cuda_vs_hybrid"] = _certification_document(
            certify_price(
                arguments.generated_price,
                arguments.generated_standard_error,
                finest["price"],
                finest["standard_error"],
            )
        )
    return output


def _rough_bergomi_dataset(arguments: argparse.Namespace) -> dict[str, Any]:
    return validate_rough_bergomi_dataset(
        arguments.price_json,
        arguments.model_json,
        arguments.product_json,
        arguments.side,
        arguments.row_offset,
        arguments.limit,
        arguments.target_dt,
        arguments.antithetic_pairs,
        arguments.batch_pairs,
        arguments.seed,
        arguments.exact_grid,
        arguments.exact_max_steps,
    )


def _rough_sabr_fukasawa_gatheral(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    return analyze_fukasawa_gatheral_probe(
        arguments.cuda_json,
        arguments.formula_allowance,
        arguments.standard_error_multiplier,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent offline validation of Volterra option prices."
    )
    subparsers = parser.add_subparsers(dest="model", required=True)

    heston = subparsers.add_parser("rough-heston")
    _option_arguments(heston)
    heston.add_argument("--initial-variance", type=float, default=0.04)
    heston.add_argument("--mean-reversion", type=float, default=0.30)
    heston.add_argument("--variance-drift", type=float, default=0.02)
    heston.add_argument("--volatility-of-variance", type=float, default=0.30)
    heston.add_argument("--hurst", type=float, default=0.10)
    heston.add_argument("--rho", type=float, default=-0.70)
    heston.add_argument("--kernel-json", type=Path)
    heston.add_argument("--generated-json", type=Path)
    heston.add_argument("--riccati-time-steps", type=int, default=1024)
    heston.add_argument("--fourier-cutoff", type=float, default=80.0)
    heston.add_argument("--fourier-points", type=int, default=1601)
    heston.set_defaults(execute=_rough_heston)

    bergomi = subparsers.add_parser("rough-bergomi")
    _option_arguments(bergomi)
    bergomi.add_argument("--xi-0", type=float, default=0.04)
    bergomi.add_argument("--eta", type=float, default=1.70)
    bergomi.add_argument("--hurst", type=float, default=0.10)
    bergomi.add_argument("--rho", type=float, default=-0.70)
    bergomi.add_argument("--steps", type=_parse_steps, default=(90, 180, 360))
    bergomi.add_argument("--antithetic-pairs", type=int, default=16_384)
    bergomi.add_argument("--batch-pairs", type=int, default=2048)
    bergomi.add_argument("--seed", type=int, default=920_001_001)
    bergomi.set_defaults(execute=_rough_bergomi)

    bergomi_dataset = subparsers.add_parser("rough-bergomi-dataset")
    bergomi_dataset.add_argument("--price-json", type=Path, required=True)
    bergomi_dataset.add_argument("--model-json", type=Path, required=True)
    bergomi_dataset.add_argument("--product-json", type=Path, required=True)
    bergomi_dataset.add_argument(
        "--side", choices=("call", "put"), required=True
    )
    bergomi_dataset.add_argument("--row-offset", type=int, default=0)
    bergomi_dataset.add_argument("--limit", type=int, default=1)
    bergomi_dataset.add_argument("--target-dt", type=float, default=1.0 / 360.0)
    bergomi_dataset.add_argument("--antithetic-pairs", type=int, default=16_384)
    bergomi_dataset.add_argument("--batch-pairs", type=int, default=2048)
    bergomi_dataset.add_argument("--seed", type=int, default=4_820_001)
    bergomi_dataset.add_argument("--exact-grid", action="store_true")
    bergomi_dataset.add_argument("--exact-max-steps", type=int, default=360)
    bergomi_dataset.set_defaults(execute=_rough_bergomi_dataset)

    rough_sabr = subparsers.add_parser("rough-sabr-fukasawa-gatheral")
    rough_sabr.add_argument("--cuda-json", type=Path, required=True)
    rough_sabr.add_argument("--formula-allowance", type=float, default=0.02)
    rough_sabr.add_argument(
        "--standard-error-multiplier", type=float, default=4.0
    )
    rough_sabr.add_argument("--output-json", type=Path)
    rough_sabr.set_defaults(execute=_rough_sabr_fukasawa_gatheral)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = arguments.execute(arguments)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    output_path = getattr(arguments, "output_json", None)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


__all__ = ("build_parser", "main")

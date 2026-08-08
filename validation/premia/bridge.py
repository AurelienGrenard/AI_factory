"""Run deterministic Premia pricing batches through a private 64-bit Wine prefix."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Protocol, Sequence

from validation.premia.build_runner import build_runner, project_root


@dataclass(frozen=True)
class PremiaInput:
    """One generic runner row after model/product-specific validation."""

    row_id: str
    values: tuple[float, ...]

    def protocol_line(self) -> str:
        return _protocol_line(self.row_id, self.values)


@dataclass(frozen=True)
class PremiaResult:
    """One Premia price and its method-reported standard error, if any."""

    price: float
    standard_error: float


@dataclass(frozen=True)
class PremiaRowFailure:
    """One row rejected by Premia while the remainder of the batch continues."""

    status: int
    reason: str


_ROW_FAILURE_REASONS = {
    10: "runner field-count mismatch",
    11: "non-finite runner input",
    12: "input outside the adapter or Premia method domain",
    13: "required Premia parameter object is unavailable",
    14: "non-finite price or standard error",
    15: "unsupported Premia model mapping",
    16: "temporary fitted-curve file could not be written",
}


@dataclass(frozen=True)
class HestonEuropeanOptionInput:
    """One Heston row in AI_factory's continuous-rate convention."""

    row_id: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    initial_variance: float
    kappa: float
    theta: float
    gamma: float
    rho: float
    strike: float
    maturity: float

    def protocol_line(self) -> str:
        """Serialize one validated row for the private runner protocol."""

        return _protocol_line(
            self.row_id,
            (
                self.spot,
                self.risk_free_rate,
                self.dividend_yield,
                self.initial_variance,
                self.kappa,
                self.theta,
                self.gamma,
                self.rho,
                self.strike,
                self.maturity,
            ),
        )


class _ProtocolRow(Protocol):
    row_id: str

    def protocol_line(self) -> str: ...


def _protocol_line(row_id: str, values: tuple[float, ...]) -> str:
    """Serialize one finite, whitespace-free row consistently."""

    if not row_id or any(character.isspace() for character in row_id):
        raise ValueError("Premia row ids must be non-empty and contain no whitespace.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Premia row '{row_id}' contains a non-finite value.")
    return "\t".join((row_id, *(format(value, ".17g") for value in values)))


@dataclass(frozen=True)
class MertonEuropeanOptionInput:
    """One Merton row in AI_factory's continuous-rate convention."""

    row_id: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    jump_intensity: float
    jump_log_mean: float
    jump_log_volatility: float
    strike: float
    maturity: float

    def protocol_line(self) -> str:
        """Serialize one validated row for the private runner protocol."""

        return _protocol_line(
            self.row_id,
            (
                self.spot,
                self.risk_free_rate,
                self.dividend_yield,
                self.volatility,
                self.jump_intensity,
                self.jump_log_mean,
                self.jump_log_volatility,
                self.strike,
                self.maturity,
            ),
        )


@dataclass(frozen=True)
class KouEuropeanOptionInput:
    """One Kou row in AI_factory's continuous-rate convention."""

    row_id: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    jump_intensity: float
    up_probability: float
    positive_jump_rate: float
    negative_jump_rate: float
    strike: float
    maturity: float

    def protocol_line(self) -> str:
        """Serialize one validated row for the private runner protocol."""

        return _protocol_line(
            self.row_id,
            (
                self.spot,
                self.risk_free_rate,
                self.dividend_yield,
                self.volatility,
                self.jump_intensity,
                self.up_probability,
                self.positive_jump_rate,
                self.negative_jump_rate,
                self.strike,
                self.maturity,
            ),
        )


@dataclass(frozen=True)
class SchobelZhuEuropeanOptionInput:
    """One Schobel-Zhu row in AI_factory's continuous-rate convention."""

    row_id: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    initial_volatility: float
    mean_reversion: float
    long_run_volatility: float
    volatility_of_volatility: float
    correlation: float
    strike: float
    maturity: float

    def protocol_line(self) -> str:
        """Serialize one validated row for the private runner protocol."""

        return _protocol_line(
            self.row_id,
            (
                self.spot,
                self.risk_free_rate,
                self.dividend_yield,
                self.initial_volatility,
                self.mean_reversion,
                self.long_run_volatility,
                self.volatility_of_volatility,
                self.correlation,
                self.strike,
                self.maturity,
            ),
        )


def _windows_path(path: Path) -> str:
    """Map one absolute WSL path through Wine's standard Z: drive."""

    absolute = path.resolve()
    return "Z:" + str(absolute).replace("/", "\\")


def _wine_environment(repository: Path) -> dict[str, str]:
    """Return the isolated environment shared by all Premia executions."""

    environment = os.environ.copy()
    environment.update(
        {
            "WINEARCH": "win64",
            "WINEPREFIX": str(repository / "build" / "premia-wine"),
            "WINEDEBUG": "-all",
        }
    )
    return environment


def _ensure_wine_prefix(repository: Path, environment: dict[str, str]) -> None:
    """Initialize the untracked 64-bit prefix on first use."""

    prefix = Path(environment["WINEPREFIX"])
    if (prefix / "system.reg").is_file():
        return
    wineboot = shutil.which("wineboot")
    if wineboot is None:
        raise RuntimeError("Wine is required; install the wine and wine64 packages.")
    subprocess.run(
        (wineboot, "-u"),
        cwd=repository,
        env=environment,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def price_rows(
    rows: Sequence[_ProtocolRow], mode: str, method: str | None = None
) -> dict[str, PremiaResult]:
    """Price one complete batch through one model/product Premia context."""

    prices, failures = price_rows_partial(rows, mode, method)
    if failures:
        row_id, failure = next(iter(failures.items()))
        raise RuntimeError(
            f"Premia row '{row_id}' failed with status {failure.status}: "
            f"{failure.reason}."
        )
    return prices


def price_rows_partial(
    rows: Sequence[_ProtocolRow], mode: str, method: str | None = None
) -> tuple[dict[str, PremiaResult], dict[str, PremiaRowFailure]]:
    """Return every successful Premia row and retain row-local failures."""

    if not rows:
        return {}, {}
    row_ids = [row.row_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Premia batch row ids must be unique.")

    repository = project_root()
    package = repository / "validation" / "premia" / "premia-19-win64"
    runner = build_runner(repository)
    wine = shutil.which("wine")
    if wine is None:
        raise RuntimeError("Wine is required; install the wine and wine64 packages.")
    environment = _wine_environment(repository)
    _ensure_wine_prefix(repository, environment)
    input_text = "\n".join(row.protocol_line() for row in rows) + "\n"
    command = [wine, str(runner), _windows_path(package), mode]
    if method is not None:
        command.append(method)
    completed = subprocess.run(
        command,
        cwd=package / "bin",
        env=environment,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Premia runner failed with status {completed.returncode}: {diagnostic}"
        )

    prices: dict[str, PremiaResult] = {}
    failures: dict[str, PremiaRowFailure] = {}
    for line in completed.stdout.splitlines():
        if not line.startswith("RESULT\t"):
            continue
        fields = line.split("\t")
        if len(fields) not in {4, 5}:
            raise RuntimeError(f"Malformed Premia result line: {line}")
        _, row_id, status_text, price_text, *error_text = fields
        if row_id in prices or row_id in failures:
            raise RuntimeError(f"Premia returned duplicate row id '{row_id}'.")
        status = int(status_text)
        if status != 0:
            reason = _ROW_FAILURE_REASONS.get(
                status,
                "Premia pricing method returned a nonzero status",
            )
            failures[row_id] = PremiaRowFailure(status, reason)
            continue
        price = float(price_text)
        if not math.isfinite(price):
            raise RuntimeError(f"Premia row '{row_id}' returned a non-finite price.")
        standard_error = float(error_text[0]) if error_text else 0.0
        if not math.isfinite(standard_error) or standard_error < 0.0:
            raise RuntimeError(
                f"Premia row '{row_id}' returned an invalid standard error."
            )
        prices[row_id] = PremiaResult(price, standard_error)
    returned = set(prices).union(failures)
    missing = set(row_ids).difference(returned)
    extra = returned.difference(row_ids)
    if missing or extra:
        raise RuntimeError(
            f"Premia protocol mismatch; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return prices, failures


def _price_european_options(
    rows: Sequence[_ProtocolRow], mode: str
) -> dict[str, float]:
    """Preserve the typed European-option façade used by existing modules."""

    return {
        row_id: result.price
        for row_id, result in price_rows(rows, mode).items()
    }


def price_heston_european_options(
    rows: Sequence[HestonEuropeanOptionInput], option_type: str
) -> dict[str, float]:
    if option_type not in {"call", "put"}:
        raise ValueError("Premia Heston option_type must be 'call' or 'put'.")
    return _price_european_options(rows, f"heston_european_{option_type}")


def price_merton_european_options(
    rows: Sequence[MertonEuropeanOptionInput], option_type: str
) -> dict[str, float]:
    if option_type not in {"call", "put"}:
        raise ValueError("Premia Merton option_type must be 'call' or 'put'.")
    return _price_european_options(rows, f"merton_european_{option_type}")


def price_kou_european_options(
    rows: Sequence[KouEuropeanOptionInput], option_type: str
) -> dict[str, float]:
    if option_type not in {"call", "put"}:
        raise ValueError("Premia Kou option_type must be 'call' or 'put'.")
    return _price_european_options(rows, f"kou_european_{option_type}")


def price_schobel_zhu_european_options(
    rows: Sequence[SchobelZhuEuropeanOptionInput], option_type: str
) -> dict[str, float]:
    if option_type not in {"call", "put"}:
        raise ValueError("Premia Schobel-Zhu option_type must be 'call' or 'put'.")
    return _price_european_options(rows, f"schobel_zhu_european_{option_type}")

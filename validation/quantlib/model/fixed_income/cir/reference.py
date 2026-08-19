"""QuantLib adapter for the standalone CIR short-rate model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number


@dataclass(frozen=True)
class StableCoxIngersollRoss:
    """Use QuantLib's OTM tail and recover the other side through parity."""

    model: ql.CoxIngersollRoss
    initial_state: float

    def discountBondOption(
        self,
        option_type: int,
        strike: float,
        option_expiry: float,
        bond_maturity: float,
    ) -> float:
        """Stabilize QuantLib when its deep-ITM direct tail loses all mass."""

        call = self.model.discountBondOption(
            ql.Option.Call, strike, option_expiry, bond_maturity
        )
        put = self.model.discountBondOption(
            ql.Option.Put, strike, option_expiry, bond_maturity
        )
        expiry_bond = self.model.discountBond(
            0.0, option_expiry, self.initial_state
        )
        underlying_bond = self.model.discountBond(
            0.0, bond_maturity, self.initial_state
        )
        parity = underlying_bond - strike * expiry_bond
        if parity >= 0.0:
            put = max(put, 0.0)
            call = parity + put
        else:
            call = max(call, 0.0)
            put = call - parity
        return call if option_type == ql.Option.Call else put


def quantlib_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    _: Mapping[str, Any],
) -> StableCoxIngersollRoss:
    """Map one workbench CIR row to QuantLib's analytical model."""

    if curve is not None:
        raise ValueError("Standalone CIR prices must not reference a curve dataset.")
    context = "CIR model"
    initial_state = finite_number(model, "initial_state", context)
    if initial_state < 0.0:
        raise ValueError(f"{context}: initial_state must be non-negative.")
    mean_reversion = positive_number(model, "mean_reversion", context)
    long_term_mean = positive_number(model, "long_term_mean", context)
    volatility = positive_number(model, "volatility", context)

    # QuantLib's constructor enforces Feller, while its analytical bond-option
    # formula remains valid when the boundary is attainable.  Construct a safe
    # seed object, then install the requested parameter vector in QuantLib's
    # native [theta, k, sigma, r0] order.
    seed_volatility = min(
        volatility,
        0.5 * math.sqrt(2.0 * mean_reversion * long_term_mean),
    )
    reference = ql.CoxIngersollRoss(
        initial_state,
        long_term_mean,
        mean_reversion,
        seed_volatility,
    )
    reference.setParams(
        ql.Array(
            [long_term_mean, mean_reversion, volatility, initial_state]
        )
    )
    return StableCoxIngersollRoss(reference, initial_state)


__all__ = ("StableCoxIngersollRoss", "quantlib_model")

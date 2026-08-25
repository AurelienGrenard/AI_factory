"""Explicit prototype manifest for thin Monte Carlo pricing bindings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Binding:
    model: str
    model_display: str
    product: str
    product_type: str
    pricing_policy: str
    schedule: str
    time_kind: str
    sided: bool = True


# Schedule selection is deliberate product-family metadata. It is never
# inferred merely because a dynamics happens to satisfy several concepts.
BINDINGS = (
    Binding(
        model="merton",
        model_display="Merton",
        product="european_option",
        product_type="EuropeanOption",
        pricing_policy="EuropeanOptionPricingPolicy",
        schedule="simulation::ExactTransitionTerminalSchedule",
        time_kind="exact",
    ),
    Binding(
        model="merton",
        model_display="Merton",
        product="asian_option",
        product_type="AsianOption",
        pricing_policy="AsianOptionPricingPolicy",
        schedule="simulation::FixedStepDenseSchedule",
        time_kind="fixed",
    ),
    Binding(
        model="cev",
        model_display="CEV",
        product="european_option",
        product_type="EuropeanOption",
        pricing_policy="EuropeanOptionPricingPolicy",
        schedule="simulation::FixedStepTerminalSchedule",
        time_kind="fixed",
    ),
    Binding(
        model="cev",
        model_display="CEV",
        product="asian_option",
        product_type="AsianOption",
        pricing_policy="AsianOptionPricingPolicy",
        schedule="simulation::FixedStepDenseSchedule",
        time_kind="fixed",
    ),
)

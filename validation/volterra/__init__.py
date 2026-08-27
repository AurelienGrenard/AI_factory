"""Independent offline references for Volterra volatility models.

This package deliberately has no dependency on ``src`` or on CUDA code.  Its
purpose is to expose slow, auditable reference algorithms for validation.
"""

from validation.volterra.common import (
    MonteCarloEstimate,
    PriceCertification,
    certify_price,
)
from validation.volterra.fukasawa_gatheral import (
    fukasawa_gatheral_implied_volatility,
)
from validation.volterra.rough_bergomi import RoughBergomiParameters
from validation.volterra.rough_heston import (
    ExponentialKernel,
    RoughHestonParameters,
)
from validation.volterra.quadratic_rough_heston import (
    QuadraticRoughHestonParameters,
)
from validation.volterra.rough_sabr import RoughSabrParameters

__all__ = (
    "ExponentialKernel",
    "MonteCarloEstimate",
    "PriceCertification",
    "QuadraticRoughHestonParameters",
    "RoughBergomiParameters",
    "RoughHestonParameters",
    "RoughSabrParameters",
    "certify_price",
    "fukasawa_gatheral_implied_volatility",
)

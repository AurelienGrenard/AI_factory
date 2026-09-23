"""Pricing-aware model representations composed around generic networks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from learning.common.data.cache import PreparedPricingDataset
from learning.common.data.contracts import PricingTensorSchema
from learning.common.data.transforms import Standardization
from learning.common.networks import build_network


class UnitSpotHomogeneousModel(nn.Module):
    """Represent a homogeneous scalar price as P(S, K, z) = S p(K / S, z).

    The outer training engine still consumes and differentiates the original
    normalized coordinates.  Consequently, automatic differentiation through
    this wrapper returns the price delta with respect to the original spot.
    """

    def __init__(
        self,
        core: nn.Module,
        *,
        spot_index: int,
        strike_index: int,
        transform: Standardization,
    ) -> None:
        super().__init__()
        self.core = core
        self.spot_index = spot_index
        self.strike_index = strike_index
        retained = [
            index
            for index in range(transform.feature_mean.size)
            if index != spot_index
        ]
        if strike_index not in retained:
            raise ValueError("Spot and strike must be different input coordinates")
        self.retained_indices = tuple(retained)
        self.moneyness_index = retained.index(strike_index)
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(transform.feature_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(transform.feature_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "derived_mean",
            torch.as_tensor(transform.feature_mean[retained], dtype=torch.float32),
        )
        self.register_buffer(
            "derived_scale",
            torch.as_tensor(transform.feature_scale[retained], dtype=torch.float32),
        )
        self.register_buffer(
            "value_mean",
            torch.as_tensor(transform.value_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "value_scale",
            torch.as_tensor(transform.value_scale, dtype=torch.float32),
        )

    def forward(self, normalized_inputs: torch.Tensor) -> torch.Tensor:
        raw_inputs = normalized_inputs * self.feature_scale + self.feature_mean
        spot = raw_inputs[:, self.spot_index : self.spot_index + 1]
        strike = raw_inputs[:, self.strike_index : self.strike_index + 1]
        moneyness = strike / spot
        derived_columns = [
            (
                moneyness[:, 0]
                if position == self.moneyness_index
                else raw_inputs[:, source_index]
            )
            for position, source_index in enumerate(self.retained_indices)
        ]
        derived_inputs = torch.stack(derived_columns, dim=1)
        normalized_derived = (
            derived_inputs - self.derived_mean
        ) / self.derived_scale
        normalized_unit_price = self.core(normalized_derived)
        unit_price = normalized_unit_price * self.value_scale + self.value_mean
        price = spot * unit_price
        return (price - self.value_mean) / self.value_scale


def _feature_index(schema: PricingTensorSchema, name: str) -> int:
    matches = [
        index for index, candidate in enumerate(schema.feature_names) if candidate == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Representation feature {name!r} resolves to {len(matches)} coordinates"
        )
    return matches[0]


def build_pricing_model(
    network_specification: dict[str, Any],
    representation_specification: dict[str, Any] | None,
    *,
    prepared: PreparedPricingDataset,
    train_indices: np.ndarray,
    transform: Standardization,
) -> nn.Module:
    """Build a generic network, optionally wrapped in a pricing representation."""

    representation = representation_specification or {"name": "identity"}
    name = str(representation.get("name", "identity"))
    input_dimension = len(prepared.schema.feature_names)
    output_dimension = len(prepared.schema.value_names)
    if name == "identity":
        return build_network(
            network_specification,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
        )
    if name != "unit_spot_homogeneous":
        raise ValueError(
            "Unknown pricing representation "
            f"{name!r}; choose identity or unit_spot_homogeneous"
        )
    if output_dimension != 1:
        raise ValueError("unit_spot_homogeneous requires one scalar price target")
    spot_index = _feature_index(
        prepared.schema, str(representation.get("spot_feature", "model.spot"))
    )
    strike_index = _feature_index(
        prepared.schema, str(representation.get("strike_feature", "product.strike"))
    )
    training_spots = np.asarray(
        prepared.features[train_indices, spot_index], dtype=np.float64
    )
    if not np.allclose(training_spots, 1.0, rtol=0.0, atol=1e-7):
        raise ValueError(
            "unit_spot_homogeneous currently requires every selected training spot "
            "to equal 1 so the fitted strike and price transforms also apply to K/S "
            "and P/S"
        )
    core = build_network(
        network_specification,
        input_dimension=input_dimension - 1,
        output_dimension=output_dimension,
    )
    return UnitSpotHomogeneousModel(
        core,
        spot_index=spot_index,
        strike_index=strike_index,
        transform=transform,
    )

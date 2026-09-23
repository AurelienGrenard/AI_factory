"""Universal regression metrics for price and gradient predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from collections.abc import Iterable

from learning.common.data.contracts import PricingTensorSchema
from learning.common.data.transforms import Standardization


@dataclass
class _Moments:
    count: int = 0
    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    relative_sum: float = 0.0
    absolute_errors: list[np.ndarray] | None = None
    standardized_errors: list[np.ndarray] | None = None
    sign_matches: int = 0
    sign_count: int = 0

    def __post_init__(self) -> None:
        self.absolute_errors = []
        self.standardized_errors = []

    def add(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        standard_error: torch.Tensor | None = None,
        *,
        compare_sign: bool = False,
    ) -> None:
        error = prediction - target
        absolute = torch.abs(error)
        self.count += error.numel()
        self.absolute_sum += float(torch.sum(absolute).cpu())
        self.squared_sum += float(torch.sum(torch.square(error)).cpu())
        denominator = torch.clamp(torch.abs(target), min=1e-6)
        self.relative_sum += float(torch.sum(absolute / denominator).cpu())
        self.absolute_errors.append(absolute.detach().cpu().numpy().reshape(-1))
        if standard_error is not None:
            valid = torch.isfinite(standard_error) & (standard_error > 0.0)
            if torch.any(valid):
                standardized = absolute[valid] / standard_error[valid]
                self.standardized_errors.append(
                    standardized.detach().cpu().numpy().reshape(-1)
                )
        if compare_sign:
            self.sign_matches += int(torch.sum(torch.sign(prediction) == torch.sign(target)))
            self.sign_count += target.numel()

    def result(self, prefix: str) -> dict[str, float | int]:
        if self.count == 0:
            return {f"{prefix}_count": 0}
        result: dict[str, float | int] = {
            f"{prefix}_count": self.count,
            f"{prefix}_mae": self.absolute_sum / self.count,
            f"{prefix}_rmse": math.sqrt(self.squared_sum / self.count),
            f"{prefix}_mean_relative_error": self.relative_sum / self.count,
        }
        absolute = np.concatenate(self.absolute_errors)
        for quantile, label in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
            result[f"{prefix}_{label}_absolute_error"] = float(
                np.quantile(absolute, quantile)
            )
        result[f"{prefix}_max_absolute_error"] = float(np.max(absolute))
        if self.standardized_errors:
            standardized = np.concatenate(self.standardized_errors)
            result[f"{prefix}_standard_error_unit_count"] = int(standardized.size)
            result[f"{prefix}_mean_absolute_standard_error_units"] = float(
                np.mean(standardized)
            )
            result[f"{prefix}_p90_absolute_standard_error_units"] = float(
                np.quantile(standardized, 0.9)
            )
        if self.sign_count:
            result[f"{prefix}_sign_accuracy"] = self.sign_matches / self.sign_count
        return result


def evaluate(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    schema: PricingTensorSchema,
    transform: Standardization,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    feature_mean = torch.as_tensor(transform.feature_mean, device=device)
    feature_scale = torch.as_tensor(transform.feature_scale, device=device)
    value_mean = torch.as_tensor(transform.value_mean, device=device)
    value_scale = torch.as_tensor(transform.value_scale, device=device)
    gradient_indices = torch.as_tensor(
        [gradient.wrt_index for gradient in schema.gradients],
        dtype=torch.long,
        device=device,
    )
    gradient_scale = torch.as_tensor(
        transform.normalized_gradient_scales(
            np.asarray([gradient.wrt_index for gradient in schema.gradients], dtype=np.int64)
        ),
        device=device,
    )
    price = _Moments()
    gradient = _Moments()
    for batch in loader:
        raw_inputs = batch["features"].to(device)
        normalized_inputs = ((raw_inputs - feature_mean) / feature_scale).requires_grad_(
            bool(schema.gradients)
        )
        normalized_prediction = model(normalized_inputs)
        prediction = normalized_prediction * value_scale + value_mean
        price.add(
            prediction.detach(),
            batch["values"].to(device),
            batch["value_standard_errors"].to(device),
        )
        if schema.gradients:
            normalized_derivatives = torch.autograd.grad(
                normalized_prediction.sum(), normalized_inputs, create_graph=False
            )[0].index_select(1, gradient_indices)
            raw_derivatives = normalized_derivatives / gradient_scale
            gradient.add(
                raw_derivatives.detach(),
                batch["gradients"].to(device),
                batch["gradient_standard_errors"].to(device),
                compare_sign=True,
            )
    return {**price.result("price"), **gradient.result("gradient")}

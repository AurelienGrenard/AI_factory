"""Composable training objectives and penalties."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn


@dataclass
class LossContext:
    model: nn.Module
    normalized_inputs: torch.Tensor
    normalized_predictions: torch.Tensor
    normalized_targets: torch.Tensor
    normalized_predicted_gradients: torch.Tensor | None
    normalized_target_gradients: torch.Tensor | None
    raw_inputs: torch.Tensor
    raw_predictions: torch.Tensor
    raw_targets: torch.Tensor
    raw_predicted_gradients: torch.Tensor | None
    raw_target_gradients: torch.Tensor | None
    batch: dict[str, torch.Tensor]
    epoch: int
    global_step: int


class LossTerm(Protocol):
    requires_input_gradients: bool

    def __call__(self, context: LossContext) -> torch.Tensor: ...


LossBuilder = Callable[[dict[str, Any]], LossTerm]
_LOSS_BUILDERS: dict[str, LossBuilder] = {}


def register_loss(name: str) -> Callable[[LossBuilder], LossBuilder]:
    """Register a new objective or penalty without editing the trainer."""

    def decorator(builder: LossBuilder) -> LossBuilder:
        if name in _LOSS_BUILDERS:
            raise ValueError(f"Loss term {name!r} is already registered")
        _LOSS_BUILDERS[name] = builder
        return builder

    return decorator


class _ValueMSE:
    requires_input_gradients = False

    def __call__(self, context: LossContext) -> torch.Tensor:
        return torch.mean(
            torch.square(context.normalized_predictions - context.normalized_targets)
        )


class _GradientMSE:
    requires_input_gradients = True

    def __call__(self, context: LossContext) -> torch.Tensor:
        if (
            context.normalized_predicted_gradients is None
            or context.normalized_target_gradients is None
        ):
            raise ValueError("gradient_mse requires published gradient targets")
        return torch.mean(
            torch.square(
                context.normalized_predicted_gradients
                - context.normalized_target_gradients
            )
        )


class _L2Parameters:
    requires_input_gradients = False

    def __init__(self, specification: dict[str, Any]):
        self.normalize = bool(specification.get("normalize", True))

    def __call__(self, context: LossContext) -> torch.Tensor:
        total = torch.zeros((), device=context.normalized_predictions.device)
        count = 0
        for parameter in context.model.parameters():
            total = total + torch.sum(torch.square(parameter))
            count += parameter.numel()
        return total / max(count, 1) if self.normalize else total


@register_loss("value_mse")
def _build_value_mse(specification: dict[str, Any]) -> LossTerm:
    return _ValueMSE()


@register_loss("gradient_mse")
def _build_gradient_mse(specification: dict[str, Any]) -> LossTerm:
    return _GradientMSE()


@register_loss("l2_parameters")
def _build_l2_parameters(specification: dict[str, Any]) -> LossTerm:
    return _L2Parameters(specification)


@dataclass(frozen=True)
class _WeightedTerm:
    name: str
    weight: float
    term: LossTerm


class CompositeLoss:
    def __init__(self, terms: list[_WeightedTerm]):
        if not terms:
            raise ValueError("At least one loss term is required")
        self.terms = terms

    @property
    def requires_input_gradients(self) -> bool:
        return any(term.term.requires_input_gradients for term in self.terms)

    def __call__(self, context: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=context.normalized_predictions.device)
        metrics: dict[str, float] = {}
        for weighted in self.terms:
            value = weighted.term(context)
            total = total + weighted.weight * value
            metrics[weighted.name] = float(value.detach().cpu())
        metrics["total"] = float(total.detach().cpu())
        return total, metrics


def build_composite_loss(specifications: list[dict[str, Any]]) -> CompositeLoss:
    terms: list[_WeightedTerm] = []
    names: set[str] = set()
    for specification in specifications:
        name = str(specification.get("name", ""))
        if not name:
            raise ValueError("Each loss term requires a name")
        if name in names:
            raise ValueError(f"Duplicate loss term {name!r}")
        names.add(name)
        try:
            builder = _LOSS_BUILDERS[name]
        except KeyError as error:
            raise ValueError(
                f"Unknown loss term {name!r}; registered terms: {sorted(_LOSS_BUILDERS)}"
            ) from error
        weight = float(specification.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError(f"Loss weight for {name!r} must be nonnegative")
        terms.append(_WeightedTerm(name, weight, builder(specification)))
    return CompositeLoss(terms)

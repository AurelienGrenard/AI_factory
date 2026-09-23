"""Method-independent tensor contracts for learning datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GradientTarget:
    """A published derivative and the input coordinate it differentiates."""

    source_name: str
    value_name: str
    wrt_name: str
    wrt_index: int
    standard_error_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "value_name": self.value_name,
            "wrt_name": self.wrt_name,
            "wrt_index": self.wrt_index,
            "standard_error_name": self.standard_error_name,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "GradientTarget":
        return cls(
            source_name=str(value["source_name"]),
            value_name=str(value["value_name"]),
            wrt_name=str(value["wrt_name"]),
            wrt_index=int(value["wrt_index"]),
            standard_error_name=(
                None
                if value.get("standard_error_name") is None
                else str(value["standard_error_name"])
            ),
        )


@dataclass(frozen=True)
class PricingTensorSchema:
    """Names and dimensions seen by model-independent training code."""

    database_id: str
    row_count: int
    feature_names: tuple[str, ...]
    value_names: tuple[str, ...]
    gradients: tuple[GradientTarget, ...]
    entity_roles: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "database_id": self.database_id,
            "row_count": self.row_count,
            "feature_names": list(self.feature_names),
            "value_names": list(self.value_names),
            "gradients": [gradient.to_dict() for gradient in self.gradients],
            "entity_roles": list(self.entity_roles),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PricingTensorSchema":
        return cls(
            database_id=str(value["database_id"]),
            row_count=int(value["row_count"]),
            feature_names=tuple(str(name) for name in value["feature_names"]),
            value_names=tuple(str(name) for name in value["value_names"]),
            gradients=tuple(
                GradientTarget.from_dict(item) for item in value["gradients"]
            ),
            entity_roles=tuple(str(role) for role in value["entity_roles"]),
        )

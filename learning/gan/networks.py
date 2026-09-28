"""Dataset-independent conditional generator and critic registries."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import torch
from torch import nn


GeneratorBuilder = Callable[[int, int, dict[str, Any]], nn.Module]
CriticBuilder = Callable[[int, int, dict[str, Any]], nn.Module]
_GENERATORS: dict[str, GeneratorBuilder] = {}
_CRITICS: dict[str, CriticBuilder] = {}


def register_generator(
    name: str,
) -> Callable[[GeneratorBuilder], GeneratorBuilder]:
    def decorator(builder: GeneratorBuilder) -> GeneratorBuilder:
        if name in _GENERATORS:
            raise ValueError(f"Generator {name!r} is already registered")
        _GENERATORS[name] = builder
        return builder

    return decorator


def register_critic(
    name: str,
) -> Callable[[CriticBuilder], CriticBuilder]:
    def decorator(builder: CriticBuilder) -> CriticBuilder:
        if name in _CRITICS:
            raise ValueError(f"Critic {name!r} is already registered")
        _CRITICS[name] = builder
        return builder

    return decorator


class ConditionalGenerator(nn.Module):
    latent_dimension: int
    noise_distribution: str
    noise_minimum: float
    noise_maximum: float

    def forward(
        self, conditions: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class ConditionalCritic(nn.Module):
    def forward(
        self, conditions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


def _activation(name: str) -> nn.Module:
    builders: dict[str, Callable[[], nn.Module]] = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "softplus": lambda: nn.Softplus(beta=10.0, threshold=20.0),
        "tanh": nn.Tanh,
    }
    try:
        return builders[name]()
    except KeyError as error:
        raise ValueError(
            f"Unknown GAN activation {name!r}; "
            f"choose from {sorted(builders)}"
        ) from error


def _linear(
    input_dimension: int,
    output_dimension: int,
    spectral_norm: bool,
) -> nn.Module:
    layer: nn.Module = nn.Linear(input_dimension, output_dimension)
    if spectral_norm:
        layer = nn.utils.parametrizations.spectral_norm(layer)
    return layer


class _ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        condition_size: int,
        activation: str,
        layer_norm: bool,
    ):
        super().__init__()
        self.normalization = (
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        )
        self.film = nn.Linear(condition_size, 2 * hidden_size)
        self.first = nn.Linear(hidden_size, hidden_size)
        self.second = nn.Linear(hidden_size, hidden_size)
        self.activation = _activation(activation)

    def forward(
        self, hidden: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.film(condition).chunk(2, dim=1)
        residual = self.normalization(hidden)
        residual = residual * (1.0 + 0.1 * scale) + 0.1 * shift
        residual = self.activation(self.first(residual))
        residual = self.second(residual)
        return (hidden + residual) / math.sqrt(2.0)


class _ResidualGenerator(ConditionalGenerator):
    def __init__(
        self,
        condition_dimension: int,
        observation_dimension: int,
        specification: dict[str, Any],
    ):
        super().__init__()
        self.latent_dimension = int(specification["latent_dimension"])
        self.noise_distribution = str(
            specification.get("noise_distribution", "normal")
        )
        self.noise_minimum = float(
            specification.get("noise_minimum", -1.0)
        )
        self.noise_maximum = float(
            specification.get("noise_maximum", 1.0)
        )
        hidden_size = int(specification["hidden_size"])
        depth = int(specification["depth"])
        activation = str(specification.get("activation", "silu"))
        layer_norm = bool(specification.get("layer_norm", True))
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dimension, hidden_size),
            _activation(activation),
            nn.Linear(hidden_size, hidden_size),
            _activation(activation),
        )
        self.input = nn.Linear(
            hidden_size + self.latent_dimension, hidden_size
        )
        self.blocks = nn.ModuleList(
            _ConditionalResidualBlock(
                hidden_size,
                hidden_size,
                activation,
                layer_norm,
            )
            for _ in range(depth)
        )
        self.output_norm = (
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        )
        self.output_activation = _activation(activation)
        self.output = nn.Linear(hidden_size, observation_dimension)

    def forward(
        self, conditions: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dimension:
            raise ValueError(
                "Generator noise has the wrong latent dimension"
            )
        if conditions.shape[0] != noise.shape[0]:
            raise ValueError(
                "Generator conditions and noise batch sizes differ"
            )
        encoded = self.condition_encoder(conditions)
        hidden = self.input(torch.cat((encoded, noise), dim=1))
        for block in self.blocks:
            hidden = block(hidden, encoded)
        hidden = self.output_activation(self.output_norm(hidden))
        return self.output(hidden)


class _EmbeddedGenerator(ConditionalGenerator):
    """Smooth conditional MLP with separate noise/parameter embeddings."""

    def __init__(
        self,
        condition_dimension: int,
        observation_dimension: int,
        specification: dict[str, Any],
    ):
        super().__init__()
        self.latent_dimension = int(specification["latent_dimension"])
        self.noise_distribution = str(
            specification.get("noise_distribution", "normal")
        )
        self.noise_minimum = float(
            specification.get("noise_minimum", -1.0)
        )
        self.noise_maximum = float(
            specification.get("noise_maximum", 1.0)
        )
        parameter_embedding_size = int(
            specification.get("parameter_embedding_size", 150)
        )
        noise_embedding_size = int(
            specification.get("noise_embedding_size", 150)
        )
        hidden_size = int(specification["hidden_size"])
        depth = int(specification["depth"])
        activation = str(specification.get("activation", "softplus"))
        if depth < 2:
            raise ValueError(
                "conditional_embedded_mlp depth must be at least two"
            )
        self.condition_embedding = nn.Linear(
            condition_dimension, parameter_embedding_size
        )
        self.noise_embedding = nn.Linear(
            self.latent_dimension, noise_embedding_size
        )
        layers: list[nn.Module] = []
        input_size = parameter_embedding_size + noise_embedding_size
        for layer_index in range(depth - 1):
            layers.extend(
                (
                    nn.Linear(
                        input_size if layer_index == 0 else hidden_size,
                        hidden_size,
                    ),
                    _activation(activation),
                )
            )
        layers.append(nn.Linear(hidden_size, observation_dimension))
        self.network = nn.Sequential(*layers)
        self.embedding_activation = _activation(activation)

    def forward(
        self, conditions: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dimension:
            raise ValueError(
                "Generator noise has the wrong latent dimension"
            )
        if conditions.shape[0] != noise.shape[0]:
            raise ValueError(
                "Generator conditions and noise batch sizes differ"
            )
        embedded = torch.cat(
            (
                self.noise_embedding(noise),
                self.condition_embedding(conditions),
            ),
            dim=1,
        )
        return self.network(self.embedding_activation(embedded))


class _ConcatCritic(ConditionalCritic):
    def __init__(
        self,
        condition_dimension: int,
        observation_dimension: int,
        specification: dict[str, Any],
    ):
        super().__init__()
        hidden_size = int(specification["hidden_size"])
        depth = int(specification["depth"])
        activation = str(
            specification.get("activation", "leaky_relu")
        )
        spectral_norm = bool(
            specification.get("spectral_norm", False)
        )
        layers: list[nn.Module] = [
            _linear(
                condition_dimension + observation_dimension,
                hidden_size,
                spectral_norm,
            ),
            _activation(activation),
        ]
        for _ in range(depth - 1):
            layers.extend(
                (
                    _linear(hidden_size, hidden_size, spectral_norm),
                    _activation(activation),
                )
            )
        layers.append(_linear(hidden_size, 1, spectral_norm))
        self.network = nn.Sequential(*layers)

    def forward(
        self, conditions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        if conditions.shape[0] != observations.shape[0]:
            raise ValueError(
                "Critic conditions and observations batch sizes differ"
            )
        return self.network(
            torch.cat((conditions, observations), dim=1)
        ).squeeze(1)


class _EmbeddedCritic(ConditionalCritic):
    """Conditional critic with separate observation/parameter embeddings."""

    def __init__(
        self,
        condition_dimension: int,
        observation_dimension: int,
        specification: dict[str, Any],
    ):
        super().__init__()
        parameter_embedding_size = int(
            specification.get("parameter_embedding_size", 150)
        )
        observation_embedding_size = int(
            specification.get("observation_embedding_size", 150)
        )
        hidden_size = int(specification["hidden_size"])
        depth = int(specification["depth"])
        activation = str(specification.get("activation", "softplus"))
        spectral_norm = bool(
            specification.get("spectral_norm", False)
        )
        if depth < 2:
            raise ValueError(
                "conditional_embedded_mlp depth must be at least two"
            )
        self.condition_embedding = _linear(
            condition_dimension,
            parameter_embedding_size,
            spectral_norm,
        )
        self.observation_embedding = _linear(
            observation_dimension,
            observation_embedding_size,
            spectral_norm,
        )
        layers: list[nn.Module] = []
        input_size = (
            parameter_embedding_size + observation_embedding_size
        )
        for layer_index in range(depth - 1):
            layers.extend(
                (
                    _linear(
                        input_size if layer_index == 0 else hidden_size,
                        hidden_size,
                        spectral_norm,
                    ),
                    _activation(activation),
                )
            )
        layers.append(_linear(hidden_size, 1, spectral_norm))
        self.network = nn.Sequential(*layers)
        self.embedding_activation = _activation(activation)

    def forward(
        self, conditions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        if conditions.shape[0] != observations.shape[0]:
            raise ValueError(
                "Critic conditions and observations batch sizes differ"
            )
        embedded = torch.cat(
            (
                self.observation_embedding(observations),
                self.condition_embedding(conditions),
            ),
            dim=1,
        )
        return self.network(
            self.embedding_activation(embedded)
        ).squeeze(1)


class _ProjectionCritic(ConditionalCritic):
    def __init__(
        self,
        condition_dimension: int,
        observation_dimension: int,
        specification: dict[str, Any],
    ):
        super().__init__()
        hidden_size = int(specification["hidden_size"])
        depth = int(specification["depth"])
        activation = str(
            specification.get("activation", "leaky_relu")
        )
        spectral_norm = bool(
            specification.get("spectral_norm", False)
        )
        observation_layers: list[nn.Module] = [
            _linear(
                observation_dimension, hidden_size, spectral_norm
            ),
            _activation(activation),
        ]
        for _ in range(depth - 1):
            observation_layers.extend(
                (
                    _linear(hidden_size, hidden_size, spectral_norm),
                    _activation(activation),
                )
            )
        self.observation_encoder = nn.Sequential(
            *observation_layers
        )
        self.condition_encoder = nn.Sequential(
            _linear(condition_dimension, hidden_size, spectral_norm),
            _activation(activation),
            _linear(hidden_size, hidden_size, spectral_norm),
        )
        self.unconditional = _linear(hidden_size, 1, spectral_norm)
        self.scale = 1.0 / math.sqrt(hidden_size)

    def forward(
        self, conditions: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        if conditions.shape[0] != observations.shape[0]:
            raise ValueError(
                "Critic conditions and observations batch sizes differ"
            )
        features = self.observation_encoder(observations)
        condition = self.condition_encoder(conditions)
        projection = torch.sum(features * condition, dim=1)
        return (
            self.unconditional(features).squeeze(1)
            + self.scale * projection
        )


@register_generator("conditional_residual_mlp")
def _build_residual_generator(
    condition_dimension: int,
    observation_dimension: int,
    specification: dict[str, Any],
) -> ConditionalGenerator:
    return _ResidualGenerator(
        condition_dimension,
        observation_dimension,
        specification,
    )


@register_generator("conditional_embedded_mlp")
def _build_embedded_generator(
    condition_dimension: int,
    observation_dimension: int,
    specification: dict[str, Any],
) -> ConditionalGenerator:
    return _EmbeddedGenerator(
        condition_dimension,
        observation_dimension,
        specification,
    )


@register_critic("conditional_concat_mlp")
def _build_concat_critic(
    condition_dimension: int,
    observation_dimension: int,
    specification: dict[str, Any],
) -> ConditionalCritic:
    return _ConcatCritic(
        condition_dimension,
        observation_dimension,
        specification,
    )


@register_critic("conditional_projection_mlp")
def _build_projection_critic(
    condition_dimension: int,
    observation_dimension: int,
    specification: dict[str, Any],
) -> ConditionalCritic:
    return _ProjectionCritic(
        condition_dimension,
        observation_dimension,
        specification,
    )


@register_critic("conditional_embedded_mlp")
def _build_embedded_critic(
    condition_dimension: int,
    observation_dimension: int,
    specification: dict[str, Any],
) -> ConditionalCritic:
    return _EmbeddedCritic(
        condition_dimension,
        observation_dimension,
        specification,
    )


def build_generator(
    specification: dict[str, Any],
    *,
    condition_dimension: int,
    observation_dimension: int,
) -> ConditionalGenerator:
    name = str(
        specification.get("name", "conditional_residual_mlp")
    )
    try:
        builder = _GENERATORS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown generator {name!r}; "
            f"registered: {sorted(_GENERATORS)}"
        ) from error
    generator = builder(
        condition_dimension,
        observation_dimension,
        specification,
    )
    if not isinstance(generator, ConditionalGenerator):
        raise TypeError(
            f"Generator builder {name!r} returned an incompatible module"
        )
    return generator


def build_critic(
    specification: dict[str, Any],
    *,
    condition_dimension: int,
    observation_dimension: int,
) -> ConditionalCritic:
    name = str(
        specification.get("name", "conditional_projection_mlp")
    )
    try:
        builder = _CRITICS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown critic {name!r}; "
            f"registered: {sorted(_CRITICS)}"
        ) from error
    critic = builder(
        condition_dimension,
        observation_dimension,
        specification,
    )
    if not isinstance(critic, ConditionalCritic):
        raise TypeError(
            f"Critic builder {name!r} returned an incompatible module"
        )
    return critic

"""Adversarial objectives for conditional terminal distributions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .networks import ConditionalCritic


@dataclass(frozen=True)
class CriticLoss:
    total: torch.Tensor
    wasserstein: torch.Tensor
    gradient_penalty: torch.Tensor
    gradient_norm_mean: torch.Tensor
    drift: torch.Tensor


@dataclass(frozen=True)
class GeneratorLoss:
    total: torch.Tensor
    adversarial: torch.Tensor
    mode_seeking_ratio: torch.Tensor
    moment_matching: torch.Tensor


def gradient_penalty(
    critic: ConditionalCritic,
    conditions: torch.Tensor,
    real_observations: torch.Tensor,
    fake_observations: torch.Tensor,
    *,
    target: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize observation gradients while holding the condition fixed."""

    if real_observations.shape != fake_observations.shape:
        raise ValueError(
            "Real and generated observations must have the same shape"
        )
    alpha_shape = (real_observations.shape[0],) + (
        1,
    ) * (real_observations.ndim - 1)
    alpha = torch.rand(
        alpha_shape,
        device=real_observations.device,
        dtype=real_observations.dtype,
    )
    interpolated = (
        alpha * real_observations
        + (1.0 - alpha) * fake_observations.detach()
    ).requires_grad_(True)
    scores = critic(conditions, interpolated)
    gradients = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
    )[0]
    norms = gradients.flatten(start_dim=1).norm(2, dim=1)
    penalty = torch.mean(torch.square(norms - target))
    return penalty, norms.mean()


def critic_wasserstein_gp(
    critic: ConditionalCritic,
    conditions: torch.Tensor,
    real_observations: torch.Tensor,
    fake_observations: torch.Tensor,
    *,
    gradient_penalty_weight: float,
    gradient_penalty_target: float,
    drift_weight: float,
) -> CriticLoss:
    real_scores = critic(conditions, real_observations)
    fake_scores = critic(conditions, fake_observations.detach())
    wasserstein = fake_scores.mean() - real_scores.mean()
    penalty, norm_mean = gradient_penalty(
        critic,
        conditions,
        real_observations,
        fake_observations,
        target=gradient_penalty_target,
    )
    drift = torch.mean(torch.square(real_scores))
    total = (
        wasserstein
        + gradient_penalty_weight * penalty
        + drift_weight * drift
    )
    return CriticLoss(
        total,
        wasserstein,
        penalty,
        norm_mean,
        drift,
    )


def critic_logistic(
    critic: ConditionalCritic,
    conditions: torch.Tensor,
    real_observations: torch.Tensor,
    fake_observations: torch.Tensor,
) -> CriticLoss:
    """Non-saturating logistic discriminator loss, returned as logits."""

    real_logits = critic(conditions, real_observations)
    fake_logits = critic(conditions, fake_observations.detach())
    adversarial = (
        F.softplus(-real_logits).mean()
        + F.softplus(fake_logits).mean()
    )
    score_gap = fake_logits.mean() - real_logits.mean()
    zero = adversarial.new_zeros(())
    return CriticLoss(
        adversarial,
        score_gap,
        zero,
        zero,
        zero,
    )


def generator_wasserstein(
    critic: ConditionalCritic,
    conditions: torch.Tensor,
    fake_observations: torch.Tensor,
) -> torch.Tensor:
    return -critic(conditions, fake_observations).mean()


def mode_seeking_ratio(
    first_observations: torch.Tensor,
    second_observations: torch.Tensor,
    first_noise: torch.Tensor,
    second_noise: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Measure output response to latent changes for equal conditions."""

    if first_observations.shape != second_observations.shape:
        raise ValueError(
            "Mode-seeking observations must have equal shapes"
        )
    if first_noise.shape != second_noise.shape:
        raise ValueError("Mode-seeking noise tensors must have equal shapes")
    if first_observations.shape[0] != first_noise.shape[0]:
        raise ValueError(
            "Mode-seeking observations and noise batch sizes differ"
        )
    output_distance = torch.mean(
        torch.abs(first_observations - second_observations),
        dim=1,
    )
    noise_distance = torch.mean(
        torch.abs(first_noise - second_noise),
        dim=1,
    )
    return torch.mean(output_distance / (noise_distance + epsilon))


def moment_matching_loss(
    real_observations: torch.Tensor,
    fake_observations: torch.Tensor,
) -> torch.Tensor:
    """Match normalized batch means and standard deviations."""

    if real_observations.shape != fake_observations.shape:
        raise ValueError(
            "Moment-matching observations must have equal shapes"
        )
    real_mean = real_observations.mean(dim=0)
    fake_mean = fake_observations.mean(dim=0)
    real_std = real_observations.std(dim=0, unbiased=False)
    fake_std = fake_observations.std(dim=0, unbiased=False)
    return torch.mean(torch.square(fake_mean - real_mean)) + torch.mean(
        torch.square(fake_std - real_std)
    )


def generator_wasserstein_regularized(
    critic: ConditionalCritic,
    conditions: torch.Tensor,
    real_observations: torch.Tensor,
    fake_observations: torch.Tensor,
    first_noise: torch.Tensor,
    *,
    second_observations: torch.Tensor | None = None,
    second_noise: torch.Tensor | None = None,
    mode_seeking_weight: float = 0.0,
    mode_seeking_epsilon: float = 1.0e-6,
    moment_matching_weight: float = 0.0,
) -> GeneratorLoss:
    """Combine WGAN loss with optional generic anti-collapse terms."""

    adversarial = generator_wasserstein(
        critic, conditions, fake_observations
    )
    return _regularized_generator_loss(
        adversarial,
        real_observations,
        fake_observations,
        first_noise,
        second_observations=second_observations,
        second_noise=second_noise,
        mode_seeking_weight=mode_seeking_weight,
        mode_seeking_epsilon=mode_seeking_epsilon,
        moment_matching_weight=moment_matching_weight,
    )

def _regularized_generator_loss(
    adversarial: torch.Tensor,
    real_observations: torch.Tensor,
    fake_observations: torch.Tensor,
    first_noise: torch.Tensor,
    *,
    second_observations: torch.Tensor | None,
    second_noise: torch.Tensor | None,
    mode_seeking_weight: float,
    mode_seeking_epsilon: float,
    moment_matching_weight: float,
) -> GeneratorLoss:
    zero = adversarial.new_zeros(())
    ratio = zero
    if mode_seeking_weight > 0.0:
        if second_observations is None or second_noise is None:
            raise ValueError(
                "Mode-seeking regularization requires paired generations"
            )
        ratio = mode_seeking_ratio(
            fake_observations,
            second_observations,
            first_noise,
            second_noise,
            epsilon=mode_seeking_epsilon,
        )
    moments = (
        moment_matching_loss(real_observations, fake_observations)
        if moment_matching_weight > 0.0
        else zero
    )
    total = (
        adversarial
        - mode_seeking_weight * ratio
        + moment_matching_weight * moments
    )
    return GeneratorLoss(total, adversarial, ratio, moments)


def generator_logistic_regularized(
    critic: ConditionalCritic,
    conditions: torch.Tensor,
    real_observations: torch.Tensor,
    fake_observations: torch.Tensor,
    first_noise: torch.Tensor,
    *,
    second_observations: torch.Tensor | None = None,
    second_noise: torch.Tensor | None = None,
    mode_seeking_weight: float = 0.0,
    mode_seeking_epsilon: float = 1.0e-6,
    moment_matching_weight: float = 0.0,
) -> GeneratorLoss:
    """Non-saturating logistic generator loss with optional regularizers."""

    adversarial = F.softplus(
        -critic(conditions, fake_observations)
    ).mean()
    return _regularized_generator_loss(
        adversarial,
        real_observations,
        fake_observations,
        first_noise,
        second_observations=second_observations,
        second_noise=second_noise,
        mode_seeking_weight=mode_seeking_weight,
        mode_seeking_epsilon=mode_seeking_epsilon,
        moment_matching_weight=moment_matching_weight,
    )

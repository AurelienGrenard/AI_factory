# CIR

| At a glance | Value |
|---|---|
| Process | One-factor affine square-root short rate |
| Transition | Exact state law at requested dates |
| Path state | `state` |
| Random laws | Adaptive Poisson plus Gamma |
| Pricing | Affine zero-coupon, forward, and swap analytics |
| Early exercise | Not implemented |

## Role and reference

This directory implements the Cox-Ingersoll-Ross short-rate model

```text
dr_t = kappa(theta - r_t) dt + sigma sqrt(r_t) dW_t.
```

The exact transition preserves non-negativity. The Feller condition
`2*kappa*theta >= sigma^2` is not required by the implementation: when it is
violated, zero is accessible but the exact transition law remains valid.

See [Cox, Ingersoll, and Ross (1985)](https://doi.org/10.2307/1911242).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load
  the process plus initial rate.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement the
  exact endpoint transition.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) expose the
  affine bond coefficients, zero-coupon, forward, and swap formulas.

## Dataset row

`CirModelParameters` combines the process parameters with its initial short
rate.

| Symbol | Dataset field |
|---|---|
| $r_0$ | `initial_state` |
| $\kappa$ | `mean_reversion` |
| $\theta$ | `long_term_mean` |
| $\sigma$ | `volatility` |

All process parameters are positive and the initial rate is non-negative.

## Affine bond formula

For $\tau=T-t$ and

$$
\gamma=\sqrt{\kappa^2+2\sigma^2},
$$

the model is affine:

$$
P(t,T)=A(t,T)\exp\{-B(t,T)r_t\},
$$

with

$$
B(t,T)=\frac{2(e^{\gamma\tau}-1)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma},
$$

and

$$
A(t,T)=\left[
\frac{2\gamma e^{(\kappa+\gamma)\tau/2}}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}
\right]^{2\kappa\theta/\sigma^2}.
$$

The implementation evaluates equivalent formulas in terms of
`exp(-gamma*tau)` to avoid overflow. `A`, `B`, and `log_A` are public device
functions; `log_zero_coupon_bond` uses `log_A-B*r` semantics without an
`exp`/`log` round trip.

## Exact dynamics

Over one interval $\Delta$, define

$$
c=\frac{\sigma^2(1-e^{-\kappa\Delta})}{4\kappa},\qquad
\nu=\frac{4\kappa\theta}{\sigma^2},\qquad
\lambda=\frac{e^{-\kappa\Delta}r_t}{c}.
$$

Then

$$
r_{t+\Delta}=cX,\qquad X\sim\chi'^2_\nu(\lambda).
$$

The device sampler uses the exact mixture

```text
N ~ Poisson(lambda / 2)
r_next ~ Gamma(nu / 2 + N, 2 * c)
```

so `c` is applied directly as the Gamma scale. Small Poisson means use
inversion; large means use Hoermann PTRS. One `UniformSequence(key, path)` and
one `NormalPairCache` live for the complete path.

<details>
<summary>Dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
CirExactTransition prepare_model(const CirProcessParameters&, float interval);
void one_step_transition(const CirExactTransition&, philox::UniformSequence&, philox::NormalPairCache&, float& state);
float simulate_terminal_state(const CirExactTransition&, float initial_state, philox::PhiloxKey, std::size_t path);
float simulate_on_regular_grid(const CirExactTransition& stub, const CirExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states);

namespace joint {
struct CirJointTransition;
struct CirJointState { float state; float state_integral; };
CirJointTransition prepare_model(const CirProcessParameters&, float interval);
void one_step_transition(const CirJointTransition&, philox::UniformSequence&, philox::NormalPairCache&, CirJointState&);
CirJointState simulate_terminal_state(const CirJointTransition&, float initial_state, philox::PhiloxKey, std::size_t path);
CirJointState simulate_on_regular_grid(const CirJointTransition& stub, const CirJointTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states, float* integrated_states);
}
```

</details>

## Analytics interface

```cpp
float log_A(const CirModelParameters&, float valuation_time, float maturity);
float A(const CirModelParameters&, float valuation_time, float maturity);
float B(const CirModelParameters&, float valuation_time, float maturity);
float log_zero_coupon_bond(const CirModelParameters&, float state, float valuation_time, float maturity);
float zero_coupon_bond(const CirModelParameters&, float state, float valuation_time, float maturity);
float forward_rate(const CirModelParameters&, float state, float valuation_time, float start, float end, float accrual);
float swap_rate(const CirModelParameters&, float state, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

Zero-coupon calls and puts additionally require a deterministic non-central
chi-square CDF. That numerical primitive and its tail validation are kept
separate from the random sampler and are not yet exposed here.

## Path discounting and swaptions

The exact endpoint transition does not jointly sample
$\int_t^{t+\Delta}r_s\,ds$. The common joint types and four function
signatures are exposed so CIR follows the fixed-income interface, but their
definitions remain deliberately unavailable until the integration scheme is
chosen and validated. `log_discount_factor` and `discount_factor` are likewise
not exposed yet.
Boundary-only formulas use the direct conditional bond value. A future
Monte-Carlo Bermudan swaption must specify and validate its integration scheme
or use a lattice/PDE approach; exact exercise-date rates alone are not enough
for pathwise discounting.

Related navigation: [dynamics contract](../../../../docs/cuda-model-dynamics-contract.md)
and [pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

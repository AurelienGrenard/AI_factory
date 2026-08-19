# Vasicek

| At a glance | Value |
|---|---|
| Process | One-factor affine Gaussian short rate |
| Transition | Exact state and state/integral laws |
| Path state | `state`, optionally `state_integral` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements the affine Gaussian short-rate model

```text
dr_t = a(b - r_t) dt + sigma dW_t.
```

`W` is a standard Brownian motion. `r` is a Gaussian Markov short rate and
`b` is its constant mean-reversion level.

See [Vasicek (1977)](https://doi.org/10.1016/0304-405X(77)90016-2).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the process plus initial rate.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact state and joint state/integral transitions.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) provide discount, bond, forward, swap, and option formulas.
- [`rate_option.cuh`](rate_option.cuh) / [`rate_option.cu`](rate_option.cu) and [`zero_coupon_bond_option.cuh`](zero_coupon_bond_option.cuh) / [`zero_coupon_bond_option.cu`](zero_coupon_bond_option.cu) own launchers.

## Dataset row

`VasicekModelParameters` combines the process parameters with its initial
short rate.

| Symbol | Dataset field |
|---|---|
| $r_0$ | `initial_state` |
| $a$ | `mean_reversion` |
| $b$ | `long_term_mean` |
| $\sigma$ | `volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `decay` | $e^{-a\Delta t}$ |
| `state_mean_increment` | $b(1-e^{-a\Delta t})$ |
| `state_standard_deviation` | Exact Gaussian endpoint variance |
| `integral_state_loading`, `integral_mean_increment` | Conditional integral mean |
| `integral_state_normal_loading` | Exact state/integral covariance |
| `integral_independent_standard_deviation` | Residual integral variance |

The state-only transition names its deterministic field `mean_increment`; the
joint transition uses `state_mean_increment`.

The state-only API uses one `float`; `joint::VasicekJointState` adds only
`state_integral` for discounting.

## Dynamics interface

| Function | Role |
|---|---|
| `integral_state_loading` | Load the current rate into its future integral mean |
| `integral_variance`, `integral_moments` | Return exact conditional integral moments |
| `prepare_model` | Precompute an exact state transition |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Apply one prepared transition directly |
| `simulate_on_regular_grid` | Store pre-terminal date-major states |
| `joint::*` variants | Evolve the rate and accumulated integral together |

The state and joint state/integral transitions are exact. A payoff needing only
dated rates samples directly at those dates; one needing path discounting uses
the exact joint Gaussian law. No fine numerical time grid is required.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float integral_state_loading(float mean_reversion, float interval);
float integral_variance(const VasicekProcessParameters&, float interval);
VasicekIntegralMoments integral_moments(const VasicekProcessParameters&, float interval);
VasicekExactTransition prepare_model(const VasicekProcessParameters&, float interval);
void one_step_transition(const VasicekExactTransition&, float normal, float& state);
float simulate_terminal_state(const VasicekExactTransition&, float initial_state, float normal);
float simulate_on_regular_grid(const VasicekExactTransition& stub, const VasicekExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states);
joint::VasicekJointExactTransition joint::prepare_model(const VasicekProcessParameters&, float interval);
void joint::one_step_transition(const joint::VasicekJointExactTransition&, float state_normal, float integral_normal, joint::VasicekJointState&);
joint::VasicekJointState joint::simulate_terminal_state(const joint::VasicekJointExactTransition&, float initial_state, float state_normal, float integral_normal);
joint::VasicekJointState joint::simulate_on_regular_grid(const joint::VasicekJointExactTransition& stub, const joint::VasicekJointExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states, float* integrals);
```

</details>

## Random-number strategy

Direct terminal helpers accept caller-provided normals. A regular-grid helper
owns one `philox::UniformSequence(key, path)` and one normal cache for the whole
path. The state-only transition consumes one normal and the joint transition
two.

## Affine bond formula

For $\tau=T-t$,

$$
P(t,T)=A(t,T)e^{-B(t,T)r_t},\qquad
B(t,T)=\frac{1-e^{-a\tau}}{a},
$$

with

$$
\log A(t,T)=\frac12V_I(\tau)-b[\tau-B(t,T)],
$$

where $V_I$ is the conditional variance of the future rate integral.
`log_A`, `A`, and `B` expose the textbook coefficients; bond evaluation
computes `log_A` and `B` together.

## Pricing kernels

Rate options and zero-coupon-bond options use closed-form analytics with one
thread per result. Every file follows `PreparedRow` →
`evaluate_price<Side>` → launcher. Call/put is a compile-time `OptionSide`.

<details>
<summary>Exact analytics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float log_A(const VasicekModelParameters&, float valuation_time, float maturity);
float A(const VasicekModelParameters&, float valuation_time, float maturity);
float B(const VasicekModelParameters&, float valuation_time, float maturity);
float log_zero_coupon_bond(const VasicekModelParameters&, float state, float valuation_time, float maturity);
float log_discount_factor(float state_integral);
float discount_factor(float state_integral);
float zero_coupon_bond(const VasicekModelParameters&, float state, float valuation_time, float maturity);
float zero_coupon_bond_call_price(const VasicekModelParameters&, float state, float valuation_time, float expiry, float maturity, float strike);
float zero_coupon_bond_put_price(const VasicekModelParameters&, float state, float valuation_time, float expiry, float maturity, float strike);
float forward_rate(const VasicekModelParameters&, float state, float valuation_time, float start, float end, float accrual);
float swap_rate(const VasicekModelParameters&, float state, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

</details>

## Memory and numerical policy

Prepared exact coefficients and the shared formulas in
`../common/mean_reverting_gaussian.cuh` avoid repeated transcendental work and
small-time cancellation. Grids write separate date-major state and integral
arrays only when needed. Fast-math is forbidden.

## American and Bermudan options

No Vasicek American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/vasicek/),
[validation](../../../../validation/model/fixed_income/vasicek/),
[fixed-income Gaussian helpers](../common/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

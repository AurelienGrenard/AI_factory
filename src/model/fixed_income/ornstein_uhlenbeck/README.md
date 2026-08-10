# Ornstein–Uhlenbeck short-rate model

| At a glance | Value |
|---|---|
| Process | One-factor centered Gaussian short rate |
| Transition | Exact state and state/integral laws |
| Path state | `state`, optionally `state_integral` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory uses a centered Ornstein–Uhlenbeck process directly as the
short rate:

```text
dX_t = -a X_t dt + sigma dW_t,    r_t = X_t.
```

`W` is a standard Brownian motion. `X` is a centered Gaussian Markov process;
the standalone model identifies that state directly with the short rate.

The Gaussian mean-reverting process originates in
[Uhlenbeck and Ornstein (1930)](https://doi.org/10.1103/PhysRev.36.823).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the process plus initial state.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact state and joint state/integral transitions.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) provide discount, bond, forward, swap, and option formulas.
- [`rate_option.cuh`](rate_option.cuh) / [`rate_option.cu`](rate_option.cu) and [`zero_coupon_bond_option.cuh`](zero_coupon_bond_option.cuh) / [`zero_coupon_bond_option.cu`](zero_coupon_bond_option.cu) own launchers.

## Dataset row

`OrnsteinUhlenbeckModelParameters` combines the process parameters with its
initial state.

| Symbol | Dataset field |
|---|---|
| $X_0$ | `initial_state` |
| $a$ | `mean_reversion` |
| $\sigma$ | `volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `decay` | $e^{-a\Delta t}$ |
| `state_standard_deviation` | $\sigma\sqrt{(1-e^{-2a\Delta t})/(2a)}$ |
| `integral_state_loading` | Conditional mean loading of $\int X_s ds$ |
| `integral_state_normal_loading` | Exact state/integral covariance |
| `integral_independent_standard_deviation` | Residual integral variance |

The last three fields belong to the `joint` transition.

The state-only API uses one `float`. `joint::OrnsteinUhlenbeckJointState` adds
only the accumulated integral needed for path discounting.

## Dynamics interface

| Function | Role |
|---|---|
| `integral_state_loading` | Load the current state into its future integral mean |
| `integral_variance`, `integral_moments` | Return exact conditional integral moments |
| `prepare_model` | Precompute an exact state transition |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Apply one prepared transition directly |
| `simulate_on_regular_grid` | Store pre-terminal date-major states |
| `joint::*` variants | Evolve the state and accumulated integral together |

All transitions are exact over the requested interval. Code needing only a
short rate samples the state directly at observation dates. Code needing a
discount integral uses the exact joint law; neither case requires a finer
artificial `dt`.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float integral_state_loading(float mean_reversion, float interval);
float integral_variance(const OrnsteinUhlenbeckProcessParameters&, float interval);
OrnsteinUhlenbeckIntegralMoments integral_moments(const OrnsteinUhlenbeckProcessParameters&, float interval);
OrnsteinUhlenbeckExactTransition prepare_model(const OrnsteinUhlenbeckProcessParameters&, float interval);
void one_step_transition(const OrnsteinUhlenbeckExactTransition&, float normal, float& state);
float simulate_terminal_state(const OrnsteinUhlenbeckExactTransition&, float initial_state, float normal);
float simulate_on_regular_grid(const OrnsteinUhlenbeckExactTransition& stub, const OrnsteinUhlenbeckExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states);
joint::OrnsteinUhlenbeckJointExactTransition joint::prepare_model(const OrnsteinUhlenbeckProcessParameters&, float interval);
void joint::one_step_transition(const joint::OrnsteinUhlenbeckJointExactTransition&, float state_normal, float integral_normal, joint::OrnsteinUhlenbeckJointState&);
joint::OrnsteinUhlenbeckJointState joint::simulate_terminal_state(const joint::OrnsteinUhlenbeckJointExactTransition&, float initial_state, float state_normal, float integral_normal);
joint::OrnsteinUhlenbeckJointState joint::simulate_on_regular_grid(const joint::OrnsteinUhlenbeckJointExactTransition& stub, const joint::OrnsteinUhlenbeckJointExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states, float* integrals);
```

</details>

## Random-number strategy

Direct terminal helpers receive their normal variates from the caller.
Regular-grid helpers create one `philox::UniformSequence(key, path)` and one
normal cache for the complete path. The joint law consumes two normals per
interval; the state-only law consumes one.

## Pricing kernels

The current rate and zero-coupon-bond options are closed form. Their common
shape is `PreparedRow` → `evaluate_price<Side>` → one thread per result.
Call/put behavior is a compile-time `OptionSide`; no Monte Carlo state or
standard-error buffer is allocated.

<details>
<summary>Exact analytics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float log_discount_factor(const joint::OrnsteinUhlenbeckJointState&);
float discount_factor(const joint::OrnsteinUhlenbeckJointState&);
float zero_coupon_bond(const OrnsteinUhlenbeckModelParameters&, float state, float valuation_time, float maturity);
float zero_coupon_bond_call_price(const OrnsteinUhlenbeckModelParameters&, float state, float valuation_time, float expiry, float maturity, float strike);
float zero_coupon_bond_put_price(const OrnsteinUhlenbeckModelParameters&, float state, float valuation_time, float expiry, float maturity, float strike);
float forward_rate(const OrnsteinUhlenbeckModelParameters&, float state, float valuation_time, float start, float end, float accrual);
float swap_rate(const OrnsteinUhlenbeckModelParameters&, float state, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

</details>

## Memory and numerical policy

The exact Gaussian coefficients are prepared once and shared mathematical
moments come from `../common/mean_reverting_gaussian.cuh`. Small-time formulas
avoid cancellation. Simulation grids use separate date-major state and
integral arrays only when requested. Fast-math is forbidden.

## American and Bermudan options

No Ornstein–Uhlenbeck American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/ornstein_uhlenbeck/),
[validation](../../../../validation/model/fixed_income/ornstein_uhlenbeck/),
[fixed-income Gaussian helpers](../common/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

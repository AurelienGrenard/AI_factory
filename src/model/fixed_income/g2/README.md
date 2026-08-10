# G2

| At a glance | Value |
|---|---|
| Process | Two-factor correlated Gaussian short rate |
| Transition | Exact states and state/integral laws |
| Path state | `state_x`, `state_y`, optionally `state_integral` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements a standalone two-factor Gaussian short-rate model:

```text
dx_t = -a x_t dt + sigma dW_t^x
dy_t = -b y_t dt + eta dW_t^y
d<W^x,W^y>_t = rho dt,    r_t = x_t + y_t.
```

`W^x` and `W^y` are standard Brownian motions with instantaneous correlation
`rho`. `x` and `y` are correlated centered Gaussian Ornstein–Uhlenbeck
factors. Their sum is the standalone short rate.

It is the unshifted two-factor Gaussian process also reused by G2++. The
mean-reverting factors are exact Ornstein–Uhlenbeck processes; the fitted
time-dependent extension follows the Gaussian term-structure framework of
[Hull and White (1990)](https://doi.org/10.1093/rfs/3.4.573).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load both factors plus initial state.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact correlated state and state/integral laws.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) provide short-rate, discount, bond, forward, swap, and option formulas.
- [`rate_option.cuh`](rate_option.cuh) / [`rate_option.cu`](rate_option.cu) and [`zero_coupon_bond_option.cuh`](zero_coupon_bond_option.cuh) / [`zero_coupon_bond_option.cu`](zero_coupon_bond_option.cu) own launchers.

## Dataset row

`G2ModelParameters` combines `G2ProcessParameters` with both initial factor
states.

| Symbol | Dataset field |
|---|---|
| $x_0$ | `initial_state_x` |
| $y_0$ | `initial_state_y` |
| $a$ | `mean_reversion_x` |
| $\sigma$ | `volatility_x` |
| $b$ | `mean_reversion_y` |
| $\eta$ | `volatility_y` |
| $\rho$ | `correlation` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `decay_x`, `decay_y` | $e^{-a\Delta t}$, $e^{-b\Delta t}$ |
| `state_x_standard_deviation` | Exact variance of $x_{t+\Delta t}$ |
| `state_y_x_normal_loading`, `state_y_independent_standard_deviation` | Cholesky loadings of the correlated endpoints |
| `integral_state_x_loading`, `integral_state_y_loading` | Conditional integral mean |
| `integral_x_normal_loading`, `integral_y_normal_loading`, `integral_independent_standard_deviation` | Joint state/integral Cholesky loadings |

The integral fields belong to the `joint` transition. The joint state adds one
accumulated integral to the two factor values.

## Dynamics interface

| Function | Role |
|---|---|
| `integral_moments` | Return both factor loadings and exact integral variance |
| `prepare_model` | Precompute an exact correlated two-factor transition |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Apply one prepared transition directly |
| `simulate_on_regular_grid` | Store pre-terminal date-major factor states |
| `joint::*` variants | Evolve both factors and their accumulated integral |

All laws are exact over the requested interval. Products needing only rates
sample `(x,y)` at observation dates. Products needing discounting use the
exact three-dimensional joint law instead of a finer time grid.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
G2IntegralMoments integral_moments(const G2ProcessParameters&, float interval);
G2ExactTransition prepare_model(const G2ProcessParameters&, float interval);
void one_step_transition(const G2ExactTransition&, float x_normal, float y_normal, G2State&);
G2State simulate_terminal_state(const G2ExactTransition&, G2State initial_state, float x_normal, float y_normal);
G2State simulate_on_regular_grid(const G2ExactTransition& stub, const G2ExactTransition& regular, G2State initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states_x, float* states_y);
joint::G2JointExactTransition joint::prepare_model(const G2ProcessParameters&, float interval);
void joint::one_step_transition(const joint::G2JointExactTransition&, float x_normal, float y_normal, float integral_normal, joint::G2JointState&);
joint::G2JointState joint::simulate_terminal_state(const joint::G2JointExactTransition&, G2State initial_state, float x_normal, float y_normal, float integral_normal);
joint::G2JointState joint::simulate_on_regular_grid(const joint::G2JointExactTransition& stub, const joint::G2JointExactTransition& regular, G2State initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states_x, float* states_y, float* integrals);
```

</details>

## Random-number strategy

Direct terminal helpers receive normals from the caller. Grid helpers keep one
`philox::UniformSequence(key, path)` and one normal cache alive. The state law
uses two normals per interval and the joint state/integral law uses three.

## Pricing kernels

Current rate options and zero-coupon-bond options are closed form and use one
thread per result. Files follow `PreparedRow` → `evaluate_price<Side>` →
launcher, with compile-time call/put specialization.

<details>
<summary>Exact analytics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float short_rate(const G2State&);
float log_discount_factor(const joint::G2JointState&);
float discount_factor(const joint::G2JointState&);
float zero_coupon_bond(const G2ModelParameters&, const G2State&, float valuation_time, float maturity);
float zero_coupon_bond_call_price(const G2ModelParameters&, const G2State&, float valuation_time, float expiry, float maturity, float strike);
float zero_coupon_bond_put_price(const G2ModelParameters&, const G2State&, float valuation_time, float expiry, float maturity, float strike);
float forward_rate(const G2ModelParameters&, const G2State&, float valuation_time, float start, float end, float accrual);
float swap_rate(const G2ModelParameters&, const G2State&, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

</details>

## Memory and numerical policy

Correlated Gaussian loadings are prepared once. Each OU factor reuses the
stable formulas in `../common/mean_reverting_gaussian.cuh`. State, joint state,
and analytics remain register-sized; optional grids use separate date-major
arrays for `x`, `y`, and their integral. Fast-math is forbidden.

## American and Bermudan options

No G2 American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/g2/),
[validation](../../../../validation/model/fixed_income/g2/),
[fixed-income Gaussian helpers](../common/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

# Schöbel–Zhu

| At a glance | Value |
|---|---|
| Process | Gaussian stochastic volatility |
| Transition | Exact OU endpoint + log-spot Euler step |
| Path state | `log_spot`, `volatility` |
| Random laws | Normal |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements stochastic volatility driven by an
Ornstein–Uhlenbeck process:

```text
dS_t / S_t = (r - q) dt + v_t dW_t^S
dv_t       = kappa(theta - v_t) dt + gamma dW_t^v
d<W^S,W^v>_t = rho dt.
```

`W^S` and `W^v` are standard Brownian motions with instantaneous correlation
`rho`. `v_t` is a Gaussian Ornstein–Uhlenbeck volatility process, not a CIR
variance process; it is therefore a signed state in the implementation.

See Schöbel and Zhu, *Stochastic Volatility With an Ornstein–Uhlenbeck
Process: An Extension* (1999), available from the authors' institution as a
[primary working-paper version](https://www.econstor.eu/handle/10419/104833).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement the volatility/spot scheme and path summaries.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $v_0$ | `initial_volatility` |
| $\kappa$ | `mean_reversion` |
| $\theta$ | `long_run_volatility` |
| $\gamma$ | `volatility_of_volatility` |
| $\rho$ | `correlation` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot`, `initial_volatility` | $\log S_0$, $v_0$ |
| `long_run_volatility` | $\theta$ |
| `exp_mean_reversion_dt` | $e^{-\kappa\Delta t}$ |
| `ou_std` | Exact OU endpoint variance |
| `endpoint_increment_correlation`, `endpoint_increment_residual` | Exact OU endpoint/Brownian coupling |
| `drift_dt`, `sqrt_dt` | $(r-q)\Delta t$, $\sqrt{\Delta t}$ |
| `correlation`, `correlation_residual` | $\rho$, $\sqrt{1-\rho^2}$ |

`SchobelZhuState` contains `log_spot` and the signed OU `volatility`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one OU/Euler step |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

The volatility endpoint is sampled exactly with its correct joint Gaussian
coupling to the interval Brownian increment. The log spot is advanced with the
left-end volatility, so terminal simulation still follows a numerical grid.
Each step uses three independent normals.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
SchobelZhuPreparedParameters prepare_model(const SchobelZhuModelParameters&, float maturity, std::size_t steps);
SchobelZhuState initial_state(const SchobelZhuPreparedParameters&);
void one_step_transition(const SchobelZhuPreparedParameters&, float ou_normal, float increment_residual_normal, float asset_residual_normal, SchobelZhuState&);
SchobelZhuState simulate_terminal_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuMeanPathResult simulate_mean_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuGeometricMeanPathResult simulate_geometric_mean_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuTwoTimePathResult simulate_at_two_times(const SchobelZhuPreparedParameters& first, const SchobelZhuPreparedParameters& second, philox::PhiloxKey, std::size_t path, std::size_t first_steps, std::size_t second_steps);
SchobelZhuMaximumPathResult simulate_maximum_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuState simulate_on_regular_grid(const SchobelZhuPreparedParameters& stub, const SchobelZhuPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t stub_steps, std::uint32_t steps_per_exercise, std::uint32_t exercise_count, std::size_t path_count, float* spots, float* volatilities);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
The three step normals are consumed in a fixed order and the sequence is never
restarted.

## Pricing kernels

All current products use the standard one-block-per-price Monte Carlo kernel:
one prepared row, strided paths, FP64 moments, block reduction, and FP32
outputs. Call/put behavior is a compile-time `OptionSide`.

## Memory and numerical policy

Ordinary payoffs keep `(log_spot, volatility)` in registers and retain only
their payoff statistic. The regular-grid helper exposes separate date-major
spot and volatility arrays when both are requested. Prepared Gaussian
loadings avoid repeated exponentials in hot loops. Fast-math is forbidden.

## American and Bermudan options

No Schöbel–Zhu American/Bermudan launcher is currently present in this
directory.

Related navigation: [model catalog](../../../../catalog/model/equity/schobel_zhu/),
[Premia validation](../../../../validation/premia/model/equity/schobel_zhu/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

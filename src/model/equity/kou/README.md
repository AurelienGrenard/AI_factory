# Kou jump diffusion

| At a glance | Value |
|---|---|
| Process | Lognormal diffusion + double-exponential jumps |
| Transition | Exact over each observation interval |
| Path state | `log_spot` |
| Random laws | Normal + Poisson + exponential |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements geometric Brownian diffusion with compound-Poisson
double-exponential log jumps:

```text
dS_t / S_(t-) = (r - q - lambda E[J-1]) dt + sigma dW_t + (J-1) dN_t.
```

A log jump is positive exponential with probability `p` and negative
exponential otherwise. See [Kou (2002)](https://doi.org/10.1287/mnsc.48.8.1086.166).

`W` is a standard Brownian motion. `N` is an independent Poisson process with
intensity `lambda`. At a jump, the spot is multiplied by `J = exp(Y)`. With
probability `p`, `Y = E_+` and `E_+ ~ Exponential(eta_+)`; otherwise
`Y = -E_-` and `E_- ~ Exponential(eta_-)`. All jump variables are independent
of `W`. The drift subtracts `lambda E[J-1]`.

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact finite-interval increments.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\sigma$ | `volatility` |
| $\lambda$ | `jump_intensity` |
| $p$ | `up_probability` |
| $\eta_+$ | `positive_jump_rate` |
| $\eta_-$ | `negative_jump_rate` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot` | $\log S_0$ |
| `drift_dt` | $(r-q-\lambda E[J-1]-\sigma^2/2)\Delta t$ |
| `diffusion_std` | $\sigma\sqrt{\Delta t}$ |
| `poisson_mean` | $\lambda\Delta t$ |
| `zero_jump_probability` | $e^{-\lambda\Delta t}$ |
| `up_probability` | $p$ |
| `inverse_positive_jump_rate` | $1/\eta_+$ |
| `inverse_negative_jump_rate` | $1/\eta_-$ |

`KouState` stores only `log_spot`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one exact interval law |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from caller-supplied variates |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return maturity state and arithmetic mean |
| `simulate_geometric_mean_state` | Return maturity state and geometric mean |
| `simulate_at_two_times` | Return two requested boundary states |
| `simulate_maximum_state` | Return maturity state and monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

Every interval is exact: draw its Poisson count, then draw the sign and
exponential magnitude of each realized jump, and finally add the Gaussian
diffusion increment. Direct terminal and two-time helpers avoid artificial
substeps; path-dependent products use exact observation-to-observation laws.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
KouPreparedParameters prepare_model(const KouModelParameters&, float interval);
KouPreparedParameters prepare_model(const KouModelParameters&, float maturity, std::size_t steps);
KouState initial_state(const KouPreparedParameters&);
void one_step_transition(const KouPreparedParameters&, float diffusion_normal, float jump_log_sum, KouState&);
KouState simulate_terminal_state(const KouPreparedParameters&, philox::PhiloxKey, std::size_t path);
KouMeanPathResult simulate_mean_state(const KouPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
KouGeometricMeanPathResult simulate_geometric_mean_state(const KouPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
KouTwoTimePathResult simulate_at_two_times(const KouPreparedParameters& first, const KouPreparedParameters& second, philox::PhiloxKey, std::size_t path);
KouMaximumPathResult simulate_maximum_state(const KouPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
KouState simulate_on_regular_grid(const KouPreparedParameters& stub, const KouPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
The variable number of uniforms required by realized jumps advances only that
path's local sequence and cannot shift another path's mapping.

## Pricing kernels

All current products use the standard one-block-per-price Monte Carlo kernel:
one `PreparedRow`, strided path evaluation, FP64 moment accumulation, block
reduction, and FP32 outputs. Call/put is a compile-time `OptionSide`.

## Memory and numerical policy

Only `log_spot` and payoff statistics remain live per path. Observation grids
contain spots only. Drift, Poisson, and reciprocal-rate constants are prepared
once outside hot loops. Fast-math is forbidden.

## American and Bermudan options

No Kou American/Bermudan launcher is currently present in this directory.

Related navigation: [model catalog](../../../../catalog/model/equity/kou/),
[validation](../../../../validation/model/equity/kou/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

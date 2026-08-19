# Merton jump diffusion

| At a glance | Value |
|---|---|
| Process | Lognormal diffusion + lognormal jumps |
| Transition | Exact over each observation interval |
| Path state | `log_spot` |
| Random laws | Normal + Poisson |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements geometric Brownian diffusion with independent
compound-Poisson Gaussian log jumps:

```text
dS_t / S_(t-) = (r - q - lambda E[J-1]) dt + sigma dW_t + (J-1) dN_t.
```

`W` is a standard Brownian motion. `N` is an independent Poisson process with
intensity `lambda`. At each event, the spot is multiplied by `J = exp(Y)`,
where the jump logs are independent variables
`Y ~ Normal(mu_J, sigma_J^2)`. The term `lambda E[J-1]`, with
`E[J] = exp(mu_J + sigma_J^2/2)`, is the martingale compensator.

See [Merton (1976)](https://doi.org/10.1016/0304-405X(76)90022-2).

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
| $\mu_J$ | `jump_log_mean` |
| $\sigma_J$ | `jump_log_volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot` | $\log S_0$ |
| `drift_dt` | $(r-q-\lambda E[J-1]-\sigma^2/2)\Delta t$ |
| `diffusion_std` | $\sigma\sqrt{\Delta t}$ |
| `poisson_mean` | $\lambda\Delta t$ |
| `zero_jump_probability` | $e^{-\lambda\Delta t}$ |
| `jump_log_mean` | $\mu_J$ |
| `jump_log_volatility` | $\sigma_J$ |

`MertonState` stores only `log_spot`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one exact interval law |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from caller-supplied variates |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

Every interval is exact. One Poisson draw gives the jump count; conditional on
that count, the sum of Gaussian jump logs is itself Gaussian. Direct terminal
and two-time helpers therefore do not use an artificial `dt`; monitored
payoffs use exact increments between observation dates.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
MertonPreparedParameters prepare_model(const MertonModelParameters&, float interval);
MertonPreparedParameters prepare_model(const MertonModelParameters&, float maturity, std::size_t steps);
MertonState initial_state(const MertonPreparedParameters&);
void one_step_transition(const MertonPreparedParameters&, std::uint32_t jump_count, float diffusion_normal, float jump_normal, MertonState&);
MertonState simulate_terminal_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path);
MertonMeanPathResult simulate_mean_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
MertonGeometricMeanPathResult simulate_geometric_mean_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
MertonTwoTimePathResult simulate_at_two_times(const MertonPreparedParameters& first, const MertonPreparedParameters& second, philox::PhiloxKey, std::size_t path);
MertonMaximumPathResult simulate_maximum_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
MertonState simulate_on_regular_grid(const MertonPreparedParameters& stub, const MertonPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
An interval consumes one Poisson uniform, one diffusion normal, and a jump
normal only when the count is nonzero. No helper creates a second sequence.

## Pricing kernels

All current products use one block per price. Threads evaluate strided paths,
accumulate payoff and squared-payoff sums in FP64, and reduce to an FP32 price
and standard error. Call/put behavior is a compile-time `OptionSide`.

## Memory and numerical policy

The evolving state is one register scalar. Path summaries retain only their
requested statistic; regular-grid storage contains spots only. Exact interval
simulation minimizes both work and discretization state. Fast-math is
forbidden.

## American and Bermudan options

No Merton American/Bermudan launcher is currently present in this directory.

Related navigation: [model catalog](../../../../catalog/model/equity/merton/),
[validation](../../../../validation/model/equity/merton/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

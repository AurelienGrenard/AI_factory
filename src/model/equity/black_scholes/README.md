# Black–Scholes

| At a glance | Value |
|---|---|
| Process | Lognormal diffusion |
| Transition | Exact over each observation interval |
| Path state | `log_spot` |
| Random laws | Normal |
| Pricing | Closed form or Monte Carlo, depending on the product |
| Early exercise | Not implemented |

## Role and reference

This directory implements the risk-neutral geometric Brownian motion

```text
dS_t / S_t = (r - q) dt + sigma dW_t.
```

`W` is a standard Brownian motion under the risk-neutral measure. `r`, `q`,
and `sigma` are constant for one model row.

Its log-price transition is exact for every requested interval. The model is
from [Black and Scholes (1973)](https://doi.org/10.1086/260062).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the compact host-side model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact device-side transitions and path summaries.
- each other `<product>.cuh/.cu` pair owns one pricing launcher and its kernels.

## Dataset row

`BlackScholesModelParameters` is trivially copyable and transferred as one
contiguous FP32 array.

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\sigma$ | `volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot` | $\log S_0$ |
| `drift` | $(r-q-\sigma^2/2)\Delta t$ |
| `standard_deviation` | $\sigma\sqrt{\Delta t}$ |

`BlackScholesState` stores only `log_spot`; products never carry an unused
volatility state.

The mean, geometric-mean, two-time, and maximum result structures add exactly
the statistic requested by the payoff.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one interval or regular-step law |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from caller-supplied variates |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

`prepare_model(parameters, time_interval)` prepares a direct exact increment.
The `(maturity, num_steps)` overload prepares an exact increment on a monitored
regular grid. Terminal and two-time products therefore do not introduce an
artificial daily grid; path-dependent products step only at their observation
dates.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
BlackScholesPreparedParameters prepare_model(const BlackScholesModelParameters&, float interval);
BlackScholesPreparedParameters prepare_model(const BlackScholesModelParameters&, float maturity, std::size_t steps);
BlackScholesState initial_state(const BlackScholesPreparedParameters&);
void one_step_transition(const BlackScholesPreparedParameters&, float normal, BlackScholesState&);
BlackScholesState simulate_terminal_state(const BlackScholesPreparedParameters&, philox::PhiloxKey, std::size_t path);
BlackScholesMeanPathResult simulate_mean_state(const BlackScholesPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BlackScholesGeometricMeanPathResult simulate_geometric_mean_state(const BlackScholesPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BlackScholesTwoTimePathResult simulate_at_two_times(const BlackScholesPreparedParameters& first, const BlackScholesPreparedParameters& second, philox::PhiloxKey, std::size_t path);
BlackScholesMaximumPathResult simulate_maximum_state(const BlackScholesPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BlackScholesState simulate_on_regular_grid(const BlackScholesPreparedParameters& stub, const BlackScholesPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Every path owns one `philox::UniformSequence(key, path)` and one
`philox::NormalPairCache`. Exact terminal simulation consumes one normal.
Multi-date helpers keep the same sequence alive across all observations.

## Pricing kernels

Closed-form products use one thread per price: asset-or-nothing, digital,
European, forward-start, gap, geometric Asian, range accrual, and straddle.
Their files share the same `PreparedRow` → `compute_price` → launcher shape.

The remaining products use the standard Monte Carlo organization: one block
works on one price, threads evaluate paths, moments accumulate in FP64, and a
block reduction produces the FP32 price and standard error. Call and put sides
are compile-time `OptionSide` specializations, not runtime branches.

## Memory and numerical policy

Path state and payoff statistics remain in registers. Only products that need
dated observations write spot-only, date-major arrays. Exact transitions are
preferred whenever the payoff permits them. Fast-math is forbidden and the
Philox mapping is independent of CUDA scheduling.

## American and Bermudan options

No Black–Scholes American/Bermudan launcher is currently present in this
directory.

Related navigation: [model catalog](../../../../catalog/model/equity/black_scholes/),
[validation](../../../../validation/model/equity/black_scholes/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

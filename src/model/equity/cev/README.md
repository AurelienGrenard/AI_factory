# CEV

| At a glance | Value |
|---|---|
| Process | Local-volatility diffusion |
| Transition | Absorbed Milstein numerical scheme |
| Path state | `spot` |
| Random laws | Normal |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements the constant-elasticity-of-variance local-volatility
model

```text
dS_t = (r - q) S_t dt + sigma S_t^beta dW_t.
```

`W` is a standard Brownian motion under the risk-neutral measure. The local
diffusion coefficient is the power function `sigma S_t^beta`.

The model belongs to the alternative diffusions introduced by
[Cox and Ross (1976)](https://doi.org/10.1016/0304-405X(76)90023-4).
The CUDA implementation uses an absorbed Milstein discretization.

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement the absorbed Milstein scheme and path summaries.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\sigma$ | `sigma` |
| $\beta$ | `beta` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_spot` | $S_0$ |
| `drift_dt` | $(r-q)\Delta t$ |
| `diffusion_scale` | $\sigma\sqrt{\Delta t}$ |
| `milstein_scale` | $\beta\sigma^2\Delta t/2$ |
| `beta` | $\beta$ |

`CevState` contains only `spot`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one Milstein step |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from a supplied normal |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

`one_step_transition` evaluates `S^beta` once, reuses it in the diffusion and
Milstein correction, and absorbs a non-positive candidate at zero. Unlike the
exact-increment models, terminal simulation uses the requested numerical grid.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
CevPreparedParameters prepare_model(const CevModelParameters&, float maturity, std::size_t steps);
CevState initial_state(const CevPreparedParameters&);
void one_step_transition(const CevPreparedParameters&, float normal, CevState&);
CevState simulate_terminal_state(const CevPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
CevMeanPathResult simulate_mean_state(const CevPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
CevGeometricMeanPathResult simulate_geometric_mean_state(const CevPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
CevTwoTimePathResult simulate_at_two_times(const CevPreparedParameters& first, const CevPreparedParameters& second, philox::PhiloxKey, std::size_t path, std::size_t first_steps, std::size_t second_steps);
CevMaximumPathResult simulate_maximum_state(const CevPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
CevState simulate_on_regular_grid(const CevPreparedParameters& stub, const CevPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t stub_steps, std::uint32_t steps_per_exercise, std::uint32_t exercise_count, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one
`philox::NormalPairCache`. One normal is consumed per Milstein step and the
sequence remains alive for the entire path.

## Pricing kernels

All current products use one block per price. Threads evaluate strided paths,
accumulate payoff moments in FP64, and reduce to an FP32 price and standard
error. Call and put sides are compile-time `OptionSide` specializations.

## Memory and numerical policy

The path state is one scalar and path summaries retain only their requested
statistic. Regular-grid storage contains spots only. The step size is part of
the approximation and must follow the repository simulation convention.
Fast-math is forbidden.

## American and Bermudan options

No CEV American/Bermudan launcher is currently present in this directory.

Related navigation: [model catalog](../../../../catalog/model/equity/cev/),
[QuantLib validation](../../../../validation/quantlib/model/equity/cev/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

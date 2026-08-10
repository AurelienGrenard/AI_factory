# Variance Gamma

| At a glance | Value |
|---|---|
| Process | Pure-jump Lévy process through a Gamma clock |
| Transition | Exact over each observation interval |
| Path state | `log_spot` |
| Random laws | Gamma + normal |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Longstaff–Schwartz |

## Role and reference

This directory implements the exponential Variance-Gamma model as Brownian
motion evaluated on an independent Gamma clock:

```text
X_t = theta G_t + sigma W_(G_t),    E[G_t] = t,
S_t = S_0 exp((r - q + omega)t + X_t).
```

`W` is a standard Brownian motion. `G` is an independent Gamma subordinator
with independent increments and
`G_t ~ Gamma(shape=t/nu, scale=nu)`, hence `E[G_t]=t`. Conditional on `G_t`,
the increment `X_t` is Gaussian. `omega` is the deterministic martingale
correction derived from `sigma`, `nu`, and `theta`.

See [Madan, Carr, and Chang (1998)](https://doi.org/10.1023/A:1009703431535).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact Lévy increments and path summaries.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

The Gamma shape and scale are derived from the single clock parameter `nu`;
they are not independent dataset inputs.

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\sigma$ | `sigma` |
| $\nu$ | `nu` |
| $\theta$ | `theta` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot` | $\log S_0$ |
| `gamma_shape` | $\Delta t/\nu$ |
| `gamma_scale` | $\nu$ |
| `theta`, `sigma` | $\theta$, $\sigma$ |
| `drift_dt` | $(r-q+\omega)\Delta t$ |

The state contains only `log_spot`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one exact interval law |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from a clock draw and normal |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return maturity state and arithmetic mean |
| `simulate_geometric_mean_state` | Return maturity state and geometric mean |
| `simulate_at_two_times` | Return two requested boundary states |
| `simulate_maximum_state` | Return maturity state and monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

Each interval increment is exact. Terminal and two-time products use direct
interval laws; monitored products use exact independent increments between
observation dates. `one_step_transition` receives the sampled Gamma clock and
one normal for the subordinated Brownian motion.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
VarianceGammaPreparedParameters prepare_model(const VarianceGammaModelParameters&, float interval);
VarianceGammaPreparedParameters prepare_model(const VarianceGammaModelParameters&, float maturity, std::size_t steps);
VarianceGammaState initial_state(const VarianceGammaPreparedParameters&);
void one_step_transition(const VarianceGammaPreparedParameters&, float gamma_increment, float normal, VarianceGammaState&);
VarianceGammaState simulate_terminal_state(const VarianceGammaPreparedParameters&, philox::PhiloxKey, std::size_t path);
VarianceGammaMeanPathResult simulate_mean_state(const VarianceGammaPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
VarianceGammaGeometricMeanPathResult simulate_geometric_mean_state(const VarianceGammaPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
VarianceGammaTwoTimePathResult simulate_at_two_times(const VarianceGammaPreparedParameters& first, const VarianceGammaPreparedParameters& second, philox::PhiloxKey, std::size_t path);
VarianceGammaMaximumPathResult simulate_maximum_state(const VarianceGammaPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
VarianceGammaState simulate_on_regular_grid(const VarianceGammaPreparedParameters& stub, const VarianceGammaPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t exercise_count, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
Gamma increments use the Marsaglia–Tsang rejection algorithm
([Marsaglia and Tsang, 2000](https://doi.org/10.1145/358407.358414)) on that
same scalar stream. Variable rejection counts remain local to the path.

## Pricing kernels

All current products use the standard one-block-per-price Monte Carlo kernel.
Threads evaluate strided paths, accumulate FP64 payoff moments, and reduce to
FP32 price and standard error. Call/put is a compile-time `OptionSide`.

## Memory and numerical policy

The evolving path state is one scalar. Path summaries retain only the required
statistic and early-exercise grids contain spots only. Exact interval laws
avoid artificial `dt` for terminal claims. Fast-math is forbidden.

## American and Bermudan options

`american_option.cuh/.cu` use the shared Longstaff–Schwartz pipeline. Only
spots are stored because the Lévy clock is an increment construction, not an
additional Markov state used by the regression basis.

Related navigation: [model catalog](../../../../catalog/model/equity/variance_gamma/),
[QuantLib validation](../../../../validation/quantlib/model/equity/variance_gamma/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[American/Bermudan contract](../../../../docs/cuda-american-and-bermudan-pricing-contract.md).

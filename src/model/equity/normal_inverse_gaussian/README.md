# Normal Inverse Gaussian

| At a glance | Value |
|---|---|
| Process | Pure-jump Lévy process through an inverse-Gaussian clock |
| Transition | Exact over each observation interval |
| Path state | `log_spot` |
| Random laws | Inverse Gaussian + normal |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Longstaff–Schwartz |

## Role and reference

This directory implements the exponential Normal-Inverse-Gaussian model as a
Brownian motion with drift evaluated on an inverse-Gaussian clock:

```text
X_t = beta G_t + W_(G_t),
S_t = S_0 exp((r - q + omega)t + X_t).
```

`W` is a standard Brownian motion. `G` is an independent inverse-Gaussian
subordinator. With `gamma = sqrt(alpha^2-beta^2)`, the implementation uses
`G_t ~ IG(mean=delta t/gamma, shape=(delta t)^2)`. Conditional on `G_t`,
`X_t` is Gaussian. `omega` is the deterministic martingale correction.

The input uses the standard NIG `alpha`, `beta`, and `delta` parameters and
derives the risk-neutral location correction. See
[Barndorff-Nielsen (1997)](https://doi.org/10.1111/1467-9469.00045).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact Lévy increments and path summaries.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

There is no redundant free location parameter in the dataset.

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\alpha$ | `alpha` |
| $\beta$ | `beta` |
| $\delta$ | `delta` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot` | $\log S_0$ |
| `inverse_gaussian_mean` | $\delta\Delta t/\sqrt{\alpha^2-\beta^2}$ |
| `inverse_gaussian_shape` | $(\delta\Delta t)^2$ |
| `beta` | $\beta$ |
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

Every interval increment is exact. Terminal and two-time products use direct
interval laws; monitored products advance through exact independent increments
at observation dates. The transition receives one inverse-Gaussian clock
increment and one normal.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
NormalInverseGaussianPreparedParameters prepare_model(const NormalInverseGaussianModelParameters&, float interval);
NormalInverseGaussianPreparedParameters prepare_model(const NormalInverseGaussianModelParameters&, float maturity, std::size_t steps);
NormalInverseGaussianState initial_state(const NormalInverseGaussianPreparedParameters&);
void one_step_transition(const NormalInverseGaussianPreparedParameters&, float inverse_gaussian_increment, float normal, NormalInverseGaussianState&);
NormalInverseGaussianState simulate_terminal_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path);
NormalInverseGaussianMeanPathResult simulate_mean_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
NormalInverseGaussianGeometricMeanPathResult simulate_geometric_mean_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
NormalInverseGaussianTwoTimePathResult simulate_at_two_times(const NormalInverseGaussianPreparedParameters& first, const NormalInverseGaussianPreparedParameters& second, philox::PhiloxKey, std::size_t path);
NormalInverseGaussianMaximumPathResult simulate_maximum_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
NormalInverseGaussianState simulate_on_regular_grid(const NormalInverseGaussianPreparedParameters& stub, const NormalInverseGaussianPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t exercise_count, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
The clock uses the Michael–Schucany–Haas generator
([Michael, Schucany, and Haas, 1976](https://doi.org/10.1080/00031305.1976.10479147))
on that same stream; the subordinated Brownian draw follows from the same
normal cache.

## Pricing kernels

All current products use the standard one-block-per-price Monte Carlo kernel:
one prepared row, strided paths, FP64 moment accumulation, block reduction,
and FP32 outputs. Call/put is a compile-time `OptionSide`.

## Memory and numerical policy

The evolving state is one scalar. Path summaries retain only their requested
statistic and early-exercise grids store spots only. Exact interval laws avoid
artificial `dt` for terminal claims. Fast-math is forbidden.

## American and Bermudan options

`american_option.cuh/.cu` use the shared Longstaff–Schwartz pipeline and a
spot-only exercise grid.

Related navigation: [model catalog](../../../../catalog/model/equity/normal_inverse_gaussian/),
[validation infrastructure](../../../../validation/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[American/Bermudan contract](../../../../docs/cuda-american-and-bermudan-pricing-contract.md).

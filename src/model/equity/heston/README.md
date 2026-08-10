# Heston

| At a glance | Value |
|---|---|
| Process | Stochastic variance diffusion |
| Transition | Andersen QE-M numerical scheme |
| Path state | `log_spot`, `variance` |
| Random laws | Normal + uniform |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Longstaff–Schwartz |

## Role and reference

This directory implements the risk-neutral Heston stochastic-volatility model

```text
dS_t / S_t = (r - q) dt + sqrt(v_t) dW_t^S
dv_t       = kappa(theta - v_t) dt + gamma sqrt(v_t) dW_t^v
d<W^S,W^v>_t = rho dt.
```

`W^S` and `W^v` are standard Brownian motions with instantaneous correlation
`rho`. `v_t` is the CIR square-root variance process; `theta` is its level,
`kappa` its mean-reversion speed, and `gamma` its diffusion coefficient.

The model is from [Heston (1993)](https://doi.org/10.1093/rfs/6.2.327).
Simulation uses Andersen's quadratic-exponential scheme with martingale
correction (QE-M), described in
[Andersen (2008)](https://doi.org/10.21314/JCF.2008.189).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement QE-M transitions and path summaries.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $v_0$ | `initial_variance` |
| $\kappa$ | `kappa` |
| $\theta$ | `theta` |
| $\gamma$ | `gamma` |
| $\rho$ | `rho` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot`, `initial_variance` | $\log S_0$, $v_0$ |
| `theta`, `exp_kdt` | $\theta$, $e^{-\kappa\Delta t}$ |
| `variance_linear_scale`, `variance_constant_scale` | CIR conditional moments |
| `drift_dt` | $(r-q)\Delta t$ |
| `k0`, `k1`, `k2`, `k3`, `k4` | QE-M log-spot coefficients |
| `martingale_a` | QE-M martingale correction |

These values are prepared once per price and reused by all paths.

`HestonState` contains only `log_spot` and `variance`. The path-summary
structures add one payoff statistic without retaining the complete trajectory.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one numerical-step law |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from caller-supplied variates |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return maturity state and arithmetic mean |
| `simulate_geometric_mean_state` | Return maturity state and geometric mean |
| `simulate_at_two_times` | Return two requested boundary states |
| `simulate_maximum_state` | Return maturity state and monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

`one_step_transition` receives the two normals and one uniform required by
QE-M. Complete-path helpers create the random stream and repeatedly apply that
transition. `simulate_on_regular_grid` stores pre-maturity spots and variances
in two separate date-major SoA regions; the maturity state is returned.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
HestonQeParameters prepare_model(const HestonModelParameters&, float maturity, std::size_t steps);
HestonState initial_state(const HestonQeParameters&);
void one_step_transition(const HestonQeParameters&, float variance_normal, float variance_uniform, float stock_normal, HestonState&);
HestonState simulate_terminal_state(const HestonQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
HestonMeanPathResult simulate_mean_state(const HestonQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
HestonGeometricMeanPathResult simulate_geometric_mean_state(const HestonQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
HestonTwoTimePathResult simulate_at_two_times(const HestonQeParameters& first, const HestonQeParameters& second, philox::PhiloxKey, std::size_t path, std::size_t first_steps, std::size_t second_steps);
HestonMaximumPathResult simulate_maximum_state(const HestonQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
HestonState simulate_on_regular_grid(const HestonQeParameters& stub, const HestonQeParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t stub_steps, std::uint32_t steps_per_exercise, std::uint32_t exercise_count, std::size_t path_count, float* spots, float* variances);
```

</details>

## Random-number strategy

Each path constructs one `philox::UniformSequence(key, path)` and one
`philox::NormalPairCache`. The sequence is never restarted inside a path.
Every QE-M step consumes two normals and one uniform in a fixed order.

## Pricing kernels

All current Heston products use Monte Carlo. One persistent CUDA block handles
one price at a time, prepares one `PreparedRow`, evaluates strided paths, sums
payoff moments in FP64, and reduces them to an FP32 price and standard error.
Call and put sides are compile-time `OptionSide` specializations.

## Memory and numerical policy

Ordinary European and path-dependent products keep `(log_spot, variance)` in
registers and retain only the payoff statistic. Global state grids exist only
for early exercise. Their spot and variance fields are separate to preserve
coalesced access and avoid AoS traffic. Fast-math is forbidden.

## American and Bermudan options

`american_option.cuh/.cu` use the shared Longstaff–Schwartz infrastructure in
`src/common/longstaff_schwartz`. Both spot and variance grids are retained
because the active two-factor regression basis uses them.

Related navigation: [model catalog](../../../../catalog/model/equity/heston/),
[validation](../../../../validation/model/equity/heston/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[American/Bermudan contract](../../../../docs/cuda-american-and-bermudan-pricing-contract.md).

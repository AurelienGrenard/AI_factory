# Bates

| At a glance | Value |
|---|---|
| Process | Heston stochastic variance + lognormal jumps |
| Transition | Heston QE-M + exact compound-Poisson interval sum |
| Path state | `log_spot`, `variance` |
| Random laws | Normal + uniform + Poisson |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Longstaff–Schwartz |

## Role and reference

This directory extends Heston with independent compound-Poisson lognormal
jumps:

```text
dS_t / S_(t-) = (r - q - lambda E[J-1]) dt
                + sqrt(v_t) dW_t^S + (J-1) dN_t,
```

while `v_t` follows the Heston CIR variance process. See
[Bates (1996)](https://doi.org/10.1093/rfs/9.1.69). The diffusion part uses
Andersen QE-M.

`W^S` and `W^v` are the correlated Brownian motions defined by Heston. `N` is
a Poisson process with intensity `lambda`. At a jump, the spot is multiplied
by `J = exp(Y)`, where the independent jump logs are
`Y ~ Normal(mu_J, sigma_J^2)`. The Poisson process, jump sizes, and Heston
Brownian motions are mutually independent. The drift subtracts
`lambda E[J-1]` to preserve the discounted martingale.

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load Heston-plus-jump rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) combine Heston QE-M with exact compound-Poisson increments.
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
| $\lambda$ | `jump_intensity` |
| $\mu_J$ | `jump_log_mean` |
| $\sigma_J$ | `jump_log_volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `heston` | Prepared Heston QE-M parameters |
| `poisson_mean` | $\lambda\Delta t$ |
| `poisson_zero_probability` | $e^{-\lambda\Delta t}$ |
| `jump_log_mean` | $\mu_J$ |
| `jump_log_volatility` | $\sigma_J$ |
| `jump_compensator` | $\lambda E[J-1]\Delta t$ |

`BatesState` is exactly `HestonState`: `log_spot` and `variance`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute Heston and jump laws for one step |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from caller-supplied variates |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

For a payoff that observes only an interval boundary, the code simulates all
Heston QE steps and draws one Poisson count for the whole interval. Conditional
on that count, the sum of lognormal jump logs is one Gaussian draw. Fully
pathwise payoffs retain the one-step transition. This removes unnecessary
Poisson simulations without changing the boundary law.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
BatesQeParameters prepare_model(const BatesModelParameters&, float maturity, std::size_t steps);
BatesState initial_state(const BatesQeParameters&);
void one_step_transition(const BatesQeParameters&, float variance_normal, float variance_uniform, float stock_normal, std::uint32_t jump_count, float jump_normal, BatesState&);
BatesState simulate_terminal_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesMeanPathResult simulate_mean_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesGeometricMeanPathResult simulate_geometric_mean_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesTwoTimePathResult simulate_at_two_times(const BatesQeParameters& first, const BatesQeParameters& second, philox::PhiloxKey, std::size_t path, std::size_t first_steps, std::size_t second_steps);
BatesMaximumPathResult simulate_maximum_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesState simulate_on_regular_grid(const BatesQeParameters& stub, const BatesQeParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t stub_steps, std::uint32_t steps_per_exercise, std::uint32_t exercise_count, std::size_t path_count, float* spots, float* variances);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
The Poisson count is obtained by CDF inversion; the jump normal is drawn only
when the count is nonzero. Variable consumption remains local to the path.

## Pricing kernels

All current products use the standard one-block-per-price Monte Carlo kernel:
one `PreparedRow`, strided paths, FP64 payoff moments, block reduction, and
FP32 price/standard error. Call and put use compile-time `OptionSide`.

## Memory and numerical policy

Ordinary products retain only the two-scalar state and their payoff statistic.
Early-exercise grids use separate date-major spot and variance regions.
Prepared interval laws avoid repeated exponentials in hot loops. Fast-math is
forbidden.

## American and Bermudan options

`american_option.cuh/.cu` use the shared Longstaff–Schwartz pipeline and retain
both spot and variance for the two-factor regression basis.

Related navigation: [model catalog](../../../../catalog/model/equity/bates/),
[validation](../../../../validation/model/equity/bates/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[American/Bermudan contract](../../../../docs/cuda-american-and-bermudan-pricing-contract.md).

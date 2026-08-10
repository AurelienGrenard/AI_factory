# Rough Bergomi

| At a glance | Value |
|---|---|
| Process | Non-Markovian rough stochastic variance |
| Transition | `kappa=1` hybrid scheme, direct history convolution |
| Path state | `log_spot`, `variance`, external Brownian history |
| Random laws | Normal |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements rough Bergomi with a constant forward variance
`xi_0`:

```text
Y_t = sqrt(2H) integral_0^t (t-s)^(H-1/2) dW_s
v_t = xi_0 exp(eta Y_t - eta^2 t^(2H) / 2)
dS_t / S_t = (r-q) dt + sqrt(v_t) dZ_t,    d<Z,W>_t = rho dt.
```

`W` and `Z` are standard Brownian motions with instantaneous correlation
`rho`. `Y` is the Volterra Gaussian process obtained by convolving `W` with
the fractional power kernel. `v_t` is lognormal conditional on that rough
driver. The current implementation uses the constant forward-variance curve
`xi_0(t) = xi_0`.

The model follows [Bayer, Friz, and Gatheral (2016)](https://doi.org/10.1080/14697688.2015.1099717).
Simulation uses the `kappa=1` hybrid scheme of
[Bennedsen, Lunde, and Pakkanen (2017)](https://arxiv.org/abs/1507.03004):
the singular current cell is treated exactly and older cells use L2-optimal
power-kernel weights.

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load flat-`xi_0` model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement the non-Markovian hybrid simulation.
- [`european_option.cuh`](european_option.cuh) / [`european_option.cu`](european_option.cu) implement the current Monte Carlo product launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\xi_0$ | `xi_0` |
| $\eta$ | `eta` |
| $H$ | `hurst_exponent` |
| $\rho$ | `rho` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot`, `initial_variance`, `log_initial_variance` | $\log S_0$, $\xi_0$, $\log\xi_0$ |
| `eta`, `half_eta_squared` | $\eta$, $\eta^2/2$ |
| `hurst_exponent`, `alpha`, `alpha_plus_one` | $H$, $H-1/2$, $H+1/2$ |
| `two_h`, `sqrt_two_h` | $2H$, $\sqrt{2H}$ |
| `time_step`, `sqrt_time_step`, `time_step_to_alpha` | $\Delta t$, $\sqrt{\Delta t}$, $\Delta t^{H-1/2}$ |
| `drift_time_step` | $(r-q)\Delta t$ |
| `rho`, `orthogonal_correlation` | $\rho$, $\sqrt{1-\rho^2}$ |
| `singular_driver_loading`, `singular_independent_loading` | Exact current-cell Gaussian coupling |

`RoughBergomiState` keeps only `log_spot` and current `variance` in registers.

`RoughBergomiHistoryView` addresses one path's Brownian-increment lane in
global memory. `RoughBergomiHybridGridView` addresses deterministic far
weights and log-variance corrections in shared memory. History is therefore
not embedded in the state or allocated as a thread-local C++ array.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute scalar hybrid-scheme constants |
| `prepare_hybrid_grid` | Build deterministic weights and variance corrections in shared memory |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from supplied normals and path history |
| `simulate_one_step` | Draw one step from the single path-local random stream |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return maturity state and arithmetic mean |
| `simulate_geometric_mean_state` | Return maturity state and geometric mean |
| `simulate_at_two_times` | Return two states without restarting history |
| `simulate_maximum_state` | Return maturity state and monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated spots |

The non-Markovian model additionally exposes `prepare_hybrid_grid` and
`simulate_one_step`, and passes a grid view plus a history view to every path
helper. `simulate_at_two_times` advances once through a continuous history; it
never restarts the rough driver. The current direct convolution costs
`O(num_steps^2)` per path.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
RoughBergomiPreparedParameters prepare_model(const RoughBergomiModelParameters&, float maturity, std::size_t steps);
void prepare_hybrid_grid(const RoughBergomiPreparedParameters&, std::uint32_t steps, float* weights, float* corrections, std::uint32_t thread, std::uint32_t threads);
RoughBergomiState initial_state(const RoughBergomiPreparedParameters&);
void one_step_transition(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, std::uint32_t step, float driver_normal, float singular_normal, float spot_normal, RoughBergomiState&);
void simulate_one_step(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, std::uint32_t step, philox::UniformSequence&, philox::NormalPairCache&, RoughBergomiState&);
RoughBergomiState simulate_terminal_state(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, philox::PhiloxKey, std::size_t path, std::size_t steps);
RoughBergomiMeanPathResult simulate_mean_state(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, philox::PhiloxKey, std::size_t path, std::size_t steps);
RoughBergomiGeometricMeanPathResult simulate_geometric_mean_state(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, philox::PhiloxKey, std::size_t path, std::size_t steps);
RoughBergomiTwoTimePathResult simulate_at_two_times(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, philox::PhiloxKey, std::size_t path, std::uint32_t first_step, std::uint32_t terminal_step);
RoughBergomiMaximumPathResult simulate_maximum_state(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, philox::PhiloxKey, std::size_t path, std::size_t steps);
RoughBergomiState simulate_on_regular_grid(const RoughBergomiPreparedParameters&, const RoughBergomiHybridGridView&, RoughBergomiHistoryView, philox::PhiloxKey, std::size_t path, std::uint32_t stub_steps, std::uint32_t steps_per_observation, std::uint32_t observation_count, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
Every step consumes three normals: rough Brownian increment, independent part
of the singular-cell integral, and independent part of the spot increment.
The sequence is never restarted.

## Pricing kernels

The European option uses the standard one-block-per-price Monte Carlo shape,
extended with a workspace plan. A persistent block prepares one row, builds
the shared hybrid grid cooperatively, evaluates strided paths, accumulates
FP64 payoff moments, and reduces to FP32 outputs. Call/put is a compile-time
`OptionSide`.

## Memory and numerical policy

Registers hold the state, random sequence, normal cache, and payoff
accumulators. Shared memory holds `far_weights[num_steps]`,
`log_variance_corrections[num_steps]`, and the reduction workspace. Global
memory holds time-major history as `history[block][step][thread]`; threads of a
warp access contiguous values at a fixed step. The host allocates
`block_count * threads_per_block * maximum_step_count` floats once, and lanes
are reused across paths and prices. Fast-math is forbidden.

## American and Bermudan options

No rough-Bergomi American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/equity/rough_bergomi/),
[validation infrastructure](../../../../validation/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

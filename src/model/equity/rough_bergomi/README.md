# Rough Bergomi

| At a glance | Value |
|---|---|
| Process | Non-Markovian rough stochastic variance |
| Transition | `kappa=1` hybrid scheme, direct or cuFFTDx convolution |
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

## Formula index

- [Dynamics and hybrid simulation](#dynamics-interface)
- [European option payoff and Monte Carlo price](#european-option)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load flat-`xi_0` model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement the non-Markovian hybrid simulation.
- [`european_option.cuh`](european_option.cuh) / [`european_option.cu`](european_option.cu) implement the current Monte Carlo product launcher.
- [`european_option_fft.cuh`](european_option_fft.cuh) / [`european_option_fft.cu`](european_option_fft.cu) implement the optional SM-8.9 cuFFTDx launcher.

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
| `alpha_plus_one` | $H+1/2$ |
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
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return two spots without restarting history |
| `simulate_maximum_state` | Return only the monitored maximum |
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

## European option

For strike $K$, maturity $T$, and $\varepsilon=+1$ for a call or $-1$ for a
put,

$$
H=[\varepsilon(S_T-K)]^+,
\qquad
V_0=\mathbb E^{\mathbb Q}[e^{-rT}H].
$$

Both launchers estimate

$$
\widehat V_0=\frac1M\sum_{m=1}^M e^{-rT}H^{(m)}
$$

and its Monte Carlo standard error. The direct launcher constructs the rough
Volterra convolution from path history; the optional cuFFTDx launcher computes
the same hybrid convolution in frequency space.

## Pricing kernels

The European option uses the standard one-block-per-price Monte Carlo shape,
extended with a workspace plan. A persistent block prepares one row, builds
the shared hybrid grid cooperatively, evaluates strided paths, accumulates
FP64 payoff moments, and reduces to FP32 outputs. Call/put is a compile-time
`OptionSide`.

The optional FFT launcher prices one result row per host call and fills the GPU
with path pairs instead of price rows. It generates the rough Brownian cells by
random access to the canonical Philox stream, packs two real paths in one C2C,
and performs forward FFT, multiplication by the row-specific far-weight
spectrum, normalization, and inverse FFT with cuFFTDx. The inverse is written
to one compact chunk buffer. A second kernel assigns one thread to each path,
replays the same Philox normals sequentially, evolves variance and spot, and
reduces call or put payoffs in FP64. Only block moments survive the chunk.

This split is intentional. Consuming the inverse inside the FFT block removes
the chunk buffer, but leaves only one useful thread per packed path pair during
the sequential spot evolution. On the target GPU that fully fused experiment
was much slower than one compact global round trip followed by a dense payoff
kernel.

The FFT length is dispatched at compile time to the next power of two covering
the linear convolution: 16, 64, 128, 256, 512, 1024, 2048, 4096, or 8192. The
public call/put templates have explicit instantiations and no runtime payoff
branch.

### cuFFTDx build

mathDx is an optional external dependency. Enable the pricer on the measured
Ada target with:

```bash
cmake -S . -B build \
  -DCUDA_WORKBENCH_ARCHITECTURES=89 \
  -DAI_FACTORY_MATHDX_ROOT=/path/to/nvidia/mathdx/version
```

Without `AI_FACTORY_MATHDX_ROOT`, the original direct-history pricer and all
existing targets remain unchanged.

### Measured comparison

On an RTX 4090 Laptop with CUDA 12.8, 1,000 aligned catalogue rows and 8,192
paths per row took 9.545 s for direct-history calls versus 0.482 s for cuFFTDx;
puts took 9.599 s versus 0.481 s. Prices agreed within `1.2e-7` and standard
errors within `7.0e-10` because both paths use the same Philox mapping.

The chunk sweep at 65,536 paths per row measured:

| paths/chunk | 1,000 call rows | FFT workspace at N=2520 |
|---:|---:|---:|
| 4,096 | 6.350 s | 41.4 MB |
| 8,192 | 3.739 s | 82.7 MB |
| 16,384 | 2.540 s | 165.2 MB |
| 32,768 | 2.351 s | 330.4 MB |
| 65,536 | 2.337 s | 660.7 MB |

`32,768` is the recommended throughput/memory compromise. Linear scaling of
the measured catalogue workload gives roughly 38 seconds at `2^20` paths,
versus about 20 minutes for the direct implementation. This is a projection;
the checked-in CUDA test deliberately uses a smaller bounded workload.

## Memory and numerical policy

In the direct implementation, registers hold the state, random sequence,
normal cache, and payoff
accumulators. Shared memory holds `far_weights[num_steps]`,
`log_variance_corrections[num_steps]`, and the reduction workspace. Global
memory holds time-major history as `history[block][step][thread]`; threads of a
warp access contiguous values at a fixed step. The host allocates
`block_count * threads_per_block * maximum_step_count` floats once, and lanes
are reused across paths and prices. Fast-math is forbidden.

The FFT implementation keeps no Brownian history. Its reusable workspace holds
one spectrum, deterministic variance corrections, one compact convolution
chunk, and one FP64 moment pair per 256 paths. Workspace therefore depends on
the chosen chunk rather than the total Monte Carlo path count. For `N=2520`, a
32,768-path chunk occupies about 330 MB.

## American and Bermudan options

No rough-Bergomi American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/equity/rough_bergomi/),
[validation infrastructure](../../../../validation/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

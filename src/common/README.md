# Common CUDA infrastructure

<details>
<summary>Implementation</summary>

```text
common/
├── README.md
├── check_cuda.cuh
├── cuda_kernel_diagnostics.cuh
├── cuda_kernel_diagnostics.cpp
├── noncentral_chi_square.cuh
├── normal_distribution.cuh
├── option_side.cuh
├── philox.cuh
├── reductions.cuh
├── result_index.cuh
├── sample.cuh
├── time_grid.cuh
└── longstaff_schwartz/
    ├── exercise_schedule.cuh
    ├── exercise_schedule.cu
    ├── laguerre.cuh
    ├── launch.cuh
    ├── launch.cu
    ├── linear_solver.cuh
    ├── linear_solver.cu
    ├── regression.cuh
    ├── regression.cu
    ├── workspace.cuh
    └── workspace.cu
```

</details>

[`check_cuda.cuh`](#check-cuda) ·
[`cuda_kernel_diagnostics.cuh/.cpp`](#cuda-kernel-diagnostics) ·
[`option_side.cuh`](#option-side) ·
[`result_index.cuh`](#result-index) ·
[`time_grid.cuh`](#time-grid) ·
[`sample.cuh`](#sample) ·
[`philox.cuh`](#philox) ·
[`normal_distribution.cuh`](#normal-distribution) ·
[`noncentral_chi_square.cuh`](#noncentral-chi-square) ·
[`reductions.cuh`](#reductions) ·
[`longstaff_schwartz/`](#longstaff-schwartz)

<a id="check-cuda"></a>
## [`check_cuda.cuh`](check_cuda.cuh)

| Function | Definition |
|---|---|
| `check_cuda(status, operation)` | Throws a C++ exception containing `operation` and the CUDA Runtime error represented by `status`. |
| `checked_workspace_product(left, right, message)` | Returns `left * right` after rejecting `std::size_t` overflow. |
| `bounded_block_count(result_count, block_count)` | Returns `min(result_count, block_count)` after rejecting zero dimensions. |
| `validate_block_count(result_count, block_count)` | Requires `1 <= block_count <= result_count`. |
| `validate_device_pointer(pointer, name)` | Requires a non-null pointer whose CUDA attributes report device memory. |
| `validate_model_product_construction(...)` | Validates aligned rows or the Cartesian count `model_count * product_count`. |
| `validate_model_curve_product_construction(...)` | Validates aligned rows or the Cartesian count `model_count * curve_count * product_count`. |
| `validate_monte_carlo_path_count(paths_per_result)` | Requires at least two paths so a sample variance exists. |
| `validate_day_fraction(day_fraction)` | Requires a positive finite contractual day fraction. |
| `validate_monte_carlo_parameters(paths_per_result, dt)` | Validates the path count and the positive finite simulation step `dt`. |
| `validate_simulation_steps_per_day(steps)` | Requires at least one simulation step per contractual day. |
| `validate_cuda_block_size(threads_per_block)` | Checks the positive block size against the active device limit. |
| `validate_reduction_block_size(threads_per_block)` | Additionally requires a whole number of 32-thread warps. |
| `validate_grid_x_size(block_count)` | Checks `block_count` against the active device `gridDim.x` limit. |
| `validate_row_seed_range(result_count, base_seed)` | Requires `base_seed + result_count - 1` to fit in `std::uint64_t`. |

<a id="cuda-kernel-diagnostics"></a>
## [`cuda_kernel_diagnostics.cuh`](cuda_kernel_diagnostics.cuh) / [`cuda_kernel_diagnostics.cpp`](cuda_kernel_diagnostics.cpp)

For `B` active blocks per streaming multiprocessor, `W_B` warps per block and
`W_max` supported warps per streaming multiprocessor, the reported theoretical
occupancy is

```math
\mathrm{occupancy}=\frac{B W_B}{W_{\max}}.
```

| Function or type | Definition |
|---|---|
| `CudaKernelLaunchDiagnostics` | Device identity, launch geometry, kernel resources and theoretical occupancy. |
| `cuda_kernel_diagnostics_enabled()` | Reads the opt-in `AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS` flag. |
| `reserve_cuda_kernel_launch_diagnostics(...)` | Deduplicates one `(kernel, variant, grid, block, shared memory)` report. |
| `inspect_cuda_kernel_launch(kernel, grid, block, shared_bytes)` | Returns device limits, registers, local/shared memory, active blocks and theoretical occupancy for the exact specialization. |
| `emit_cuda_kernel_launch_diagnostics(...)` | Writes the reserved report as one JSON object. |
| `report_cuda_kernel_launch_if_enabled(...)` | Performs reservation, inspection and emission only when diagnostics are enabled. |
| `environment_flag_enabled(value)` | Internal parser accepting `1`, `true` or `on`. |
| `report_key(...)` | Internal stable key used to deduplicate reports. |

<a id="option-side"></a>
## [`option_side.cuh`](option_side.cuh)

| Symbol | Definition |
|---|---|
| `OptionSide::call` | Call orientation. |
| `OptionSide::put` | Put orientation. |
| `option_side_name(side)` | Returns the stable host label `"call"` or `"put"`. |

<a id="result-index"></a>
## [`result_index.cuh`](result_index.cuh)

Let `i` be a flattened result index and `P` the number of products.

| Function | Definition |
|---|---|
| `decode_model_product_result_index(i, P, cartesian)` | Returns `ModelProductIndices {model_index, product_index}`. |

Aligned construction uses `(i,i)`. Cartesian construction lets products vary
fastest:

```math
\mathrm{model\_index}=\left\lfloor\frac{i}{P}\right\rfloor,
\qquad
\mathrm{product\_index}=i\bmod P.
```

<a id="time-grid"></a>
## [`time_grid.cuh`](time_grid.cuh)

Let `n` be the positive number of grid steps per year and `k` an integer grid
index. `TimeGrid(n)` stores

```math
\Delta t=\frac{1}{n}.
```

| Function | Definition |
|---|---|
| `TimeGrid(n)` | Builds `{steps_per_year = n, step_size = 1/n}`. |
| `year_fraction(k, grid)` | Returns the grid time `t_k = k * grid.step_size`. |
| `validate(grid)` | Requires a positive `n` and exactly `step_size = 1/n`. |
| `index(t, grid, name)` | Returns the nearest integer index to `n t` after checking that `t` lies on the grid within `1e-4` grid step. |

<a id="sample"></a>
## [`sample.cuh`](sample.cuh)

The three Philox domains `kParameterDomain`, `kScheduleDomain` and
`kDynamicsDomain` keep parameter, calendar and path streams independent.

### Rows and bounds

| Type | Definition |
|---|---|
| `UniformBounds` | Closed interval `[minimum, maximum]`. |
| `RandomCalendarRules` | Observation-time bounds and minimum interval between observations. |
| `GeneratedTerminalTime` | Integer grid day and corresponding year fraction. |
| `TerminalSampleRow<ModelParameters, SampleValues>` | Model parameters, maturity and terminal values. |
| `CalendarSampleRow<ModelParameters, SampleValues, N>` | Model parameters, `N` observation times and `N` sampled values. |
| `ModelPathIndices` | Decoded model-row and conditional-path indices. |

Let `d_min` and `d_max` be the integer grid indices of the terminal bounds.
`generate_terminal_time` draws

```math
d\sim\mathcal{U}\{d_{\min},\ldots,d_{\max}\},
\qquad
T=d\,\Delta t.
```

For a calendar of `N` observations, `generate_random_calendar` draws every
integer day uniformly from its currently feasible interval. Let `g` be the
minimum gap, let `j` run from zero to `N-1`, and define

```math
\ell_j=
\begin{cases}
d_{\min}, & j=0,\\
d_{j-1}+g, & j>0,
\end{cases}
\qquad
u_j=d_{\max}-(N-j-1)g.
```

The next observation day is

```math
d_j\sim\mathcal{U}\{\ell_j,\ldots,u_j\}.
```

| Function | Definition |
|---|---|
| `generate_terminal_time(integers, bounds, grid)` | Draws one terminal grid day and its year fraction. |
| `generate_random_calendar<N>(...)` | Draws `N` strictly increasing feasible observation days. |
| `validate_uniform_bounds(bounds, name)` | Requires finite ordered bounds. |
| `validate_terminal_bounds(maturity, grid)` | Requires both terminal bounds to lie on the grid in increasing order. |
| `validate_generated_sample_launch(...)` | Validates output memory, package boundaries, launch slice and geometry. |
| `validate_random_calendar_rules<N>(rules, grid)` | Requires the requested minimum gaps to fit inside the allowed interval. |
| `validate_fixed_calendar<N>(times, grid)` | Requires `N` non-null, on-grid, strictly increasing times. |
| `decode_sample_index(i, P)` | Returns `{i / P, i % P}`, where `P` is the number of paths per model. |
| `sample_count(M, P)` | Returns `M P`, where `M` is the model count, after zero and overflow checks. |
| `validate_sample_launch(...)` | Validates model/sample arrays, launch slice, geometry and Philox seed range. |
| `validate_terminal_time(T)` | Requires a positive finite maturity `T`. |
| `validate_regular_calendar(t_1, delta, N)` | Requires positive finite first time and spacing, with `N > 0`. |
| `rounded_step_count(interval, target_dt)` | Returns the nearest positive integer to `interval / target_dt`. |
| `calendar_step_count(s_0, s, N)` | Returns `s_0 + (N-1)s` after `std::uint32_t` overflow checks. |

<a id="philox"></a>
## [`philox.cuh`](philox.cuh)

The counter-based generator is Philox-4x32-10 from
[Salmon et al. (2011)](https://doi.org/10.1145/2063384.2063405). A 64-bit seed
defines the two-word key; a path index `p` and local group index `g` define the
four-word counter:

```math
K=(\mathrm{low}_{32}(\mathrm{seed}),\mathrm{high}_{32}(\mathrm{seed})),
```

```math
C=(\mathrm{low}_{32}(p),\mathrm{high}_{32}(p),
   \mathrm{low}_{32}(g),\mathrm{high}_{32}(g)).
```

### Counter and streams

| Function or type | Definition |
|---|---|
| `PhiloxCounter` | Four 32-bit counter/output words. |
| `PhiloxKey` | Two 32-bit key words. |
| `RandomQuad` | Four FP32 uniforms. |
| `NormalPair` | Two independent FP32 standard normals. |
| `make_key(seed)` | Splits the 64-bit seed into a `PhiloxKey`. |
| `philox4x32_10(key, counter)` | Applies ten Philox multiplication, permutation and key-bump rounds. |
| `random_bits(key, p, g)` | Evaluates Philox at the counter identified by path `p` and group `g`. |
| `uint32_to_uniform(x)` | Maps one 32-bit word to the midpoint `(x+1/2)2^{-32}` and clamps below one. |
| `uniform_quad(key, p, g)` | Converts the four Philox output words into four FP32 uniforms in `(0,1)`. |
| `UniformSequence(key, p)` | Initializes a scalar-uniform stream at group zero for path `p`. |
| `UniformSequence::next()` | Exposes the cached uniform groups as one ordered scalar stream. |
| `Uint32Sequence(key, p)` | Initializes a raw-integer stream at group zero for path `p`. |
| `Uint32Sequence::next()` | Exposes the cached Philox groups as one ordered raw-integer stream. |
| `bounded_uint32(integers, b)` | Draws exactly uniformly from `{0,...,b-1}` by rejecting the prefix of size `2^32 mod b`. |

### Distribution transforms

For independent uniforms `U_1,U_2` in `(0,1)`, `box_muller` returns two
independent standard normals:

```math
R=\sqrt{-2\log U_2},
\qquad
\Theta=2\pi U_1,
\qquad
(Z_1,Z_2)=R(\cos\Theta,\sin\Theta).
```

For a Gamma shape `alpha >= 1` and a standard normal variate `Z`, the
Marsaglia–Tsang core sets

```math
d=\alpha-\frac{1}{3},
\qquad
c=\frac{1}{\sqrt{9d}},
\qquad
V=(1+cZ)^3,
```

then accepts `d V` with the method's squeeze or logarithmic test. Let
`G_alpha` denote a unit-scale Gamma variate with shape `alpha`, and let `U` be
an independent uniform variate. For `0 < alpha < 1`, the method uses

```math
G_{\alpha}=G_{\alpha+1}U^{1/\alpha}.
```

Let `X` follow a non-central chi-square law with `nu` degrees of freedom and
noncentrality `lambda`, and let `s > 0` be the requested scale. With
`Gamma(alpha, theta)` denoting shape `alpha` and scale `theta`, the exact
Poisson–Gamma representation is

```math
N\sim\mathrm{Poisson}\!\left(\frac{\lambda}{2}\right),
\qquad
sX\mid N\sim\mathrm{Gamma}\!\left(\frac{\nu}{2}+N,2s\right),
```

For inverse-Gaussian mean `mu > 0` and shape `lambda > 0`, let `Z` be standard
normal and `U` uniform on `(0,1)`. The Michael–Schucany–Haas construction sets

```math
w=\frac{\mu Z^2}{2\lambda},
\qquad
r=1+w+\sqrt{w(2+w)},
```

then returns

```math
X=
\begin{cases}
\mu/r, & U\le r/(r+1),\\
\mu r, & U>r/(r+1).
\end{cases}
```

| Function or type | Definition |
|---|---|
| `poisson_from_uniform(U, lambda, p_0)` | Inverts the Poisson CDF from one uniform, with `p_0 = exp(-lambda)` supplied by the caller. |
| `poisson_from_uniform_sequence(uniforms, lambda)` | Uses inversion for `lambda < 10` and Hörmann PTRS otherwise. |
| `box_muller(U_1, U_2)` | Returns the pair `(Z_1,Z_2)` above. |
| `NormalPairCache` | Stores the unused second Box–Muller normal. |
| `next_normal(uniforms, cache)` | Returns the cached normal or consumes two uniforms to refill the cache. |
| `detail::marsaglia_tsang_gamma_shape_at_least_one(...)` | Draws a unit-scale Gamma variate for `alpha >= 1`. |
| `marsaglia_tsang_gamma(..., alpha, theta)` | Draws `Gamma(alpha, theta)` for every positive shape `alpha` and scale `theta`. |
| `scaled_noncentral_chi_square(..., nu, lambda, s)` | Draws the scaled non-central chi-square variate `sX` through the Poisson–Gamma mixture. |
| `michael_schucany_haas_inverse_gaussian(..., mu, lambda)` | Draws `IG(mu, lambda)` with the exact normal/uniform construction. |

References: [Box and Muller (1958)](https://doi.org/10.1214/aoms/1177706645),
[Hörmann (1993)](https://doi.org/10.1016/0167-6687%2893%2990997-4),
[Marsaglia and Tsang (2000)](https://doi.org/10.1145/358407.358414), and
[Michael, Schucany and Haas (1976)](https://doi.org/10.1080/00031305.1976.10479147).

<a id="normal-distribution"></a>
## [`normal_distribution.cuh`](normal_distribution.cuh)

Let `Z` be a standard normal random variable. `normal_cdf(z)` evaluates its
CDF directly in FP32.

| Function | Definition |
|---|---|
| `normal_cdf(z)` | Returns the standard-normal CDF at `z`. |

The implementation uses

```math
\Phi(z)=\mathbb{P}[Z\le z]
=\frac{1}{2}\,\mathrm{erfc}\!\left(-\frac{z}{\sqrt{2}}\right).
```

<a id="noncentral-chi-square"></a>
## [`noncentral_chi_square.cuh`](noncentral_chi_square.cuh)

`DistributionProbabilities {cdf, survival}` carries both tails so callers do
not reconstruct a small probability by FP32 subtraction.

### Regularized Gamma law

For a shape `a > 0` and an argument `x >= 0`, with `Gamma(a)` denoting the
Gamma function, define

```math
P(a,x)=\frac{1}{\Gamma(a)}\int_0^x t^{a-1}e^{-t}\,\mathrm dt,
\qquad
Q(a,x)=\frac{1}{\Gamma(a)}\int_x^{\infty}t^{a-1}e^{-t}\,\mathrm dt.
```

| Function or type | Definition |
|---|---|
| `clamp_probability(x)` | Clamps a final probability to `[0,1]`. |
| `CompensatedSum(initial)` | Initializes a compensated FP32 sum. |
| `CompensatedSum::add(x)` | Adds `x` with Kahan compensation. |
| `log_one_plus_minus_argument(x)` | Evaluates `log(1+x)-x` by an 18-term local series when `abs(x) <= 1/4`. |
| `stirling_correction(a)` | Evaluates the Stirling remainder used for `log Gamma(a)` when `a >= 8`. |
| `gamma_log_scale(a, x)` | Evaluates `-x + a log(x) - log Gamma(a)` without subtracting large nearby terms. |
| `regularized_gamma_series(a, x)` | Evaluates `P(a,x)` directly on the left of the Gamma transition region. |
| `regularized_gamma_continued_fraction(a, x)` | Evaluates `Q(a,x)` directly on the right with the modified-Lentz fraction. |
| `regularized_gamma_probability_pair(a, x)` | Selects the numerically stable direct tail and returns both `(P,Q)`. |
| `regularized_gamma_probabilities(a, x)` | Public device interface returning `(P(a,x),Q(a,x))`. |

The common scale factor is

```math
e^{-x+a\log x-\log\Gamma(a)}.
```

For `a >= 8`, write `r=(x-a)/a` and let `C(a)` be
`stirling_correction(a)`. The same logarithm is evaluated as

```math
a\,[\log(1+r)-r]
+\frac{1}{2}\log\!\left(\frac{a}{2\pi}\right)-C(a),
```

### Non-central chi-square law

Let `X` follow a non-central chi-square law with degrees of freedom `nu > 0`
and noncentrality `lambda >= 0`. Its CDF at `x >= 0` is the Poisson mixture

```math
\mathbb{P}[X\le x]
=\sum_{k=0}^{\infty}
e^{-\lambda/2}\frac{(\lambda/2)^k}{k!}
P\!\left(\frac{\nu}{2}+k,\frac{x}{2}\right).
```

| Function | Definition |
|---|---|
| `poisson_mode_log_weight(lambda_over_two, mode)` | Evaluates the logarithm of the modal Poisson weight without large-term cancellation. |
| `poisson_gamma_mixture(nu, lambda, x)` | Sums both tails outward from the modal Poisson term with compensated FP32 sums. |
| `log_one_plus_minus_ratio(delta)` | Evaluates the stable deviance term `log(1+delta)-delta/(1+delta)`. |
| `saddlepoint_probabilities(nu, lambda, x)` | Applies the Lugannani–Rice saddlepoint approximation for large `lambda`. |
| `noncentral_chi_square_probabilities(nu, lambda, x)` | Uses the exact mixture for `lambda <= 1024` and the saddlepoint approximation above it. |

All device arithmetic in this file is FP32. The exact series implementation
follows the recurrence strategy of [Ding (1992)](https://doi.org/10.2307/2347584);
the large-noncentrality branch uses
[Lugannani and Rice (1980)](https://doi.org/10.2307/1426607), and the continued
fraction uses [Lentz (1976)](https://doi.org/10.1364/AO.15.000668).

<a id="reductions"></a>
## [`reductions.cuh`](reductions.cuh)

Let `Y_1,...,Y_M` be `M >= 2` discounted Monte Carlo payoffs. `MomentSums`
stores the FP64 moments

```math
s_1=\sum_{i=1}^{M}Y_i,
\qquad
s_2=\sum_{i=1}^{M}Y_i^2.
```

| Function | Definition |
|---|---|
| `reduce_block(sum, sumsq)` | Deterministically reduces one pair `(s_1,s_2)` across a whole CUDA block. |
| `reduce_block_values<N>(values)` | Deterministically reduces `N` FP64 values into separate shared-memory totals. |
| `compute_statistics(total, M, price, standard_error)` | Converts final moments into the sample mean and its standard error. |

```math
\widehat V=\frac{s_1}{M},
\qquad
\widehat{\mathrm{Var}}(Y)
=\frac{s_2-M\widehat V^2}{M-1},
\qquad
\mathrm{SE}(\widehat V)
=\sqrt{\frac{\max(\widehat{\mathrm{Var}}(Y),0)}{M}}.
```

State evolution remains FP32; FP64 is used here for long reductions and final
statistics.

<a id="longstaff-schwartz"></a>
## [`longstaff_schwartz/`](longstaff_schwartz)

[`exercise_schedule.cuh`](longstaff_schwartz/exercise_schedule.cuh) /
[`exercise_schedule.cu`](longstaff_schwartz/exercise_schedule.cu) ·
[`laguerre.cuh`](longstaff_schwartz/laguerre.cuh) ·
[`launch.cuh`](longstaff_schwartz/launch.cuh) /
[`launch.cu`](longstaff_schwartz/launch.cu) ·
[`linear_solver.cuh`](longstaff_schwartz/linear_solver.cuh) /
[`linear_solver.cu`](longstaff_schwartz/linear_solver.cu) ·
[`regression.cuh`](longstaff_schwartz/regression.cuh) /
[`regression.cu`](longstaff_schwartz/regression.cu) ·
[`workspace.cuh`](longstaff_schwartz/workspace.cuh) /
[`workspace.cu`](longstaff_schwartz/workspace.cu)

Contains all exercise scheduling, Laguerre basis, FP64 regression and Cholesky
solve, workspace planning, batching, resources and launch metrics required by
the Longstaff–Schwartz method. See
[Longstaff and Schwartz (2001)](https://doi.org/10.1093/rfs/14.1.113) and the
[`CUDA American and Bermudan pricing contract`](../../docs/cuda-american-and-bermudan-pricing-contract.md).

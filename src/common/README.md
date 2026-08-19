# Shared CUDA and pricing infrastructure

This directory contains model-independent building blocks used by the CUDA
pricing code. Model mathematics stays under `src/model`; product rows stay
under `src/product`. A new utility belongs here only when several independent
model/product implementations use the same contract.

## Contents

- [CUDA validation and diagnostics](#cuda-validation-and-diagnostics)
- [Result construction and option orientation](#result-construction-and-option-orientation)
- [Random numbers and distributions](#random-numbers-and-distributions)
- [Monte Carlo reductions](#monte-carlo-reductions)
- [Longstaff–Schwartz](#longstaffschwartz)
- [Compatibility implementation](#compatibility-implementation)
- [Reproducibility rules](#reproducibility-rules)

## File index

| File | Responsibility |
|---|---|
| [`check_cuda.cuh`](check_cuda.cuh) | CUDA errors, launch validation, dataset-shape validation, and overflow checks |
| [`cuda_kernel_diagnostics.cuh`](cuda_kernel_diagnostics.cuh) / [`cuda_kernel_diagnostics.cpp`](cuda_kernel_diagnostics.cpp) | Opt-in resources and occupancy report for an exact kernel specialization |
| [`result_index.cuh`](result_index.cuh) | Aligned/Cartesian result-index decoding |
| [`option_side.cuh`](option_side.cuh) | Compile-time call/put orientation |
| [`philox.cuh`](philox.cuh) | Philox stream, uniforms, normals, Poisson, Gamma, and inverse-Gaussian draws |
| [`normal_distribution.cuh`](normal_distribution.cuh) | FP32 standard-normal CDF |
| [`noncentral_chi_square.cuh`](noncentral_chi_square.cuh) | Deterministic regularized-Gamma and non-central-chi-square CDF/survival probabilities |
| [`reductions.cuh`](reductions.cuh) | Deterministic FP64 block reductions and Monte Carlo statistics |
| [`longstaff_schwartz/basis.cuh`](longstaff_schwartz/basis.cuh) / [`basis.cu`](longstaff_schwartz/basis.cu) | Active regression basis |
| [`longstaff_schwartz/exercise_schedule.cuh`](longstaff_schwartz/exercise_schedule.cuh) / [`exercise_schedule.cu`](longstaff_schwartz/exercise_schedule.cu) | Maturity-anchored exercise dates |
| [`longstaff_schwartz/regression.cuh`](longstaff_schwartz/regression.cuh) / [`regression.cu`](longstaff_schwartz/regression.cu) | Normal-equation accumulation, reduction, and solve orchestration |
| [`longstaff_schwartz/linear_solver.cuh`](longstaff_schwartz/linear_solver.cuh) / [`linear_solver.cu`](longstaff_schwartz/linear_solver.cu) | Small dense FP64 Cholesky solve |
| [`longstaff_schwartz/workspace.cuh`](longstaff_schwartz/workspace.cuh) / [`workspace.cu`](longstaff_schwartz/workspace.cu) | Typed regions, aligned workspace layout, and batch planning |
| [`longstaff_schwartz/launch.cuh`](longstaff_schwartz/launch.cuh) / [`launch.cu`](longstaff_schwartz/launch.cu) | Workspace ownership, memory budget, timing, and launch metrics |
| [`least_squares.cuh`](least_squares.cuh) / [`least_squares.cu`](least_squares.cu) | Inactive compatibility implementation retained for now |

## CUDA validation and diagnostics

### `check_cuda.cuh`

Host-side validation shared by launchers and generators:

- `check_cuda` turns CUDA Runtime failures into readable C++ exceptions;
- `checked_workspace_product` rejects allocation-size overflow;
- `bounded_block_count`, `validate_block_count`, `validate_cuda_block_size`,
  `validate_reduction_block_size`, and `validate_grid_x_size` validate launch
  geometry;
- `validate_device_pointer` rejects host/null pointers passed as device arrays;
- `validate_model_product_construction` and
  `validate_model_curve_product_construction` validate aligned or Cartesian
  datasets;
- `validate_monte_carlo_path_count`, `validate_monte_carlo_parameters`, and
  `validate_row_seed_range` protect the common simulation contract.

### `cuda_kernel_diagnostics.cuh/.cpp`

Optional launch diagnostics for the exact kernel specialization and geometry.
`report_cuda_kernel_launch_if_enabled` reports registers per thread, local and
shared memory, theoretical occupancy, device limits, and binary/PTX versions.

Set `AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1` (also accepts `true` or `on`) to emit
one JSON object per distinct `(kernel, variant, grid, block, dynamic shared
bytes)` tuple on standard error. Diagnostics are deduplicated and have no GPU
query cost when disabled. Occupancy is a resource diagnostic, not a direct
performance measurement.

## Result construction and option orientation

### `result_index.cuh`

`decode_model_product_result_index` returns `ModelProductIndices`. In aligned
mode result `i` uses model `i` and product `i`. In Cartesian mode products vary
fastest:

```text
model_index   = result_index / product_count
product_index = result_index % product_count
```

Model/curve/product launchers apply the same ordering with the curve dimension
between model and product, as validated by `check_cuda.cuh`.

### `option_side.cuh`

`OptionSide::{call,put}` is the compile-time orientation shared by option
families. Public launchers are explicitly instantiated for both sides;
`option_side_name` supplies the stable host-side label. This removes duplicated
call/put kernels and avoids a runtime branch in payoff code.

## Random numbers and distributions

### `philox.cuh`

Implements Philox-4x32-10 from
[Salmon et al. (2011)](https://doi.org/10.1145/2063384.2063405). The public
mapping is deterministic:

```text
key                 = make_key(base_seed + result_index)
counter words 0..1  = path_index
counter words 2..3  = local_group_index
```

`random_bits(key, path, group)` returns four 32-bit words;
`uniform_quad` maps their midpoints to FP32 values strictly inside `(0,1)`.
`UniformSequence(key, path)` hides four-value groups and exposes one ordered
`next()` stream. Each Monte Carlo path creates exactly one such sequence.

`box_muller`, `NormalPairCache`, and `next_normal` produce normals while
retaining the second Box–Muller value. `poisson_from_uniform` performs inverse
CDF sampling from a caller-provided uniform and prepared `exp(-mean)`.
`poisson_from_uniform_sequence` retains that inversion for small means and
uses Hoermann PTRS transformed rejection for large means, where direct
inversion would be slow and `exp(-mean)` can underflow.
`marsaglia_tsang_gamma` and
`michael_schucany_haas_inverse_gaussian` implement the Gamma and
inverse-Gaussian samplers used by VG and NIG. Rejection loops consume the same
path-local sequence; they never create or restart a generator.
`scaled_noncentral_chi_square` composes the adaptive Poisson and Gamma draws
through the exact Poisson-Gamma mixture. Its scale is applied by the Gamma
draw itself, which avoids a separate post-draw multiply in CIR callers.

### `normal_distribution.cuh`

`normal_cdf` evaluates the standard-normal CDF in FP32 through `erfcf`. It is
the shared primitive for closed-form device analytics.

### `noncentral_chi_square.cuh`

`regularized_gamma_probabilities` and
`noncentral_chi_square_probabilities` return CDF and survival probability
together, so tail-sensitive option formulas never form `1-CDF` in FP32.
Regularized Gamma uses the convergent series on the left and a modified-Lentz
continued fraction on the right. The non-central chi-square uses its exact
Poisson--Gamma mixture centered at the modal Poisson term while the intensity
is moderate, then switches to a Lugannani--Rice saddlepoint evaluation for
large noncentralities. The internal evaluation is FP64 and the public result is
FP32. CUDA tests compare both tails with SciPy across degrees of freedom down
to `0.2`, the switching boundary, extreme tails, and noncentralities up to
`1e8`.

## Monte Carlo reductions

### `reductions.cuh`

`MomentSums` stores FP64 payoff sum and squared-payoff sum.
`reduce_block` performs a deterministic two-stage warp/block reduction using
dynamic shared memory. `reduce_block_values<N>` applies the same scheme to a
fixed register array. `compute_statistics` converts final moments into the
sample mean and standard error, clamping only a negative round-off variance to
zero.

Callers must use a whole number of warps and provide the dynamic shared-memory
size required by the selected reduction. State evolution stays FP32; FP64 is
reserved for long sums, regression equations, and final statistics.

## Longstaff–Schwartz

The active early-exercise implementation follows
[Longstaff and Schwartz (2001)](https://doi.org/10.1093/rfs/14.1.113) and lives
in `longstaff_schwartz/`.

### `longstaff_schwartz/basis.cuh/.cu`

Defines the fixed six-term two-factor Laguerre basis and its polynomial
primitives. `RegressionBasis` in `regression.cuh` is the single active alias;
changing the regression family is intentionally a separate design task.

### `longstaff_schwartz/exercise_schedule.cuh/.cu`

`maturity_anchored_exercise_count` validates a regular exercise interval and
counts the dates `T-(E-1)delta, ..., T` shared by American/Bermudan products.

### `longstaff_schwartz/regression.cuh/.cu`

Accumulates fixed-basis normal equations in FP64, reduces partial equations
across path blocks, applies a small relative ridge, solves one exercise-level
regression, and records whether its coefficients are valid.
`regression_shared_bytes` returns the reduction storage required by a block.

### `longstaff_schwartz/linear_solver.cuh/.cu`

`cholesky_solve_normal_equations` is the small dense FP64 Cholesky solver used
by the regression stage. It works in place and rejects an unsafe diagonal
instead of returning unstable coefficients.

### `longstaff_schwartz/workspace.cuh/.cu`

Describes and plans the caller-owned device workspace. `WorkspaceDescriptor`
declares the prepared-row type, model-specific SoA state fields, and regression
dimensions. `make_workspace_layout` aligns all common regions inside one byte
buffer. `plan_batches` groups consecutive rows under a memory budget and
returns offsets, counts, maximum exercise counts, and allocation maxima.

State fields remain model-specific: Heston/Bates can declare spot and variance,
whereas a one-state model declares only spot. `workspace_pointer<T>` turns a
planned region into a typed device pointer.

### `longstaff_schwartz/launch.cuh/.cu`

`query_workspace_budget` reserves the larger of 1 GiB or ten percent of free
device memory as a safety margin. `LaunchResources` owns the single workspace
allocation and CUDA timing events with RAII. `LaunchResult` reports kernel
time, batches, launches, maximum batch size, blocks per price, and bytes.

## Compatibility implementation

### `least_squares.cuh/.cu`

Contains the earlier six-term Laguerre basis and Cholesky implementation in the
`least_squares` namespace. Current American/Bermudan launchers use
`longstaff_schwartz/` instead; this pair has no active consumers and should not
be selected for new code without an explicit compatibility reason.

## Reproducibility rules

- never enable fast-math;
- derive one Philox key from `base_seed + result_index` and one sequence per
  path;
- preserve the documented order of random consumption and reductions;
- keep state evolution FP32 and use FP64 only where accumulation stability
  justifies it;
- validate device pointers, sizes, Cartesian shapes, and launch geometry before
  launching a kernel.

The detailed contracts are in
[`docs/cuda-model-dynamics-contract.md`](../../docs/cuda-model-dynamics-contract.md),
[`docs/cuda-closed-form-and-monte-carlo-pricing-contract.md`](../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md),
and [`docs/cuda-american-and-bermudan-pricing-contract.md`](../../docs/cuda-american-and-bermudan-pricing-contract.md).

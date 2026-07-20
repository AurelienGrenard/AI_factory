# CUDA Engineering Contract

This document defines the target structure, numerical policy, execution model,
and review rules for AI Factory CUDA source code. Read `architecture.md`, the
dataset protocols, `certification.md`, and `code_guide.md` first.

CUDA is the production engine of the project. A kernel must therefore be fast,
reproducible, explainable, and structurally familiar to a reader who already
understands another model/product implementation.

## Scope

This contract currently covers:

- analytic pricing kernels;
- European terminal Monte Carlo products;
- European path-dependent Monte Carlo products without early exercise.

American options and Bermudan products are deliberately excluded. Their state
storage, backward induction, regression, and exercise stages require a separate
contract. Existing early-exercise implementations remain subject to numerical
certification but are not reference layouts for this document.

## Canonical Examples

Three implementations define the target CUDA vocabulary:

1. `black_scholes/european_calls.cu` for analytic kernels;
2. `heston/european_calls.cu` for European Monte Carlo kernels;
3. `black_scholes/asian_arithmetic_calls.cu` for European path-dependent
   Monte Carlo kernels.

These files become canonical only after their dedicated rewrite and
certification. Until then, existing `.cu` files are numerical and performance
baselines, not style references. New European implementations must select the
closest canonical family and preserve its outer execution structure.

The canonical examples are intentionally simple. A different model changes the
dynamics. A different product changes the observer state and payoff. Neither
change should reinvent row mapping, path mapping, precision, reduction,
workspace ownership, timing, or error handling.

### Teaching workbench

`cuda_workbench/` is an autonomous, heavily commented implementation of the
Heston European-call reference pattern. It rebuilds the common Philox,
reduction, runtime, model-dynamics, and specialized-product layers from first
principles. Its `float` and `double` choices are written explicitly at every
use site. Use it to learn or review the execution model;
it is not linked into `src_cpp`, registry generation, or production results.

The workbench must remain small and explanatory. Once an implementation is
understood, certified, and adopted by the active engine, `src_cpp` remains the
sole production source of truth.

## Numerical Precision

The C++ CPU and CUDA engines use the same execution precision policy.

```text
registry model/product values           float
C API model/product inputs              float
prepared execution parameters          float
Philox uniform and normal outputs       float
path state and model dynamics           float
individual payoff                      float
Monte Carlo sums and squared sums       double
final mean, variance, standard error    double
result outputs                          double
```

Registry loading, CPU, and CUDA must preserve the same FP32 model/product
representation and apply the same Philox counter mapping.
Converting a `float` payoff to `double` before accumulation preserves the FP32
payoff exactly and prevents additional FP32 summation loss:

```cpp
const float payoff = evaluate_path(...);
const double value = static_cast<double>(payoff);
double sum = value;
double sumsq = value * value;
```

Do not write `double sumsq = payoff * payoff`; that multiplication occurs in
FP32 before conversion. FP32 literals and functions must be explicit in hot
path code, for example `0.5f`, `expf`, `logf`, `sqrtf`, and `fmaxf`, so silent
promotion to FP64 cannot recreate the current performance problem.

Bit operations are appropriate when CUDA or RNG semantics are naturally
binary, including warp lanes, masks, fixed-width words, and counter-based RNG
rounds. Do not introduce bit tricks as speculative micro-optimizations for
ordinary arithmetic: compilers already optimize constant powers of two. Give
the arithmetic equivalent in a short comment whenever the binary expression is
not immediately obvious.

Use explicit `float` and `double` declarations in numerical source so precision
is visible at each operation. Do not hide these choices behind project aliases.

Changing execution precision changes the numerical engine. Production and
validation results must be regenerated, and the FP32 implementation must first
be certified against the previous FP64 baseline over normal and stressed
parameter rows. The accepted implementation must show no material bias relative
to Monte Carlo uncertainty.

## Source Ownership

CUDA source ownership follows this structure:

```text
src_cpp/ai_factory/cuda/
  common/
    philox.cuh                 # one Philox implementation and counter mapping
    reductions.cuh             # block-level FP64 reductions
    runtime.cuh                # workspaces, CUDA errors, events
    <named_mechanic>.cuh       # genuinely shared execution mechanics
  products/
    <product>.hpp              # model-independent product parameters
  <model>/
    types.cuh                  # model-specific input parameters
    api.cuh                    # public model CUDA entry points
    dynamics.cuh               # reusable model dynamics and path evolution
    paths.cu                   # model-only path export
    <product>.cu               # specialized fused pricing implementation
    <product>.cuh              # only when declarations/helpers are shared
```

`common` contains mechanics shared by several models or products. Model
dynamics belong in `<model>/dynamics.cuh`. Product observation and payoff logic
remain in `<model>/<product>.cu` unless the same logic is genuinely shared by
several implementations.

Moving a device function into a header does not create another kernel launch.
Inline device helpers remain fused into the specialized pricing kernel.

## Common File Vocabulary

Every European `.cu` file uses the following section order:

1. public model or product API include;
2. model dynamics and named common mechanics;
3. CUDA and standard-library includes;
4. `ai_factory::cuda` and anonymous namespaces;
5. launch constants and `WorkspaceTag`;
6. optional prepared-row and product-state structures;
7. device preparation and path-evaluation functions;
8. specialized kernel;
9. public host launcher.

Use the same names where the responsibility is the same:

```text
PreparedRow       FP32 coefficients reused during path evaluation
ProductState      running path-dependent payoff state
prepare_row       boundary parameters to execution coefficients
evaluate_path     one realization from RNG counters to discounted payoff
pricing_kernel    one-row block with fused simulation, payoff, and reduction
WorkspaceTag      owner of reusable device buffers and timing events
```

Names may include a model/product prefix when required to avoid symbol
ambiguity. Avoid generic historical names such as `kernel`, `mc_kernel`, or
`monte_carlo_kernel` when a more precise role is available.

## Analytic Kernel Pattern

The canonical analytic example is the Black-Scholes European call. An analytic
kernel has no path count, seed, time loop, partial moments, or Monte Carlo
standard error.

```text
one thread = one aligned model/product row
one grid = one analytic result batch
```

The kernel uses a grid-stride loop when row count may exceed the initial grid:

```cpp
for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
     row < row_count;
     row += blockDim.x * gridDim.x) {
    outputs[row] = evaluate_analytic(rows[row]);
}
```

The public launcher validates inputs, reuses a workspace, copies rows, records
kernel events, launches the specialized kernel, and copies outputs back. It
must not introduce fake Monte Carlo work to manufacture a timing hierarchy.

Analytic formulas that reproduce an input curve must remain documented as such
in result metadata. Arithmetic intensity, not a universal engine ordering,
determines whether an analytic batch benefits from CUDA.

## European Monte Carlo Pattern

The canonical European Monte Carlo example is the Heston European call. Large
production databases provide row-level parallelism, so one CUDA block owns one
aligned model/product row and every thread processes several paths.

The mandatory production mapping is:

```cpp
const std::size_t row_index = blockIdx.x;
for (std::size_t path = threadIdx.x;
     path < paths_per_row;
     path += blockDim.x) {
    const float payoff = evaluate_path(row, path);
    const double value = static_cast<double>(payoff);
    sum += value;
    sumsq += value * value;
}
```

The single kernel performs:

```text
prepare row coefficients
simulate several paths per thread
evaluate and discount each payoff
accumulate payoff and payoff squared in FP64 thread-local sums
reduce thread-local sums inside the block
write price and standard error directly from thread 0
```

This avoids global partial-moment storage, global atomics, and a second kernel.
It is the default for large dataset generation. A separate multi-block-per-row
kernel may be justified for interactive single-row pricing, but it is not the
production reference and must not complicate the canonical dataset path.

The time loop inside one stochastic path remains sequential because the next
model state depends on the previous state. Parallelism comes from independent
paths and independent rows.

## European Path-Dependent Pattern

The canonical path-dependent example is the Black-Scholes arithmetic Asian
call. It uses exactly the same one-block-per-row mapping, thread-stride path
loop, block reduction, workspace, and timing as the Heston European call.

Only the path evaluator changes. It maintains a compact `ProductState` in
registers while the model advances:

```text
Asian call          running arithmetic sum and observation count
lookback call       running maximum spot
barrier option      hit flag and terminal spot
autocall            observation state and redemption metrics
volatility swap     running squared log-return sum
```

The standard path loop is conceptually:

```cpp
auto model_state = initialize_model(row);
ProductState product_state{};

for (std::size_t step = 0; step < num_steps; ++step) {
    advance_model(row, prepared, path, step, model_state);
    observe_product(row, step, model_state, product_state);
}

return finish_product(row, model_state, product_state);
```

These functions are device-inline concepts, not separate kernels. Simulation,
observation, and payoff remain fused. `paths.cu` is the only model source whose
purpose is to materialize paths for reconstruction and diagnostics.

## Preparation And Shared State

Constants reused at every time step should be prepared once per block or loaded
from a compact prepared representation. Thread zero may initialize a small
shared context followed by `__syncthreads()`. Cooperative initialization is
appropriate for arrays such as rough-kernel weights or fixed-income payment
coefficients.

A separate preparation kernel and global prepared database require profiling
evidence. Avoid replacing a few repeated arithmetic operations with additional
launches and global-memory traffic.

Model-independent product state should remain visibly separate from model
state. This makes the difference between a new model and a new payoff obvious
when two source files are compared.

## RNG Contract

Philox counter generation is centralized. Product kernels never redefine
Philox rounds, uniform conversion, Box-Muller transformation, or stream ids.

Every random draw is a pure function of at least:

```text
row seed, stream id, path index, step index, draw component
```

Chunking, row grouping, block size, grid size, and scheduling must not change
the random numbers assigned to a row/path/step. CRN modes reuse counters by
construction rather than storing random arrays.

The FP32 RNG transformation must be implemented once for CPU and CUDA and
validated statistically and pathwise before it replaces the FP64 mapping.

## Reduction Contract

Monte Carlo reductions use a named common implementation. Specialized product
files provide values; they do not copy warp-shuffle or final variance code.

The common reduction owns:

- warp and block summation;
- sample variance and standard error;
- non-negative protection against final roundoff.

Products with additional statistics extend a clearly named moment vector, as
autocalls do, but retain the same block-reduction vocabulary. A shared file
must describe its actual responsibility, for example
`autocall_reduction.cuh`; avoid ambiguous catch-all names.

## Workspace And Transfers

Every public launcher follows the same lifecycle:

1. validate row count, path count, step count, and product constraints;
2. obtain a typed reusable workspace;
3. resize/reuse row and output buffers;
4. copy input rows from host to device;
5. record the start event;
6. launch all kernels that define kernel time;
7. record and synchronize the stop event;
8. copy outputs from device to host;
9. populate the optional `kernel_seconds` output.

Production loops must not allocate and free device memory for every group when a
reusable workspace can own the buffers. The C API translates exceptions and
exposes symbols; it does not own simulation, payoff, reduction, or workspace
logic.

## Launch Configuration

`1024` is a hardware limit, not a default. Candidate block sizes are normally
`128` and `256`:

- state-heavy or register-heavy kernels commonly start at 128;
- simple terminal or analytic kernels commonly start at 256.

The selected value must be supported by release-build measurements and checked
for registers per thread, register spilling, shared memory per block, resident
blocks per SM, and total kernel time. A model/product kernel may differ from a
canonical example only when the performance reason is real and recorded in
source-level benchmark notes or tests, not in registry metadata.

Tail threads use neutral values in reductions. Product logic must not perform
out-of-range reads before checking `path < num_paths` or `row < row_count`.

## Timing Contract

CUDA events surround only the kernel sequence. Host-to-device copies occur
before the start event and device-to-host copies after the synchronized stop
event. Thus:

```text
kernel_seconds = timed CUDA kernel sequence
wall_seconds   = complete hot engine call measured by orchestration
```

Warm-up belongs to the benchmark runner and occurs before recorded repetitions.
Timing code does not belong inside the stochastic path loop. The certification
rules in `certification.md` remain authoritative for workloads, repetitions,
aggregation, and hierarchy checks.

## Product Headers

`<product>.cuh` exists only when another translation unit consumes a product
declaration, specialized row type, or reusable device helper. A `.cu` file does
not receive a same-name `.cuh` merely for visual symmetry. Public model pricing
entry points normally live in `<model>/api.cuh`.

## Forbidden Patterns

The following patterns fail CUDA architecture review:

- dead or superseded kernels retained with `[[maybe_unused]]`;
- multiple active execution strategies in one product file without dispatch
  evidence and tests;
- local copies of Philox, warp reductions, final variance logic, or workspace
  management;
- one-block-per-row Monte Carlo as an unexplained default;
- path materialization before payoff evaluation;
- separate simulation and payoff launches without measured necessity;
- FP64 path dynamics or accidental FP64 promotion after the FP32 migration;
- hidden financial logic in `c_api.cpp`, tools, registry generators, or timing
  code;
- model A importing specialized pricing code from model B;
- arbitrary block sizes with no occupancy and timing evidence;
- comments or names that hide whether a function prepares, simulates, observes,
  pays, reduces, or launches.

## Migration And Certification

European CUDA migration proceeds by family, not through a mechanical global
rewrite:

1. preserve the current FP64 output and timing baseline;
2. rewrite and certify the three canonical examples;
3. add architecture tests for their layout and dependencies;
4. migrate each remaining model/product to the nearest canonical pattern;
5. compare FP32 CPU and CUDA pathwise under the shared Philox mapping;
6. compare mixed-precision outputs against the FP64 baseline statistically;
7. regenerate production and validation results;
8. execute validation notebooks and strict timing certification;
9. remove superseded kernels and helpers only after no active dependency
   remains.

For each migrated pair, certification checks:

- source ownership and canonical execution family;
- CPU/CUDA numerical agreement under the new precision policy;
- statistical agreement with PyTorch and the previous FP64 baseline;
- deterministic row/seed/path/step mapping;
- no regression in CUDA kernel time on the reference hardware;
- no register spilling or occupancy collapse without an explicit tradeoff;
- correct kernel and wall timing semantics;
- complete production, validation, notebook, and schema tests.

The objective is not identical source text. It is a stable execution grammar:
the reader should immediately locate preparation, model dynamics, product
observation, payoff, reduction, launch, and timing in every European example.

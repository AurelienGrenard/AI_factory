# AI Factory Code Guide

This guide maps the current execution code. The engineering philosophy and
dependency rules are authoritative in `docs/architecture.md`; the production
and validation checklists live in their respective dataset protocol documents.
This file only maps the concrete source tree.

## Source Layout

```text
src_python/ai_factory/
  pytorch/
    common/
      american_options.py
      device.py
      monte_carlo.py
      pathwise_products.py
    black_scholes/
      <product>.py
      pathwise.py
    heston/
      common.py
      <product>.py
      pathwise.py
    rough_bergomi/
      common.py
      <product>.py
      pathwise.py
    black_76/
      caplets.py
      swaptions.py
    hull_white/ cir/ cir_plus_plus/ g2_plus_plus/
      <fixed_income_product>.py

src_cpp/ai_factory/
  cpu/
    common/
      philox.hpp / philox.cpp
      monte_carlo.hpp
      time_grid.hpp
      payoffs/
        <product>.hpp
    black_scholes/
      common.hpp / common.cpp
      <product>.hpp / <product>.cpp
    heston/
      common.hpp / common.cpp
      <product>.hpp / <product>.cpp
    rough_bergomi/
      common.hpp / common.cpp
      <product>.hpp / <product>.cpp
    black_76/ hull_white/ cir/ cir_plus_plus/ g2_plus_plus/
      <fixed_income_product>.hpp / <fixed_income_product>.cpp
  cuda/
    common/
      api.cuh
      philox.cuh
      reductions.cuh
      runtime.cuh
      types.cuh
    heston/
      api.cuh
      dynamics.cuh
      <product>.cu
      paths.cu
    rough_bergomi/
      api.cuh
      dynamics.cuh
      <product>.cu
      paths.cu
    black_scholes/
      api.cuh
      dynamics.cuh
      <product>.cu
      paths.cu
    black_76/ hull_white/ cir/ cir_plus_plus/ g2_plus_plus/
      api.cuh
      <fixed_income_product>.cuh / <fixed_income_product>.cu
  c_api/
    c_api.cpp
```

```text
tools/
  registry/
    common/
      database_writing.py
      grids.py
      io.py
      paths.py
      slicing.py
    model/
      core_parameter_databases.py
      slicing.py
    product/
      core_parameter_databases.py
      recipe_helpers.py
      slicing.py
    result/
      common/
        metadata.py
        production_pipeline.py
      heston/
        lookback_fixed_calls.py
        asian_arithmetic_calls.py
        volatility_swaps.py
        american_puts.py
      rough_bergomi/
        lookback_fixed_calls.py
        asian_arithmetic_calls.py
        volatility_swaps.py
      black_76/ hull_white/ cir/ cir_plus_plus/ g2_plus_plus/
        <fixed_income_product>.py
  paths/
    heston.py
    rough_bergomi.py
```

`src_python` is intentionally PyTorch-only. The `registry/` tree contains
generated databases, specifications, and same-basename generation scripts.
Reusable registry-generation logic lives under `tools/registry`, split into
`common`, `model`, `product`, and `result`.

## Execution Paths

The current Heston/lookback and Rough Bergomi/lookback result generation
recipes support four engines:

- Python CPU with PyTorch RNG.
- Python GPU with PyTorch RNG.
- C++ CPU with project Philox-4x32-10.
- C++ CUDA with the same project Philox-4x32-10.

The Python engines are readable statistical references. They are not expected
to match C++ pathwise because PyTorch RNG is a different backend.

They are still expected to be good PyTorch implementations. Use one vectorized
code path parameterized by `device`, generate random tensors directly on the
target device, batch rows with compatible time-grid shapes, and avoid Python
loops over rows or paths when tensor operations express the same work clearly.
Python row seeds are not a reproducibility contract; they should not force slow
row-by-row execution.

The C++ CPU and C++ CUDA engines are expected to match to floating-point
precision for the same row, seed, model parameters, product parameters, time
grid rule, and Philox counter mapping.

C++ implementations should be uniform across models. CPU code should use the
shared Philox helpers and the same validate/precompute/simulate structure from
one model to the next. CUDA code should be treated as production code: kernels
should fuse simulation and payoff accumulation when appropriate, avoid avoidable
global-memory traffic, and keep launch structure consistent across validation
databases.

The normative CUDA file layout, row/path mapping, FP32 simulation with FP64
accumulation, reduction, workspace, timing, and migration rules live in
`cuda.md`. This code guide describes ownership and entry points; it does not
override that CUDA contract.

## Registry Reference Cases

`tools/registry/result/heston/lookback_fixed_calls.py` owns the reusable Heston
lookback production-audit recipe for `heston_01 x lookback_fixed_calls_01`,
including price-only and price plus CRN delta result databases.

`tools/registry/result/rough_bergomi/lookback_fixed_calls.py` mirrors the same
shape for `rough_bergomi_01 x lookback_fixed_calls_01` with
Bennedsen-Lunde-Pakkanen hybrid simulation.

The Python implementation batches rows by time-grid step count, simulates on
the requested PyTorch device, and keeps only the running maximum spot for this
lookback payoff. It avoids materializing full trajectories for the priced
database.

The C++ CUDA implementation is specialized for performance: each kernel launch
handles rows with a common step count, each path is simulated independently with
Philox, and payoff/reduction are fused in device code.

The C++ CPU implementation should mirror the same modeling structure across
models: validate parameters, precompute per-row constants, generate Philox
normal/uniform arrays through `src_cpp/ai_factory/cpu/common/philox.*`, then
run a tight path/time loop over those arrays. Shared CPU payoff formulas live
under `src_cpp/ai_factory/cpu/common/payoffs/`; for example Heston and Rough Bergomi
fixed-lookback engines use the same `lookback_fixed_calls` payoff helper. This
keeps models readable side by side and avoids model-specific RNG or payoff
mechanics hidden in inner loops.

The CUDA implementation intentionally specializes model/product kernels for
performance. Shared CUDA code is limited to reusable mechanics such as
`cuda/common/philox.cuh`, `cuda/common/reductions.cuh`,
`cuda/common/runtime.cuh`, and `cuda/common/types.cuh`. Model-level CUDA entry
points are declared in `cuda/<model>/api.cuh`. Active kernels live under model folders, for example
`cuda/heston/lookback_fixed_calls.cu`, `cuda/heston/paths.cu`, and
`cuda/rough_bergomi/lookback_fixed_calls.cu`. Model dynamics that are reused by
pricing and path export live beside the model kernels, for example
`cuda/heston/dynamics.cuh`. The old combined CUDA implementation has been
removed; new kernels must live under the matching model/product folder.
The three canonical CUDA source references are separately defined in
`cuda.md`; existing kernels are not style references until migrated and
certified under that contract.

## Time Grid Policy

Result YAML files express the time grid through `target_dt`, not a global
`num_steps`:

```yaml
time_grid:
  rule: nearest integer step count to target dt
  target_dt: 1/52
  step_count: round(maturity / target_dt)
  effective_dt: maturity / step_count
```

For each result row, the runner computes:

```text
step_count = max(1, round(maturity / target_dt))
effective_dt = maturity / step_count
```

Kernels still receive an integer `num_steps`; the registry runner groups rows
by computed `step_count` before calling the engine. This keeps GPU kernels
homogeneous and avoids per-row loop divergence inside a launch.

## Large Dataset Policy

Large random product databases may contain many distinct maturities. The
production runner should therefore separate numerical convention from execution
mechanics:

- The YAML time-grid rule defines the numerical convention.
- Exact grouping by computed `step_count` is an internal execution detail and
  does not need extra YAML metadata.
- Chunking large groups is also internal, provided row ids and seeds are fixed
  before chunking.
- Bucketizing or capping `step_count` changes the numerical grid and therefore
  must be declared in the result YAML as a different time-grid rule.
- Small validation slices are not reliable throughput benchmarks when maturities
  are random. A 100-row slice can contain dozens of distinct `step_count` values,
  which creates many tiny CPU/GPU calls and exaggerates orchestration overhead.
  Use production-sized C++ GPU results, or a dedicated profiling run, to judge
  whether the kernel "bombards".

For million-row result generation, the intended execution pattern is:

```text
build rows with stable row_id -> seed mapping
compute step_count(row) from the YAML time_grid rule
group rows by exact step_count
split each group into memory-safe chunks
call the selected engine per chunk
append outputs without changing row order or seeds
```

If exact grouping produces too many groups for a very fine `target_dt`, create
a new result database with an explicit bucketed rule, for example:

```yaml
time_grid:
  rule: nearest bucketed step count to target dt
  target_dt: 1e-3
  step_count_bucket: 16
  step_count: 16 * round(maturity / (16 * target_dt))
  effective_dt: maturity / step_count
```

Do not hide bucketization as an implementation detail.

## Reproducibility Contract

A result row is reproducible when these inputs are fixed:

- model database id and row id;
- product database id and row id;
- result database id and row id;
- result row seed;
- time-grid rule from the result YAML;
- Heston scheme or other model simulation scheme;
- RNG backend and counter mapping;
- engine source files listed in the result YAML;
- path count, dtype policy, and device backend.

For C++ CPU/CUDA Philox engines, chunking and grouping must never affect the
seed assigned to a row. They may affect performance, but not trajectories.

## Adding A New Priced Result

Start in `registry/validation`. This is where the four-engine benchmark and
notebook checks live.

1. Add or reuse a model parameter database under
   `registry/validation/models/<model_family>`.
2. Add or reuse a product parameter database under
   `registry/validation/products/<product_family>`.
3. Add a same-basename result generator under
   `registry/validation/results/<model_family>/<product_family>/generators`.
4. Put reusable generation logic in `tools/registry` if more than one result
   database will need it.
5. For C++ reproducible engines, call `src_cpp` through the shared library or a
   dedicated tool backed by project Philox.
6. Write compact result YAML with `summary`, `time_grid`, `outputs`,
   model/product database references, construction rule, timing, and
   reproducibility scripts when applicable.
7. Add tests for:
   - C++ CPU/GPU agreement when both use Philox;
   - statistical agreement against Python references;
   - path-export repricing when path tools exist.

After validation passes, add the large target database under
`registry/production`. Production generators may be chunked and restartable, but
they must keep the validated row id, seed, time-grid, RNG, and pricing
conventions.

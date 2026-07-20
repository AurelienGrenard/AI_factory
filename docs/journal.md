# AI Factory Journal

This journal records implementation decisions and benchmark observations that
would otherwise be lost in conversation.

Older entries are historical snapshots and may mention files or conventions
that were later replaced. The current contracts in `docs/architecture.md`,
`docs/production_dataset.md`, `docs/validation_dataset.md`, and
`docs/certification.md` always take precedence.

## 2026-07-12: C++ Source Layout Refactor

- Reorganized `src_cpp/ai_factory` around CPU, CUDA, and C API boundaries:
  - `cpu/common` contains CPU Philox and shared CPU execution utilities;
  - `cpu/common/payoffs` contains small contract formulas used by CPU engines;
  - `cpu/heston` and `cpu/rough_bergomi` contain model-specific CPU code;
  - `cuda/common` contains inline CUDA Philox and reduction helpers;
  - `c_api` contains the stable ABI loaded from Python registry tools.
- Extracted CUDA Philox and block reductions from the historical combined
  CUDA file into `cuda/common/philox.cuh` and `cuda/common/reductions.cuh`.
- Moved active CUDA model/product kernels into
  `cuda/heston/lookback_fixed_calls.cu`, `cuda/heston/paths.cu`, and
  `cuda/rough_bergomi/lookback_fixed_calls.cu`.
- Removed the old combined CUDA implementation once the model/product CUDA
  split became the source of truth.

## 2026-07-11: Validation Dataset Coding Standards

- Renamed the dataset checklist to `docs/validation_dataset.md` so it clearly
  describes validation databases rather than future production runs.
- Made the validation standard explicit:
  - PyTorch CPU/GPU code must be batched, vectorized, and implemented as one
    device-parametrized code path; Python seeds are not a pathwise guarantee.
  - C++ CPU code must use the same validate/precompute/shared-Philox/tight-loop
    structure across models.
  - C++ CUDA code is production code and should fuse simulation and payoff
    accumulation when that avoids unnecessary memory traffic.
  - Validation notebooks must keep the same compact structure and visual style
    from one model/product pair to the next.

## 2026-07-11: Time Grid And Large Random Product Databases

- Added Rough Bergomi fixed-lookback as the next validation case after Heston
  fixed-lookback:
  - Python CPU/GPU reference path uses PyTorch;
  - C++ CPU/GPU reproducible path uses project Philox;
  - the C++ simulation follows the Bennedsen-Lunde-Pakkanen hybrid scheme;
  - result databases reuse the same compact JSON/YAML/generator pattern.
- Removed the active Black-Scholes source path from `src_python`, `src_cpp`,
  and validation tests. Historical journal entries below may still mention
  older Black-Scholes benchmarks.
- Added `docs/validation_dataset.md` as the canonical checklist for new validation and
  production result databases.

- Split the registry into two tiers:
  - `registry/validation` for benchmark databases, four-engine checks, tests,
    and notebooks;
  - `registry/production` for large final databases generated after validation,
    usually with the C++ CUDA engine.
- Split notebooks into `notebooks/validation` and `notebooks/production`.
- Added the result-YAML `time_grid` convention:
  `target_dt: 1/52`, `step_count: round(maturity / target_dt)`, and
  `effective_dt: maturity / step_count`.
- Result JSON rows stay compact; per-row step counts are reconstructed from the
  product maturity and YAML time-grid rule.
- Large random product databases should assign row ids and seeds first, then
  group by exact computed `step_count`, then chunk for memory.
- Exact grouping and chunking are execution details. Bucketizing or capping
  `step_count` changes the numerical grid and must be declared as a different
  YAML rule.
- The production checklist for new databases now lives in
  `docs/validation_dataset.md`.

## 2026-07-10: Notebook And Cleanup Pass

- Consolidated the active validation story around one English Heston lookback
  notebook. This notebook was later superseded by production-audit notebooks
  using the `01` database convention.
- Removed obsolete benchmark notebooks and generated benchmark artifacts from
  the active tree; the Heston Euler/QE/QE-M comparison now lives in tests rather
  than in a notebook.
- Kept only reusable example entry points:
  `examples/benchmark_python_cpp.py` and
  `examples/profile_heston_lookback_runtime.py`.
- Updated the Black-Scholes Python/C++ consistency test to read fixtures from
  the registry model/product databases instead of old generated benchmark
  files.
- The next intended workflow is to repeat the Heston/lookback pattern one
  payoff/model pair at a time: four result databases, compact YAML specs, one
  short validation notebook, and regression tests.

## 2026-07-10: Path Reconstruction Tools

- Moved reusable registry-generation helpers out of `registry` and into
  `tools/registry`, keeping `registry/validation/models`, `registry/validation/products`, and
  `registry/validation/results` symmetric around `data`, `specifications`, and
  `generators`.
- Added C++ path-only reconstruction for the Heston fixed-lookback reference
  case:
  - CPU and CUDA C API functions now export full spot paths without touching the
    pricing kernels.
  - path helpers under `tools/paths` reconstruct CPU/CUDA Philox paths on
    demand.
- Result YAML files now reference both C++ pricing generators and both C++
  path-export scripts under `reproducibility_scripts`.
- Added tests that:
  - reprice the stored Heston lookback result from exported CPU paths;
  - compare CPU/GPU exported paths pathwise when CUDA is visible.

## 2026-07-09: Reproducible Registry Recipes

- Added an early long-form registry convention for reproducible parameter
  databases. This was later replaced by the compact `01` production convention:
  - model database ids include the model family;
  - product database ids include the payoff family, e.g.
    `asian_arithmetic_calls`;
  - result ids reference both inputs and the engine/plan role, e.g.
    `<model_id>__<product_id>__<engine>_<version>`.
- Model YAML files now document only model parameter construction:
  - no RNG, numerical scheme, path count, device, or pricing metadata;
  - `S0 = 1`, `r = 0`, and `q = 0` are fixed in the early validation grids.
- Product YAML files now document only payoff and product-parameter
  construction:
  - payoff expression;
  - homogeneity degree and scaling rule;
  - deterministic Cartesian grid used to build the JSON.
- Result YAML files now own the experiment recipe:
  - exact result id and JSON path;
  - model/product database ids and YAML/JSON paths;
  - generation script path;
  - seed policy and Cartesian planning metadata;
  - placeholder for simulation/pricing source files and backend metadata.
- Added same-basename generation scripts under:
  - `registry/validation/models/generators`;
  - `registry/validation/products/generators`;
  - `registry/validation/results/generators`.
- Split each registry family into `data`, `specifications`, and `generators`
  directories.
- Added `tools/registry/model/core_parameter_databases.py` as generic write
  helpers only. Concrete grids live in the same-basename generator scripts, for
  example same-basename model generators under
  `registry/validation/models/<model_family>/generators`.

## 2026-07-08: Registry And Source Split

- Python source moved to `src_python`.
- C++ source added under `src_cpp`.
- Registry JSON/YAML remains the durable data layer.
- Result rows keep only row-varying data: `id`, `model_id`, `product_id`,
  `seed`, and computed `outputs`.
- Planning and pricing were initially separated into a planner and a batch
  runner. These legacy helpers were later replaced by the aligned-row
  `ProductionPipeline` described in the current architecture documentation.

## 2026-07-08: RNG Policy

- Python benchmark path intentionally uses PyTorch RNG via `torch.randn`.
- C++ CPU and C++ CUDA paths use a project Philox-4x32-10 implementation.
- The Philox reference is:
  Salmon, John K., et al. "Parallel random numbers: as easy as 1, 2, 3."
  Proceedings of 2011 International Conference for High Performance Computing,
  Networking, Storage and Analysis. 2011.
- Random123 reference implementation:
  https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h

## 2026-07-08: CUDA Path

- Added early CUDA Black-Scholes and Heston benchmark kernels in a combined
  CUDA source file that was later removed.
- Kernel design:
  - one CUDA block per database row;
  - threads inside a block split Monte Carlo paths;
  - Philox and Box-Muller run in device code;
  - block reduction computes price and standard error.
- Added `ai_factory_cuda_benchmark` CLI for batch CUDA benchmarks.
- Added `ai_factory_cpu_benchmark` CLI for batch C++ CPU benchmarks.

## 2026-07-08: C++ CPU/GPU Alignment

- Initial C++ CPU/GPU comparison showed statistical agreement but not
  floating-point agreement.
- Root cause: CUDA consumed Philox output words differently from CPU.
- Fix: CUDA now uses `v0/v1` for one Box-Muller pair and `v2/v3` for the next,
  matching the CPU flattening order.
- After the fix, Black-Scholes C++ CPU and C++ CUDA agree to floating-point
  precision in tested benchmark outputs:
  - large benchmark max absolute price difference around `1e-16`;
  - small benchmark max absolute price difference around `4e-16`.

## 2026-07-08: Black-Scholes Benchmark Snapshot

Run configuration:

- model: Black-Scholes;
- large database: `100400` rows;
- paths per row: `256`;
- steps: `1`;
- total simulated trajectories: `25,702,400`.

Observed wall/kernel timings:

- Python CPU PyTorch: about `0.728 s`;
- Python GPU PyTorch: about `0.093 s`;
- C++ CPU Philox: about `4.725 s`;
- C++ CUDA Philox kernel: about `0.0148 s`;
- C++ CUDA process including startup/transfers: about `0.284 s`.

Interpretation:

- C++ CUDA is fastest in compute time.
- Python GPU beats Python CPU on the large benchmark.
- Current C++ CPU is scalar and slower than PyTorch CPU on the large benchmark;
  optimize with OpenMP/vectorization before drawing conclusions about C++ CPU
  potential.
- Python and C++ prices are compared statistically because their RNGs differ.
- C++ CPU and C++ CUDA prices should match to floating-point precision because
  they share Philox.

## 2026-07-08: CUDA Workspace Optimization

- Added reusable CUDA workspaces for Black-Scholes and Heston benchmark paths.
- The CUDA benchmark now allocates device rows/outputs once, before warmups and
  measured repeats.
- `cpp_gpu_cuda_call` now measures the hot path with reusable device buffers:
  host-to-device row copy, kernel launch, device-to-host output copy.
- `cpp_gpu_workspace_setup` separately reports CUDA context/buffer setup.
- A large GPU-only Black-Scholes benchmark was run with:
  - rows: `100400`;
  - paths per row: `4096`;
  - total trajectories: `411,238,400`;
  - warmups: `2`;
  - measured C++ repeats: `8`.
- Observed timings:
  - Python GPU PyTorch wall: `0.8891 s`;
  - C++ CUDA Philox hot call: `0.1501 s`;
  - C++ CUDA Philox kernel: `0.1430 s`;
  - C++ CUDA workspace setup: `0.3892 s`;
  - C++ CUDA process total: `2.1285 s`.
- Interpretation:
  - C++ CUDA hot call is about `5.92x` faster than Python GPU.
  - C++ CUDA kernel-only time is about `6.22x` faster than Python GPU.
  - The remaining hot-call overhead around the kernel is about `7 ms`.
  - The process total is still not a fair pricing metric because it includes
    short-lived executable startup, CUDA initialization, warmups, repeats, and
    teardown.
- Price comparison for the first reported rows remained statistically sound:
  `within_3_combined_standard_errors`, with max difference about `0.91`
  combined standard errors.

## 2026-07-08: Shared-Library CUDA Integration

- Added `libai_factory_cuda.so`, a C ABI shared library wrapping the C++ CUDA
  pricing workspaces.
- Python benchmark now supports `--cuda-backend library` and uses it by default.
- The shared-library path is loaded with Python `ctypes`, so measured C++ CUDA
  calls happen inside the Python process instead of launching
  `ai_factory_cuda_benchmark` as a subprocess.
- The CLI backend remains available with `--cuda-backend cli` for A/B checks.
- Final large Black-Scholes GPU-only benchmark:
  - rows: `100400`;
  - paths per row: `4096`;
  - total trajectories: `411,238,400`;
  - Python GPU PyTorch wall: `1.3190 s`;
  - C++ CUDA shared-library hot call: `0.1505 s`;
  - C++ CUDA kernel: `0.1431 s`;
  - C++ CUDA workspace setup: `0.0018 s`;
  - C++ CUDA measured loop total: `1.5305 s` for warmups plus repeats.
- Speed comparison:
  - C++ CUDA hot call is about `8.76x` faster than Python GPU;
  - C++ CUDA kernel-only is about `9.22x` faster than Python GPU.
- Price comparison against Python GPU remains statistically sound:
  `within_3_combined_standard_errors`; max first-output difference about
  `0.91` combined standard errors.

## 2026-07-08: Heston CUDA Kernel Optimization

- Optimized the Heston CUDA kernel without changing the Philox stream layout.
- Previous Heston implementation called Philox/Box-Muller once per normal.
- New implementation consumes Box-Muller pairs for consecutive time steps when
  the normal index is even, preserving the exact normal ordering:
  - one Philox/Box-Muller pair gives step `t` and step `t+1` for the spot shock;
  - one Philox/Box-Muller pair gives step `t` and step `t+1` for the independent
    variance shock.
- Added a device helper for one Heston Euler step so repeated calculations are
  centralized.
- Precomputed per-row constants in the kernel:
  - drift scale;
  - `kappa * dt`;
  - `volatility_of_variance * sqrt(dt)`;
  - discount factor.
- Benchmark after optimization:
  - model: Heston;
  - rows priced: `1000`;
  - paths per row: `1024`;
  - steps: `64`;
  - total path-steps: `65,536,000`.
- Observed timings:
  - Python GPU PyTorch wall: `0.1067 s`;
  - C++ CUDA shared-library hot call: `0.0306 s`;
  - C++ CUDA kernel: `0.0290 s`.
- Compared with the previous Heston kernel snapshot:
  - hot call improved from about `0.0432 s` to `0.0306 s`;
  - kernel time improved from about `0.0413 s` to `0.0290 s`;
  - this is roughly a `30%` kernel-time reduction.
- Price comparison against Python GPU remained statistically sound:
  `within_3_combined_standard_errors`; max first-output difference about `2.23`
  combined standard errors.

## 2026-07-08: Heston CPU/GPU Pathwise Reproducibility

- Added a C ABI debug/test function for Heston terminal spots:
  - C++ CPU terminal spots from `generate_heston_terminal_spots`;
  - C++ CUDA terminal spots from a dedicated terminal-spot kernel.
- Added `tests/cpp/test_heston_cuda_pathwise.py`.
  - The test compiles `libai_factory_cuda.so`.
  - It calls CPU and GPU Heston terminal-spot generation through `ctypes`.
  - It compares terminal spots path by path for the same model, seed, path
    count, and time grid.
  - The test is skipped when CUDA is not visible.
- GPU run result:
  - paths: `512`;
  - steps: `64`;
  - seed: `900000001`;
  - max absolute terminal-spot difference: `8.88e-16`;
  - mean absolute terminal-spot difference: `6.61e-17`.
- Added `examples/heston_cpu_gpu_pathwise.py` to generate a durable JSON report:
  `examples/generated_benchmarks/heston_cpu_gpu_pathwise.json`.
- Added `notebooks/heston_benchmark_overview.ipynb`.
  - It shows Python GPU vs C++ GPU Heston timings.
  - It shows Python/C++ price agreement in Monte Carlo standard-error units.
  - It shows C++ CPU vs C++ GPU pathwise terminal-spot agreement.

## 2026-07-08: Comparative Validation Notebook

- Added a compact notebook at `notebooks/2026-07-08_comparative_validation.ipynb`.
- The notebook consolidates the main evidence for the current benchmark stack:
  - C++ CPU and C++ GPU produce the same Monte Carlo paths and prices for both
    Black-Scholes and Heston, up to floating-point precision.
  - Python and C++ remain statistically consistent, but are not bitwise-identical
    because they use different RNGs.
  - The measured timing hierarchy is clear: the C++ CUDA kernel is the fastest
    path, ahead of Python GPU and the other C++ paths.
- The notebook is intentionally compact and focuses on the certification points
  needed for reporting: pathwise agreement, statistical agreement with Python,
  and a simple performance ranking.

## Open Follow-Ups

- Add pathwise dump tests for first normals and full intermediate paths.
- Add chunked Python GPU path for large Heston cases.
- Add OpenMP/vectorized C++ CPU implementation.
- Add a report generator that turns benchmark summaries into Markdown tables.
- Turn the current `ctypes` C ABI into the production integration layer, or
  replace it with a richer Python extension if tighter array interop becomes
  necessary.

## 2026-07-08: Python/C++ GPU Dataset Validation Fix

- Rechecked C++ Philox reproducibility with CUDA enabled:
  - `tests/cpp/test_philox_cpu_gpu_consistency.py` passed for Black-Scholes and
    Heston prices.
  - `tests/cpp/test_heston_cuda_pathwise.py` passed for Heston terminal spots
    path by path.
- Fixed the Python benchmark seed policy:
  - previous benchmark used one PyTorch RNG stream seeded by the first row;
  - current default reseeds Python GPU row by row from the result-plan seed;
  - `--python-single-stream` remains available for raw single-stream timing.
- Added full result-file comparisons to `benchmark_summary_*.json` under
  `dataset_comparisons`.
  - Python GPU still uses PyTorch RNG and C++ GPU uses project Philox, so they
    are compared as independent Monte Carlo estimators, not pathwise.
- Regenerated GPU-only benchmark datasets with `2048` paths on `1000`
  large-sample rows.
  - Black-Scholes large sample: `2 / 1000` rows outside 3 combined standard
    errors; status `statistically_consistent`.
  - Heston large sample: `3 / 1000` rows outside 3 combined standard errors;
    status `statistically_consistent`.
- Updated `notebooks/2026-07-08_comparative_validation.ipynb` so it shows:
  - dataset-level Python GPU vs C++ GPU statistical validation;
  - GPU timing summary;
  - the separate strict C++ CPU/GPU Philox validation.

## 2026-07-09: Heston Andersen QE / QE-M Schemes

- Added three configurable Heston schemes:
  - `euler`: existing full-truncation Euler baseline.
  - `qe`: Andersen Quadratic-Exponential variance scheme.
  - `qe_martingale`: QE plus martingale drift correction for the log spot.
- Implemented the schemes in:
  - Python CPU/GPU benchmark path through PyTorch.
  - C++ CPU path with project Philox.
  - C++ CUDA kernel path with project Philox.
- C++ implementation details:
  - `psi_c = 1.5`.
  - central log-spot discretization with `gamma1 = gamma2 = 0.5`.
  - QE-M falls back to uncorrected QE on a step if the martingale moment
    regularity condition fails.
- Added CLI support:
  - `--heston-scheme euler|qe|qe_martingale` in benchmark tooling.
- Added `examples/heston_qe_validation.py`.
  - Generates Python CPU, Python GPU, C++ CPU, and C++ GPU datasets for all
    three schemes.
  - Writes `examples/generated_benchmarks/heston_qe/heston_qe_validation_summary.json`.
- Added `notebooks/heston_andersen_qe_validation.ipynb` as the explanatory
  resource notebook.
- Validation run with `16` rows, `8192` paths, `64` steps:
  - C++ CPU/GPU matched at floating-point precision for Euler, QE, and QE-M.
  - Python GPU vs C++ GPU was statistically consistent for all three schemes.
  - C++ GPU scheme differences are now reported as pairwise pricing errors,
    not raw mean/min/max prices:
    - Euler vs QE: mean absolute price difference `8.96e-4`, max absolute
      difference `4.09e-3`, RMSE `1.28e-3`.
    - Euler vs QE-M: mean absolute price difference `8.96e-4`, max absolute
      difference `4.09e-3`, RMSE `1.28e-3`.
    - QE vs QE-M: mean absolute price difference `1.81e-7`, max absolute
      difference `6.34e-7`, RMSE `3.15e-7`.
  - Timing interpretation in the notebook distinguishes:
    - Python GPU `wall_seconds` and `simulation_seconds`.
    - C++ GPU `wall_seconds`, which includes warmups and measured repeats.
    - C++ GPU hot-path `cuda_call_seconds` and kernel-only `kernel_seconds`,
      which are the fair metrics to compare against Python GPU simulation time.

## 2026-07-09: Rough Bergomi Hybrid Scheme

- Added a rough Bergomi benchmark model with flat forward variance curve:
  - parameters: `spot`, `risk_free_rate`, `dividend_yield`,
    `forward_variance`, `eta`, `alpha`, `rho`;
  - supported range: `alpha in (-0.5, 0)`, `rho in (-1, 1)`.
- Implemented the Bennedsen-Lunde-Pakkanen hybrid scheme for the truncated BSS
  process used by rough Bergomi:
  - current implementation uses `kappa = 1`;
  - the singular current-cell integral is sampled jointly with the Brownian
    increment;
  - past cells use the optimal evaluation points
    `b_k = ((k^(alpha+1) - (k-1)^(alpha+1)) / (alpha+1))^(1/alpha)`.
- Added rough Bergomi to:
  - Python CPU/GPU PyTorch benchmark path;
  - C++ CPU Philox path;
  - C++ CUDA Philox kernel path;
  - shared-library `ctypes` benchmark backend.
- Added full dataset comparisons for:
  - Python CPU vs Python GPU;
  - Python CPU vs C++ CPU;
  - Python GPU vs C++ GPU;
  - C++ CPU vs C++ GPU.
- Added `examples/2026_07_09_cross_model_validation.py` and
  `notebooks/2026-07-09_cross_model_validation.ipynb`.
- Optimized the rough Bergomi CUDA kernel:
  - previous kernel recomputed past Brownian shocks and hybrid weights inside
    the convolution loop;
  - current kernel precomputes per-path Brownian increments and singular-cell
    terms for step counts up to `256`;
  - hybrid weights are computed once per row/block in shared memory.
- Consolidated validation run updated to use a true small/large split:
  - `small`: `50` rows (`5` models by `10` products);
  - `large_full`: `10,000` rows (`100` models by `100` products);
  - `512` Monte Carlo paths per row in the current notebook summary.
  - Black-Scholes, Heston QE-M, and rough Bergomi all passed the dataset
    statistical checks.
  - C++ CPU/GPU matched near floating-point precision for all three models.
  - C++ CPU/GPU max absolute price differences were below `5e-16` on the
    `large_full` grid for Black-Scholes, Heston QE-M, and rough Bergomi.
  - `large_full` hot-path simulation timings:
    - Black-Scholes: C++ GPU kernel `0.0025 s`, Python GPU `0.325 s`,
      C++ CPU `0.186 s`, Python CPU `0.418 s`.
    - Heston QE-M: C++ GPU kernel `0.162 s`, Python GPU `1.292 s`,
      C++ CPU `14.627 s`, Python CPU `23.574 s`.
    - Rough Bergomi: C++ GPU kernel `0.181 s`, Python GPU `2.450 s`,
      C++ CPU `14.244 s`, Python CPU `25.813 s`.

## 2026-07-08: Payoff Registry Expansion

- Added generic PyTorch payoff functions, now located in
  vanilla calls, asset-or-nothing digitals, cash-or-nothing digitals,
  up-and-out calls, arithmetic Asian calls, fixed/floating lookbacks, log
  contracts, realized variance/volatility swaps, and quadratic power calls.
- Added an initial generated-registry prototype. This was later replaced by the
  `data/specifications/generators` registry convention.
- Generated normalized product databases `000002` with YAML documentation for:
  payoff expression, homogeneity degree, scaling rule, construction grid, and
  Python payoff function.
- Generated large normalized model databases:
  - Black-Scholes: `800` models from a deterministic Cartesian grid.
  - Heston: `800` models from a deterministic modular grid over broad parameter
    ranges.
- Generated Cartesian result plans for every large model/product pair:
  - `22` result-plan JSON/YAML pairs;
  - largest single plans: `108,000` rows for BS/Heston up-and-out calls;
  - rows intentionally contain no prices yet, only `(model_id, product_id, seed)`.
- Added tests that check payoff scaling degrees, generated registry
  documentation, large model row counts, and selected result-plan Cartesian
  counts.

# AI Factory

AI Factory is a C++20/CUDA workbench for quantitative model simulation,
financial-product pricing, and reproducible dataset generation. It provides
shared CUDA engines for Markovian, rough-volatility, closed-form, Monte Carlo,
and early-exercise workloads.

The repository contains source code and versioned dataset recipes. Large JSON
datasets are generated locally or downloaded separately; they are not stored
in Git.

## Start here

Choose the shortest path for your task:

| Goal | Entry point |
|---|---|
| Use a price, Greek, SE or dataset as a reference | [Unresolved closures](docs/audit/unresolved-closures.md) and [open findings](docs/audit/response.md) |
| Build and run a first test | [Quick start](#quick-start) |
| Understand CMake and the `build/` folder | [CMake build guide](docs/cmake-build-workflow.md) |
| Understand the repository | [Documentation map](docs/README.md) |
| Understand CUDA composition | [Pricing-policy composition](docs/cuda/pricing-policy-composition.md) |
| Add a model, curve, product, or dataset | [Catalogue extension workflow](docs/catalog-extension-and-validation-workflow.md) |
| Generate model parameters or product rows | [Parameter-dataset contract](docs/model-and-product-parameter-dataset-generation.md) |
| Generate model-only samples | [Model-sample contract](docs/model-sample-dataset-generation.md) |
| Train or compare deep pricers | [Learning guide](learning/README.md) |
| Organize local studies | [Local workspaces](docs/local-workspaces.md) |
| Run or resume price/sample generation | [Dataset-generation workflow](docs/dataset-generation-workflow.md) |
| Validate generated prices independently | [Price-validation pipeline](docs/independent-price-validation-pipeline.md) |
| Diagnose or tune CUDA kernels | [Kernel diagnostics](docs/cuda/launch-validation-and-kernel-diagnostics.md) and [performance protocol](docs/performance-regression-protocol.md) |
| Modify generated bindings | [Code-generation guide](tools/codegen/pricing_bindings/README.md) |

## Capabilities

- Exact and fixed-step Markovian simulation.
- Gaussian-Volterra FFT simulation and Markovian N-factor rough lifts.
- Closed-form equity and fixed-income pricing.
- Standard Monte Carlo and Longstaff--Schwartz pricing.
- Model-only sample generation for generative-model training.
- Reproducible parameter, product, sample, and price datasets.
- Independent cached price validation through Premia and QuantLib adapters.
- Architecture-specific CUDA diagnostics and performance baselines.

The typed capability manifest is the authoritative inventory of models,
products, engines, generated bindings, and catalogue recipes:
[`tools/codegen/pricing_bindings/capability_manifest.py`](tools/codegen/pricing_bindings/capability_manifest.py).
Documentation does not duplicate that evolving matrix.

## Quick start

### Requirements

- CMake 3.20 or newer and a build tool;
- A CUDA-compatible C++20 host compiler (GCC 10 through 14 with CUDA 12.9);
- CUDA Toolkit 12.9.x;
- `nlohmann-json3-dev`;
- Python 3 with PyYAML and `jsonschema` for catalogue workflows and their tests;
- an NVIDIA GPU for CUDA runtime tests.

The Volterra FFT targets use the separate mathDx/cuFFTDx package. Use the
[CUDA 12 archive of mathDx 25.12.1](https://developer.nvidia.com/cufftdx-downloads)
with CUDA 12.9.

### Local RTX 4090 build

Use C++20/CUDA 12.9.0 (nvcc 12.9.41) and mathDx in an SM89 build directory.
With CUDA 12.9 installed, configure an isolated build using its `nvcc`:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda-12.9/bin/nvcc \
  -DCUDA_WORKBENCH_ARCHITECTURES=89
cmake --build build --target ai_factory_host_tests
ctest --test-dir build -R '^dataset_catalog$' --output-on-failure
# Optional: compile a catalogue generator by target name.
cmake --build build --target generate_black_scholes_01
```

The `local-sm89` preset additionally selects the developer's pinned toolchain
and mathDx paths when those are available. `CUDA_VISIBLE_DEVICES` selects a GPU that can run the compiled
binary. The default `all` target builds libraries; `local-tests` and
`local-generators` cover the test and catalogue aggregates explicitly.
Optional recipes in `work/catalog/` remain buildable by target name when
present locally, but are excluded from the permanent generator aggregates.

### Run one catalog generator

The local preset can compile and run any of the 710 catalogue generators
by CMake target name. The launcher keeps outputs in a new directory under
`work/generation/runs/`, so an existing dataset is never overwritten. It
copies required parameter inputs and builds missing parameter generators when
needed. The native generator writes to relative `catalog/` and `datasets/`
paths in that run directory. The launcher prints the dataset and receipt paths
when finished.

```bash
./tools/run_generator.py local-sm89 --list
./tools/run_generator.py local-sm89 generate_black_scholes_european_calls_01 --gpu 0
./tools/run_generator.py local-sm89 generate_black_scholes_samples_01 --gpu 0 --smoke-test
```

A notebook can create a unique directory under `work/` and pass it as
`--run-dir`. Write its `run.log` and optional `progress.json` there as well;
then the journal, dataset, metadata, recipe and inputs stay under the same run
directory. The launcher accepts an existing run directory only if it contains
those journal or progress files. Compilation updates the selected CMake build
directory under `builds/`; generation does not modify the root `catalog/` or
`datasets/` unless `--publish` is explicitly requested.

The output file is `generation.yaml` (the source is `generator.cpp`). Its
`provenance.hardware_used` records the actual CPU or GPU, and
`provenance.build_options` records the CUDA architecture, compiler versions,
C++20/CUDA20, Release flags, the NVCC flags of linked CUDA libraries, and the
SHA-256 of the CUDA math header used during compilation.
The `tuning_profile.measured_on` field describes the origin of a tuning profile;
it is not the device observed during this run. `--publish` is optional and
only accepts a new immutable dataset and receipt pair whose parameter inputs
are already published in the project.

### Another GPU

With the local CUDA 12.9 toolchain installed, reuse the shared preset in a
separate build directory and set that GPU's compute capability and tuning ID:

```bash
cmake --preset local-sm89 -B builds/sm_XX \
  -DCUDA_WORKBENCH_ARCHITECTURES=<compute-capability> \
  -DAI_FACTORY_CUDA_TUNING_PROFILE_ID=smXX_unqualified_v1
cmake --build builds/sm_XX --target ai_factory_host_tests -j2
ctest --test-dir builds/sm_XX --output-on-failure -R '^dataset_catalog$'
```

On another machine, also set `CMAKE_CUDA_COMPILER`,
`CMAKE_CUDA_HOST_COMPILER`, `CMAKE_CXX_COMPILER`, and
`AI_FACTORY_MATHDX_ROOT` to its CUDA 12.9/C++20/mathDx CUDA 12 installation.

Before publishing tuning values for another GPU, follow the
[performance regression protocol](docs/performance-regression-protocol.md) and
record a separate architecture profile.

## Repository map

```text
src/          Runtime C++/CUDA models, products, curves, and shared primitives
learning/     Python/PyTorch training, shared data contracts and evaluation
tools/        Offline generation, publication, code generation, and diagnostics
work/         Ignored, local recipes, generation campaigns and experiments
catalog/      Published recipes and generation metadata
datasets/     Published or downloaded dataset artifacts only; ignored by Git
build/        Legacy local CMake build; ignored by Git
builds/       Separate CMake builds per CUDA architecture; ignored by Git
artifacts/    Local audit evidence, performance runs and tool caches; ignored by Git
tests/        Host, CUDA, architecture, and performance tests
cmake/        Build ownership by runtime, catalogue, tests, and performance
docs/         Task-oriented workflows, contracts, references, and audit records
```

The reusable code under `src`, `learning`, and `tools` never depends on a
concrete experiment. Experiment-specific datasets, seeds, training budgets,
notebooks, and reports live under `work/experiments/<asset_class>/<model>/`.
Frozen generation campaigns live under `work/generation/`; a result enters
`datasets/` only through explicit publication.
The runtime under `src` never depends on `tools`, `catalog`, or `validation`.
Model-product launch units live under each model's `product/` directory.
`catalog` and `datasets` mirror the canonical model, curve, and product paths
defined by `src`.

For the detailed ownership and file conventions, use the
[documentation map](docs/README.md) rather than inferring them from this
summary.

## Common workflows

Build the narrowest target for the change. These examples select the local
SM89 build:

```bash
cmake --build --preset local-core
cmake --build --preset local-equity
cmake --build --preset local-fixed-income
cmake --build --preset local-tests
ctest --preset local-tests
```

Inspect registered targets and tests in the active local build:

```bash
cmake --build builds/local-cuda12-9-0-sm89 --target help | rg 'generate_heston_'
ctest --test-dir builds/local-cuda12-9-0-sm89 -N
```

To compile and run a single recipe while keeping its outputs under `work/`, use
the generator launcher shown above, for example:

```bash
./tools/run_generator.py local-sm89 generate_heston_01 --gpu 0
```

The generator owns its JSON and adjacent YAML output; do not edit generated
metadata by hand. Use the
[catalogue extension workflow](docs/catalog-extension-and-validation-workflow.md)
for the complete publication and validation sequence.

Generated pricing and sampling bindings must round-trip without a diff:

```bash
python3 tools/codegen/pricing_bindings/generate.py \
  --family all \
  --output /tmp/ai_factory-pricing-bindings \
  --compare-root .
```

## Reproducibility and portability

- CUDA fast math is intentionally disabled.
- Philox keys and counters are independent of batching and launch geometry.
- Contractual dates use integer business-day counts; discretized models derive
  their fixed step from the declared global time grid.
- Performance evidence is scoped to the recorded GPU, toolchain, binary, and
  power state. Other GPUs require their own measurements.
- Independent validation normally reads versioned caches. External engines run
  only during explicit reference regeneration.

Numerical and CUDA invariants are normative in the contracts under
[`docs/cuda`](docs/cuda/README.md).

## Current limitations

Before using numerical outputs or datasets as references, read the
[unresolved closures](docs/audit/unresolved-closures.md),
[open audit findings](docs/audit/response.md) and the relevant
[resolved contract](docs/audit/closed.md). NUM-037 and PERF-027 remain numerically unresolved despite their exclusion
mask or withdrawn Hessian. Accepted capability limits and tradeoffs are
recorded with the resolved findings.

- Equity pricing can also compute spot delta, the price sensitivity to the
  initial asset level. Markovian and six rough models have separate paired
  launchers and recipes. Tests cover selected cases. Catalogue-wide delta
  bias and production performance still need qualification. Rough early
  exercise is not implemented.
  See the [price-delta contract](docs/cuda/equity-price-delta-contract.md) and
  [open audit work](docs/audit/response.md).
- Catalogue URLs using `datasets.ai-factory.example` are placeholders until a
  production data host is configured.
- The checked-in runtime performance baseline covers only the RTX 4090 Laptop
  `sm_89` profile.
- CUDA tests require compatible NVIDIA hardware; host-only tests remain
  available without running GPU kernels.

For every substantial change, start from the relevant workflow in
[`docs/README.md`](docs/README.md) and build only the affected targets before
running the broader suites.

# AI Factory

AI Factory is a C++23/CUDA workbench for quantitative model simulation,
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
| Build and run a first test | [Quick start](#quick-start) |
| Understand CMake and the `build/` folder | [CMake build guide](docs/cmake-build-workflow.md) |
| Understand the repository | [Documentation map](docs/README.md) |
| Understand CUDA composition | [Pricing-policy composition](docs/cuda/pricing-policy-composition.md) |
| Add a model, curve, product, or dataset | [Catalogue extension workflow](docs/catalog-extension-and-validation-workflow.md) |
| Generate model parameters or product rows | [Parameter-dataset contract](docs/model-and-product-parameter-dataset-generation.md) |
| Generate model-only samples | [Model-sample contract](docs/model-sample-dataset-generation.md) |
| Train or compare deep pricers | [Learning guide](learning/README.md) |
| Browse concrete studies | [Experiments guide](experiments/README.md) |
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

- CMake 3.20 or newer and Ninja;
- GCC 14 or another CUDA-compatible C++23 compiler;
- CUDA Toolkit 13.3 or newer;
- `nlohmann-json3-dev`;
- an NVIDIA GPU for CUDA runtime tests.

cuFFTDx is optional. It is required only for mathDx-backed Volterra FFT
targets and is enabled through `AI_FACTORY_MATHDX_ROOT`.

### Reference SM89 build

The checked-in `dev` preset targets the repository's RTX 4090 Laptop reference
machine (`sm_89`, GCC 14, CUDA 13.3):

```bash
cmake --preset dev
cmake --build --preset host-tests
ctest --test-dir build --output-on-failure -R '^dataset_catalog$'
```

The final command provides a small observable smoke result without generating
large datasets.

### Another GPU

Configure a separate build directory with the target GPU's compute capability:

```bash
cmake -S . -B builds/sm_XX -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_WORKBENCH_ARCHITECTURES=<compute-capability> \
  -DBUILD_TESTING=ON
cmake --build builds/sm_XX --target ai_factory_host_tests -j2
ctest --test-dir builds/sm_XX --output-on-failure -R '^dataset_catalog$'
```

The SM89 launch profile is a safe reference, not a universal optimum. Before
publishing tuning values for another GPU, follow the
[performance regression protocol](docs/performance-regression-protocol.md) and
record a separate architecture profile.

## Repository map

```text
src/          Runtime C++/CUDA models, products, curves, and shared primitives
learning/     Python/PyTorch training, shared data contracts and evaluation
tools/        Offline generation, publication, code generation, and diagnostics
experiments/  Model-oriented studies, concrete campaigns, notebooks, and reports
catalog/      Versioned executable recipes and adjacent dataset metadata
datasets/     Generated or downloaded JSON artifacts; ignored by Git
build/        Main local CMake build; ignored by Git
builds/       Optional separate CMake builds; ignored by Git
artifacts/    Local audit evidence, performance runs and tool caches; ignored by Git
tests/        Host, CUDA, architecture, and performance tests
validation/   Independent price-reference pipelines and backend adapters
cmake/        Build ownership by runtime, catalogue, tests, and performance
docs/         Task-oriented workflows, contracts, references, and audit records
```

The reusable code under `src`, `learning`, and `tools` never depends on a
concrete experiment. Experiment-specific datasets, seeds, training budgets,
notebooks, and reports live under `experiments/<asset_class>/<model>/`.
The runtime under `src` never depends on `tools`, `catalog`, or `validation`.
Model-product launch units live under each model's `product/` directory.
`catalog` and `datasets` mirror the canonical model, curve, and product paths
defined by `src`.

For the detailed ownership and file conventions, use the
[documentation map](docs/README.md) rather than inferring them from this
summary.

## Common workflows

Build the narrowest target for the change. Useful aggregate targets are
available as presets:

```bash
cmake --build --preset core
cmake --build --preset equity
cmake --build --preset fixed-income
cmake --build --preset tests
ctest --preset tests
```

Inspect primary targets, then search the complete Ninja target list for
individual generators:

```bash
cmake --build build --target help
ninja -C build -t targets all | rg '^generate_heston_'
ctest --test-dir build -N
```

One dataset recipe can be built and executed directly, for example:

```bash
cmake --build build --target generate_heston_01 -j2
./build/generate_heston_01
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

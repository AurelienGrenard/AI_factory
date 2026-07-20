# Project Architecture And Engineering Philosophy

This document is the architectural contract of AI Factory. It explains where
code belongs, how dependencies must flow, and how a new model/product dataset
must fit into the project. File-level details are listed in `code_guide.md` and
the CUDA execution contract is defined in `cuda.md`;
production and validation procedures are defined in `production_dataset.md`
and `validation_dataset.md`.

This document is authoritative for ownership and dependency rules. Read it
before `production_dataset.md`, `validation_dataset.md`, `certification.md`,
`code_guide.md`, and `cuda.md`. `journal.md` records history and `roadmap.md` records possible
future work; they never override the current contracts.

## Core Principles

1. Production datasets are the objective of the project. Validation datasets,
   alternate engines, tests, and notebooks exist to establish confidence in
   the production CUDA implementation.
2. Numerical source code is reusable and independent of the registry. Dataset
   recipes orchestrate source code; they do not contain pricing algorithms.
3. Every model/product pair has an explicit source implementation. Market
   inputs such as initial curves remain separate registry entities. A generic
   dispatcher must not hide the absence of the corresponding source file.
4. Code shared by several implementations has one owner in `common`. Code that
   is genuinely specific to a model/product pair remains specialized.
5. Similar examples follow the same naming, signatures, execution strategy,
   metadata, tests, and notebook layout.
6. Reproducibility is part of the implementation contract, not an informal
   property of a generated JSON file.

## Architectural Layers

The project has four distinct layers. Dependencies flow from registry scripts
to tools, from tools to source engines, and never in the opposite direction.

### Source Engines

`src_cpp` and `src_python` contain simulation and pricing implementations.

```text
src_cpp/ai_factory/
  cpu/
    common/
    <model>/
      common.hpp/.cpp
      <product>.hpp/.cpp
  cuda/
    common/
    <model>/
      api.cuh
      dynamics.cuh
      paths.cu
      <product>.cuh/.cu       # .cuh only when declarations are shared
  c_api/
    c_api.cpp

src_python/ai_factory/pytorch/
  common/
  <model>/
    common.py
    pathwise.py
    <product>.py
```

Simulation schemes, path evolution, payoff evaluation, Monte Carlo reduction,
regression, pricing, and gradients belong in `src`. They must not depend on
`tools`, `registry`, or notebooks.

CUDA kernels are specialized for the model/product pair. Simulation and payoff
accumulation remain fused when that is the efficient design. Moving inline
device helpers into `common/*.cuh` or `<model>/dynamics.cuh` does not imply a
separate kernel launch or materialized path database.
Analytic, European Monte Carlo, and European path-dependent kernels follow the
canonical execution families and mixed-precision policy in `cuda.md`.

PyTorch uses one implementation for CPU and GPU through a `device` argument.
It must be vectorized, batched, readable, and faithful to PyTorch conventions.
Python loops over paths or time steps are not acceptable when tensor operations
provide the same calculation. Thin model/product wrappers are acceptable when
they make the selected simulator and payoff explicit.

### Tools

`tools` contains reusable orchestration and infrastructure.

```text
tools/
  common/                    # native loading, time-grid rules
  paths/<model>.py           # model-only path reconstruction helpers
  registry/
    common/                  # registry paths, writers, slicing primitives
    curve/                   # market-curve generation and slicing facade
    model/                   # model generation and model slicing facade
    product/                 # product generation and product slicing facade
    result/
      common/                # metadata, grouping, production pipeline
      <model>/
        common.py            # model-wide orchestration helpers when needed
        <product>.py         # public pipeline entry point
```

Tools may load parameters, align rows, group them by numerical grid, call a
source engine, record timings, and write JSON/YAML files. They must not contain
a second implementation of a stochastic model or payoff.

The public result entry point is always
`tools.registry.result.<model>.<product>`. A shared model dispatcher may live in
`<model>/common.py`, but registry generators import the product facade. This
keeps every supported pair visible and discoverable.

### Registry

`registry` is the durable dataset layer.

```text
registry/
  production/
    models/<model>/{data,specifications,generators}
    curves/<curve>/{data,specifications,generators}
    products/<product>/{data,specifications,generators}
    results/<model>/<product>/{data,specifications,generators}
  validation/
    ... same family layout ...
```

JSON is machine-readable data. YAML is compact human-readable provenance.
Same-basename generators are small executable recipes; they import one tool
entry point and contain no numerical implementation.

Production model, curve when required, and product databases normally contain
`N` rows. Equity results pair model row `i` with product row `i`. Fixed-income
results pair model row `i`, curve row `i`, and product row `i`. Validation
generators deterministically slice every production input; they do not recreate
the parameter sampling rule independently.

Curve-fitting short-rate models store only stochastic model parameters, while
the curve registry stores curve parameters. Shared fixed-income source helpers
derive `discount_factor(T)` and `instantaneous_forward(T)` analytically;
model-specific code derives the deterministic shift `phi(t)`. Discount factors
and forward-rate grids are not duplicated in model JSON files.

Production results are generated with the validated CUDA engine. Validation
results independently reprice the production slice with CUDA, C++ CPU, PyTorch
GPU, and PyTorch CPU.

### Registry Schema Contract

Every database has a same-basename JSON, YAML, and generator. The JSON and YAML
share `format`, `database_id`, `generation_script`, and row count. JSON stores
rows; YAML documents their meaning and construction.

Curve YAML requires `title`, `format`, `database_id`, `curve_family`,
`json_path`, `generation_script`, `parameters`, `equations`, and `construction`.
Model YAML uses the same contract with `model_family` and `dynamics`; product
YAML uses `product_family` and `payoff`. Their JSON files respectively contain
`curves`, `models`, or `products`, and every row contains exactly the common
identity structure `id` plus `parameters`. Specialized parameter names remain
database-specific.

Result YAML requires `title`, `format`, `database_id`, `json_path`,
`generation_script`, `summary`, `time_grid`, `outputs`, `model_database`,
`product_database`, `result_construction`, and `timing`. `summary` always
contains `row_count` and `source_files`. A result that consumes a market curve
also requires `curve_database`.

Input database references use the same nested representation in result JSON
and YAML:

```yaml
model_database:
  id: heston_03
  json_path: registry/production/models/heston/data/heston_03.json
```

The flat legacy fields `model_database_id`, `curve_database_id`, and
`product_database_id` are forbidden. A database reference id identifies the
whole input database; `model_id`, `curve_id`, and `product_id` inside a result
row identify rows of those databases. `result_construction` is also present and
identical in result JSON and YAML so aligned, Cartesian, or explicit mappings
remain understandable from either artifact.

The top-level contract is exact: a registry document may contain only its
mandatory fields and explicitly allowed top-level extensions such as
`monitoring`, `exercise`, or `delta_method`. Nested financial blocks remain
family-specific. Analytic results retain the common `time_grid` key with a rule
stating that no numerical grid is applicable.

`summary.source_files` lists only the primary specialized implementation unit
for the selected engine:

```text
C++ CUDA  -> src_cpp/ai_factory/cuda/<model>/<product>.cu
C++ CPU   -> src_cpp/ai_factory/cpu/<model>/<product>.cpp
PyTorch   -> src_python/ai_factory/pytorch/<model>/<product>.py
```

Headers, common numerical dependencies, the C API, and registry orchestration
are intentionally omitted. They are discoverable through includes, imports,
and `generation_script`; copying the transitive dependency closure into every
result YAML is noisy and becomes stale after refactoring. The field remains a
list only for an exceptional implementation genuinely split across several
specialized source units.

### Notebooks And Tests

Validation notebooks present evidence; they are not source code. Equivalent
examples use the same cells, table schemas, plots, labels, and ordering. A
notebook may call helpers but must not implement a private pricing routine.
Loading, timing presentation, and result comparisons are owned by
`tools.validation.audit`, which is shared by notebooks and automated tests.

Automated tests own invariants that should not rely on visual inspection:

- deterministic model and product parameter slicing;
- direct comparison of the first production result rows with regenerated validation rows;
- C++ CPU/CUDA Philox reproducibility;
- production CUDA regeneration;
- price-only versus price-and-gradient consistency with the same seed;
- path reconstruction and repricing;
- metadata and source-path integrity.

## Dependency Rules

These rules are mandatory:

- `src` imports only source-level common modules, never `tools` or `registry`.
- product tools do not import model tools;
- one model's source or result module does not import another model's module;
- model and product validation generators import their respective slicing
  facades only;
- result generators import their exact model/product result facade;
- notebooks do not become dependencies of generators or tools;
- `common` modules never import a specialized model/product implementation.

Examples of forbidden dependencies include Rough Bergomi importing a Heston
result helper, an American-put pipeline importing a lookback pipeline, or a
product writer importing a model writer merely to reuse generic file I/O.

## What Belongs In Common

Move code to `common` when all of the following are true:

- at least two implementations use the same semantics;
- the code has one stable owner independent of a model/product pair;
- extraction preserves a clear public interface;
- extraction does not compromise kernel fusion or performance.

Typical common code includes Philox, reductions, CUDA runtime helpers, native
library loading, time-grid rules, Monte Carlo summaries, slicing, JSON/YAML
writers, Laguerre bases, and generic payoff summaries.

Do not extract code merely because two files look similar. Model dynamics,
CUDA kernels, ctypes symbol configuration, model-specific state, and specialized
regression logic may legitimately share a shape while requiring separate
implementations. Thin wrappers are preferable to a generic abstraction that
hides which model and product are being priced.

## Reproducibility Contract

C++ CPU and CUDA use the same project Philox mapping. Given the same row,
parameters, seed, time grid, scheme, and path count, they must represent the
same random experiment and agree to floating-point precision where specified.

Price-and-gradient CRN engines reuse the base random drivers for down, base,
and up spots. Their base price must match the corresponding price-only engine
when seed and simulation settings are identical.

`src_cpp/ai_factory/cuda/<model>/paths.cu` reconstructs the same spot paths used
by pricing kernels. `tools/paths/<model>.py` is a model-only helper for either
reconstructing a result row or simulating requested paths. Path reconstruction
does not depend on the payoff and never slows the pricing kernel.

PyTorch provides an independent statistical validation layer. It need not share
Philox or produce pathwise-identical samples, but CPU and GPU implementations
must implement the same algorithm and yield statistically coherent outputs.

## Adding A Model/Product Pair

A complete pair follows this order:

1. agree on model and product parameter distributions or grids;
2. generate production model and product databases;
3. implement CPU C++, CUDA, and shared-device PyTorch source files;
4. expose the native functions through the C API;
5. add the product result facade under `tools/registry/result/<model>`;
6. generate the production CUDA result;
7. slice the first validation rows from production model and product parameters;
8. independently reprice the slice with the four validation engines;
9. validate paths when the model supports reconstruction;
10. add the standard validation notebook and automated tests.

Price-only and price-plus-gradient datasets are separate deliverables. A new
pair must support only the outputs requested by its product, but each supported
variant follows the same pipeline.

## Architectural Sanity Checklist

Before accepting a restructuring or a new pair, verify:

- every registry path, generation script, and YAML `source_files` entry exists;
- every `.cpp` and `.cu` source is included in CMake;
- CPU product files have their expected headers;
- no obsolete ids, audit modules, empty directories, caches, or generated
  bytecode remain in the project tree;
- all registry generator modules import successfully;
- validation slices equal the corresponding production prefixes exactly;
- no cross-model or product-to-model orchestration imports exist;
- shared code has one owner and specialized code remains discoverable;
- CPU and CUDA smoke tests pass after moving native interfaces;
- documentation reflects the current tree rather than historical layouts.

Performance audits are separate from architecture audits. Structural changes
must preserve numerical behavior and should not alter hot kernels unless the
task explicitly includes performance work.

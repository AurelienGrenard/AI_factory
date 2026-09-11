# Offline tools

`tools` owns offline construction, generation, publication, code generation,
and diagnostics. Runtime code under `src` never depends on this tree.

## Find a tool

| Task | Owner |
|---|---|
| Generate or verify pricing and sampling bindings | [`codegen/pricing_bindings`](codegen/pricing_bindings/README.md) |
| Build parameter, sample, or price datasets | `datasets/` |
| Inspect dataset provenance and reuse | [`datasets/check_dataset_compatibility.py`](datasets/check_dataset_compatibility.py), [contract](../docs/dataset-provenance-contract.md) |
| Run CUDA pricing from an offline recipe | `cuda/pricing_runner.cuh` |
| Compose product-specific price generation | `pricing/` |
| Generate model parameters and samples | `sampling/` |
| Run performance campaigns and profiling | [`performance/`](../docs/performance-regression-protocol.md) |

## Ownership boundaries

- `datasets/sampling.*` owns pure row, grid, and stress sampling.
- `datasets/*_dataset.*` owns dataset assembly and publication by artifact
  family.
- `datasets/artifact_io.*` owns JSON/YAML serialization.
- `cuda/` owns reusable offline CUDA execution and architecture checks.
- `pricing/` owns product-specific price-generation orchestration.
- `sampling/` owns Philox parameter generation and model-sample orchestration.
- `codegen/` owns generated bindings, recipes, manifests, and drift checks.
- `performance/` owns benchmark execution, comparison, rebaseline, and
  profiling tools.

Catalogue `generator.cpp` files are thin executable recipes. They select
inputs, launch arguments, and publication metadata; they do not own generic
CUDA resources, serialization, or numerical implementations.

The [typed capability manifest](codegen/pricing_bindings/capability_manifest.py)
is the source of truth for declared recipes and generated compositions.
`cuda/check_catalog_generators.py` verifies the catalogue against that
manifest in both directions.

# Offline tooling boundaries

The offline dataset pipeline is split by responsibility:

- `datasets/sampling.*` contains pure row, grid and stress sampling;
- `datasets/parameter_dataset.*` assembles and publishes parameter datasets;
- `datasets/price_dataset.*` assembles and publishes price datasets;
- `datasets/sample_dataset.*` streams model-only sample JSON and adjacent YAML;
- `datasets/artifact_io.*` owns JSON/YAML serialization and validation metadata;
- `cuda/pricing_runner.cuh` owns ordinary CUDA allocations, transfers, events,
  timing and cleanup through RAII;
- `pricing/` contains reusable, product-specific pricing orchestration.
- `sampling/` contains Philox parameter generation and common CUDA sample
  recipe orchestration.

Catalog `generator.cpp` files are recipes: they choose inputs, launcher
arguments and publication metadata, but do not own ordinary CUDA resources.
All equity recipes, including American/LSM, are generated from the composed
manifests under `codegen/pricing_bindings/`. Their common orchestration is in
`pricing/equity_price_generation.cuh` and
`pricing/american_option_price_generation.cuh`. Recipe execution produces both
the price JSON and its timed catalog YAML.

`cuda/check_catalog_generators.py` enforces this rule without a raw-CUDA recipe
escape hatch. It also checks every available `DatasetSpec`, the
typed capability resolver and the exact sample-binding set. Fixed-income and
parameter recipe bodies remain bounded, declared families; their paths cannot
be extended outside the manifest.

The runtime tree under `src/` never depends on these offline components.

# Offline tooling boundaries

The offline dataset pipeline is split by responsibility:

- `datasets/sampling.*` contains pure row, grid and stress sampling;
- `datasets/parameter_dataset.*` assembles and publishes parameter datasets;
- `datasets/price_dataset.*` assembles and publishes price datasets;
- `datasets/artifact_io.*` owns JSON/YAML serialization and validation metadata;
- `cuda/pricing_runner.cuh` owns ordinary CUDA allocations, transfers, events,
  timing and cleanup through RAII;
- `pricing/` contains reusable, product-specific pricing orchestration.

Catalog `generator.cpp` files are recipes: they choose inputs, launcher
arguments and publication metadata, but do not own ordinary CUDA resources.
`cuda/check_catalog_generators.py` enforces this rule and keeps the small list
of reviewed algorithmic escape hatches explicit. A new escape hatch must be a
pipeline whose device workspace or execution graph cannot be expressed by the
ordinary runner; convenience alone is not sufficient.

The runtime tree under `src/` never depends on these offline components.

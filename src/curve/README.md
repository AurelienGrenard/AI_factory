# Curve implementations

`src/curve/<curve>` owns a compact parameter row, its host loader and reusable
term-structure analytics. A curve does not own stochastic-model parameters,
product payoffs or a concrete model-product launcher.

Each curve keeps the same boundary:

- `parameters.hpp` defines the transferable row;
- `dataset.hpp/.cpp` load and validate rows on the host;
- `term_structure.cuh` declares the public device analytics;
- `term_structure_impl.cuh` contains their included device definitions;
- the local README documents only parameter meaning, equations and numerical
  treatment.

Curve-fitted models consume this interface through a static provider and keep
the curve row separate from the stochastic-model row. Supported compositions
belong to the
[capability manifest](../../tools/codegen/pricing_bindings/capability_manifest.py).
Canonical analytics names and provider requirements belong to the
[analytics contract](../../docs/cuda/model-analytics-contract.md).

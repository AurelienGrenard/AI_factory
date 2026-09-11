# Rough equity models

This directory owns equity models defined by rough or Volterra dynamics. A
model remains here when it uses a finite-factor Markovian approximation: the
mathematical family, not the current simulator, determines placement.

The `rough` component is part of the canonical source, catalogue, and dataset
paths. Public C++ namespaces do not repeat it.

## Find an implementation

Within `src/model/equity/rough/<model>/`:

- `parameters.hpp` defines the compact model row;
- `dataset.hpp/.cpp` load model rows on the host;
- `dynamics.cuh` and `dynamics_impl.cuh` own model path transformations;
- a strategy-qualified preparation or pricing file owns Volterra FFT or
  N-factor details;
- `sample.cuh/.cu` compose model-only sampling when available;
- `product/` contains generated model-product launch units.

Shared Volterra kernels and engines live under `src/common/volterra`; product
parameters, schedules, and payoffs live under `src/product`. Model READMEs
document only equations and model-specific numerical choices.

## Authoritative references

- [Model reference index](../../../../docs/model-and-curve-reference-index.md)
  links every maintained mathematical note.
- [Product entry point](../../../product/README.md) explains product ownership.
- [Pricing-policy composition](../../../../docs/cuda/pricing-policy-composition.md)
  distinguishes Volterra FFT and Markovian N-factor execution.
- [Dynamics contract](../../../../docs/cuda/model-dynamics-contract.md) defines
  common state, time-grid, and Philox rules.
- [Capability manifest](../../../../tools/codegen/pricing_bindings/capability_manifest.py)
  is the source of truth for available models, engines, products, and bindings.

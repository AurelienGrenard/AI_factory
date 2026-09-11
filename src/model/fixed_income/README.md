# Fixed-income models

This directory owns short-rate model parameters, dynamics, analytics, model
sampling, and model-product composition. Shared financial-product contracts
remain under `src/product`; reusable rate primitives remain under
`src/common/fixed_income`.

## Find an implementation

Within `src/model/fixed_income/<model>/`:

- `parameters.hpp` defines the compact model row;
- `dataset.hpp/.cpp` load model rows on the host;
- `dynamics.cuh` and `dynamics_impl.cuh` own simulation when applicable;
- `analytics.cuh` and `analytics_impl.cuh` own model analytics;
- `sample.cuh/.cu` compose model-only sampling when available;
- `product/<product>.cuh/.cu` owns a standalone model-product launcher;
- `product/<curve>/<product>.cuh/.cu` owns a curve-fitted composition.

A model README documents only its equation, parameter mapping, and
model-specific analytical formulas. Common caplet, bond-option, swaption, and
early-exercise contracts are not copied between models.

## Authoritative references

- [Model reference index](../../../docs/model-and-curve-reference-index.md)
  links every maintained mathematical note.
- [Product entry point](../../product/README.md) explains product-owned rows,
  schedules, and pricing policies.
- [Dynamics contract](../../../docs/cuda/model-dynamics-contract.md) and
  [analytics contract](../../../docs/cuda/model-analytics-contract.md) define
  common model interfaces.
- [Shared fixed-income rate identities](../../common/fixed_income/fixed-income-rate-identities-reference.md)
  define model-independent forward and swap quantities.
- [Closed-form and Monte Carlo contract](../../../docs/cuda/closed-form-and-monte-carlo-pricing-contract.md)
  defines ordinary pricing composition.
- [American and Bermudan contract](../../../docs/cuda/american-and-bermudan-pricing-contract.md)
  defines early-exercise composition.
- [Capability manifest](../../../tools/codegen/pricing_bindings/capability_manifest.py)
  is the source of truth for model, curve, product, engine, and binding support.

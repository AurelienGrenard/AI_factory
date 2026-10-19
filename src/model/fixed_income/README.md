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

## Initial short rate and curve sensitivities

In CIR++, G2++ and Hull–White, the deterministic shift fits the input zero
curve. At inception their short rate is the initial instantaneous forward:

```text
flat:                    r(0) = curve.rate
Nelson–Siegel, Svensson: r(0) = curve.beta0 + curve.beta1
```

The `curve.rate` and `curve.beta0` gradients therefore measure a parallel
shift of the **whole initial curve**, with the fitted shift recomputed. They
are not sensitivities to the short rate at time zero while holding the rest
of the curve fixed. `curve.beta1` also changes `r(0)`, but changes the curve
shape as well; it cannot be interpreted as the same risk as `curve.beta0`.
Likewise, CIR++ `model.initial_state` changes the latent CIR factor and its
fitted shift, so it is not an independent `r(0)` bump.

The dataset gradients are derivatives per unit of the named curve parameter,
where rates are decimal annual rates. Multiply a parallel-rate derivative by
`0.0001` for the price change under a +1 bp parallel shift (subject to the
finite-difference approximation). An isolated `r(0)` sensitivity requires a
specified deformation of the entire initial curve; it is not an additional
coordinate of these fitted models.

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

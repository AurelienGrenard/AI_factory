# Financial product contracts

This directory owns model-independent product parameters, schedules, path
state, and payoff policies. It never owns model dynamics or a concrete
model-product launcher.

## File roles

Each `src/product/<product>/` uses the smallest applicable subset of this
layout:

- `parameters.hpp` — compact product row transferred to CUDA;
- `dataset.hpp/.cpp` — host loader and row validation;
- `pricing_policy.cuh` — product preparation, path observation, and payoff;
- `schedule.cuh` — contractual dates when a generic schedule is insufficient;
- `continuation_state.cuh` — regression state for early exercise when needed.

The product path and file names identify the financial contract. Details that
depend on a model or curve belong under
`src/model/<asset-class>/<family>/<model>/product/[<curve>/]`.

## Find a supported composition

The authoritative model-product matrix is
[`tools/codegen/pricing_bindings/capability_manifest.py`](../../tools/codegen/pricing_bindings/capability_manifest.py).
Generated composition units use the templates documented in the
[code-generation entry point](../../tools/codegen/pricing_bindings/README.md).

Use the relevant contract for implementation details:

- [closed-form and Monte Carlo pricing](../../docs/cuda/closed-form-and-monte-carlo-pricing-contract.md);
- [American and Bermudan pricing](../../docs/cuda/american-and-bermudan-pricing-contract.md);
- [complete policy composition](../../docs/cuda/pricing-policy-composition.md).

Product availability, model support, and generated target names are not copied
into this README because they are derived from the capability manifest.

For one product, `parameters.hpp` and `pricing_policy.cuh` are the executable
contract for its terms and payoff. Equation sheets in the separately
maintained website are a derived presentation and must remain consistent with
that source; the repository boundary is documented by the
[protected-download proposal](../../docs/proposed-protected-dataset-download-design.md).

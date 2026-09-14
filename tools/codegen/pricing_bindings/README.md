# Pricing and model-sample code generation

This directory generates thin model-product pricing bindings, model-only
sampling bindings, catalogue recipes, and their CMake registration. Numerical
algorithms, dynamics, schedules, and product payoffs remain hand-written in
`src`.

## Run and verify code generation

Regenerate every owned artifact in place:

```bash
python3 tools/codegen/pricing_bindings/generate.py \
  --family all \
  --output .
```

Check the repository without modifying it:

```bash
python3 tools/codegen/pricing_bindings/generate.py \
  --family all \
  --output /tmp/ai_factory-pricing-bindings \
  --compare-root .
```

CTest exposes the same zero-diff check as `pricing_binding_codegen`.
Regeneration preserves the timestamps of unchanged outputs, avoiding needless
CUDA rebuilds.

## Find the source of truth

| Responsibility | File or directory |
|---|---|
| Render artifacts and enforce bounded substitutions | `generate.py` |
| Define canonical model and sample specifications | `sample_manifest.py` |
| Define equity pricing products, variants, and bindings | `manifest.py` |
| Compose models, curves, products, engines, recipes, and CMake targets | `capability_manifest.py` |
| Store complete C++ and recipe templates | `templates/` |
| Record generated-file fingerprints | `PricingCapabilityManifest.json` |
| Test manifest composition | `test_capability_manifest.py` |

`capability_manifest.py` is the public inventory. It resolves each declared
`(model, curve, product, variant)` to one engine, binding, target, and recipe,
and rejects absent or ambiguous compositions.

## Template layout

Templates are grouped first by generated artifact and then by numerical
engine. New templates must follow this map:

```text
templates/
|-- pricing/
|   |-- markovian/
|   |   `-- fixed_income/{standalone,curve_fitted}/
|   |-- rough/markovian_n_factor/
|   |-- rough/volterra_fft/
|   |-- longstaff_schwartz/equity/
|   |-- longstaff_schwartz/fixed_income/terminal_forward/curve_fitted/
|   `-- closed_form/
|       |-- black_scholes/
|       `-- fixed_income/
|           |-- affine_one_factor/{cir,gaussian}/
|           |-- affine_two_factor/
|           |-- curve_fitted_one_factor/
|           `-- curve_fitted_two_factor/
|-- sampling/
|   |-- markovian/
|   |-- rough/markovian_n_factor/
|   |-- rough/volterra_fft/
|   `-- catalog/
`-- catalog/pricing/
    |-- price_delta/
    |-- markovian/
    |-- rough/markovian_n_factor/
    |-- rough/volterra_fft/
    |-- black_scholes_closed_form/
    |-- fixed_income/
    |   |-- longstaff_schwartz/terminal_forward/curve_fitted/
    |   `-- monte_carlo/{standalone,curve_fitted}/
    `-- american_longstaff_schwartz/
```

Every complete `.cuh`, `.cu`, and `generator.cpp` body lives in a template
file. Renderer code computes paths, substitutions, and bounded instantiation
fragments; it does not hide C++ in multiline Python strings.

## Generated outputs

`PRICE_DELTA_BINDING_SPECS` derives separate
`<product>_price_delta.cuh/.cu` launchers from existing pricing contracts.
Its strategies are explicit and validated; the generator does not infer
homogeneity from a model name. MC templates live beside their price-only
counterparts, and Black-Scholes closed-form templates stay in their engine
folder. Shared BS product formula policies live in generated `<product>_impl.cuh`
headers beside their launchers, so price and delta use the same formula body.
All nine American bindings use templates under
`pricing/longstaff_schwartz/equity/`, composing the common frozen-date policy.
The 12 Markovian models have price-delta recipes under `catalog/.../price_delta`,
with generated `generator.cpp` and planned `recipe.yaml`; execution alone writes
`dataset.yaml`. CRN aliases preserve price-only seeds, and production MC/LSM
uses 2^20 paths. The qualification remains bounded checks, not certified bias
or delta-specific tuning. See the
[implementation contract](../../../docs/cuda/equity-price-delta-contract.md).
Every price recipe has an aligned target and a distinct Cartesian target. Every
equity price-delta source has the same pair. Cartesian rows use
model-major/product-fastest order, or model-major/curve/product order for fitted
rates. Use `tools/datasets/generate_cartesian_datasets.py` to build, inspect,
execute or resume either family without constructing a command line by hand.

- Pricing bindings are written below each model's `product/` directory.
- Model-sample bindings are written as `<model>/sample.cuh` and `sample.cu`.
- Price and sample recipes are written below `catalog/model/**`.
- Model, curve, product, binding, and recipe registration is written to
  `cmake/generated/CapabilityManifest.cmake`.
- Output hashes are written to `PricingCapabilityManifest.json`.

Generated files compose existing policies and instantiate public launchers.
They must not contain hand-written numerical branches or product exceptions.
Sample parameter-law descriptions live beside their factories in
`sample_manifest.py`; validation rejects an undocumented derived scalar or
public parameter. FFT bindings also expose a host-only block-dimensions query
from the compiled sampling specialization, used for truthful recipe metadata
and grid-count replay without changing FFT tuning.
Price recipe execution writes both JSON and YAML because timing metadata is
known only at runtime.

Pricing recipes take their production path count (`2^20`) and launch settings
from `tools/cuda/tuning_profile.hpp`, through the shared host launch planner.
`pricing_launch_family()` derives each binding's family from its declared
engine and dynamics. The generated `pricing_launch_families` inventory is
consumed by `inspect_pricing_launch_plan`; it contains no tuning numbers.
See the [launch-plan contract](../../../docs/cuda/launch-validation-and-kernel-diagnostics.md#planifier-le-pricing-du-catalogue)
for inspection and native memory ownership. Regeneration changes source
recipes, not already-published datasets or their truthful historical YAML.

Fixed-income European swaptions select their engine per binding: existing
one-factor Jamshidian providers remain closed form; the two-factor standalone
and fitted families select the common terminal MC kernel. Each family has
explicit `.cuh/.cu` and recipe templates at the paths shown above. The shared
product policy, not these templates, owns the calendar and payoff.

CIR++ composes the one-factor fitted closed-form templates and the fitted
terminal-forward LSM templates. Its sample specification inherits the CIR
factor law; its two curve entries declare the same underlying forward dynamics.
No new product algorithm belongs in these bindings.

Philox reservations are append-only by extension epoch. New models declare
their epoch in the canonical sample specification; immutable prefix hashes
test that adding a model does not rekey any existing recipe.

## Minimum hand-written extension

For an ordinary model:

1. implement its parameter, dataset, dynamics, analytics, and optional
   preparation contracts under `src/model`;
2. add its canonical model specification and parameter law;
3. regenerate and run the zero-diff check.

For an ordinary product:

1. implement its parameter, dataset, schedule, and pricing policy under
   `src/product`;
2. add one product/variant specification;
3. regenerate and run the zero-diff check.

The generator then owns model-product bindings, catalogue recipes, CMake
registration, and provenance. Specialized analytical algorithms and
Longstaff--Schwartz implementations remain explicit templates or hand-written
owners when their contracts differ.

## Family selection

`--family all` is the normal complete operation. Narrow selectors exist for
development: `markovian`, rough families, `fixed_income`, `catalog`, and
sampling. Use `--help` for the current accepted values rather than copying the
list into another document.

Any generated diff must be explained by a manifest or template change. A
manual edit to an output is invalid because the next regeneration will replace
it.

# Pricing and model-sample code generation

The generator emits model-product pricing bindings, model-only sampling
bindings, and their catalog recipes from composed typed Python manifests.
Product payoff logic,
schedules, model dynamics and CUDA engines remain hand-written; generated
files compose those policies, instantiate the public launchers and describe
the offline dataset run. The specialized Black-Scholes and fixed-income
closed-form units are copied from generator-owned analytical templates without
altering their algorithms; early-exercise bindings remain explicitly
hand-written.

The complete generic binding matrix is generated in place with:

```bash
python3 tools/codegen/pricing_bindings/generate.py \
    --family all \
    --output .
```

Generate into a temporary directory and compare it with the checked-in tree:

```bash
python3 tools/codegen/pricing_bindings/generate.py \
    --family all \
    --output /tmp/ai_factory-pricing-bindings \
    --compare-root .
```

The same comparison is registered as the `pricing_binding_codegen` CTest so
CI detects any hand-edited binding or generated recipe that is no longer
reproducible. The same manifest emits
`cmake/generated/EquityPricingBindings.cmake`; model families, the 21-product
matrix, owned early-exercise units, sample units, fixed-income units and
mathDx-only Volterra units therefore cannot drift from the declared
capabilities.
`PricingCapabilityManifest.json` records schema version plus source and output
SHA-256 fingerprints for the complete generated tree.

Templates are organized first by artifact, then by numerical engine. This is
the navigation contract; new templates must not be added at the root:

```text
templates/
|-- pricing/
|   |-- markovian/product_binding.{cuh,cu}.tpl
|   |-- rough/markovian_n_factor/product_binding.{cuh,cu}.tpl
|   |-- rough/volterra_fft/product_binding.{cuh,cu}.tpl
|   `-- closed_form/
|       |-- black_scholes/<product>.{cuh,cu}.tpl
|       `-- fixed_income/
|           |-- affine_one_factor/{cir,gaussian}/<product>.{cuh,cu}.tpl
|           |-- affine_two_factor/<product>.{cuh,cu}.tpl
|           |-- curve_fitted_one_factor/<product>.{cuh,cu}.tpl
|           `-- curve_fitted_two_factor/<product>.{cuh,cu}.tpl
|-- sampling/
|   |-- markovian/model_binding.{cuh,cu}.tpl
|   |-- rough/markovian_n_factor/model_binding.{cuh,cu}.tpl
|   |-- rough/volterra_fft/model_binding.{cuh,cu}.tpl
|   `-- catalog/{recipe_support.cuh,generator.cpp}.tpl
`-- catalog/pricing/
    |-- markovian/generator.cpp.tpl
    |-- rough/markovian_n_factor/generator.cpp.tpl
    |-- rough/volterra_fft/generator.cpp.tpl
    |-- black_scholes_closed_form/generator.cpp.tpl
    |-- fixed_income/{affine_one_factor,affine_two_factor,...}/
    `-- american_longstaff_schwartz/generator.cpp.tpl
```

Complete generated C++ artifacts live in this tree, not as multiline source
strings in `generate.py`. Renderer code computes only bounded substitutions,
instantiation fragments and paths.

`sample_manifest.py` owns the 24 canonical model contracts: identity, source
family, transition, state, observables, analytics, supported architectures,
sample binding and parameter laws. Despite its historical filename, it is not
a sample-only identity table. `manifest.py` derives its pricing model views
from those entries and owns only the mechanical equity product/variant
cross-products and recipe profiles. `capability_manifest.py` composes them
into `EngineSpec`, `ModelSpec`,
`ProductSpec`, `CurveSpec`, `ProductBindingSpec` and `DatasetSpec` entries for
all repository domains. Its resolver
maps a declared `(model, curve, product, variant)` through one engine, one
binding, one CMake target and one recipe, and rejects an absent or ambiguous
composition. Import-time graph validation applies this uniqueness rule to
every published price recipe. The boundary checker compares all 697
available recipe paths, all 832 product-binding files and every sample
launcher to that matrix in both directions.

## Generated price recipes

The typed publication manifest crosses 18 equity models with 29 ordinary
product variants and adds 16 American variants. The fixed-income matrix adds
42 closed-form recipes, for 580 generated price `generator.cpp` files in all:

- Black-Scholes analytical closed form;
- Markovian fixed-step Monte Carlo;
- exact-transition Monte Carlo;
- seven-factor Markovian rough approximation;
- hybrid Volterra FFT with an explicitly planned device workspace;
- fixed-step Longstaff-Schwartz;
- exact-transition Longstaff-Schwartz;
- affine one- and two-factor fixed-income closed form, standalone or fitted to
  a Nelson-Siegel/Svensson curve.

The generated source only binds datasets, policies, numerical profiles and
publication metadata. `tools/pricing/equity_price_generation.cuh` owns the
common non-American execution and publication flow;
`tools/pricing/american_option_price_generation.cuh` owns the American
Longstaff-Schwartz flow; `tools/cuda/pricing_runner.cuh` owns CUDA buffers,
transfers, timing and cleanup. The price JSON and
`dataset.yaml` are written when the recipe executes, because the YAML contains
the measured wall and kernel times and therefore cannot be a static codegen
artifact.

Adding an ordinary equity model now requires its dynamics/dataset contract and
one canonical `SampleModelSpec`; the pricing and sample views are derived from
it. Adding a product requires its payoff/schedule/dataset contract and one
product/variant entry. No model/product cross-product source is then written
manually. Model-parameter and product-parameter generators
remain domain-owned sampling programs: their distributions are data-design
choices rather than mechanical pricing bindings. Their exact paths and
multiplicity are nevertheless declared by `DatasetSpec`, so an undeclared or
missing recipe fails the checker. The 16 Bermudan fixed-income recipes and 53
parameter/curve recipe bodies are the remaining hand-written families, with
explicit ownership in the capability manifest.

Sample publication is a distinct generated capability. `sample_manifest.py`
owns the 24 binding compositions, core parameter laws and observable schemas;
the codegen emits all 24 `sample.cuh`/`sample.cu` pairs, their shared recipe
helpers and two thin `generator.cpp` recipes per model. Generated recipes
support a 1,000-row `--smoke-test` and a non-publishing, full-shape
`--preflight` with deterministic replay under a second launch geometry.

The product manifest covers all 21 non-American equity products. It records
the terminal, dense, regular or two-date schedule family, whether the payoff
is sided, and its canonical path-product policy. The Markovian cross-product
generates fixed-step bindings for diffusion models and exact terminal,
regular, or calendar bindings for models with exact increments; dense
monitoring remains fixed-step. The rough cross-product generates:

- hybrid-FFT bindings for rough Bergomi, log-modulated rough Bergomi, rough
  SABR, and rough Stein--Stein;
- 2/3/7-factor prepared-Markovian bindings for rough Heston and quadratic
  rough Heston.

Black-Scholes path-dependent Monte Carlo bindings are generated as well. Its
eight non-American product-specific closed forms remain analytical
implementation units, stored as code-generation assets so the same command
still regenerates every non-American equity `.cuh`/`.cu`. The specialized
algorithms are preserved; only their source ownership moves to the generator.

Generated equity pricing bindings follow the source-family layout under
`src/model/equity/rough/<model>/product/` for rough models and
`src/model/equity/markovian/<model>/product/` for finite-state dynamics.
Model-only sampling stays at `<model>/sample.cuh/.cu`. The `product` folder
component is an ownership boundary and is deliberately absent from public
namespaces and stable CMake target names.

The 21 fixed-income closed-form binding pairs and 42 catalogue recipes are
template-owned by four explicit factorization branches: affine one factor,
affine two factors, curve-fitted one factor and curve-fitted two factors. CIR
and Gaussian one-factor implementations remain visibly separated where their
public APIs differ. Bermudan Longstaff--Schwartz bindings and recipes are a
separate, explicitly hand-written early-exercise family.

`--family catalog` emits the 538 equity and 42 fixed-income closed-form price
recipes. `--family fixed_income` emits the 21 closed-form binding pairs.
`--family markovian` and
the backward-compatible `--family prototype` select the same complete
Markovian binding matrix; `--family all` emits bindings, recipes, CMake and the
fingerprinted provenance manifest.

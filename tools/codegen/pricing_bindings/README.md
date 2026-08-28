# Pricing and model-sample code generation

The generator emits model-product pricing bindings, model-only sampling
bindings, and their catalog recipes from composed typed Python manifests.
Product payoff logic,
schedules, model dynamics and CUDA engines remain hand-written; generated
files compose those policies, instantiate the public launchers and describe
the offline dataset run. The specialized Black-Scholes closed-form units are
copied from generator-owned analytical templates without altering their
algorithms; American recipes bind the shared Longstaff-Schwartz host pipeline.

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
matrix, early-exercise and sample units, fixed-income units and mathDx-only
Volterra units therefore cannot drift from the declared capabilities.
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
|   `-- closed_form/black_scholes/<product>.{cuh,cu}.tpl
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
    `-- american_longstaff_schwartz/generator.cpp.tpl
```

Complete generated C++ artifacts live in this tree, not as multiline source
strings in `generate.py`. Renderer code computes only bounded substitutions,
instantiation fragments and paths.

`manifest.py` owns the mechanical equity cross-products and recipe profiles;
`sample_manifest.py` owns the 24 model-only sample bindings and parameter laws.
`capability_manifest.py` composes it into `EngineSpec`, `ModelSpec`,
`ProductSpec` and `DatasetSpec` entries for all repository domains. Its resolver
maps a declared `(model, curve, product, variant)` to one engine and rejects an
absent or ambiguous combination. The boundary checker compares all 689
available recipe paths and every sample launcher to that matrix.

## Generated price recipes

The typed publication manifest crosses 18 equity models with 29 ordinary
product variants and adds eight American variants. It generates 530 versioned
`generator.cpp` recipes for seven execution contracts:

- Black-Scholes analytical closed form;
- Markovian fixed-step Monte Carlo;
- exact-transition Monte Carlo;
- seven-factor Markovian rough approximation;
- hybrid Volterra FFT with an explicitly planned device workspace;
- fixed-step Longstaff-Schwartz;
- exact-transition Longstaff-Schwartz.

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
one `ModelRecipeSpec`; adding a product requires its payoff/schedule/dataset
contract and one product/variant entry. No model/product cross-product source
is then written manually. Model-parameter and product-parameter generators
remain domain-owned sampling programs: their distributions are data-design
choices rather than mechanical pricing bindings. Their exact paths and
multiplicity are nevertheless declared by `DatasetSpec`, so an undeclared or
missing recipe fails the checker. The 58 fixed-income and 53 parameter/curve
recipe bodies are two bounded families with explicit ownership and activation
criteria in the capability manifest.

Sample publication is a distinct generated capability. `sample_manifest.py`
owns the 24 binding compositions, core parameter laws and observable schemas;
the codegen emits all 24 `sample.cuh`/`sample.cu` pairs, their shared recipe
helpers and two thin `generator.cpp` recipes per model.

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

Fixed-income units are currently only inventoried by the typed capability
manifest and generated CMake fragment. Their closed-form `.cuh/.cu` bindings
and catalogue recipe bodies are not yet template-owned; main-audit finding
`STRUCT-015` tracks that explicit extension gap. Bermudan Longstaff--Schwartz
bindings are a separate early-exercise family and are not closed-form units.

`--family catalog` emits only the 530 equity price recipes. `--family markovian` and
the backward-compatible `--family prototype` select the same complete
Markovian binding matrix; `--family all` emits bindings, recipes, CMake and the
fingerprinted provenance manifest.

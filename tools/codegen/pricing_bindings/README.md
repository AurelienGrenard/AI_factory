# Pricing-binding code generation

The generator emits model-product binding files from an explicit Python
manifest. Product payoff logic, schedules and CUDA engines remain
hand-written; generic generated files compose those policies and instantiate
the public launchers. The eight specialized Black-Scholes closed-form units
are copied from generator-owned analytical templates without altering their
algorithms.

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
CI detects any hand-edited binding that is no longer reproducible. The same
manifest emits `cmake/generated/EquityPricingBindings.cmake`; model families,
the 21-product matrix, ordinary units and mathDx-only Volterra units therefore
cannot drift from the generated `.cu/.cuh` files.

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

Generated equity bindings follow the source-family layout:
`src/model/equity/rough/<model>` for this matrix and
`src/model/equity/markovian/<model>` for finite-state dynamics. The folder
component is not part of public namespaces or target names.

`--family markovian` and the backward-compatible `--family prototype` select
the same complete Markovian binding matrix.

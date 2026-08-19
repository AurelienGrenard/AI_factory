# AI Factory

AI Factory is a C++/CUDA library for quantitative pricing and price dataset
generation. The repository tracks source code, reproducible generation
recipes, and YAML catalog entries. Complete datasets are kept locally or
published to external storage.

## Project Structure

```text
AI_factory/
|-- src/          C++/CUDA simulation and pricing code
|-- tools/        parameter generation and dataset-writing utilities
|-- catalog/      one reproducible folder per published dataset
|-- datasets/     complete JSON datasets, ignored by Git
|-- docs/         implementation contracts and operational documentation
|-- tests/        dataset contracts and CUDA tests
|-- validation/   unified model/product validation and backend adapters
`-- CMakeLists.txt
```

All implementation contracts, workflows, derivations, and work-tracking notes
live in [`docs/`](docs/README.md). The main CUDA contracts are:

- [`cuda-closed-form-and-monte-carlo-pricing-contract.md`](docs/cuda-closed-form-and-monte-carlo-pricing-contract.md)
  for closed-form and Monte Carlo pricers;
- [`cuda-american-and-bermudan-pricing-contract.md`](docs/cuda-american-and-bermudan-pricing-contract.md)
  for American and Bermudan pricers;
- [`cuda-model-dynamics-contract.md`](docs/cuda-model-dynamics-contract.md) for
  reusable model-state simulation interfaces;
- [`cuda-launch-validation-and-kernel-diagnostics.md`](docs/cuda-launch-validation-and-kernel-diagnostics.md)
  for launch guards, kernel resource diagnostics, and their test coverage.

### `src`

`src` contains the numerical code and does not depend on catalog files:

- `src/common`: Philox, CUDA reductions, least squares, and CUDA checks;
- `src/curve/<curve>`: curve dataset loaders and CUDA term-structure analytics;
- `src/model/<asset_class>/<model>`: standalone model dynamics, analytics, loaders, and pricing kernels, split between `equity` and `fixed_income`;
- `src/product/<product>`: FP32 contract rows and JSON dataset loaders.

Each model, curve, or product uses `dataset.hpp/.cpp` for its compact row and
host loader. CUDA declarations and implementations retain descriptive names
such as `dynamics.cuh/.cu` or `term_structure.cuh/.cu`. Curve-specific dataset
construction helpers live under `tools/datasets`; catalog generators contain
only their recipe constants and `main`.

Pricing functions receive contiguous arrays that have already been loaded.
They do not know output paths, dataset URLs, or catalog formats.

Models retain matching type, function, and kernel layouts whenever their
mathematics and data dependencies permit it.

### CUDA kernel diagnostics

Every pricing launcher can report the resources and theoretical occupancy of
the exact kernel specialization and launch geometry it uses. Diagnostics are
disabled by default and do not alter dataset files. Enable them for any test or
generator with:

```bash
AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1 \
    ./build/test_heston_terminal_payoffs_cuda \
    2> build/heston-kernel-diagnostics.jsonl
```

Each JSON line identifies the kernel and variant, CUDA device, grid and block
geometry, registers, local and shared memory, active blocks and warps per SM,
and theoretical occupancy. Repeated batches with the same geometry are emitted
only once. A generator can be inspected in exactly the same way; it continues
to write its normal dataset while the diagnostic is written separately to
standard error.

### `tools`

`tools/datasets` provides the operations shared by all generators:

- uniform sampling and parameter grids;
- aligned and Cartesian product constructions;
- complete JSON dataset writing;
- YAML catalog writing.

It also contains host-only construction rules tied to a dataset family, such
as the constrained Nelson-Siegel and Svensson forward-level reconstructions.

Every dataset must declare an HTTP(S) URL. A generator fails explicitly when
the URL is missing or invalid.

### `catalog`

Each catalog entry is a self-contained, versioned dataset recipe:

```text
catalog/
|-- curve/<curve>/<dataset_id>/
|   |-- dataset.yaml
|   `-- generator.cpp
|-- model/<asset_class>/<model>/<dataset_id>/
|   |-- dataset.yaml
|   `-- generator.cpp
|-- product/equity/<product>/<dataset_id>/
|-- product/fixed_income/<product>/<dataset_id>/
|   |-- dataset.yaml
|   `-- generator.cpp
`-- price/<asset_class>/<model>/[<curve>/]<product>/<dataset_id>/
    |-- dataset.yaml
    `-- generator.cpp
```

`generator.cpp` is the executable recipe. Curve, model, and product generators
define parameter bounds and grids; price generators load complete input
datasets and run the CUDA pricer. The adjacent `dataset.yaml` records the
resulting metadata.

Every `dataset.yaml` exposes only two locations:

- `catalog`: repository directory containing `dataset.yaml` and `generator.cpp`;
- `url`: external download URL of the complete dataset.

Local JSON paths remain implementation details of `generator.cpp`.

The current URLs use the temporary `datasets.ai-factory.example` domain. They
must be replaced with the final data server URLs.

### `datasets`

`datasets/` follows the same model and product hierarchy:

```text
datasets/
|-- curve/nelson_siegel/nelson_siegel_01.json
|-- curve/svensson/svensson_01.json
|-- model/equity/heston/heston_01.json
|-- model/fixed_income/g2/g2_01.json
|-- model/fixed_income/g2_plus_plus/g2_plus_plus_01.json
|-- model/fixed_income/hull_white/hull_white_01.json
|-- model/fixed_income/ornstein_uhlenbeck/ornstein_uhlenbeck_01.json
|-- model/fixed_income/vasicek/vasicek_01.json
|-- model/fixed_income/cir/cir_01.json
|-- product/equity/european_options/european_options_01.json
|-- product/equity/american_options/american_options_01.json
|-- product/fixed_income/rate_options/rate_options_01.json
|-- price/equity/heston/<product>/<price_dataset_id>.json
|-- price/fixed_income/g2/<product>/<price_dataset_id>.json
|-- price/fixed_income/g2_plus_plus/<curve>/<product>/<price_dataset_id>.json
|-- price/fixed_income/ornstein_uhlenbeck/<product>/<price_dataset_id>.json
|-- price/fixed_income/vasicek/<product>/<price_dataset_id>.json
`-- price/fixed_income/hull_white/<curve>/<product>/<price_dataset_id>.json
```

This directory is ignored by Git. Its files can be generated locally or
downloaded from the `url` declared in the catalog.

## Curves And Short Rates

A curve dataset and a short-rate model dataset are deliberately independent.
`nelson_siegel_01` stores `beta0`, `beta1`, `beta2`, and the positive decay
time `tau`. The source implementation evaluates:

- the continuously compounded zero rate `z(0,T)`;
- the discount factor `P(0,T)`;
- the instantaneous forward `f(0,T)`;
- its analytical maturity derivative.

`svensson_01` exposes the same curve functions while adding `beta3` and a
second decay scale. Its generator reconstructs four forward-rate anchors and
rejects curves whose instantaneous forwards leave the configured range.

`ornstein_uhlenbeck_01` is a standalone short-rate model with
`r(t) = x(t)`. It samples the initial rate and mean reversion directly, then
reconstructs volatility from a bounded stationary standard deviation. This
avoids unstable low-reversion/high-volatility combinations.

`vasicek_01` extends the same Gaussian process with a long-term mean:

```text
dr(t) = a (b - r(t)) dt + sigma dW(t)
```

It samples `a`, `b`, and `r(0)`, then reconstructs `sigma` from a bounded
stationary standard deviation. Its dynamics, analytics, pricing launchers,
and tests deliberately mirror the OU layout; only the deterministic mean
increments and Vasicek bond levels differ.

`cir_01` uses the positive square-root short-rate process

```text
dr(t) = kappa (theta - r(t)) dt + sigma sqrt(r(t)) dW(t).
```

It samples `kappa`, `theta`, and `r(0)` first, then draws `sigma`
conditionally. The core rows cover Feller ratios from `1/6` to `10`; the
stress rows widen the range from `1/10` to `16`. The exact non-central
chi-square transition remains valid on both sides of the Feller threshold.

`hull_white_01` stores only the mean-reversion speed `a` and volatility
`sigma`. It uses the same stationary-dispersion reconstruction as OU, without
an initial state because the centered Hull-White state starts from zero. For
pricing, one Hull-White row is paired with one curve row. The implementation uses

```text
r(t) = x(t) + phi(t)
dx(t) = -a x(t) dt + sigma dW(t)
```

The reusable OU layer jointly simulates the Gaussian state and its time
integral. `src/model/fixed_income/hull_white/<curve>` composes that process with the
selected curve analytics and computes `phi(t)` from `f(0,t)`, so the full
model reproduces the supplied initial curve. Nelson-Siegel and Svensson are
curve providers, not parameters embedded in Hull-White.

`g2_01` is the standalone correlated two-factor Gaussian model:

```text
r(t) = x(t) + y(t)
dx(t) = -a x(t) dt + sigma dW_x(t)
dy(t) = -b y(t) dt + eta dW_y(t)
d<W_x,W_y>(t) = rho dt
```

Both initial factor states are stored because `r(0)` alone does not determine
future bond prices when the two mean-reversion speeds differ. The generator
orders `b` above `a` and samples stationary factor dispersions before
reconstructing `sigma` and `eta`, avoiding redundant factors and unstable
parameter combinations.

`g2_plus_plus_01` stores the same curve-independent process without initial
states. `src/model/fixed_income/g2_plus_plus/<curve>` adds a deterministic shift `phi(t)`
to the centered factors, exactly reproducing the supplied initial curve. Its
public analytical interface mirrors G2 just as the Hull-White interface
mirrors OU.

The fitted Hull-White and G2++ price datasets are independently checked first
with Premia's HW1D/HW2D closed forms. The validation runner supplies the exact
Nelson-Siegel or Svensson discounts at the contract dates through Premia's
external-curve interface; specialized QuantLib formulas provide row-local
fallback only when the Premia backend fails technically.

As with Heston, `dataset.hpp/.cpp` files contain compact rows and host JSON
loaders. Numerical functions used by kernels live in `.cuh/.cu` files.

Bates composes the Heston QE-M transition with an independent compound-Poisson
lognormal jump process. Each path owns one scalar uniform sequence and one
normal-pair cache. A jump normal is requested only when the Poisson count is
non-zero; an already cached normal is reused before another Box-Muller pair is
drawn. Unused values from the current Philox group remain cached for the next
step. The compensator
`lambda * (exp(nu + delta^2 / 2) - 1)` preserves the risk-neutral drift.
Terminal and scheduled-observation simulations keep all required Heston QE-M
steps but draw one exact compound-Poisson sum per observed interval. Products
that inspect every numerical step retain the pathwise one-jump-draw-per-step
transition.

Variance-Gamma and Normal-Inverse-Gaussian use exact Lévy increments. Terminal,
two-time, scheduled-observation, and exercise-grid simulations draw directly
over their requested intervals without an artificial daily `target_dt`.
Products that truly monitor a path still use exact increments on each monitored
step. VG samples its Gamma clock with Marsaglia-Tsang; NIG samples its
inverse-Gaussian clock with Michael-Schucany-Haas. Both use the same single
`UniformSequence` and `NormalPairCache` contract as Heston and Bates.

## Philox Random Mapping

Every Monte Carlo result row builds one key from
`base_seed + result_index`. Philox then addresses each four-value random group
with the complete 128-bit counter

```text
(path_index_low, path_index_high,
 local_group_index_low, local_group_index_high)
```

Every path starts at local group zero and advances its own local group as
needed. Paths therefore never reserve ranges based on a predicted number of
draws. Fixed-consumption simulations and algorithms with conditional draws or
rejection use the same mapping.

`UniformSequence` caches each group and exposes one continuous scalar stream.
Purely Gaussian simulations use a non-owning `NormalPairCache` to reuse the
second Box-Muller result without creating another random sequence. These
helpers are device-only, force-inlined, and deterministic for a fixed key,
path, algorithm, toolchain, and launch geometry.

## Dataset Artifacts

### Heston Model

The `heston_01` catalog entry begins as follows:

```yaml
title: "Heston parameter dataset heston_01"
database_id: "heston_01"
model_family: "Heston"
catalog: "catalog/model/equity/heston/heston_01"
url: "https://datasets.ai-factory.example/v1/model/equity/heston/heston_01.json"
row_count: 1000
```

The complete JSON dataset contains rows with stable identifiers:

```json
{
  "database_id": "heston_01",
  "row_count": 1000,
  "models": [
    {
      "id": "000001",
      "parameters": {
        "spot": 1.0,
        "risk_free_rate": 0.02982048,
        "dividend_yield": 0.04603526,
        "initial_variance": 0.07711430,
        "kappa": 1.59042335,
        "theta": 0.14153057,
        "rho": -0.46044695,
        "gamma": 0.76693642
      }
    }
  ]
}
```

### Product

The `european_options_01` dataset contains `strike` and `maturity`. For each
maturity `T`, its grid builds linearly spaced log-strikes over `[-aT, aT]`,
then applies `K = exp(x)`.

```yaml
catalog: "catalog/product/equity/european_options/european_options_01"
url: "https://datasets.ai-factory.example/v1/product/european_options/european_options_01.json"
row_count: 1000
construction:
  method: "maturity-dependent exponential grid"
```

### Price

A price row references its model, optional curve, and product identifiers
instead of duplicating their parameters:

```json
{
  "id": "000001",
  "model_id": "000001",
  "product_id": "000001",
  "seed": 900000001,
  "outputs": {
    "price": 0.04025437,
    "standard_error": 0.00013884
  }
}
```

The price catalog entry records the method, timing, and references to both
input datasets:

```yaml
database_id: "heston_01__european_calls_01__01"
catalog: "catalog/price/equity/heston/european_calls/heston_01__european_calls_01__01"
url: "https://mlp.lpma.math.upmc.fr/DataCarlo/Assets/Heston/EuropeanCall/heston_01__european_calls_01__01.json"
row_count: 1000
model_dataset:
  id: "heston_01"
  catalog: "catalog/model/equity/heston/heston_01"
  url: "https://datasets.ai-factory.example/v1/model/equity/heston/heston_01.json"
product_dataset:
  id: "european_options_01"
  catalog: "catalog/product/equity/european_options/european_options_01"
  url: "https://datasets.ai-factory.example/v1/product/european_options/european_options_01.json"
price_construction:
  method: "Aligned"
```

`Aligned` pairs rows with the same index and requires equal dataset sizes.
`CartesianProduct` generates every combination in model, optional curve,
then product order.

Call and put prices remain distinct price datasets, but share one product
parameter dataset whenever the row construction is identical. Their pricing
generators select `OptionSide::call` or `OptionSide::put`, which instantiates a
small compile-time payoff specialization; no side flag is stored per row or
branched on per simulated path. Gap options retain separate call-oriented and
put-oriented parameter datasets inside `gap_options/` because their payoff
strike grids differ.

Side-aware CUDA launchers expose this choice directly in their public API, for
example `launch_heston_european_option_cuda<OptionSide::call>(...)`. Their
`.cu` file explicitly instantiates the call and put versions, so ordinary C++
generators can link either specialization without including CUDA
implementations or keeping a runtime dispatch wrapper.

## Build

Requirements:

- a C++17 compiler;
- the CUDA Toolkit;
- `nlohmann-json3-dev`;
- CMake 3.18 or newer.

For an RTX 4090:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_WORKBENCH_ARCHITECTURES=89
cmake --build build -j
```

The architecture list remains configurable, for example:
`-DCUDA_WORKBENCH_ARCHITECTURES="70;86;89"`.

## Generate Datasets

Parameter datasets are quick to regenerate:

```bash
./build/generate_heston_01
./build/generate_bates_01
./build/generate_variance_gamma_01
./build/generate_normal_inverse_gaussian_01
./build/generate_g2_01
./build/generate_g2_plus_plus_01
./build/generate_nelson_siegel_01
./build/generate_svensson_01
./build/generate_hull_white_01
./build/generate_ornstein_uhlenbeck_01
./build/generate_vasicek_01
./build/generate_cir_01
./build/generate_european_options_01
./build/generate_american_options_01
./build/generate_gap_call_options_01
./build/generate_gap_put_options_01
./build/generate_rate_options_01
./build/generate_zero_coupon_bond_options_01
```

Each command replaces the local dataset and its YAML catalog entry together.
Every model and product generator follows the ordered 900-row core plus 100-row
stress policy documented in
[`docs/model-and-product-parameter-dataset-generation.md`](docs/model-and-product-parameter-dataset-generation.md).
Price datasets follow the same workflow:

```bash
./build/generate_heston_european_calls_01
./build/generate_heston_american_puts_01
./build/generate_bates_european_calls_01
./build/generate_bates_american_puts_01
./build/generate_variance_gamma_european_calls_01
./build/generate_normal_inverse_gaussian_european_calls_01
./build/generate_g2_caplets_01
./build/generate_g2_floorlets_01
./build/generate_g2_zero_coupon_bond_calls_01
./build/generate_g2_zero_coupon_bond_puts_01
./build/generate_g2_plus_plus_nelson_siegel_caplets_01
./build/generate_g2_plus_plus_nelson_siegel_floorlets_01
./build/generate_g2_plus_plus_nelson_siegel_zero_coupon_bond_calls_01
./build/generate_g2_plus_plus_nelson_siegel_zero_coupon_bond_puts_01
./build/generate_g2_plus_plus_svensson_caplets_01
./build/generate_g2_plus_plus_svensson_floorlets_01
./build/generate_g2_plus_plus_svensson_zero_coupon_bond_calls_01
./build/generate_g2_plus_plus_svensson_zero_coupon_bond_puts_01
./build/generate_ornstein_uhlenbeck_caplets_01
./build/generate_ornstein_uhlenbeck_floorlets_01
./build/generate_ornstein_uhlenbeck_zero_coupon_bond_calls_01
./build/generate_ornstein_uhlenbeck_zero_coupon_bond_puts_01
./build/generate_vasicek_caplets_01
./build/generate_vasicek_floorlets_01
./build/generate_vasicek_zero_coupon_bond_calls_01
./build/generate_vasicek_zero_coupon_bond_puts_01
./build/generate_cir_caplets_01
./build/generate_cir_floorlets_01
./build/generate_cir_zero_coupon_bond_calls_01
./build/generate_cir_zero_coupon_bond_puts_01
./build/generate_hull_white_nelson_siegel_caplets_01
./build/generate_hull_white_nelson_siegel_floorlets_01
./build/generate_hull_white_nelson_siegel_zero_coupon_bond_calls_01
./build/generate_hull_white_nelson_siegel_zero_coupon_bond_puts_01
./build/generate_hull_white_svensson_caplets_01
./build/generate_hull_white_svensson_floorlets_01
./build/generate_hull_white_svensson_zero_coupon_bond_calls_01
./build/generate_hull_white_svensson_zero_coupon_bond_puts_01
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

`dataset_catalog` validates two- and three-input constructions and mandatory
catalog fields. CUDA tests cover reusable OU, Vasicek, CIR, G2, Hull-White, and
G2++ analytics; caplets, floorlets, and zero-coupon options; deterministic
Gamma/non-central-chi-square tails; the uniform Heston,
Bates, VG, NIG, Merton, Kou, CEV, and Schöbel-Zhu dynamics and product
launchers, including path averages, forward starts, jumps, and barriers; and
the early-exercise pipelines. They use small in-memory fixtures and skip
automatically without a CUDA GPU.

Every fixed-income and Black-Scholes price dataset has an immutable 1,000-row
reference under `validation/datasets/price`. Routine CTest checks these caches,
their source fingerprints, row provenance, metrics, and compact catalogue YAML
without importing QuantLib or starting Premia/Wine. Vasicek, centered OU,
Hull-White, G2++, and 25 Black-Scholes product families use Premia references;
standalone G2 and CIR use specialized QuantLib references; four Black-Scholes
structured families use QuantLib Monte Carlo. CIR records that Premia is
callable but unreliable before selecting QuantLib. Direct backend checks remain
useful numerical diagnostics when the corresponding dependency is installed.
The shared validator reports row errors, combined Monte-Carlo uncertainty,
directional counts, and systematic bias. Premia continuous-monitoring prices
remain the primary analytical reference for discrete Black-Scholes barriers:
the JSON records the proven ordering and explains the expected signed bias.
Slower direct checks for equity models not yet migrated are available with:

```bash
cmake -S . -B build -DAI_FACTORY_QUANTLIB_EXOTIC_VALIDATION=ON
```

Every price validation follows specialized Premia, specialized QuantLib, then
QuantLib Monte Carlo. Migrated JSON records all three slots in that order,
details only methods that actually produced rows, and verifies both the 900-row
core and 100-row stress regimes. Its YAML deliberately contains only `status`,
`verified`, and the reference-dataset path. External engines run only during an
explicit `--generate`; normal validation is cache-only. Equity models other
than Black-Scholes retain their legacy validation only until they are migrated
to this same contract. See
[`validation/premia`](validation/premia/README.md),
[`validation/quantlib`](validation/quantlib/README.md), and the
[catalog extension workflow](docs/catalog-extension-and-validation-workflow.md)
for supported products and direct command-line usage.

## Add a Dataset

1. Identify whether the extension adds a model, curve, product family, pricing
   pair, or only a new price dataset; reuse every unaffected layer.
2. Add the compact loader and numerical implementation under `src`, following
   the closest CUDA contract and its public function order.
3. Add the reproducible `generator.cpp` and generated `dataset.yaml` under the
   matching `catalog/` hierarchy, then register their CMake target and tests.
4. For a price dataset, add the unified model-product validator and apply
   Premia, QuantLib specialized, then QuantLib Monte Carlo. Persistent
   references belong under `validation/datasets/price` and are regenerated
   explicitly; do not add validation JSON or notebooks beside their YAML.
5. Run the generator, loader checks, isolated validation, relevant CUDA tests,
   and the complete CTest suite before publication.
6. Update the separately maintained website project with the new public entry.

Call and put contracts share one product family and one templated pricer when
only the payoff orientation changes, while their price datasets remain
distinct. The complete, authoritative checklist is
[`docs/catalog-extension-and-validation-workflow.md`](docs/catalog-extension-and-validation-workflow.md);
the report and fallback contract is
[`docs/independent-price-validation-pipeline.md`](docs/independent-price-validation-pipeline.md).

Storage credentials must not appear in YAML files or the static website.
Private storage should use signed URLs or server-side authentication.

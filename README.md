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
|-- tests/        dataset contracts and CUDA tests
`-- CMakeLists.txt
```

### `src`

`src` contains the numerical code and does not depend on catalog files:

- `src/common`: Philox, CUDA reductions, least squares, and CUDA checks;
- `src/curve/<curve>`: curve dataset loaders and CUDA term-structure analytics;
- `src/model/<model>`: model dataset loaders, dynamics, and pricing kernels;
- `src/product/<product>`: FP32 contract rows and JSON dataset loaders.

Each model, curve, or product uses `dataset.hpp/.cpp` for its compact row and
host loader. CUDA declarations and implementations retain descriptive names
such as `dynamics.cuh/.cu` or `term_structure.cuh/.cu`.
Reusable sampling rules live in `generation.hpp/.cpp`; catalog generators
contain only their recipe constants and `main`.

Pricing functions receive contiguous arrays that have already been loaded.
They do not know output paths, dataset URLs, or catalog formats.

### `tools`

`tools/datasets` provides the operations shared by all generators:

- uniform sampling and parameter grids;
- aligned and Cartesian product constructions;
- complete JSON dataset writing;
- YAML catalog writing.

Every dataset must declare an HTTP(S) URL. A generator fails explicitly when
the URL is missing or invalid.

### `catalog`

Each catalog entry is a self-contained, versioned dataset recipe:

```text
catalog/
|-- curve/<curve>/<dataset_id>/
|   |-- dataset.yaml
|   `-- generator.cpp
|-- model/<model>/<dataset_id>/
|   |-- dataset.yaml
|   `-- generator.cpp
|-- product/<product>/<dataset_id>/
|   |-- dataset.yaml
|   `-- generator.cpp
`-- price/<model>/<product>/<dataset_id>/
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
|-- model/heston/heston_01.json
|-- model/hull_white/hull_white_01.json
|-- product/european_calls/european_calls_01.json
|-- product/american_puts/american_puts_01.json
`-- price/heston/<product>/<price_dataset_id>.json
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

`hull_white_01` stores only the mean-reversion speed `a` and volatility
`sigma`. For pricing, one Hull-White row is paired with one curve row. The
implementation uses

```text
r(t) = x(t) + phi(t)
dx(t) = -a x(t) dt + sigma dW(t)
```

The CUDA dynamics jointly simulate the Gaussian factor and its time integral,
which gives each path an exact accumulated discount factor. `phi(t)` is
computed from `f(0,t)` so that the model reproduces the supplied initial
curve. Nelson-Siegel is therefore one curve provider, not a parameter embedded
in Hull-White.

As with Heston, `dataset.hpp/.cpp` files contain compact rows and host JSON
loaders. Numerical functions used by kernels live in `.cuh/.cu` files.

## Dataset Artifacts

### Heston Model

The `heston_01` catalog entry begins as follows:

```yaml
title: "Heston parameter dataset heston_01"
database_id: "heston_01"
model_family: "Heston"
catalog: "catalog/model/heston/heston_01"
url: "https://datasets.ai-factory.example/v1/model/heston/heston_01.json"
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

The `european_calls_01` dataset contains `strike` and `maturity`. For each
maturity `T`, its grid builds linearly spaced log-strikes over `[-aT, aT]`,
then applies `K = exp(x)`.

```yaml
catalog: "catalog/product/european_calls/european_calls_01"
url: "https://datasets.ai-factory.example/v1/product/european_calls/european_calls_01.json"
row_count: 1000
construction:
  method: "maturity-dependent exponential grid"
```

### Price

A price row references the model and product identifiers instead of
duplicating their parameters:

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
catalog: "catalog/price/heston/european_calls/heston_01__european_calls_01__01"
url: "https://mlp.lpma.math.upmc.fr/DataCarlo/Assets/Heston/EuropeanCall/heston_01__european_calls_01__01.json"
row_count: 1000
model_dataset:
  id: "heston_01"
  catalog: "catalog/model/heston/heston_01"
  url: "https://datasets.ai-factory.example/v1/model/heston/heston_01.json"
product_dataset:
  id: "european_calls_01"
  catalog: "catalog/product/european_calls/european_calls_01"
  url: "https://datasets.ai-factory.example/v1/product/european_calls/european_calls_01.json"
price_construction:
  rule: "aligned row pairing"
```

`Aligned` pairs rows with the same index and requires equal dataset sizes.
`CartesianProduct` generates every pair in model-major order.

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
./build/generate_nelson_siegel_01
./build/generate_hull_white_01
./build/generate_european_calls_01
./build/generate_american_puts_01
```

Each command replaces the local dataset and its YAML catalog entry together.
Price datasets follow the same workflow:

```bash
./build/generate_heston_european_calls_01
./build/generate_heston_american_puts_01
```

The Cartesian product containing one million prices is intentionally separate:

```bash
./build/generate_heston_european_calls_02
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

`dataset_catalog` validates construction rules and mandatory catalog fields.
The CUDA tests use small in-memory fixtures and are skipped automatically when
no CUDA GPU is available.

## Add a Dataset

1. Add or reuse the required structures and kernels under `src`.
2. Create its catalog folder under `catalog/model`, `product`, or `price`.
3. Add `generator.cpp` and its adjacent `dataset.yaml`.
4. Declare local output paths inside `generator.cpp`.
5. Declare an external HTTP(S) URL.
6. Add the CMake target.
7. Run the generator and validate the dataset and catalog artifacts.
8. Add a self-contained test that does not require a published dataset.

Storage credentials must not appear in YAML files or the static website.
Private storage should use signed URLs or server-side authentication.

# AI Factory

AI Factory is a C++/CUDA library for quantitative pricing and price dataset
generation. The repository tracks source code, reproducible generation
recipes, YAML catalog entries, and small JSON previews. Complete datasets are
kept locally or published to external storage.

## Project Structure

```text
AI_factory/
|-- src/          C++/CUDA simulation and pricing code
|-- tools/        parameter generation and dataset-writing utilities
|-- catalog/      one reproducible folder per published dataset
|-- previews/     versioned JSON previews limited to 100 rows
|-- datasets/     complete JSON datasets, ignored by Git
|-- tests/        dataset contracts and CUDA tests
`-- CMakeLists.txt
```

### `src`

`src` contains the numerical code and does not depend on catalog files:

- `src/common`: Philox, CUDA reductions, least squares, and CUDA checks;
- `src/heston`: parameters, QE-M dynamics, and product-specific kernels;
- `src/products`: FP32 product structures and JSON loaders.

Pricing functions receive contiguous arrays that have already been loaded.
They do not know output paths, dataset URLs, or catalog formats.

### `tools`

`tools/datasets` provides the operations shared by all generators:

- uniform sampling and parameter grids;
- aligned and Cartesian product constructions;
- complete JSON dataset writing;
- automatic generation of valid JSON previews;
- YAML catalog writing.

Every dataset must declare an HTTP(S) URL. A generator fails explicitly when
the URL is missing or invalid.

### `catalog`

Each catalog entry is a self-contained, versioned dataset recipe:

```text
catalog/
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

`generator.cpp` is the executable recipe. Model and product generators define
parameter bounds and grids; price generators load complete input datasets and
run the CUDA pricer. The adjacent `dataset.yaml` records the resulting
metadata.

Every `dataset.yaml` exposes only two locations:

- `catalog`: repository directory containing `dataset.yaml` and `generator.cpp`;
- `url`: external download URL of the complete dataset.

Local JSON and preview paths remain implementation details of `generator.cpp`.

The current URLs use the temporary `datasets.ai-factory.example` domain. They
must be replaced with the final data server URLs.

### `previews`

Previews preserve the complete dataset schema while limiting the payload to
100 rows. They add the following metadata:

```json
{
  "source_row_count": 1000,
  "row_count": 100,
  "preview_of": "https://datasets.ai-factory.example/v1/...",
  "results": []
}
```

C++ loaders and tests use these files. A Git clone therefore remains buildable
and testable without downloading complete datasets.

### `datasets`

`datasets/` follows the same model and product hierarchy:

```text
datasets/
|-- model/heston/heston_01.json
|-- product/european_calls/european_calls_01.json
|-- product/american_puts/american_puts_01.json
`-- price/heston/<product>/<price_dataset_id>.json
```

This directory is ignored by Git. Its files can be generated locally or
downloaded from the `url` declared in the catalog.

## Dataset Artifacts

### Heston Model

The `heston_01` catalog entry begins as follows:

```yaml
title: "Heston parameter dataset heston_01"
database_id: "heston_01"
model_family: "Heston"
catalog: "catalog/model/heston/heston_01"
url: "https://datasets.ai-factory.example/v1/model/heston/heston_01.json"
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
construction:
  row_count: 1000
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

The price catalog entry records the construction, numerical method, CUDA
configuration, sources, timings, and references to both input datasets:

```yaml
database_id: "heston_01__european_calls_01__01"
catalog: "catalog/price/heston/european_calls/heston_01__european_calls_01__01"
url: "https://datasets.ai-factory.example/v1/price/heston/european_calls/heston_01__european_calls_01__01.json"
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
./build/generate_european_calls_01
./build/generate_american_puts_01
```

Each command replaces the local dataset, its preview, and its YAML catalog
entry together. Price datasets follow the same workflow:

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

`dataset_catalog` validates schemas, previews, and mandatory catalog fields.
`heston_american_put_cuda` runs a small reproducible pricing test and is
skipped automatically when no CUDA GPU is available.

## Add a Dataset

1. Add or reuse the required structures and kernels under `src`.
2. Create its catalog folder under `catalog/model`, `product`, or `price`.
3. Add `generator.cpp` and its adjacent `dataset.yaml`.
4. Declare local output paths inside `generator.cpp`.
5. Declare an external HTTP(S) URL.
6. Add the CMake target.
7. Run the generator and validate all three artifacts.
8. Add a test based on the preview, never on the ignored complete dataset.

Storage credentials must not appear in YAML files, previews, or the static
website. Private storage should use signed URLs or server-side authentication.

# Dataset catalogue

`catalog/` contains the published dataset recipes. Each selected leaf has a
`generator.cpp` and a `recipe.yaml`; leaves that have actually been executed
also have a `generation.yaml` receipt. `catalog/manifest.json` lists the
selected outputs and marks whether each dataset/receipt pair is ready. Generated JSON datasets are written under `datasets/` by the
generators and are distributed separately from the source tree.

```text
catalog/{curve,model,product}/.../generator.cpp
catalog/{curve,model,product}/.../recipe.yaml
datasets/{curve,model,product}/...json
```

Use the CMake target named in the capability manifest to compile a generator.
`tools/run_generator.py` runs one target in an isolated directory;
`tools/datasets/generate_catalog.py` plans and resumes price and sample
campaigns. The schema definitions are in `tools/datasets/schemas/`. Pricing
bindings and repeated recipes are maintained by
`tools/codegen/pricing_bindings/`.

The published generator aggregates select only the recipes in `catalog/`.
Local experiments may use `work/catalog/` when that directory exists, but it
is not needed by a delivered source checkout. Published recipe and receipt
contents retain their original paths and hashes. The 14 fixed-income bond
barrier leaves contain recipes and generators; their current aligned datasets
and `generation.yaml` receipts are pending execution against the refreshed
parameter inputs.

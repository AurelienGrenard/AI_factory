# Production catalog layout

The repository publishes selected recipes directly under `catalog/`:

```text
catalog/{curve,model,product}/...
catalog/manifest.json
datasets/prod/{curve,model,product}/...
```

`catalog/` contains one published sample and one model parameter dataset per
model when available, plus one aligned price dataset per model and product
variant. Fixed-income prices have one selected dataset per available curve and
product variant. The product and curve parameter inputs needed by selected
prices are included. The manifest lists 696 catalog leaves: 24 samples, 25
model parameter datasets, 618 aligned prices, and 29 price inputs. The sample
for `quadratic_rough_heston` has not been generated. Price gradients are not
published yet.

`datasets/` is ignored by Git. To move the production data to a server, copy
`catalog/` and `datasets/prod/`, keeping their internal paths.
`catalog/manifest.json` records the selected dataset paths.

The recipe and generation receipt files retain their historical logical
`catalog/...` and `datasets/...` paths. `tools/datasets/catalog_layout.py`
resolves these to the physical selected catalog and dataset paths. The JSON
files in `datasets/prod/` were checked against the SHA-256 values in their
completed generation receipts. Their historical recipe hashes differ from the
current recipes for all 696 production leaves. This verifies stored bytes;
it does not establish that current recipes reproduce them or certify prices.

After changing the production selection, update `catalog/manifest.json`.

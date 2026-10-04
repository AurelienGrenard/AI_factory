# Production catalog layout

The published recipes follow the same taxonomy in two branches:

```text
catalog/prod/{curve,model,product}/...
catalog/other/{curve,model,product}/...
datasets/prod/{curve,model,product}/...
datasets/other/{curve,model,product}/...
```

`catalog/prod` contains one published sample and one model parameter dataset
per model when available, plus one aligned price dataset per model and product
variant. Curve-specific prices have one leaf per curve and product variant.
The product and curve parameter inputs needed by selected prices also live in
`prod`. There are 696 catalog leaves and 696 JSONs in `prod`: 24 samples, 25
model parameter datasets, 618 aligned prices, and 29 price inputs. The sample
for `quadratic_rough_heston` has not been generated. No price-gradient dataset
is published; its recipes remain in `other`.

`catalog/other` holds every remaining recipe, including unpublished price and
gradient recipes and the unselected second samples. It is local and ignored by
Git. `catalog/prod` and the website are tracked. All of `datasets/` remains
ignored by Git, as before. To move the production data to a server, copy both
`catalog/prod` and `datasets/prod`, preserving their relative `curve`, `model`
and `product` paths. `datasets/prod` currently contains about 24.5 GB of JSON.
The server must host the JSONs at the path configured in the website.

The original recipe and generation receipt files were moved without changing
their bytes. Their embedded `catalog/...` and `datasets/...` paths are
historical logical paths. `tools/datasets/catalog_layout.py` resolves those
paths into the physical `prod` or `other` branch for the generation tools.
The website uses physical paths from `catalog/prod/manifest.json`.

The JSON files in `prod` were checked against the SHA-256 stored in their
completed generation receipts. The receipts have valid record checksums.
Their historical recipe hashes differ from the current recipes for all 696
production leaves. This verifies the stored JSON bytes; it does not establish
that the current recipes reproduce them or independently certify prices.

After changing the production selection, update `catalog/prod/manifest.json`
and regenerate the website index:

```sh
python3 AI_factory_website/build_dataset_index.py
```

# Production catalog layout

The repository publishes selected recipes directly under `catalog/`:

```text
catalog/{curve,model,product}/...
catalog/manifest.json
datasets/{curve,model,product}/...
work/catalog/{curve,model,product}/...
work/datasets/{curve,model,product}/...
```

`catalog/` contains one published sample and one model parameter dataset per
model when available, plus one aligned price dataset per model and product
variant. Fixed-income prices have one selected dataset per available curve and
product variant. The product and curve parameter inputs needed by selected
prices are included. The manifest lists 710 catalog leaves: 24 samples, 25
model parameter datasets, 631 aligned prices, and 30 price inputs.
The 14 bond barrier leaves contain generators and recipes but are marked
`ready: false` until current datasets and generation receipts are produced. The sample
for `quadratic_rough_heston` has not been generated. Price gradients are not
published yet.

`datasets/` is ignored by Git. To move the production data to a server, copy
`catalog/` and `datasets/`, keeping their internal paths.
`catalog/manifest.json` records the selected dataset paths. Historical release and qualification evidence is retained locally under
`maintainer/evidence/` and is not part of the GitHub deliverable.

The recipe and generation receipt files retain their historical logical
`catalog/...` and `datasets/...` paths. `tools/datasets/catalog_layout.py`
resolves these to the physical selected catalog and dataset paths. The JSON
files in `datasets/` were checked against the SHA-256 values in their
completed generation receipts. The 52 regenerated model and product parameter recipes now match their
receipts by SHA-256. The other 644 completed production recipes retain historical
receipt hashes that differ from current recipes. The 14 pending barrier
recipes have no current receipt. Artifact hashes verify stored
bytes; they do not independently certify price quality.

After changing the production selection, update `catalog/manifest.json`.

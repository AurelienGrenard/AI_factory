# Dataset catalog recipes

The catalog is divided into `prod` and `other`. Both use the same taxonomy:

```text
catalog/<branch>/curve/<curve>/<dataset-id>/
catalog/<branch>/model/<asset-class>/<family>/<model>/{parameters,samples,prices,price_gradients}/...
catalog/<branch>/product/<product>/<dataset-id>/
```

Fixed-income models omit the equity family level. A price path adds its product
and, when required, its curve before the dataset ID. Every leaf contains a
`generator.cpp` and a `recipe.yaml`. Published leaves also contain a
`generation.yaml` receipt. The matching JSON file lives under the same
relative path in `datasets/<branch>/`.

`prod` contains the selected published recipes. `other` contains every
remaining recipe and is ignored by Git; it stays local. All generated JSONs
are ignored by Git. See [the production layout guide](../docs/deployment-catalog.md)
for the selection and server copy instructions.

The recipes and receipts retain their historical logical paths and hashes.
`tools/datasets/catalog_layout.py` resolves those paths into their physical
branches. Do not rewrite historical receipts to disguise a move.

Generated bindings and repeated price/sample recipes are owned by the
[typed capability manifest and code generator](../tools/codegen/pricing_bindings/README.md).
Their schemas live under [`tools/datasets/schemas`](../tools/datasets/schemas/).

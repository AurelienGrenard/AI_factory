# Dataset catalogue recipes

`catalog` contains reproducible dataset recipes and adjacent publication
metadata. It contains no generated JSON dataset and does not own runtime
pricing or simulation algorithms.

The hierarchy mirrors the canonical identities under `src`:

```text
catalog/
|-- curve/<curve>/<dataset-id>/
|-- model/<asset-class>/<family>/<model>/{parameters,samples,prices}/...
`-- product/<product>/<dataset-id>/
```

Fixed-income models omit the equity family level. A price path adds its product
and, when required by the pricing contract, its curve before the dataset ID.
The matching generated artifact follows the same hierarchy under the ignored
`datasets/` directory.

Each generated leaf owns three files with deliberately separate roles:

- `generator.cpp`: executable implementation;
- `recipe.yaml`: canonical, minimal description of what the dataset means and
  how it must be built;
- `generation.yaml`: receipt for one completed materialization, including
  hashes, execution evidence and timing. It is written last and therefore acts
  as the publication-complete marker.

Price leaves may also own `validation.yaml`. Validation never belongs in the
recipe or generation receipt. A generator and its recipe always exist together;
`generation.yaml` exists only for a materialized published artifact.

Published dataset IDs are immutable. If data or recipe semantics change, use a
new dataset ID. Publication requires a clean Git revision, stages the JSON
first, and publishes `generation.yaml` last.

Generated bindings and repeated price/sample recipes are owned by the
[typed capability manifest and code generator](../tools/codegen/pricing_bindings/README.md).
Do not edit generated recipes or runtime-produced receipts by hand. Their
schemas live under [`tools/datasets/schemas`](../tools/datasets/schemas/).

Use the [catalogue extension workflow](../docs/catalog-extension-and-validation-workflow.md)
for the complete addition and publication sequence, and the specialized
[parameter](../docs/model-and-product-parameter-dataset-generation.md) or
[sample](../docs/model-sample-dataset-generation.md) contract for row rules.

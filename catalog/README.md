# Dataset catalogue recipes

`catalog` contains reproducible dataset recipes and adjacent publication
metadata. It contains no generated JSON dataset and does not own runtime
pricing or simulation algorithms.

The hierarchy mirrors the canonical identities under `src`:

```text
catalog/
|-- curve/<curve>/<dataset-id>/
|-- model/<asset-class>/<family>/<model>/{parameters,samples,prices,price_gradients}/...
`-- product/<product>/<dataset-id>/
```

Fixed-income models omit the equity family level. A price path adds its product
and, when required by the pricing contract, its curve before the dataset ID.
The matching generated artifact follows the same hierarchy under the ignored
`datasets/` directory.

The public derivative catalogue deliberately exposes one canonical contract:
`price_gradients/**/<dataset-id>_price_gradient_diagonal_hessian`. It contains
the price, the selected gradient, and the selected diagonal Hessian. Gradient-
only, spot-only, and full-Hessian variants remain code-generation capabilities
for experiments; they do not multiply public catalogue recipes.

Each recipe leaf always owns two source files:

- `generator.cpp`: minimal executable entry point;
- `recipe.yaml`: canonical description of the dataset identity and contents.

After a completed materialization, the leaf also owns `generation.yaml`. This
receipt contains hashes, execution evidence and timing; it is written last and
therefore acts as the publication-complete marker. Independent certification is
deferred and no `validation.yaml` belongs in the catalogue for now.

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

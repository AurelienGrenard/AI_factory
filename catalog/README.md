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

Each leaf owns `generator.cpp`, the executable source of truth for generation.
After a successful publication it also owns `dataset.yaml`, the compact record
of the executed recipe and published artifact. A newly generated or
unpublished recipe may therefore have no YAML yet; an existing YAML must never
describe a different execution.

Generated bindings and repeated price/sample recipes are owned by the
[typed capability manifest and code generator](../tools/codegen/pricing_bindings/README.md).
Do not edit generated recipes or runtime-produced YAML by hand.

Use the [catalogue extension workflow](../docs/catalog-extension-and-validation-workflow.md)
for the complete addition and publication sequence, and the specialized
[parameter](../docs/model-and-product-parameter-dataset-generation.md) or
[sample](../docs/model-sample-dataset-generation.md) contract for row rules.

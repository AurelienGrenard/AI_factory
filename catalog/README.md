# Dataset catalogue

`catalog/` contains the public generation recipes and their native `generator.cpp`
sources. A leaf has a `generation.yaml` only after its generator has actually
run. The generated JSON lives in `datasets/`, which is distributed separately
from this source package.

The selected inventory currently has 1,438 recipes: 46 model parameter
sets, 29 product parameter sets, three curves, 50 sample sets, 655 prices
and 655 price-gradient sets. The catalogue manifest names the selected
sources; the presence of a recipe does not imply a computed or qualified
dataset. Correlation matrices and multi-asset pricing prototypes live only
under local `work/catalog/` and `work/datasets/`. Their public folders remain empty
until a production recipe is selected and qualified.

The fixed-income selection has 117 price recipes and 104 price-gradient
recipes; the latter include 26 Bermudan swaptions with
`frozen_regression_policy`. CIR standalone European swaptions
publish first derivatives only. The bond up-and-out product has 13 public price generators and its public
product parameter generator, but no price-gradient binding yet.

`catalog/manifest.json` is the selected inventory and records whether a JSON
and `generation.yaml` pair exists. The price-gradient recipes are source only; no gradient result is claimed.
The 13 bond barrier price recipes also await execution against current
parameters. Use `tools/datasets/generate_catalog.py --kind all --catalog-only
--construction aligned` to inspect or execute the selected pricing recipes. Without `--publish`, results remain in the
chosen `work/generation/` run directory for review.

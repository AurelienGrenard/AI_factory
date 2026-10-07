# Dataset catalogue

`catalog/` contains the public generation recipes and their native `generator.cpp`
sources. A leaf has a `generation.yaml` only after its generator has actually
run. The generated JSON lives in `datasets/`, which is distributed separately
from this source package.

The current public selection has 1,195 recipes: 25 model parameter sets, 28
product parameter sets, three curve input sets, 655 aligned prices, and 484
aligned price-gradient sets. The curves are required by the curve-fitted fixed-income gradients.
Cartesian recipes and model-sample generators belong to the local
`work/catalog/` branch. Code generation still knows their typed
capabilities; they are not needed to run the public aligned-gradient campaign.

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

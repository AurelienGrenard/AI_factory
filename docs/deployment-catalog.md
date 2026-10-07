# Production catalog layout

The deliverable publishes `catalog/` recipes, their `generator.cpp` sources,
and the CMake/codegen machinery needed to compile them. JSON outputs in
`datasets/` are generated separately and are not tracked by Git.

The current [manifest](../catalog/manifest.json) selects 1,195 leaves: 25
model parameter generators, 28 product parameter generators, three curve
generators, 655 aligned price generators, and 484 aligned price-gradient
generators. Cartesian variants and model-sample generators remain local under
`work/catalog/` when present. The public fixed-income set has 117 prices and
104 price-gradient recipes; its 26 Bermudan gradient recipes use frozen
regression policy. CIR standalone European swaptions expose first derivatives only.
The bond up-and-out product has a public parameter generator and 13 public
price generators. It has no gradient binding pending a derivative contract.

`catalog/manifest.json` marks a leaf ready only when its JSON and generation
receipt exist. The parameter inputs and curves already executed retain their
receipts. The aligned price-gradient leaves and 13 bond barrier price leaves are source
recipes without current results. Older price JSONs remain local historical
outputs until they are regenerated against the refreshed parameters.

The [generation workflow](dataset-generation-workflow.md) stages new datasets
under `work/generation/`. A staging run leaves the canonical `datasets/` and
`catalog/` receipts unchanged. Historical audit evidence remains local under
`maintainer/evidence/`.

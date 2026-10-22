# Dataset metadata

`recipe.yaml` exists before a run. It names the generator, the output JSON and,
for pricing, the exact input dataset paths and numerical plan. Parameter
recipes keep their sampling bounds; product recipes keep the parameter units,
calendar and payoff. The generation receipt is the sibling `generation.yaml`,
so the recipe does not name it.

The JSON holds data and stable identifiers: dataset ID, families, optional
option side, row count, input dataset IDs, and rows. It contains no URL or
catalogue path. The recipe resolves input IDs to paths. Moving the catalogue
therefore does not change the JSON checksum.

Published `generation.yaml` version 4 records input and output paths with
SHA-256, plus known CPU, GPU and software versions. Price receipts also give
`wall_seconds` and a readable duration. Parameter and sample receipts have no
timing. The receipt does not fingerprint the recipe, generator, executable or
build. If a published JSON is edited coherently, update its output SHA-256;
if a published input JSON is normalized or replaced, update the dependent
input SHA-256 only after checking that the intended numerical input rows are
unchanged. See `tools/datasets/refresh_generation.py`.

Version 1–3 receipts remain readable. `campaign.schema.yaml` governs the
separate frozen checkpoint contract in `work/generation/`.

The publication checks require coherent IDs, row counts, finite prices and
standard errors, and the sensitivity columns named by the recipe. They do not
require an exact set or order of presentation fields in the JSON. The tests
exercise these data checks without depending on hard-coded catalogue counts
or on datasets from a separate generation campaign.

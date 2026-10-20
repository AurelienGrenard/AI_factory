# Dataset metadata schemas

`recipe.yaml` describes the current dataset contract. Published
`generation.yaml` version 4 records the path and historical SHA-256 of each
input dataset, the path and current SHA-256 of the output JSON, and only
hardware and software details actually recorded. It contains no checksum for
the recipe, generator, executable or build. Older versions 1, 2 and 3 remain
readable.

An edited JSON needs its output SHA-256 updated in the receipt. Historical
input SHA-256 values must remain unchanged when current parameter datasets are
replaced. Use `tools/datasets/refresh_generation.py` after a coherent edit.

`campaign.schema.yaml` governs the separate frozen checkpoint contract in
`work/generation/`, where stricter hashes protect an unfinished run.

Validate metadata with `python3 tools/datasets/metadata_schemas.py`.
Check a dataset triple with `tools/datasets/check_dataset_compatibility.py`.

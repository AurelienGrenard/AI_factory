# Dataset metadata schemas

`recipe.yaml` describes the current dataset contract. `generation.yaml`
records technical facts and SHA-256 checksums of the current JSON and its
model, product and curve inputs. Version 2 of the generation schema is compact
and editable. Historical version-1 receipts remain readable.

A publication edit is valid when the recipe describes the JSON and the
version-2 receipt contains the current file hashes. The receipt does not lock
the bytes of the recipe, generator or build. Optional `curation` metadata
identifies a previously generated JSON and gradients removed from it.

`campaign.schema.yaml` governs a separate, strict checkpoint contract under
`work/generation/`. Its frozen code, recipe and input hashes are retained to
protect resumes while a calculation is running. A campaign is not catalogue
metadata and is never copied into `datasets/`.

Check catalogue metadata with:

```bash
python3 tools/datasets/metadata_schemas.py
```

Check one published triple with `tools/datasets/check_dataset_compatibility.py`.
Update a curated or legacy generation receipt with
`tools/datasets/refresh_generation.py`.

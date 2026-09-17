# Dataset metadata schemas

These schemas define the three independent documents owned by every catalogue
leaf:

- `recipe.yaml` describes the dataset that should exist;
- `generation.yaml` records one completed materialization and is written last;
- `validation.yaml` records independent certification without changing either
  the recipe or its generation history.

The files use JSON Schema syntax encoded as YAML. They live under `tools`
because they are machine contracts used by generators, publication checks and
CI; the catalogue contains only concrete dataset instances.

`campaign.schema.yaml` separately describes resumable controller state below
`work/generation/`. A campaign is not catalogue metadata and is never copied
into `datasets/`.

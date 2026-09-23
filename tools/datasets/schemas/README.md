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

Every recipe declares the dataset's canonical public `url` before generation.
The URL is part of the frozen recipe and the generated artifact must repeat it
exactly. The current `datasets.ai-factory.example` host remains a placeholder;
its canonical construction is owned by `DatasetSpec.url` in the typed manifest.

`campaign.schema.yaml` separately describes resumable controller state below
`work/generation/`. A campaign is not catalogue metadata and is never copied
into `datasets/`.

Run the shared validator on every tracked catalogue document with:

```bash
python3 tools/datasets/metadata_schemas.py
```

The generation controller also applies these contracts when it reads a
recipe, finalizes a generation receipt, saves or resumes a campaign, and
publishes a dataset pair. CTest registers the mutation and catalogue coverage
suite as `metadata_schemas`. Local ignored campaign and experiment manifests
can be checked explicitly with `--campaign` and `--experiment`.

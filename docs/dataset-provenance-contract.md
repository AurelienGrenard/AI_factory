# Dataset recipe and generation metadata

A published dataset consists of `recipe.yaml`, a JSON dataset and
`generation.yaml`. The recipe describes the **current** JSON: identity, inputs,
row count, numerical method, selected sensitivities and bump rules. The JSON
contains the data. `generation.yaml` contains technical facts about the
materialization and the SHA-256 of the **current** JSON and input files.

## Editable publication contract

Version 2 of `generation.yaml` keeps:

- the recipe path;
- the output path, row count and SHA-256;
- each model, product and curve input path and SHA-256, when applicable;
- the GPU or CPU actually observed, driver when available, CMake, C++ and CUDA
  compiler versions, and optional execution timings;
- an optional `curation` entry with the source JSON SHA-256 and coordinates
  removed from an already calculated gradient dataset.

It contains no checksum of itself and no recipe, generator, binary, source
archive or build-graph checksum. An edit to the recipe or `generation.yaml`
does not require rerunning the generator. An edit to the JSON requires updating
its SHA-256 in `generation.yaml`; an edit to an input file requires updating its
recorded SHA-256. The recipe must continue to describe the JSON fields and
selected gradients. The checker validates those current relationships without
requiring a compiled executable:

```bash
python3 tools/datasets/check_dataset_compatibility.py \
  --recipe catalog/.../recipe.yaml \
  --generation catalog/.../generation.yaml \
  --dataset datasets/.../dataset.json
```

`compatible` means that the currently supplied metadata and files agree. It
is **not** proof that the current generator produced the historical bytes,
that an edited seed was used, or that an independent financial reference has
certified the prices. The SHA-256 values detect accidental content mismatch;
they are not signatures of authenticity.

For a legacy version-1 receipt, the checker ignores its old self-checksum and
historical recipe/binary hashes. Convert it to the compact format after
curating the JSON and recipe with:

```bash
python3 tools/datasets/refresh_generation.py \
  --recipe catalog/.../recipe.yaml \
  --dataset datasets/.../dataset.json \
  --generation catalog/.../generation.yaml \
  --removed-gradient model.risk_free_rate \
  --removed-gradient model.dividend_yield
```

The refresh tool preserves recorded hardware and timing, updates current file
hashes and verifies the current recipe/JSON contract. `--removed-gradient` is
optional; other coherent JSON edits can be refreshed without a special flag.
Missing historical software versions remain `unavailable`; today's installed
tools are never presented as those used for the old run. A curation entry
records a declared gradient removal when supplied; retain the original JSON
separately if exact verification of the transformation is needed. Never claim
that a previous receipt was produced by a new execution.

The generator and recipe in the catalogue are the instructions for recreating
the **current numerical dataset**. Exact byte equality is not guaranteed
across GPU architectures, compiler versions, or an altered bump scenario
layout; compare a targeted replay using a justified numerical tolerance and
Monte Carlo standard errors when those details change.

## Resumable campaigns

The work-in-progress campaign controller still freezes its own executable,
inputs, recipe and checkpoint contract. Those checks protect restart of an
unfinished Monte Carlo job and are separate from the editable publication
contract. Do not replace files inside a running campaign. The controller
validates native outputs against its frozen plan before writing the compact
publication receipt. Historical version-1 receipts remain schema-valid; they
are not rewritten en masse.

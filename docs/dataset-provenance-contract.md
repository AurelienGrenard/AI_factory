# Dataset recipe and generation metadata

A published dataset has a `recipe.yaml`, a JSON file and, after an actual run,
a `generation.yaml`. The recipe names the generator and datasets, defines the
output path and row count, and describes the numerical method, sensitivities and
bump rules. The JSON contains the data.

Version 4 of `generation.yaml` contains:

- `inputs`: a `path` and historical `sha256` for each dataset consumed,
  keyed by its role in the recipe (`model`, `product`, `curve`,
  `correlation`, etc.). Parameter generators have `inputs: {}`.
- `output`: the published JSON `path` and its current `sha256`.
- `hardware` and `software`: facts recorded in the execution receipt or
  recovered from a build configuration whose SHA-256 exactly matches the
  historical receipt. Toolchain paths establish a major or toolkit version,
  but do not establish an unrecorded compiler patch version. Sections or
  fields without evidence are omitted.

There is no `artifact` section and no recipe, generator, executable, source or
build checksum in the published receipt. Paths appear in both the recipe
and the receipt, while the row count stays in the recipe. Current machine
versions must never be substituted for missing historical versions.

The input SHA values describe the files consumed **at generation time**. If a
parameter dataset is later replaced, preserve those historical SHA values. The
compatibility checker reports that the current input differs. A curated JSON
needs an updated `output.sha256`; changing an input or output path in the
recipe also requires updating its path in the receipt. Neither edit claims that a new generation took place.

Check a published triple with:

```bash
python3 tools/datasets/check_dataset_compatibility.py \
  --recipe catalog/.../recipe.yaml \
  --generation catalog/.../generation.yaml \
  --dataset datasets/.../dataset.json
```

Refresh a receipt after a coherent edit to the recipe and JSON with:

```bash
python3 tools/datasets/refresh_generation.py \
  --recipe catalog/.../recipe.yaml \
  --dataset datasets/.../dataset.json \
  --generation catalog/.../generation.yaml
```

The optional `--removed-gradient` argument checks that a removed coordinate
is absent from the current recipe. It does not add curation metadata to the
technical receipt. Keep any curation history separately if it is needed.

`compatible` confirms the current recipe, JSON and receipt agree. It does not
prove that the current code reproduces the historical bytes or certify the
numerical prices. The SHA values detect changes; they are not signatures.

An unfinished campaign in `work/generation/` has a stricter frozen checkpoint
contract so that it can resume safely. Those internal hashes are separate from
the published `generation.yaml`. The native generator may write an intermediate
version-1 receipt; the campaign controller converts it to version 4 at
publication. Versions 1, 2 and 3 remain readable for historical datasets.

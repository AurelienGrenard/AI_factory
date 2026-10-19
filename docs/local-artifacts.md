# Local builds, campaigns and evidence

The project keeps generated work separate from source and public recipes:

| Location | Purpose | Git |
|---|---|---|
| `builds/<name>/` | Rebuildable CMake and CUDA objects for one toolchain and GPU architecture. | Ignored |
| `work/generation/<campaign>/` | Active or resumable dataset generation, with frozen inputs, logs and outputs. | Ignored |
| `work/experiments/` | Temporary simulation, performance and bump studies. | Ignored |
| `work/catalog/` | Local experimental recipes outside the public catalogue. | Ignored |
| `datasets/` | Generated or downloaded JSON datasets. | Ignored |
| `maintainer/` | Reusable developer utilities and reference probes. | Ignored |
| `project_journal/` | Dated audit decisions, performance reports and concise evidence. | Ignored |

The old `project_journal/evidence/` directory was retired on **2026-10-13**. Its dated
reports and selected quantitative proof files were copied to
`project_journal/`; the [evidence manifest](../project_journal/evidence/manifest-2026-10-13.json)
records original paths, retained paths, sizes and SHA-256 hashes. Builds,
checkpoints, raw traces and copies of historical datasets were discarded.
`learning/` at the repository root was left unchanged.

A `generation.yaml` is a record of a real generator run. Keep it with its
JSON output and matching `recipe.yaml`; an unexecuted recipe has no receipt.
Publication into the public catalogue and dataset tree is explicit.

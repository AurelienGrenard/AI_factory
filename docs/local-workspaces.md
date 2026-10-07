# Local generation campaigns and experiments

`work/` is the single ignored workspace for disposable activity. Deleting it
must never remove reusable source code or an intentionally published dataset.

```text
work/
├── catalog/{curve,model,product}/
├── datasets/{curve,model,product}/
├── generation/<campaign-id>/
└── experiments/<asset-class>/<model>/<study-id>/
```

## Generation campaigns

`work/generation/` is owned by the catalogue campaign controller. One campaign
contains its `campaign.json`, frozen generator and recipe inputs, compiled
binary evidence, attempts, checkpoints, logs and publication journal. It is a
resumable execution record, not a dataset.

A completed campaign may be deleted after its result has either been rejected
or published. Publication copies only the immutable JSON artifact and its
`generation.yaml` receipt to their declared destinations. Nothing below
`work/generation/` is published implicitly.

## Experiments

`work/experiments/` contains broader, intentionally disposable studies:
one-off dataset preparation, training comparisons, ablations, notebooks and
performance investigations. Keep one study self-contained:

```text
work/experiments/<asset-class>/<model>/<study-id>/
├── README.md
├── manifest.yaml
├── dataset_generation/   # optional
├── training/             # optional
└── analysis/             # optional
```

`manifest.yaml` records the study ID, purpose, inputs, commands and output
locations. A nested generation action may create or refer to a campaign under
`work/generation/`; it does not make the entire experiment a generation run.
Performance evidence remains under `artifacts/performance/`, and independent
price validation remains under `validation/`.

The main CMake graph never discovers local experiments implicitly. A study
that needs native targets may declare them in `work/experiments/CMakeLists.txt`
and opt in during configuration with
`-DAI_FACTORY_ENABLE_LOCAL_EXPERIMENTS=ON`. These targets are deliberately
excluded from every permanent catalogue aggregate, so deleting `work/` leaves
the versioned build unchanged.

## Maintainer evidence

Completed audit snapshots and historical catalogue archives, qualifications,
and releases live under ignored `maintainer/evidence/`. Their receipts and
manifests keep the original logical paths and hashes. The path resolver in
`tools/datasets/catalog_layout.py` maps those logical paths to the current
physical files. `maintainer/tools/`, `maintainer/tests/`, and
`maintainer/cmake/` contain development checks, references, benchmarks and
profilers; they are absent from the delivered build unless
`AI_FACTORY_ENABLE_MAINTAINER=ON`.

## Publication boundary

`datasets/` contains datasets only. Experiment inputs, campaign manifests,
logs, notebooks, checkpoints and reports belong under `work/` or `artifacts/`.
To retain an experimental result, give it a stable dataset ID and canonical
catalogue `recipe.yaml`, then publish it through the normal clean-revision
workflow. Otherwise it remains disposable.

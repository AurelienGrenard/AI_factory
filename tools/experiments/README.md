# Experiment tooling

This directory contains contracts and reusable tooling for disposable studies
under `work/experiments/`. It must not contain a concrete model choice,
campaign, notebook or result.

Every study starts with a small `manifest.yaml` validated by
[`schemas/experiment.schema.yaml`](schemas/experiment.schema.yaml). Dataset
generation inside a study still uses the canonical dataset recipe and campaign
contracts under `tools/datasets/`.

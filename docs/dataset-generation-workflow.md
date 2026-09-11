# Catalogue generation workflow

Use [generate_catalog.py](../tools/datasets/generate_catalog.py) to prepare and
run a sequential price/sample campaign. It calls the native generators; it does
not replace their pricing engines, choose new CUDA settings, or certify prices.

## Prepare and inspect

Existing model, curve and product JSON inputs must be present. Their recipes
remain authoritative; generation freezes these inputs without redrawing or
reordering the 900 core and 100 stress rows. Build the generators explicitly:

```bash
cmake --build build-dev --target parameter_generators price_generators \
  sample_generators ai_factory_tests inspect_pricing_launch_plan -j1
```

This builds executables; it does not generate datasets. A complete FFT catalogue
requires a configured mathDx build. The controller refuses missing or stale
executables rather than silently omitting a model.

Inspect a selection without CUDA execution or publication:

```bash
python3 tools/datasets/generate_catalog.py --kind all --model cir
```

`--model` and `--target` can be repeated. Omitting filters selects every available
recipe of `--kind prices`, `price_delta`, `samples`, or `all`. The inventory comes from the
typed capability manifest. Input paths and sample shapes come from the recipes;
the compiled launch inspector supplies proposed pricing settings. No second
Python table of CUDA geometries is maintained.

MC/LSM prices use the compiled production count, currently `2^20` paths per price.
Closed-form prices have no paths. Samples retain their independent shapes from
the [sample contract](model-sample-dataset-generation.md). This controller has
no path-count or shape override.

For equity sensitivities, build the separate `price_delta_generators` target
and select `--kind price_delta`. Building `price_generators` alone still builds
only price-only recipes. Generated `recipe.yaml` describes planned settings;
`dataset.yaml` is produced only by execution.
The recipes reuse the price-only launch profile as a candidate, not as a
delta-specific performance qualification. See the [price-delta contract](cuda/equity-price-delta-contract.md)
for CRN seed sharing, paired errors, frozen LSM dates and bias limitations.

## Run a staged pilot

Use a new run directory on the repository filesystem:

```bash
python3 tools/datasets/generate_catalog.py \
  --target generate_cir_european_payer_swaptions_01 \
  --run-dir build-dev/generation-pilot-01 --execute
```

Without `--publish`, outputs stay under the campaign's `jobs/` directory and
existing catalogue/data files remain untouched. This runs the real recipe at
its full shape, not a reduced benchmark. Sample `--smoke-test` and `--preflight`
remain separate native verification modes; consult the sample contract.

The controller requires Python 3, PyYAML and Ninja. It freezes parameter inputs,
recipe sources and executables with SHA-256 checks. Each job retains stdout,
stderr, its full process time, artifact-check time and separate publication time.
The build configuration and launch inspector are retained, with before/after GPU
observations (informative only). Version-2 campaigns also retain the dirty
implementation-source archive and attach generation provenance to staged YAML;
SDKs and external dependencies are not archived. This is not a hermetic build
or a qualified timing campaign. GPU and native runner
times remain in the generated JSON/YAML; they are not confused with full
process or publication time.

## Publish and resume

After reviewing the pilot and resolving relevant numerical failures, start the
chosen production selection in another directory with `--execute --publish`.
An unfiltered campaign can be very long; inspect its recipe list first.

```bash
python3 tools/datasets/generate_catalog.py \
  --kind all --run-dir build-dev/generation-01 --execute --publish
```

Jobs execute one at a time. The controller checks a conservative per-job disk
margin, not a guarantee of final JSON size. There is no application thermal
threshold or cooling loop. Native memory guards and numerical errors remain
blocking. Coordinate other GPU users separately; the lock only serializes
campaign controllers in this repository, even with different build directories.
Direct generator invocations and external editors do not acquire this lock;
do not run them against the same destinations during publication.

Before publication, price rows and metadata are checked; sample JSON is checked
as a stream without loading three million records at once. Generated prices must
be `pending / verified: false`, with their intended independent-reference path.
This is structural/numerical output checking, not Premia/QuantLib validation.
Never set `verified: true` manually to substitute for certification.

Publication preserves the previous JSON/YAML under the job's `backup/` directory.
Each rename is atomic, **not the pair**: a journal completes an interrupted pair
on resume. Do not consume a publishing job until its state is `complete`.
An external edit to a destination or a changed frozen input/output blocks resume
instead of being overwritten or silently accepted.
Changing a frozen controller/provenance module also blocks resume; keep the frozen controller
version until the campaign is complete.

```bash
python3 tools/datasets/generate_catalog.py \
  --run-dir build-dev/generation-01 --execute --resume
```

Resume uses the frozen selection and publication policy. Completed jobs are
hash-checked and skipped. A staged job resumes publication without running the
GPU again. An interrupted/failed generator receives a new attempt directory and
restarts that dataset from the beginning; there is no within-dataset checkpoint
or automatic retry. Existing attempts and backups are retained.

Changed pricing code or parameters require a new campaign, not a resume with
different inputs. Independent certification is performed later through the
[price-validation pipeline](independent-price-validation-pipeline.md).

Keeping an already completed dataset is a separate decision from resuming a
campaign. Follow the [dataset provenance and reuse contract](dataset-provenance-contract.md)
for its read-only checker, legacy datasets and the evidence needed after a
refactor. Codegen regeneration never updates old data or its generation history.

# Catalogue generation workflow

Use [generate_catalog.py](../tools/datasets/generate_catalog.py) to prepare and
run a sequential price/sample campaign. It calls the native generators; it does
not replace their pricing engines, choose new CUDA settings, or certify prices.

## Prepare and inspect

Existing model, curve and product JSON inputs must be present. Their recipes
remain authoritative; generation freezes these inputs without redrawing or
reordering the 900 core and 100 stress rows. Build the generators explicitly:

```bash
cmake --build build --target parameter_generators price_generators \
  sample_generators ai_factory_tests inspect_pricing_launch_plan -j1
```

This builds executables; it does not generate datasets. A complete FFT catalogue
requires a configured mathDx build. The controller refuses missing or stale
executables rather than silently omitting a model.

Inspect a selection without CUDA execution or publication:

```bash
python3 tools/datasets/generate_catalog.py --kind all --model cir
```

`--model` and `--target` can be repeated. Omitting filters selects every
available recipe of `--kind prices`, `price_delta`, `samples`, or `all`. The
inventory comes from the typed capability manifest. Input paths and sample
shapes come from the recipes. The compiled launch inspector supplies proposed
pricing settings. No second Python table of CUDA geometries is maintained.

Extend this controller when a campaign needs another manifest filter. Do not
add a Python controller for one campaign.

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
The controller accepts explicitly declared Cartesian price and price-delta
recipes. It computes their output row count as the product of the frozen input
counts, including model/curve/product products for curve-fitted rates recipes.

Compile and inspect every Cartesian price and price-delta recipe for one model:

```bash
python3 tools/datasets/generate_catalog.py \
  --kind all --construction cartesian --model heston --compile
```

Add `--run-dir <directory> --execute --publish` to run and publish this
selection. Use `--kind prices` or `--kind price_delta` to select one output
family. Omitting `--model` selects every Cartesian recipe. This can take a long
time. Resume with `--run-dir <directory> --execute --resume`.

For all aligned fixed-income prices, inspect the exact selection without
building or running anything:

```bash
python3 tools/datasets/generate_catalog.py \
  --asset-class fixed_income --construction aligned --kind prices
```

The selection comes from the typed capability manifest: currently 80 recipes,
each with 1,000 aligned price rows. The price-delta family is equity-only.
After any other campaign has finished or been stopped, build and run these
targets sequentially with publication:

```bash
python3 tools/datasets/generate_catalog.py \
  --asset-class fixed_income --construction aligned --kind prices \
  --compile \
  --run-dir datasets/generation-runs/fixed-income-aligned-01 \
  --execute --publish
```

The controller builds only the selected generators and the launch inspector in
`build/`. It then freezes the inputs and binaries. It runs one job at a time.
It checks each dataset and publishes it to its declared
`datasets/model/fixed_income/` and `catalog/model/fixed_income/` paths. A
later job's failure does not roll back earlier published jobs. Resume the
frozen campaign through `generate_catalog.py --run-dir <same directory>
--execute --resume`. A checkpoint-capable terminal Monte Carlo job continues
at its first unfinished native batch; an unsupported early-exercise job
restarts from its first price.

For all aligned equity prices in the Markovian family, inspect the selection:

```bash
python3 tools/datasets/generate_catalog.py \
  --asset-class equity --model-family markovian \
  --construction aligned --kind prices
```

This selects 364 recipes (12 models, 364,000 aligned price rows). The family
filter follows each recipe's source path, so rough-equity recipes are excluded.
Once the selection is confirmed, build and run the campaign with publication:

```bash
python3 tools/datasets/generate_catalog.py \
  --asset-class equity --model-family markovian \
  --construction aligned --kind prices \
  --compile \
  --run-dir datasets/generation-runs/equity-markovian-aligned-01 \
  --execute --publish
```

The controller builds these generators in `build/`. It publishes each completed
dataset under `datasets/model/equity/markovian/` and
its catalogue YAML under `catalog/model/equity/markovian/`. To follow the run,
use `python3 tools/datasets/watch_generation_progress.py
datasets/generation-runs/equity-markovian-aligned-01` in another terminal.
If interrupted, resume the frozen campaign with:

```bash
python3 tools/datasets/generate_catalog.py \
  --run-dir datasets/generation-runs/equity-markovian-aligned-01 \
  --execute --resume
```

## Generate model-terminal samples

The controller also owns terminal sample campaigns. `--skip-published` excludes
a recipe when its JSON and YAML already exist. A half-published pair is an
error. This only checks file presence. It does not certify old data. Inspect the
selection first:

```bash
python3 tools/datasets/generate_catalog.py --kind samples --skip-published
```

Every recipe writes three million rows. `samples_01` uses 12,000 parameter sets
and 250 paths per set. `samples_02` uses three million parameter sets and one
path per set. Every row includes the sampled maturity `T`. Large JSON files need
space for the published file and the staged copy. The controller checks free
space before each job.

After any other campaign has stopped, compile and publish the missing recipes
with one command from the repository root:

```bash
python3 tools/datasets/generate_catalog.py --kind samples --skip-published \
  --compile --compile-jobs 2 \
  --run-dir datasets/generation-runs/model-samples-01 \
  --execute --publish
```

The command builds only the selected targets. It uses two parallel compile jobs.
The campaign then runs one GPU generator at a time. It checks each
three-million-row JSON as a stream. It publishes it to its declared
`datasets/model/.../samples/` path with adjacent catalogue YAML and provenance.
`--asset-class equity`, `--model-family rough`, repeatable `--model`, and
repeatable `--target` narrow a new campaign. Omit `--skip-published` to select
existing pairs for regeneration.

In another terminal, follow the campaign with:

```bash
python3 tools/datasets/watch_generation_progress.py \
  datasets/generation-runs/model-samples-01
```

The display counts completed datasets. During a sample generator's preparation
and CUDA simulation, no within-dataset percentage is available. Once JSON
writing starts, the native writer reports samples written and an ETA for that
**writing phase** at most every ten seconds. These host-side checks add no CUDA
synchronization; they do not estimate the remaining time of the whole campaign.
The controller retains attempt journals and stdout/stderr logs.

Stop with `Ctrl+C` in the campaign terminal. Completed datasets stay published;
an interrupted sample dataset restarts from its first row on explicit resume:

```bash
python3 tools/datasets/generate_catalog.py \
  --run-dir datasets/generation-runs/model-samples-01 \
  --execute --resume
```

Resume uses the frozen binaries and selection, and checks completed output
hashes. Use a new run directory if source or recipes change. A generator's
`--smoke-test` writes only 1,000 rows under `/tmp`. Its `--preflight` executes
the full shape without publication and checks a second launch geometry. Both
require a working CUDA device. Neither mode creates a production dataset.

## Run a staged pilot

Use a new run directory on the repository filesystem:

```bash
python3 tools/datasets/generate_catalog.py \
  --target generate_cir_european_payer_swaptions_01 \
  --run-dir datasets/generation-runs/generation-pilot-01 --execute
```

Without `--publish`, outputs stay under the campaign's `jobs/` directory and
existing catalogue/data files remain untouched. This runs the real recipe at
its full shape, not a reduced benchmark. Sample `--smoke-test` and `--preflight`
remain separate native verification modes; consult the sample contract.

The controller requires Python 3, PyYAML and Ninja. It freezes parameter inputs,
recipe sources and executables with SHA-256 checks. Each job retains stdout,
stderr, its full process time, artifact-check time and separate publication time.
The build configuration and launch inspector are retained, with before/after GPU
observations (informative only). Version-3 campaigns also retain the dirty
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
  --kind all --run-dir datasets/generation-runs/generation-01 --execute --publish
```

Jobs execute one at a time. The controller checks a conservative per-job disk
margin, not a guarantee of final JSON size. There is no application thermal
threshold or cooling loop. Native memory guards and numerical errors remain
blocking. Coordinate other GPU users separately; the lock only serializes
campaign controllers in this repository, even with different build directories.
Direct generator invocations and external editors do not acquire this lock;
do not run them against the same destinations during publication.

While a native generator is running, it updates
`jobs/<target>/progress.json` in the campaign directory. The sidecar reports
completed and total prices, elapsed time, average prices per second and an ETA.
To follow a campaign in a terminal without a notebook, run from the repository
root:

```bash
python3 tools/datasets/watch_generation_progress.py \
  datasets/generation-runs/<campaign-directory>
```

The display refreshes every ten seconds and formats durations in hours,
minutes and seconds. It shows completed jobs out of the whole campaign and
the active price job's count, percentage, speed and ETA. The ETA covers the
active price dataset only: job counts are not a reliable measure of remaining
campaign time when pricing methods differ. It selects the active job
automatically; use `--target` for a specific job or `--once` for a single
display. `Ctrl+C` stops only the display, not the generator. This read-only
command can attach to a campaign that is already running and does not affect
CUDA execution.
CUDA events are queried by a separate CPU thread; progress reporting adds no
device or stream synchronization. It writes an initial point, then at most one
periodic point every ten seconds, and a final point on normal completion. The
same progress records are appended to
`jobs/<target>/attempt-NNN/progress.jsonl`, alongside timestamped controller
events for process start/exit and job completion/failure. A new attempt gets a
new journal; the previous attempt remains intact. `stdout.log` and `stderr.log`
in that directory contain the native generator's output and diagnostics, and
may be empty. `campaign.json` records the current `progress` snapshot and
`progress_journal` paths. The journal tracks generation and campaign events;
it does not contain numerical results. Supported Monte Carlo price generators
keep those results separately under
`jobs/<target>/checkpoint/results.checkpoint`.
For sample jobs, the display uses sample counts during JSON writing as
described above; CUDA preparation and simulation have no within-job ETA.
The example
[`notebook.ipynb`](../experiments/equity/cross_model/heston_rough_heston_price_delta_generation/notebook.ipynb)
verifies/builds the Heston and rough Heston Cartesian price-delta targets in
`build`, then launches their executables directly, one notebook cell per
generator. It uses the same native progress reporter and captures stdout/stderr
itself. This direct path writes to the
recipe's catalogue and dataset destinations without campaign staging,
provenance attachment, publication backups, or a controller-provided checkpoint;
use the controller above when those safeguards are needed.

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
  --run-dir datasets/generation-runs/generation-01 --execute --resume
```

Resume uses the frozen selection and publication policy. Completed jobs are
hash-checked and skipped. A staged job resumes publication without running the
GPU again. An interrupted/failed generator receives a new attempt directory;
there is no automatic retry.

Version-3 campaigns provide within-dataset checkpoints for batched terminal
Monte Carlo price generation. This covers ordinary equity prices, prepared
N-factor prices, Volterra FFT prices, fixed-income terminal Monte Carlo prices,
and stochastic European equity price-delta recipes such as Heston and Bates.
At every native batch boundary, the generator copies only the completed output
slice, appends it to `results.checkpoint`, validates it with a checksum and
flushes it to durable storage. A truncated final record is discarded after a
power loss; the preceding contiguous prefix remains usable. Price-only records
store price and standard error (8 bytes per price before small record headers);
price-delta records store price, price error, delta and delta error (16 bytes per
price). The controller deletes the checkpoint only after the final JSON/YAML
pair has passed structural checks and reached the durable `staged` state.

The checkpoint identity binds the exact frozen executable, recipe, parameter
files, row count, RNG seeds, launch plan, sensitivity and time grid. A mismatch
is a blocking error, never an implicit reset or cross-version reuse. Result
metadata records how many prices came from a previous attempt; native GPU and
wall timings cover the current process attempt, while earlier attempt durations
remain in their journals. Checkpointing changes neither a pricing kernel, RNG
mapping, reduction order nor launch geometry. When enabled, it adds a host
synchronization, a small device-to-host copy and `fsync` at each existing price
batch. When the checkpoint environment is absent, the previous non-blocking path
is preserved.

Longstaff--Schwartz price/price-delta jobs and model-sample generators do not yet
have a numerical checkpoint and restart the active dataset from the beginning.
Campaign versions 1 and 2 likewise contain progress observations only; they
cannot be upgraded after the fact because their completed GPU values were never
written. Existing attempts and publication backups are retained.

Changed pricing code or parameters require a new campaign, not a resume with
different inputs. Independent certification is performed later through the
[price-validation pipeline](independent-price-validation-pipeline.md).

Keeping an already completed dataset is a separate decision from resuming a
campaign. Follow the [dataset provenance and reuse contract](dataset-provenance-contract.md)
for its read-only checker, legacy datasets and the evidence needed after a
refactor. Codegen regeneration never updates old data or its generation history.

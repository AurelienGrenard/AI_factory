# CUDA performance regression protocol

The primary performance infrastructure has three explicit owners:

- `tests/performance` owns benchmark sources, fixtures, protocol tests and
  architecture-specific baselines;
- `tools/performance` owns campaign execution and fail-closed comparison;
- this document owns the durable protocol and retuning procedure.

`validation/**` may consume the results but owns no primary gate component.
Ordinary catalogue timing remains informational metadata.

## Protocol version 3

- Build `Release` without fast math and for one explicit CUDA architecture.
- Record the GPU model, compute capability, SM and memory counts, CUDA driver,
  runtime and compiler versions with every measurement.
- Run five warmups followed by 21 measured repetitions in each of three
  complete campaigns. Median, p95 and coefficient of variation are mandatory.
- A benchmark may group a pre-declared number of identical operations in one
  timing sample when the ungrouped interval is dominated by scheduler or clock
  jitter. All reported durations remain normalized per operation, the grouping
  count is part of the configuration and any change requires a fresh baseline.
  Lengthening a window never permits changing a declared scope-specific noise
  budget inside a campaign or selecting a favorable campaign.
- The enclosing public-API interval is
  `max(raw_host_clock, enclosing_cuda_event_interval)`. The raw host clock is
  retained separately because host and CUDA event clocks can disagree under
  dynamic clocks; this definition guarantees `kernel <= public_api` without hiding
  the observed clock-domain discrepancy.
- A change needs at least a 5% improvement to justify added complexity. A
  median or p95 regression greater than 5% fails when both reference and
  candidate coefficients of variation satisfy their scope budget. Kernel
  timing uses a 5% ceiling. The host-enclosing public API and publication use a
  separately declared 10% ceiling because OS, driver, allocator and filesystem
  scheduling remain outside the CUDA-event interval. Their median and p95
  regression budget remains 5%; this host-noise allowance cannot turn a slow
  tail into a pass. A noisier blocking comparison is inconclusive and must be
  rerun; noise never becomes a pass.
- A timing can be `informational` only when repeated attempts cannot remove
  scheduler noise. It remains mandatory, but cannot pass or fail the timing
  gate. The short closed-form launcher latency is the sole such row. Its
  numerical and resource budgets remain blocking.
- Every numerical field is exact, tolerance-bounded, maximum-bounded or
  explicitly derived. Unknown, missing and unbudgeted fields fail.
- Every result owns the exact launch diagnostics observed before it. Runtime
  attributes cover registers, shared/local allocation, geometry and theoretical
  occupancy. `cudaFuncGetName` binds that launch to its exact linked symbol;
  `cuobjdump` independently records registers, stack, local and shared bytes,
  SASS bytes/instructions and `LDL`/`STL` counts. A disagreement is an error,
  not an inferred zero. Executable and command SHA-256 identifiers are retained.
- A multi-kernel workload declares its exact phase set in
  `resource_phase_contracts`. Gaussian-Volterra hybrid pricing requires
  `row_preparation`, `convolution`, `path_evaluation`, and `finalization`;
  the bounded direct experiment omits only `convolution`. Missing, duplicate,
  or undeclared phases fail before resource values are compared.
- Device memory is budgeted by owner: persistent inputs, caller workspace,
  transient workspace allocated by the public call, outputs, CUDA
  context/plan pools and free safety margin. Total-minus-free remains a
  complementary resident check and is never presented as the owned peak.
- Keep four timing scopes distinct whenever they apply: `kernel` is the CUDA
  event interval; `public_api` is the enclosing public call; `pipeline` is the
  complete calling job outside serialization; `publication` is generation,
  streaming, serialization and artifact verification (`publication_wall`).
  These are distinct schema fields. For a microbenchmark with no
  external preparation, the public-API and pipeline boundaries may coincide;
  the manifest still records why.
- Aggregate all three campaigns with the pre-declared median of campaign
  medians. The noise coefficient is the pre-declared median of the three
  campaign coefficients: at least two campaigns must therefore satisfy the
  noise budget. The p95 remains the conservative maximum, so a slow tail is
  never hidden by the robust noise decision. Every raw campaign is retained in
  a timestamped directory with SHA-256 hashes. Selecting a minimum, a best
  attempt, or a different campaign per key is forbidden.
- Before and after each complete campaign, record the AC/battery state, GPU
  identity, P-state, clocks, temperature, power draw/limit, active throttle
  reasons and concurrent compute processes. The SM89 laptop profile refuses to
  start on battery, below its declared 140 W current power limit, under a
  hardware power-brake signal or while another compute process is active.
  Temperature and thermal-regulation signals are telemetry only: there is no
  application temperature ceiling, thermal veto, target-temperature wait or
  temperature-convergence loop. The declared per-workload warmups remain.
  The current/default/minimum/maximum limits are read from the NVIDIA XML
  status because the WSL CSV field may report `N/A`. The final snapshot checks
  the same power/concurrency conditions. A temperature alone cannot reject
  the computation or certify timing stability; raw samples, CV and comparable
  operating conditions remain necessary. Three eligible campaigns are
  required within five total attempts. A failed start preflight consumes an
  attempt and is retained with its snapshot; there is no automatic thermal
  cooldown before the next declared attempt. An environmentally
  rejected complete campaign is retained with its raw output and reason but
  never enters the timing aggregate. The same timestamped journal is updated
  after every attempt so an environmental refusal cannot discard earlier
  campaign evidence. An interrupted series may resume only from that explicit
  journal; every retained payload hash and row count is revalidated before a
  remaining declared attempt is run. The runner stops early when the remaining
  attempt count can no longer reach the required eligible-campaign count.

[`../tests/performance/baseline_sm89_v3.json`](../tests/performance/baseline_sm89_v3.json)
is both the workload and budget manifest. Its command list is the only source
of campaign commands. The same manifest partitions them exactly once
into the generated `generic_cuda`, `model_sampling`, `early_exercise` and
`rough` reports.
Output outside its declared keys, duplicate output, an orphan diagnostic or
a command without output fails immediately. It is the current measured
baseline for the available RTX 4090 Laptop GPU. SM75 and SM86 remain functional
compile targets, but no runtime threshold is inferred without their hardware;
each deployed architecture needs its own native manifest.

## Reproduction

Configure the `dev` preset, then build the dedicated targets:

```sh
cmake --preset dev
cmake --build build --target performance_benchmarks -j2
```

The generic harness covers `index`, `accumulation`, `overhead`, `geometry` and
the three `ragged` variants. The CIR harness compares `inline` and `noinline`.
The early-exercise harness covers one-factor and multi-state equity American
options, then one- and two-factor fixed-income Bermudan swaptions through the
complete Longstaff--Schwartz pipeline. Its kernel statistic is the launcher's
own enclosing CUDA-event interval from row preparation through final reduction;
host planning and workspace allocation remain visible in `public_api` and
`pipeline`; the transient workspace peak is reported separately from buffers
owned by the caller.

The model-sample harness covers exact Markovian, fixed-step Markovian,
seven-factor rough and Volterra FFT, each for
`3,000,000 x 1` and `12,000 x 250`. Markovian and N-factor runs preserve the
full production shape. Volterra retains documented saturation-preserving
reductions. Publication serializes 262,144 rows in batches of 16 and reports a
separate blocking wall time.

The Volterra harness explicitly represents Markovian Heston, seven-factor rough
Heston, rough Bergomi FFT and rough SABR FFT. A separate executable preserves
the bounded direct-convolution experiment without adding that path to
production.

## Price-count and path-count scaling qualification

The fixed regression baseline detects changes on stable representative points;
it does not by itself qualify a production scaling envelope. Before publishing
new launch geometry or claiming production throughput, run the opt-in probes
owned by `tests/performance/pricing_scaling*` and
`tools/performance/*pricing_scaling*`.

The capability manifest, rather than a second model list, selects exactly one
side and one representative product for each available configuration:

- European call for terminal Monte Carlo;
- down-and-out put for monitored Monte Carlo;
- European call for Black--Scholes closed form and caplet for each available
  fixed-income closed-form composition;
- American put and Bermudan payer swaption for the separate LSM campaign.

Absence of a published binding is recorded as not applicable. Volterra FFT and
Markovian N-factor lifts remain distinct methods even when they share the
product policy. Longstaff--Schwartz is a separate workload because its
workspace and cross-block reductions change the scaling problem; a Monte Carlo
qualification must never be presented as its evidence.
The derived inventory contains 35 MC, nine closed-form and 17 LSM cases.
Black--Scholes American has a binding but no price recipe: its probe uses the
catalogue parameter rows and an explicitly benchmark-only deterministic seed.
It cannot establish parity with a nonexistent production generator.

For Monte Carlo, cross 65,536, 262,144 and 1,048,576 paths **per price** with
100, 1,000 and 10,000 aligned benchmark prices. For closed form, vary only those
three price counts. Keep model/product rows, side, transition grid, monitoring,
seed and numerical method fixed along a scaling comparison. A geometry study
may vary threads, blocks, path chunks and batch size, but must retain every raw
output and compare price and standard-error results against tolerances declared
before the campaign.

The v2 scaling probes use a temporary 100-row stratified tile (90 core and 10
stress rows, selected at the midpoint of each ten-row catalogue stratum),
repeated without altering parameters or calendars. Every 100-row tile has the
same workload composition. This benchmark-only layout is not a new independent
parameter dataset and does not replace the ordered 1,000-row catalogue.
Source hashes, source-row mapping and a fixture signature are retained; each
global benchmark row gets its own stable Philox row key. Generated fixtures and
publication artifacts stay under the run directory. Input preparation covers
the largest input set in that process and is reported separately, never
silently charged to every smaller measured shape.
Preparation caches can reuse the tile's repeated parameters: report this
explicitly. For example, rough Heston caches exact kernel fits by Hurst bits
within one batch. Such preparation timings do not qualify a dataset with a
million distinct Hurst values; that cost needs a separate independent-input
measurement. Do not silently replace exact pricing fits with interpolation.

For a dataset-runtime estimate, use `--input-profile catalogue` with exactly
1,000 prices and `--path-counts 1048576` for stochastic pricing (the catalogue's
"1M paths" means `2^20`, not 1,000,000). This profile keeps every original
row, identifier, parameter and calendar in its ordered 900/100 split. Its
provenance is distinct from the repeated tile and must never be pooled with
that scaling evidence. Closed-form workloads have no path-count dimension.
Explicit `--cases` order is preserved for scheduling; duplicate cases fail.

The `production` stage asks the compiled `inspect_pricing_launch_plan` for
each supported probe pair and price count. It uses the resulting threads,
blocks, row bounds, LSM blocks per price or FFT chunk without a Python tuning
table. Build the inspector and the selected probes first. For example:

```bash
python3 tools/performance/run_pricing_scaling.py \
  --stage production --price-counts 1000 --path-counts 1048576 \
  --input-profile catalogue --cases kou__mc_terminal cir__closed_form \
  --output artifacts/performance/production-plan-example --plan-only
```

Use a fresh output directory and omit `--plan-only` to execute two excluded
warmups and three measurements per pair. `production` disallows `--jobs`
overrides and other stochastic path counts. It records the inspector hash
and returned plans with the campaign; it does not extend the probe inventory
to every catalogue pair. Native LSM memory batching and compiled FFT grids
remain owned by their engines. The selected profile's qualification stays
explicit: executing a production-shaped check does not certify its optimality,
cross-GPU portability or scaling to one million independently drawn prices.

`export_pricing_dataset_runtime.py` creates the compact, portable evidence
consumed by the dataset-runtime notebook. It accepts only the exact catalogue
shape and one predeclared profile per pair. Missing/failed cases remain so;
neither a smaller workload nor its extrapolation substitutes for a measurement.
The reported generation estimate is the labelled sum of input preparation,
median raw host API time, median output copy and native local JSON/YAML writing
and verification. GPU time is shown separately, not added again. This sum is
not a cold-process stopwatch and excludes repeated warmups, reference engines,
network upload and process teardown. Another GPU or another product/calendar
requires its own measurements; replay alone is not independent certification.
Report any disagreement between the raw synchronized host clock and CUDA
events explicitly. The conservative API envelope does not resolve a clock
disagreement, and a low repeat CV does not establish absolute clock accuracy.

Select geometry per model, engine, price count and path count. `screening`
uses one excluded warmup and one exploratory measurement per candidate;
`geometry` provides repeated measurements for confirmation. `--price-counts`
and `--path-counts` restrict a stage without changing its input construction.
Screening never qualifies a production configuration. Compare both workload
axes on their confirmed geometry envelope and report each actual geometry;
neither a single universal block/thread configuration nor `P * T(1)` is an
acceptance target. `single_price` supplies the latter informational comparison.
Use `--resource-evidence <completed-calibration-directory>` to bound ordinary-MC
candidates by the exact binary's compiled block-size ceiling. This adds small
64-thread blocks and the largest admissible whole-warp block to the candidate
set; it does not force a register cap, add launch bounds or change the kernel.
A changed binary or different GPU model invalidates that resource selection.
Rejected launch geometries remain recorded as such, not as numerical failures
or evidence that the supported production configuration crashes.
For ordinary MC, `plan_pricing_scaling_confirmation.py` selects one candidate
per measured shape and includes the current native 4,096-row batching reference.
It predeclares longer samples, excluded warmups and repetitions from the source
timings; it does not rerun until a CV passes. Feed its `jobs.json` to the runner
with `--jobs` and the same explicit case IDs. Shorter geometries are not pooled
with these new measurements to manufacture a favorable confirmation result.

Report GPU-event, public-API, one-time preparation, output-copy and native
publication intervals separately. Scaling probes retain the unmodified host
interval in `raw_host_clock` and every normalized sample in
`raw_host_samples_ms`, independently of the enclosing
`public_api = max(raw_host_clock, GPU)` value. Summaries leave these fields
null for older captures that did not record them; the enclosing interval must
never be substituted for missing raw host measurements.
Also retain the effective batch count,
owned memory, kernel diagnostics, compiled resources, raw repetitions and GPU
telemetry. Very short kernels group a predeclared number of identical calls and
normalize the result; an ungrouped noisy duration cannot select a geometry.
The exploratory runner stops without retry after a watchdog or an execution
guard failure, not a power-limit variation. Changed executables/inputs, concurrent
compute and incomplete output cannot pass. It writes only below a fresh build or temporary evidence directory and
never changes a catalogue dataset or the versioned baseline.

A vendor-specific performance/fan mode is not a prerequisite. The ordinary
operating mode is a valid environment to investigate production throughput.
All campaign and profiling controllers record temperatures without stopping
or waiting on a thermal threshold. This applies to long experiments and
dataset-generation/publication experiments, which application thermal gates
previously interrupted. Native catalogue generators have no such gate either.
No GPU firmware, driver regulation or external execution-authorization policy
is changed by this application policy; removing a script veto cannot override
an execution refusal from the surrounding environment.

Both `strict` and `--environment-policy observe` continue through GPU power-limit
variations, including an initially out-of-envelope limit. The bounds are timing
admission criteria, not experiment kill switches. Record excursions before,
during and after each case in `timing_ineligibility_reasons`; an excursion makes
`timing_eligible: false` persist even if power recovers. Complete numerical
results remain usable for diagnosis, not for accepting a tuning change.
Observation mode is always timing-ineligible, including at low CV. The older
LSM probe follows the same execution rule and remains exploratory.
Hardware power brake, foreign compute, external-power preflight and watchdog
guards of the scaling runner remain enforced. Neither mode applies a thermal
veto. Official baseline admission and numerical, resource, CV and regression
budgets are unchanged; `PERF-021` is not a rebaseline.
NVIDIA describes [software thermal clock regulation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
and [dynamic CPU/GPU power allocation](https://www.nvidia.com/en-us/geforce/laptops/max-q-technologies/)
separately; an observed power-limit change alone is not proof of unplugging.

The `PERF-020` policy amendment preserves the previous complete manifest under
`tests/performance/history/baseline_sm89_v3_pre_perf_020.json`. Its hash and the
scope of the amendment are recorded in the active manifest. Historical timings,
profiles and rejection evidence remain unchanged; this policy change is not a
new performance qualification or a measured rebaseline.

The current qualification ceiling is 10,000 prices by 1,048,576 paths per
price. The architectural target of one million prices by one million paths is
not a resident allocation: it requires bounded price batches, stable global
row offsets and row-to-seed mappings, resumable publication and no dependence
of numerical results on the batch partition. Measurements at 10,000 prices may
establish the steady-state unit cost and memory bound, but they do not certify
the elapsed time at one million prices.

Generate and build the derived probes with the ordinary CMake build, then run a
new evidence directory. Case IDs may be supplied to keep one GPU process active
at a time:

```sh
cmake --build build --target pricing_scaling_benchmarks -j2
python3 tools/performance/run_pricing_scaling.py \
  --build-dir build \
  --output artifacts/performance/pricing-scaling-<campaign> \
  --stage scaling \
  --cases <manifest-case-id>
```

LSM requires `--families lsm` or explicit LSM case IDs; the default selection
does not append it to an ordinary MC campaign. Use `--plan-only` to inspect
the full matrix without accessing the GPU. `--stage geometry` crosses LSM
threads, blocks per price and caller row batches. Each native launcher retains
its VRAM-aware batching within that row slice; trajectories of one price are
never split into independent regressions. The adapter offsets all aligned
input/output pointers and the base seed together. GPU time is the sum of the
native launcher's batch-event intervals; public API time additionally includes
workspace allocation, planning, synchronization, diagnostics and probe batch
accounting. Every repetition retains native batch counts, workspace peaks and
regression diagnostics. Per-kernel phase profiles still require a separate
profiling pass; aggregate batch timings do not replace them.

Summaries merge explicit evidence directories without turning missing or
environmentally rejected measurements into passes:

The per-campaign envelopes keep screening and fresh confirmations separate.
An axis comparison assembled from different execution periods is inconclusive;
the best observed times across those periods do not establish a tuning gain.
When a geometry comparison exceeds a predeclared numerical tolerance, neither
member may enter the qualified geometry envelope or a passing scaling verdict,
even at low timing CV. The summary retains the conflict and both run identities;
it does not decide which value is correct. Targeted rows with different offsets
remain distinct workloads. Numerical disagreements require diagnosis, not a
relaxed tolerance after seeing the results.

```sh
python3 tools/performance/summarize_pricing_scaling.py \
  artifacts/performance/pricing-scaling-<campaign> \
  --output artifacts/performance/pricing-scaling-<campaign>/summary.json
```

Volterra FFT geometry has one implementation owner:
`src/common/volterra/hybrid_fft_tuning.cuh`. FFT length is selected from the
validated step count; elements per thread and FFTs per block are separate
pricing/sampling fields even when the SM89 values coincide. Pricing path and
finalization block sizes are separate CMake profile variables. A retune changes
that profile, never the pricing or sampling algorithms, and publishes a native
baseline for every affected architecture.

On the exact SM89 environment, the gate builds and runs three complete
eligible campaigns (within at most five attempts), retains all raw outputs,
writes their aggregate candidate and four partitioned reports under the build
directory, and rejects
missing, duplicate, unknown, incompatible, numerically invalid,
resource-regressed, timing-regressed or blocking-inconclusive rows:

```sh
cmake --build build --target performance_regression_gate -j2
```

No timing result determines campaign eligibility and no campaign is recomposed
per key. A blocking key is inconclusive if the median aggregate CV exceeds 5%
for a kernel or 10% for a host-enclosing public API or publication measurement.
To inspect a captured candidate:

```sh
python3 tools/performance/check_baseline.py \
  tests/performance/baseline_sm89_v3.json candidate.ndjson
```

An explicit rebaseline requires a retained predecessor, a distinct output, an
exhaustive diff, a reason and an approval. For a normal rebaseline, the
retained predecessor must match the baseline used to run the campaigns byte
for byte, and the candidate must pass against it before anything is written.
Preserve the predecessor observations, hash and environment, then review the
exhaustive leaf-level diff before publication:

```sh
python3 tools/performance/run_baseline.py \
  --baseline tests/performance/history/baseline_sm89_v3_pre_struct_019.json \
  --build-dir build \
  --output artifacts/performance/performance_candidate_sm89_v3.ndjson \
  --predecessor-baseline \
    tests/performance/history/baseline_sm89_v3_pre_struct_019.json \
  --rebaseline-output tests/performance/baseline_sm89_v3.json \
  --rebaseline-diff-output \
    tests/performance/history/sm89_v3_rebaseline_diff.json \
  --rebaseline-reason "documented architecture or workload change" \
  --rebaseline-approval "reviewer and approval reference"
```

When a retained manifest has the same workload identities but observations
from an incompatible protocol, add `--initialize`. This mode checks schema
completeness, numerical/resource budgets and campaign stability, but it
deliberately does not claim a timing regression pass: observations are
initialized from the candidate. It still retains the exact input manifest and
publishes the complete diff, reason, approval and `protocol_initialization`
lineage. Its working manifest may lengthen or otherwise repair an incompatible
measurement protocol, but stable measurement identities and numerical
contract fields must remain identical to the retained predecessor. Subsequent
updates must use the normal regression-checked procedure.

The runner enables launch diagnostics itself. Manual inspection can still use
`AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1`. The `test_policy_size_budgets_cuda`
target additionally compiles probes at every storage cap.

## Jamshidian strategy experiments

The opt-in `ai_factory_jamshidian_strategy_benchmark` target compares the
existing scalar and cooperative kernels on regular European swaptions. Its
runner derives one-factor model/curve coverage from the capability manifest;
the [SM89 experiment report](performance-reports/jamshidian-strategy-scaling-sm89-2026-09-08.md)
documents commands, shapes, calendar profiles and measured limits.

Select execution mode and geometry by workload size and calendar, not model
name alone. Separate broad screening from fresh confirmations. Repeated
catalogue rows are throughput fixtures, not independent new parameters.
Keep incomplete numerical references and noisy timings visible and ineligible;
completing an experiment does not qualify every candidate or change a recipe.
Archive raw measurements, exact resources, input/binary hashes and exclusions.
This study does not replace protocol-v3 admission, native publication timing
or another GPU's own qualification.

## Representative Nsight Compute profiles

[`../tools/performance/profile_kernel.py`](../tools/performance/profile_kernel.py)
resolves the executable, arguments and exact mangled kernel symbol from the
manifest-owned candidate. It refuses a binary whose SHA-256 differs from the
timed candidate, applies the same power/concurrency preflight before
and after profiling, captures exactly one matching launch, and exports the raw
Nsight Compute metrics with a hashed provenance document. Profiling is run
only after the timing campaigns: replay overhead is never mixed with the
regression timings.

The four current representatives cover register-heavy generic CUDA, model
sampling with N-factor local traffic, the multi-state early-exercise path
kernel and rough FFT pricing. Run each command separately so the preflight can
reject any concurrent GPU use:

```sh
python3 tools/performance/profile_kernel.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --candidate artifacts/performance/performance_candidate_sm89_v3.ndjson \
  --build-dir build \
  --measurement-id cir_noinline \
  --output-dir tests/performance/profiles/sm89

python3 tools/performance/profile_kernel.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --candidate artifacts/performance/performance_candidate_sm89_v3.ndjson \
  --build-dir build \
  --measurement-id model_samples__rough_n_factor_7_12000_x_250 \
  --output-dir tests/performance/profiles/sm89

python3 tools/performance/profile_kernel.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --candidate artifacts/performance/performance_candidate_sm89_v3.ndjson \
  --build-dir build \
  --measurement-id lsm__equity_multi_state_heston \
  --resource-index 1 \
  --output-dir tests/performance/profiles/sm89

python3 tools/performance/profile_kernel.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --candidate artifacts/performance/performance_candidate_sm89_v3.ndjson \
  --build-dir build \
  --measurement-id rough_sabr_fft \
  --resource-index 2 \
  --environment-mode resource_only \
  --output-dir tests/performance/profiles/sm89
```

The versioned evidence consists of one `*.ncu.csv` raw profile and one
`*.profile.json` provenance document per representative. `resource_only`
retains pre/postflight checks but supports only register, spill, occupancy and
instruction claims. Timing claims require the default `timing` mode and the complete
campaign protocol. A different GPU or toolchain publishes a separate profile
directory and never overwrites SM89.

## Decisions represented by the SM89 profile

The versioned baseline and profiles, rather than this prose, own the exact
numbers. They currently justify these qualitative choices on SM89:

- validated 32-bit device index decoding instead of wider runtime decoding;
- the manifest-declared launch geometry for register-heavy product kernels;
- non-inlining of large non-central-chi-square helpers when it reduces register
  pressure and improves occupancy;
- blocking coverage for regular and explicit swaption schedules;
- FP64 for Monte Carlo moments and numerically sensitive accumulations where
  the recorded mixed-precision comparison exceeds the error budget;
- informational timing only for the short closed-form launcher, whose
  numerical and resource checks remain blocking;
- cuFFTDx rather than direct convolution over the measured production domain;
- one reusable Volterra stream and the manifest-declared chunk size.

These are measured decisions for the recorded SM89 environment, not universal
CUDA constants. Retune on another architecture with the same protocol and
publish a separate manifest and profile directory.

The [older Volterra CSV fixtures](../tests/performance/fixtures/volterra/README.md)
are retained only as explicitly scoped exploratory evidence. They are not a
protocol-v3 baseline or an acceptance source.

The [2026-09-05 fixed-income LSM diagnosis](performance-reports/fixed-income-lsm-2026-09-05.md)
likewise records an exploratory launch-geometry, calendar and memory campaign.
Its bounded probe uses production launchers but does not change the reference
profile or qualify a new protocol-v3 baseline.

Its [cross-model extension](performance-reports/cross-model-lsm-2026-09-05.md)
covers Vasicek, fitted rates models and representative equity LSM launchers,
including price-chunk invariance and explicit thermal/power-envelope exclusions.
The partial geometry confirmation does not authorize a production retuning.

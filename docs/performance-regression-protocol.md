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
  start on battery, below its declared 140 W current power limit, above 85 C,
  under hardware/thermal slowdown or while another compute process is active.
  Before the official starting snapshot, it repeatedly runs the declared
  conditioning command. The profile declares a minimum duration,
  minimum/maximum run counts and a trailing temperature window; conditioning
  is complete only when that window's range is at most the declared bound.
  This avoids both cold-start
  measurements and a hardware-specific target temperature. Conditioning
  outputs are excluded from timings, while their snapshots and output hashes
  are retained in the campaign journal.
  The current/default/minimum/maximum limits are read from the NVIDIA XML
  status because the WSL CSV field may report `N/A`. The final snapshot must
  remain below the maximum temperature without a forbidden throttle reason;
  the observed start/end temperature change is retained but is not confused
  with instability after a converged start. Three eligible campaigns are
  required within five total attempts. A failed start preflight consumes an attempt, is retained
  with its snapshot, then observes the manifest's 30-second cooldown before
  the next attempt; it does not abort the campaign series. An environmentally
  rejected complete campaign is retained with its raw output and reason but
  never enters the timing aggregate. The same timestamped journal is updated
  after every attempt so an environmental refusal cannot discard earlier
  campaign evidence. An interrupted series may resume only from that explicit
  journal; every retained payload hash and row count is revalidated before a
  remaining declared attempt is run. The runner stops early when the remaining
  attempt count can no longer reach the required eligible-campaign count.

[`../tests/performance/baseline_sm89_v3.json`](../tests/performance/baseline_sm89_v3.json)
is both the workload and budget manifest. Its 22-command list is the only
source of campaign commands. The same manifest partitions them exactly once
into the generated `generic_cuda`, `model_sampling`, `early_exercise` and
`rough` reports.
Output outside its 41 declared keys, duplicate output, an orphan diagnostic or
a command without output fails immediately. It is the current measured
baseline for the available RTX 4090 Laptop GPU. SM75 and SM86 remain functional
compile targets, but no runtime threshold is inferred without their hardware;
each deployed architecture needs its own native manifest.

## Reproduction

Configure the `dev` preset, then build the dedicated targets:

```sh
cmake --preset dev
cmake --build build-dev --target performance_benchmarks -j2
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

The model-sample harness contributes eight blocking rows: exact Markovian,
fixed-step Markovian, seven-factor rough and Volterra FFT, each for
`3,000,000 x 1` and `12,000 x 250`. Markovian and N-factor runs preserve the
full production shape. Volterra retains documented saturation-preserving
reductions. Publication serializes 262,144 rows in batches of 16 and reports a
separate blocking wall time.

The Volterra harness explicitly represents Markovian Heston, seven-factor rough
Heston, rough Bergomi FFT and rough SABR FFT. A separate executable preserves
the bounded direct-convolution experiment without adding that path to
production.

On the exact SM89 environment, the gate builds and runs three complete
eligible campaigns (within at most five attempts), retains all raw outputs,
writes their aggregate candidate and four partitioned reports under the build
directory, and rejects
missing, duplicate, unknown, incompatible, numerically invalid,
resource-regressed, timing-regressed or blocking-inconclusive rows:

```sh
cmake --build build-dev --target performance_regression_gate -j2
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
  --build-dir build-dev \
  --output build-dev/performance_candidate_sm89_v3.ndjson \
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

## Representative Nsight Compute profiles

[`../tools/performance/profile_kernel.py`](../tools/performance/profile_kernel.py)
resolves the executable, arguments and exact mangled kernel symbol from the
manifest-owned candidate. It refuses a binary whose SHA-256 differs from the
timed candidate, applies the same power/thermal/concurrency preflight before
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
  --candidate build-dev/performance_candidate_sm89_v3.ndjson \
  --build-dir build-dev \
  --measurement-id cir_noinline \
  --output-dir tests/performance/profiles/sm89

python3 tools/performance/profile_kernel.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --candidate build-dev/performance_candidate_sm89_v3.ndjson \
  --build-dir build-dev \
  --measurement-id model_samples__rough_n_factor_7_12000_x_250 \
  --output-dir tests/performance/profiles/sm89

python3 tools/performance/profile_kernel.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --candidate build-dev/performance_candidate_sm89_v3.ndjson \
  --build-dir build-dev \
  --measurement-id lsm__equity_multi_state_heston \
  --resource-index 1 \
  --output-dir tests/performance/profiles/sm89

python3 tools/performance/profile_kernel.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --candidate build-dev/performance_candidate_sm89_v3.ndjson \
  --build-dir build-dev \
  --measurement-id rough_sabr_fft \
  --output-dir tests/performance/profiles/sm89
```

The versioned evidence consists of one `*.ncu.csv` raw profile and one
`*.profile.json` provenance document per representative. A different GPU or
toolchain publishes a separate profile directory and never overwrites SM89.

## Decisions represented by the SM89 profile

- Validated 32-bit device index decoding is retained: 2.41 ms versus 5.39 ms
  for runtime decoding and 5.44 ms for compile-time dispatch alone, with
  identical mappings.
- Bates Phoenix and Phoenix-memory keep 512 threads: 4.07--4.22 ms versus
  6.32--6.33 ms at 256 and 11.14--11.15 ms at 128, with identical prices and
  no local-memory spill.
- Large non-central-chi-square helpers are not force-inlined. `noinline` lowers
  registers from 64 to 56, raises theoretical occupancy from 66.7% to 75%,
  shrinks the representative executable and reduces the kernel from 8.79 ms to
  8.34 ms.
- Regular, homogeneous explicit and heterogeneous explicit swaption schedules
  are all blocking workloads; their current medians are 6.05, 4.68 and 4.35 ms
  for their declared row counts.
- Monte Carlo moments and numerically sensitive accumulations remain FP64.
  Mixed precision is faster in isolation but introduces measured error on the
  non-negative mixed-scale stress stream.
- Short closed-form launch cost remains informational at roughly 7 microseconds
  per call. Existing grid-stride batches amortize it; validation remains
  fail-fast.
- cuFFTDx remains faster than direct convolution even at eight steps: 5.44 ms
  versus 9.18 ms per price on the aggregated eight-price workload, with
  identical first-price moments and no local-memory spill.
- The Volterra workspace keeps one reusable stream and a 65,536-path chunk. At
  2,097,152 paths and 252 steps it takes 26.55 ms, versus 34.34 ms for 16,384
  and 76.30 ms for 4,096, with identical outputs. At 1,048,576 paths, eight
  prices take 13.29 ms per price; live device memory and workspace footprints
  are budgeted by the manifest.

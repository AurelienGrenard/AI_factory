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
- The enclosing public-API interval is
  `max(raw_host_clock, enclosing_cuda_event_interval)`. The raw host clock is
  retained separately because host and CUDA event clocks can disagree under
  dynamic clocks; this definition guarantees `kernel <= public_api` without hiding
  the observed clock-domain discrepancy.
- A change needs at least a 5% improvement to justify added complexity. A
  median or p95 regression greater than 5% fails when both reference and
  candidate coefficients of variation are at most 5%. Host publication uses a
  separately declared 10% ceiling because filesystem scheduling dominates its
  residual noise. A noisier blocking comparison is inconclusive and must be
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
  medians. The p95 and noise coefficient use the conservative maximum. Every
  raw campaign is retained in a timestamped directory with SHA-256 hashes.
  Selecting a minimum, a best attempt, or a different campaign per key is
  forbidden.
- Before and after each complete campaign, record the AC/battery state, GPU
  identity, P-state, clocks, temperature, power draw/limit, active throttle
  reasons and concurrent compute processes. The SM89 laptop profile refuses to
  start on battery, below its declared 140 W current power limit, above 85 C,
  under hardware/thermal slowdown or while another compute process is active.
  The current/default/minimum/maximum limits are read from the NVIDIA XML
  status because the WSL CSV field may report `N/A`. Temperature may drift by at most 5 C
  between the two snapshots. Three eligible campaigns are required within five
  total attempts. A failed start preflight consumes an attempt, is retained
  with its snapshot, then observes the manifest's 30-second cooldown before
  the next attempt; it does not abort the campaign series. An environmentally
  rejected complete campaign is retained with its raw output and reason but
  never enters the timing aggregate. The same timestamped journal is updated
  after every attempt so an environmental refusal cannot discard earlier
  campaign evidence. An interrupted series may resume only from that explicit
  journal; every retained payload hash and row count is revalidated before a
  remaining declared attempt is run.

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
per key. A blocking key is
inconclusive if the conservative aggregate CV exceeds 5%, or 10% for
publication. To inspect a captured candidate:

```sh
python3 tools/performance/check_baseline.py \
  tests/performance/baseline_sm89_v3.json candidate.ndjson
```

An explicit rebaseline requires both an output path and a reason. Preserve the
predecessor observations, hash and environment, and review their exhaustive
diff before publication. The manifest is written only after every blocking key
is stable and passes all budgets:

```sh
python3 tools/performance/run_baseline.py \
  --baseline tests/performance/baseline_sm89_v3.json \
  --build-dir build-dev \
  --output build-dev/performance_candidate_sm89_v3.ndjson \
  --rebaseline-output tests/performance/baseline_sm89_v3.json \
  --rebaseline-reason "documented architecture or workload change"
```

The runner enables launch diagnostics itself. Manual inspection can still use
`AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1`. The `test_policy_size_budgets_cuda`
target additionally compiles probes at every storage cap.

## Decisions represented by the SM89 profile

- Validated 32-bit device index decoding is retained: 1.69 ms versus 3.76 ms
  for runtime decoding, with identical mappings. Compile-time dispatch of the
  construction enum alone is rejected.
- Bates Phoenix and Phoenix-memory keep 512 threads: about 4.07 ms versus
  6.32 ms at 256 and 11.15 ms at 128, with identical prices and no local-memory
  spill.
- Large non-central-chi-square helpers are not force-inlined. `noinline` lowers
  registers from 64 to 56, raises theoretical occupancy from 66.7% to 75%,
  shrinks the representative executable and reduces the kernel from 8.14 ms to
  7.38 ms.
- Regular, homogeneous explicit and heterogeneous explicit swaption schedules
  are all blocking workloads; their current medians are 4.58, 4.41 and 4.16 ms
  for their declared row counts.
- Monte Carlo moments and numerically sensitive accumulations remain FP64.
  Mixed precision is faster in isolation but introduces measured error on the
  non-negative mixed-scale stress stream.
- Short closed-form launch cost remains informational at roughly 7 microseconds
  per call. Existing grid-stride batches amortize it; validation remains
  fail-fast.
- cuFFTDx remains faster than direct convolution even at eight steps: 4.12 ms
  versus 8.74 ms per price on the aggregated eight-price workload, with
  identical first-price moments and no local-memory spill.
- The Volterra workspace keeps one reusable stream and a 65,536-path chunk. At
  2,097,152 paths and 252 steps it takes 24.16 ms, versus 27.85 ms for 16,384
  and 75.98 ms for 4,096, with identical outputs. At 1,048,576 paths, eight
  prices take 13.17 ms per price; live device memory and workspace footprints
  are budgeted by the manifest.

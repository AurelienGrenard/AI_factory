# CUDA performance baselines

This directory owns reproducible performance experiments. Catalogue timing is
informational publication metadata; it is not a regression baseline.

## Protocol version 1

- Build `Release` without fast math and for one explicit CUDA architecture.
- Record the GPU model, compute capability, SM and memory counts, CUDA driver,
  runtime and compiler versions with every measurement.
- Run five warmups followed by 21 measured repetitions. The kernel median is
  the primary statistic; p95 and coefficient of variation are mandatory.
- The enclosing wall interval is
  `max(raw_host_clock, enclosing_cuda_event_interval)`. The raw host clock is
  retained separately because host and CUDA event clocks can disagree under
  dynamic clocks; this definition guarantees `kernel <= wall` without hiding
  the observed clock-domain discrepancy.
- A change needs at least a 5% median improvement to justify added complexity.
  A regression greater than 5% fails only when both reference and candidate
  coefficients of variation are at most 5%. A noisier comparison is
  inconclusive and must be rerun; noise never becomes a failure.
- Every performance result carries a numerical invariant: identical indices,
  unchanged prices within the relevant contract, finite moments, or an
  explicit error measurement.

[`baseline_sm89_v1.json`](baseline_sm89_v1.json) is the current measured
baseline for the available RTX 4090 Laptop GPU. SM75 and SM86 remain functional
compile targets, but no runtime threshold is inferred without hardware of that
architecture. Their policy-size probes are compiled and their SASS resource
usage is recorded; a deployment on either architecture must publish its own
runtime baseline before accepting a performance-sensitive change.

## Reproduction

Configure the `dev` preset, then build the dedicated targets:

```sh
cmake --preset dev
cmake --build build-dev --target performance_benchmarks -j2
```

The generic harness supports `index`, `accumulation`, `overhead`, `geometry`
and the three `ragged` variants. The CIR harness compares `inline` and
`noinline`. The Volterra harness takes model, maturity days, steps per day,
price count, tuning value, repetitions and optional path count. The separate
`ai_factory_volterra_direct_kernel_benchmark` preserves the bounded direct
convolution experiment without adding its slower path to production.

Each process writes versioned NDJSON. Capture a candidate and compare it with
the baseline using:

```sh
python3 validation/performance/check_baseline.py \
  validation/performance/baseline_sm89_v1.json candidate.ndjson
```

Resource measurements use `AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1`. The
`test_policy_size_budgets_cuda` target compiles probes at every storage cap;
for an offline architecture build, inspect its SASS with
`cuobjdump --dump-resource-usage`.

## Decisions represented by v1

- Validated 32-bit device index decoding is retained; compile-time dispatch of
  the construction enum alone is not.
- Bates Phoenix and Phoenix-memory keep 512 threads: they are substantially
  faster than 128/256 despite one resident block, with identical prices and no
  spills.
- Large non-central-chi-square helpers are not force-inlined. This halves the
  CIR archive, lowers registers and is faster in the representative workload.
- Explicit swaption schedules use payment-major ELLPACK pools and cooperative
  Jamshidian evaluation; heterogeneous 2–30-payment rows fall from about
  6.99 ms to 0.29 ms on SM89.
- Monte Carlo moments and numerically sensitive accumulations remain FP64.
  Mixed precision is faster in isolation but introduces measurable error on a
  non-negative mixed-scale stress stream.
- Short closed-form launch cost is about 5–7 microseconds per call. Existing
  grid-stride batches amortize it; pointer validation remains fail-fast and is
  not cached across allocations.
- cuFFTDx remains faster than direct convolution even at eight steps. For FFT
  length 8192, 16 elements per thread replaces 32: median time falls from
  22.45 ms to 17.87 ms, registers from 139 to 72, and theoretical occupancy
  rises from 16.7% to 33.3%, with no local-memory spill.
- The Volterra workspace keeps one reusable stream and a 65,536-path chunk.
  At 1,048,576 paths and 252 steps this chunk takes 11.17 ms, versus 14.07 ms
  for 16,384 and 37.68 ms for 4,096, with identical outputs. Eight prices take
  11.48 ms per price, so concurrent streams are not justified by the measured
  throughput; peak reusable workspace is about 63.14 MiB for this case.

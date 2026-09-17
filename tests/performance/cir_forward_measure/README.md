# CIR terminal-forward qualification

These opt-in probes qualify the production CIR Bermudan launcher, using exact
transitions under the last-exercise bond measure. They emit evidence only;
they never publish datasets or alter tuning defaults. The original Q/prototype
comparison is retained in the linked report and its immutable row exports.

## Method

For numeraire maturity `T* = last exercise`, simulate only the exercise dates
under `Q(T*)`. The transition is scaled noncentral chi-square; see
[Brigo–Mercurio, equations 17–18 and 21](https://www.ma.imperial.ac.uk/~dbrigo/detshiftrep.pdf).
Store only the rate. Express each exercise payoff as
`H(t) * P(0,T*) / P(t,T*)`; continuation needs no pathwise discount integral.
The shared LSM kernels, cubic Hermite features and FP64 regression are unchanged.
The production schedule stores its variable-size table in the shared LSM
workspace; it has no fixed 32-exercise limit.

## Files

- `src/model/fixed_income/cir/forward_measure.cuh`: model-owned transition law.
- `src/product/bermudan_swaption/terminal_forward_pricing_policy.cuh`: numeraire adapter.
- `benchmark.cu`: calls the public production launcher on original catalogue rows.
- `transition_probe.cu`: sampler moments and discounted bond/option payoffs.
- `discount_chain_probe.cu`: old Q chains without LSM, comparing rate moments
  and paired FP32/FP64 discount accumulation; diagnostic only.
- `validation/quantlib/model/fixed_income/cir/forward_reference.py`: independent CPU drift moments,
  risk-neutral PDE and disjoint train/test LSM checks.

## Reproduce

```sh
cmake --build build --target ai_factory_cir_forward_measure_probe \
  ai_factory_cir_forward_transition_probe ai_factory_cir_discount_chain_probe -j2
python3 tools/performance/run_cir_forward_comparison.py \
  --jobs tests/performance/fixtures/cir_forward_production.json \
  --output artifacts/performance/cir-forward-smoke-<new-run>
OPENBLAS_NUM_THREADS=1 python3 -m unittest \
  tests.performance.test_cir_forward_reference \
  tests.performance.test_cir_forward_comparison -v
```

The CPU checks explicitly call QuantLib analytics but never regenerate a
validation cache. Run GPU probes sequentially. Raw measurements belong in a
fresh directory under `artifacts/performance`, not in `datasets` or `catalog`; checked compact exports
may be retained under `tests/performance/reports`.

After the numerical jobs have finished, run the bounded memory checks:

```sh
timeout 180 compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/ai_factory_cir_forward_measure_probe \
  --jobs tests/performance/fixtures/cir_forward_sanitizer.json
timeout 180 compute-sanitizer --tool racecheck --error-exitcode 99 \
  build/ai_factory_cir_forward_measure_probe \
  --jobs tests/performance/fixtures/cir_forward_sanitizer.json
```

`discount_chain_probe` deliberately tests the old method, without LSM. Its
checker (`python3 -m tools.performance.check_cir_forward_transitions ...
--discount-chains`) returns nonzero when it detects a numerical discrepancy;
this is an investigative result, not a CUDA execution failure.

`cir_forward_production.json` requests all 900 core and 100 stress rows on both
sides at `2^20` paths, then a 512-thread geometry check. Row keys always retain
the original index, including sparse selections and caller batches. The old
`cir_forward_catalogue`, `cir_forward_receiver_catalogue`, `cir_forward_smoke`
and `cir_forward_discrepancies` fixtures describe archived experiments, not
current launch interfaces. `cir_forward_sanitizer.json` is the bounded two-sided
4,096-path memory/synchronization check, not a precision or timing workload.

The initial discrepancy screen is `abs(new-old) <= 5*hypot(SE_old,SE_new)+2e-6`.
This is not a financial error budget or a proof of LSM accuracy. Inspect every
failure, repeat with independent seeds, refine the old time step and compare
with a converged independent reference. Zero observed variance, rare exercise
and a tiny absolute price do not establish relative accuracy.

Timings are exploratory, not protocol-v3 acceptance. CPU reference work may
overlap numerical campaigns; retain GPU telemetry and raw host/CUDA clocks.
Temperature is recorded without an application thermal veto.

The [comparison report](../../../docs/performance-reports/cir-forward-measure-comparison-sm89-2026-09-08.md)
owns the numerical verdict and links every measured catalogue row. Production
is bitwise equal to the qualified prototype on both 1,000-row sets at identical
seeds. Preparation uses 40 registers and no declared local memory on SM89;
128/512-thread launches have been checked. These are measured resources,
not portable architecture limits. The superseded test-only policies were
removed; the saved source snapshots preserve the historical experiment.

After this CIR decision, the agreed general campaign adds Hull–White and G2++
with Nelson–Siegel, plus targeted Bates, VG/NIG and QRH inheritance checks.
The primary geometry screen is 128/256/512 threads, with explicit exceptions
only when kernel constraints or measured gains justify them.

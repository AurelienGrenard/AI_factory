# Jamshidian strategy and price-count scaling — SM89 — 2026-09-08

## Decision and scope

Keep both execution strategies. Neither a universal scalar/cooperative choice
nor a single threads/blocks pair fits these workloads. Cooperative execution
wins on the full 1,000-row catalogue. At 2²⁰ prices, scalar execution becomes
competitive for Ornstein–Uhlenbeck and Vasicek, but their fastest measured
large-batch candidates are not statistically/numerically qualified below.
During the strategy experiment itself, no production recipe, pricing formula
or tuning default was changed. The subsequent integration is distinguished below.

This experiment covers regular European Jamshidian swaptions for all five
one-factor models, seven model/curve compositions. G2/G2++, caplets, bond
options, explicit ELLPACK schedules, Monte Carlo, LSM and samples are outside
scope. Here **2²⁰ means 1,048,576 prices, not Monte Carlo trajectories**.

Subsequent integration: recipes now use the shared
[host launch plan](../cuda/launch-validation-and-kernel-diagnostics.md#planifier-le-pricing-du-catalogue)
to request one of the existing kernels. CIR++ and Hull–White use cooperative
candidates; OU uses the measured cooperative catalogue point and its native
scalar default elsewhere. The 1,000-price case is an explicit measured shape,
not a demonstrated optimal crossover. CIR and Vasicek retain native geometry
pending fresh timings after the now-closed `NUM-021` correction. These selections are labelled candidates, not new native
timing qualifications. Runtime tests cover both modes and persistent grids;
formulas and guarded scalar fallback are unchanged. The implementation and
its bounded checks are recorded in [audit status](../audit/status.md).

## Confirmed measurements

GPU times below are medians of the three independent payer campaign medians,
not the minimum from screening. Geometry is **threads per block × blocks**.
Each size selects its geometry independently. These are the fastest screened
candidates subsequently measured, not a proof of a global optimum.

| Composition | 1,000 prices: mode, geometry, GPU ms | 2²⁰ prices: mode, geometry, GPU ms |
|---|---|---|
| CIR | cooperative, 512 × 256, **1.081** † | cooperative, 128 × 1,048,576, **79.915** † ‡ |
| CIR++ / Nelson–Siegel | cooperative, 256 × 256, **1.170** | cooperative, 64 × 1,048,576, **94.124** |
| CIR++ / Svensson | cooperative, 256 × 256, **1.209** | cooperative, 64 × 1,048,576, **104.368** |
| Hull–White / Nelson–Siegel | cooperative, 128 × 1,000, **0.881** | cooperative, 128 × 4,096, **64.775** |
| Hull–White / Svensson | cooperative, 128 × 1,000, **0.878** | cooperative, 128 × 1,048,576, **66.800** |
| Ornstein–Uhlenbeck | cooperative, 128 × 1,000, **0.895** | scalar, 64 × 16,384, **41.056** ‡ |
| Vasicek | cooperative, 128 × 1,000, **0.895** † | scalar, 256 × 4,096, **36.491** † ‡ |

† Incomplete numerical reference: never a production-qualified recommendation.
‡ At least one within-run or between-run GPU CV exceeds the predeclared 5%
bound. All samples remain in the table; no favorable period was substituted.
Other entries pass this experiment's repeated timing and cross-mode checks,
which do not replace independent financial price certification.

At 1,000 prices the scalar medians are 70.764 ms (CIR), 89.129/92.503 ms
(CIR++), 20.927/24.805 ms (Hull–White), 5.929 ms (OU), and 6.174 ms (Vasicek).
At 2²⁰ prices, the other-mode medians are respectively 232.692, 359.609,
404.623, 126.898, 156.895, 55.365 and 55.343 ms. Several of these scalar
comparators are noisy: their exact speedup ratios are not acceptance gates.

The large-batch cooperative ratio T(2²⁰)/T(2¹⁸) is about 3.86–4.33 for CIR,
CIR++ and Hull–White when the number of prices grows fourfold. This supports
approximately linear throughput in this range, not exact linearity or a
cross-GPU guarantee. The 100-row point is only informative: it uses a lighter
catalogue prefix and must not be compared as if it had the full workload mix.

## Matrix and method

- Inventory comes from `PRODUCT_BINDING_SPECS`, restricted to fixed-income
  closed-form European swaptions; a CPU test checks all seven compositions.
- 64/128/256/512 threads. Dense grids and persistent caps of 64/256/1,024/4,096
  blocks are screened where distinct; large batches re-sweep all four thread
  counts with dense, 1,024- and 4,096-block grids.
- Catalogue sizes: 100, 1,000, 16,384, 65,536, 262,144 and 1,048,576.
  The 1,000 original aligned model/product/curve rows repeat cyclically in
  larger buffers, without reordering. These are throughput fixtures, not a
  million independent parameter draws or a preparation-cache qualification.
- At 16,384 prices, separate short (≤8 payments, 470 source rows) and long
  (≥100 payments, 40 source rows) fixtures exercise calendar sensitivity.
  Actual short maximum is five payments; long maximum is 600. OU short
  screening favors scalar (0.093 vs 0.156 ms); long favors cooperative
  execution. These profile results are exploratory, not three-run retunings.
- Screening uses one warmup and three samples. Confirmation fixes five
  excluded warmup groups, 21 measured groups and three independent payer
  processes per composition. A fixed seed changes configuration order between
  campaigns. Short calls are grouped, with the recorded operation count fixed
  from the preceding stage. There is no outlier removal or retry-until-stable.
- Every measured job compares every output with its original reference tile.
  The cross-mode budget was fixed before the pilot: `2e-6 + 2e-5*abs(price)`.
  This is a comparison budget, not an independent proof of price accuracy.
- CUDA events, raw host-clock samples and enclosing API time remain separate.
  Do not add GPU time to API time. Events include submission gaps between
  grouped launcher calls. One-time expansion/allocation/H2D preparation and
  output allocation/D2H are recorded separately; JSON parsing, serialization,
  publication and cold full-generator time are not measured here.
- All launches use the existing generic closed-form kernels and product-owned
  policies. The scalar overload of the cooperative policy is instantiated by
  the benchmark; this is not a direct timing of catalogue executable startup.

At 2²⁰ prices, owned device input/output buffers occupy 40–68 MiB: OU 40,
CIR/Vasicek 44, Hull–White NS/Svensson 52/60, CIR++ NS/Svensson 60/68.
This excludes driver/context memory. Cooperative workspace uses up to 7,200
shared bytes per resident block, not a global allocation per price. Raw
records attach runtime resources and matching compiled-symbol SASS inventories;
they show 37–85 registers/thread, zero stack/local allocation and zero SASS
LDL/STL instructions. Local-memory traffic counters and achieved occupancy
were not profiled.

## Numerical and statistical limits

At the time of this campaign, `NUM-021` was open. CIR reference prices were non-finite at catalogue IDs
`000917`, `000949`, `000953`, `000958`, `000989`, `000993` in both scalar and
cooperative reference calculations. The Vasicek scalar reference is non-finite
at `000997`. The final targeted diagnostic also confirms that Vasicek's
cooperative value is non-finite there.
All seven belong to the stress region. Their repeated rows remain explicitly
`reference_incomplete`; they are neither removed nor counted as successful
comparisons. No tolerance, stress parameter or mathematical kernel was changed.

This preserves the fail-closed rule of `NUM-001`: a rejected root must not be
turned into a plausible finite price merely to finish a performance campaign.
The cause of these rejections and independent numerical certification remain
separate work. Likewise, noisy timings remain inconclusive even when prices
agree. `PERF-022` records completion of the experiment, not qualification of
every candidate or closure of the historical MC/LSM scaling findings.

## Reproduce and evidence

The completed matrix contains **1,624 configuration measurements**: 56 pilot,
588 screening, 364 calendar-profile, 280 large-batch, 252 payer-confirmation
and 84 receiver-check records. Of these, 1,278 pass the cross-mode comparison
and 346 retain an incomplete reference. Maximum finite-row price difference
is `9.536743e-7`. Of 84 independently confirmed payer geometries, 51 pass
this experiment's numerical and 5% within/between-run CV gates.

Portable evidence:

- [Summary and fingerprints](../../tests/performance/reports/jamshidian-strategy-sm89-2026-09-08/summary.json)
- [All geometry comparisons, CSV](../../tests/performance/reports/jamshidian-strategy-sm89-2026-09-08/summary.csv)
- [Raw timing and resource records](../../tests/performance/reports/jamshidian-strategy-sm89-2026-09-08/measurements.ndjson)
- [Plans, source hashes, reference tiles and telemetry](../../tests/performance/reports/jamshidian-strategy-sm89-2026-09-08/provenance.ndjson)
- [Seven numerical failures: original rows and both modes](../../tests/performance/reports/jamshidian-strategy-sm89-2026-09-08/numerical-diagnostics.json)

The measured executable is archived separately in
`build-dev/jamshidian-strategy-snapshot-01`, SHA-256
`154f173bf24d2a744463e2de4d929b1d9839b703fc672146815be27cefb5bce3`.
Its source/input archive has SHA-256
`4261bb32f353aca4249358ae13739d660f6c47a6e5bb0c7abbd33d6b695b1481`.
After timing ended, the benchmark was formatted and its diagnostics augmented
to expose actual values where the reference is invalid. A separate 56-job
check reproduces all seven reference tiles; its timings do not replace any
of the three confirmations. Ten CPU tests and the repository layout check pass.

Build with the desired native GPU architecture and no fast math:

```bash
cmake --build build-dev --target ai_factory_jamshidian_strategy_benchmark -j1
python tools/performance/run_jamshidian_strategy.py --stage pilot --output build-dev/jamshidian-pilot
python tools/performance/run_jamshidian_strategy.py --stage screen --source build-dev/jamshidian-pilot --output build-dev/jamshidian-screen
python tools/performance/run_jamshidian_strategy.py --stage profiles --source build-dev/jamshidian-screen --output build-dev/jamshidian-profiles
python tools/performance/run_jamshidian_strategy.py --stage large --source build-dev/jamshidian-screen --output build-dev/jamshidian-large
python tools/performance/run_jamshidian_strategy.py --stage confirm --source build-dev/jamshidian-screen --source build-dev/jamshidian-large --repeat 0 --output build-dev/jamshidian-confirm-0
```

Repeat the last command with independent IDs/output directories 1 and 2;
`--side receiver` adds the complementary-side check. Explicit interruption
recovery uses `--start-index` and a fresh directory; it is not an automatic
retry. Each plan has a 1,800-second watchdog. External power, foreign compute
and hardware power brake are execution guards. Temperature and thermal
regulation are telemetry only; no cooling loop or firmware setting is used.
Power-envelope excursions disqualify timing persistently without killing work.

Raw evidence lives under `build-dev/jamshidian-strategy-*`: `pilot-03`,
`screen-02`, `profiles-01`, `large-01`, `confirm-payer-01/02/03`,
`confirm-receiver-01` and `confirm-receiver-resume-01`. Manifests retain exact
jobs, source/input/binary hashes, Git status/diff and pre/live/postflight.
The initial CIR pilots and interrupted WSL-controller runs are retained but
not used to choose a favorable timing. The receiver continuation is one
complementary check, not three independent receiver confirmations.

The RTX 4090 Laptop SM89 results are local evidence, not universal defaults.
Run the same matrix on each target GPU/toolchain before changing production
geometry. No new mathematical factorization or kernel merger follows from
this experiment.

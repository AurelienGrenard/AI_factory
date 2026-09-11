# Catalogue generation readiness — SM89 — 2026-09-08

Scope: actual ordered catalogue rows, **1,000 prices × 2²⁰ paths per price**
for MC/LSM, no paths for closed form. This targeted pass finishes launch-plan
checks and staged native pilots; it is not independent price certification,
full catalogue publication, or qualification of the entire multi-size matrix.
All 29 runtime checks and seven staged native targets are complete.

## Kou: bounded regression correction

The original Kou LSM implementation changes eight of the 1,000 put prices
beyond `1e-6 + 1e-6 * abs(reference)`, with maximum absolute difference
`8.2850456e-6`. Identical forward paths do not prevent this: the regularized
Gram condition number reaches about `6e10`, amplifying FP64 summation rounding.
The earlier `NUM-017` qualification around `1e8` did not cover that domain.

Kou now opts into one normal-residual correction in the existing shared
regressor. It retains the basis, ridge, trajectories, Philox mapping and FP32
path/cashflow storage. Two compile-time kernel specializations are added per
backward date; the original workspace is reused. Other models do not pay for
this pass. The [LSM contract](../cuda/american-and-bermudan-pricing-contract.md)
owns the algorithm, diagnostics and failure rules.

The comparison reference forms compensated high/low statistics on the same
GPU paths, solves them with pivoted CPU binary128 elimination, then rounds
coefficients to FP64 before native exercise decisions. This is an independent
linear solve, **not an independent financial pricer**. Rounded-Gram reference
and compensation-only experiments were insufficient; mere agreement between
launch geometries was not the acceptance criterion.

- Six prototype geometries at 1,000 × 2²⁰: 128 threads with 32/64/128 blocks
  per price, 256 with 32/64, and 512 with 32. Every price and standard error
  matches the high-precision reference bit for bit.
- The integrated public launcher also matches all 1,000 prices and errors.
- The rebuilt production-plan probe repeats this exact match in the final
  29-case campaign: 19.158 s median GPU, 0.464% CV, two excluded warmups and
  three measurements. Its raw `kou__lsm/stdout.ndjson` SHA-256 is
  `4b0339dfffa7ea133a556be3a73d7a23fdf9382de5e282ff56e9015c60605c23`.
  This is a current dataset runtime, not another before/after cost pair.
- Permanent tests cover the eight affected rows in call/put at three
  geometries, and the regressor against a compensated CPU `long double`
  reference. Maximum scaled coefficient error is `8.1e-10` within a fixed
  `5e-7` budget. Empty/insufficient candidate sets and fatal correction
  failures check that each date has exactly one final diagnostic.
- The regressor test passes Compute Sanitizer memcheck and racecheck with
  zero errors/hazards. These checks do not certify other GPU architectures.

### FP64 cost and compiled resources

This is a numerical correction, not a speedup. The initial diagnostic full
catalogue comparison was 9.60 s before versus 13.75 s for residual refinement;
compensating all Gram statistics cost 26.94 s. These local measurements are
not a retuning baseline. The fresh check fixes one excluded warmup and three
measurements per process, with order before/corrected/corrected/before/before/corrected:

| Passage | Version | Median GPU, s | GPU CV |
|---|---|---:|---:|
| 1 | before | 9.605 | 0.17% |
| 2 | corrected | 16.062 | 5.50% |
| 3 | corrected | 17.144 | 0.79% |
| 4 | before | 12.651 | 1.85% |
| 5 | before | 12.674 | 2.37% |
| 6 | corrected | 18.512 | 3.52% |

All corrected prices/errors match the reference exactly; all old runs retain
the same eight out-of-budget prices. Paired GPU cost ratios are 1.672, 1.355
and 1.461. Passage 2 fails the 5% CV bound. Frequencies change while heating,
so these observations establish a material cost, **not a precise universal
percentage or an optimal profile**. No passage is removed or retried.

A separate single-call Nsight Systems trace attributes 4.242 s to residual
accumulation and 0.069 s to its small solve: 30.3% of corrected kernel time.
The original Gram accumulation takes 6.559 s (46.1%); path simulation 0.930 s
(6.5%). These are instrumented kernel durations, excluding CPU submission
gaps, not another uninstrumented dataset timing. The expensive added FP64
work is the residual over paths, not the tiny matrix solve.

The seven existing kernels retain their register counts: 40, 70, 86, 82, 39,
23 and 35. Residual accumulation uses **56 registers/thread**, and correction
solve **104**. All nine exact linked symbols have zero stack/local allocation
and zero SASS local-load/store instructions on SM89. The full 1,000-row
integrated run uses 16 native batches, at most 239 resident prices and
14,190,752,752 workspace bytes; no additional global workspace buffer is
introduced. This does not include the driver/context or loaded kernel code.

Do not select the lowest result across separate warming periods. All six
prototype timing series are retained. The integrated numerical check has
GPU CV **5.0028%**, just outside the predeclared 5% bound; its timing remains
diagnostic even though its numerical outputs pass.

## Production-shaped runtime confirmation

All **29 predeclared cases** complete on the current launch plans, with finite
outputs and deterministic replay. The complete ordered catalogue is used once
per call; MC/LSM has `2^20` paths per price. Two warmups are excluded and all
three measurement samples are retained. No case is retried or selected from
alternative favorable periods; no compilation runs during timing.

The [dataset notebook](pricing-dataset-runtime-sm89.ipynb) and
[portable evidence](../../tests/performance/reports/pricing-dataset-runtime-sm89-2026-09-08.json)
separate GPU time, host API, preparation, copy and local JSON/YAML publication,
with actual grids, native LSM batches and FFT chunks. Summing the measured
generation phases for one dataset of each of these 29 pairs gives **16 min 57 s**;
this excludes benchmark repeats, parameter generation, upload and certification.
It is not a cold-process timing or a prediction for the entire catalogue.

CIR++ LSM sees a power-limit change. G2++/Hull–White caplets and Kou terminal
exceed 5% GPU CV. Their timings remain visible with warnings, not accepted
for retuning. Even cases without these alerts do not constitute protocol-v3
rebaseline or proof of multi-size scaling. `PERF-016/017/019` remain open.

## Native generation and resume

Five native price recipes pass at their full 1,000-row shape: Kou call/put,
CIR/CIR++ Jamshidian and G2++ European MC. Both Kou sample shapes pass at
3M rows. All fourteen canonical JSON/YAML destinations retain their hashes;
prices stay `pending / verified: false` in staging. Native Kou put prices and
errors exactly match the binary128 solve reference. Resuming the completed
sample campaign checks hashes and skips both generators without a GPU rerun.

The first unconditional sample attempt was rejected because the host guard
counted only free pages, excluding reclaimable cache. `STRUCT-024` corrects
this with Linux `MemAvailable` and moves the guard before allocations, retaining
70% RAM / 85% VRAM limits and a conservative fallback. It changes no `src`
kernel or random draw. The failed attempt is retained; rebuilt sample binaries
have their own campaign. The conditional sample body is byte-identical across
the fix, and the full unconditional preflight replays exactly at 256/128 threads.
These native pilots and their process/check times are functional evidence,
not substitutes for the qualified-clock requirements of performance tuning.

[Native pilot evidence and resume checks](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/native-generation-pilot.json)
close `STRUCT-023/024`. The controller contract owns staging and publication;
the sample contract owns memory-availability semantics.

## Evidence and reproducibility

The complete parameter/price/sample/test build is current (Ninja dry-run has
no work remaining); the final main CTest suite passes **86/86**, excluding
independent validation. All 50 sample binaries were rebuilt after the host
memory fix and pass their 1,000-row smoke test. Earlier binary/log evidence
remains archived separately. These are functional checks, not full-shape
sample timings.
The CPU launch inspector accepts all 426 identities at 1,000 and 1M prices
(852 checks), with `2^20` pricing paths where applicable. Offline inspection
does not prove device-memory feasibility.

- [Numerical outputs and complete geometry timing samples](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/lsm/kou-normal-residual.ndjson).
- [Exact linked-symbol/runtime resources](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/lsm/kou-normal-residual-resources.ndjson).
- [All fresh cost samples, telemetry ranges and kernel-phase attribution](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/lsm/kou-normal-residual-cost.json).
- [All current sample smoke-test binaries/logs](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/sample-smoke.json).
- [852 CPU launch-plan checks](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/launch-plan-inspection.json).
- Current source snapshot, experimental reference sources, sanitizer logs and
  frozen integrated binary: `build-dev/kou-lsm-launch-confirmation.zpDtZU`.
  The 11 exploratory report files are retained in its
  `exploratory-lsm-reports.tar.gz`, SHA-256
  `9daf2a7a2fd160c09f92add1b38f7c366c548c99324c3d226af3dc316e615075`.

The commit alone does not reconstruct this dirty worktree. Snapshot and binary
hashes remain part of the evidence. Prototype timing, native generation wall
time, GPU timing and independent validation have different scopes and must
not be merged. `NUM-017` and the OU test-signature fix `BUILD-009` are closed;
`STRUCT-023/024` are also closed; wider scaling tracking continues under
`PERF-016/017/019`.

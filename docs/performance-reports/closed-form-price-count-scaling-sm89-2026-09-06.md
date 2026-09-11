# Closed-form price-count scaling — SM89 — 2026-09-06

## Decision

Keep the current 256-thread default. The 128/256/512-thread sweep does not
show a repeatable end-to-end gain large enough to justify an SM89-specific
override. For 1,000 prices, native publication costs 5.0--9.8 ms while the
normalized launcher cost is at most 0.201 ms. The pipeline is publication- or
preparation-dominated, not register- or occupancy-limited.

This report closes `PERF-018` for the requested 1/16/1,000-price scope. It does
not predict or certify one million prices, another GPU architecture, or a
protocol-v3 regression baseline.

## Coverage and method

The capability manifest selected every available analytical composition: the
Black--Scholes European call and one caplet call for CIR, G2, G2++ with
Nelson--Siegel and Svensson, Hull--White with Nelson--Siegel and Svensson,
Ornstein--Uhlenbeck, and Vasicek. Each launcher used the 1,000 aligned catalogue
rows at 1, 16 and 1,000 prices with 128, 256 and 512 threads. This gives 81
measured jobs.

Each timing sample groups 1,024 identical launcher calls and reports the time
normalized to one call. Every job has one warm-up and three measured samples.
GPU events, the synchronized public API, one-time input preparation, D2H copy,
native dataset publication, tracked device allocations, driver/context memory,
compiled resources and telemetry are recorded separately. Publication writes
only below the fresh evidence directory.

All 81 within-shape comparisons are bitwise identical, deterministic replay
passes, and every value is finite. Across all kernels and geometries there are
23--40 registers per thread, zero stack frame, zero local memory and zero
static shared memory. Tracked device allocations are 32--72 kB; the roughly
1.40 GB CUDA context/pool is reported separately.

## Representative 1,000-price results

The table uses the current 256-thread default. CV is the coefficient of
variation of the three normalized GPU samples.

| Composition | Registers | GPU ms | CV | D2H ms | Publication ms | One-time preparation ms |
|---|---:|---:|---:|---:|---:|---:|
| Black--Scholes | 23 | 0.005232 | 4.30% | 0.038 | 5.040 | 156.4 |
| CIR | 40 | 0.200265 | 0.01% | 0.151 | 6.017 | 278.0 |
| G2 | 38 | 0.007353 | 1.63% | 0.051 | 7.202 | 366.8 |
| G2++ / Nelson--Siegel | 40 | 0.009347 | 22.33% | 0.088 | 9.770 | 158.2 |
| G2++ / Svensson | 39 | 0.006102 | 1.71% | 0.044 | 9.574 | 162.8 |
| Hull--White / Nelson--Siegel | 29 | 0.006598 | 2.40% | 0.051 | 8.055 | 159.9 |
| Hull--White / Svensson | 35 | 0.006268 | 4.64% | 0.047 | 8.821 | 150.6 |
| Ornstein--Uhlenbeck | 28 | 0.006149 | 1.83% | 0.039 | 6.013 | 181.9 |
| Vasicek | 30 | 0.006084 | 3.47% | 0.030 | 6.644 | 825.7 |

G2++ / Nelson--Siegel was repeated independently because its first 256-thread
CV was above 5%. At 1,000 prices, the confirmation produced 0.006854 ms at 128
threads (4.31% CV), 0.010250 ms at 256 threads (4.56% CV) and 0.005964 ms at
512 threads (7.18% CV). The ordering is therefore not robust. Even its largest
kernel difference is negligible beside the 8--10 ms publication cost, so no
production geometry is changed.

The nearly flat costs of the shortest formulas are expected once one launch
fills only a few blocks: launch latency dominates. CIR scales visibly because
its row work includes the caplet schedule. No claim of linear price-count
scaling is made from sub-10-microsecond kernels.

## Provenance

- Revision: `872a986b1f0947a1a832af0615ffc6d80dbedb81`, dirty worktree captured in
  both immutable campaign snapshots.
- GPU: NVIDIA GeForce RTX 4090 Laptop, compute capability 8.9; CUDA compiler
  13.3.73, runtime 13.3 and driver API 13.2 as reported by the probes.
- Main evidence: `build-dev/pricing-scaling-closed-form-final-01`; summary
  SHA-256 `1de1dc68ac2d71efc88d8489df2c06fa0a7e68e46c02767c7e68e1efa7021323`,
  plan SHA-256 `f9dd13e9ba5d525cc1352b74a22f7a617518143446d95ceb9211830e3d610bdb`,
  snapshot SHA-256 `65705a297268605a764b932c75b3dbacc39fc5c63216fa9cbe41372ac156f894`.
- Confirmation: `build-dev/pricing-scaling-closed-form-confirmation-01`;
  combined summary SHA-256
  `13151584be6b2ad47dbd7756e62c08cb13a9238da661f445910cffdb29b0b0b8`,
  plan SHA-256 `3ceee06f3294fe18b60f86c2cf7ee0059d6ab0b256c709e48b2d2c4d9e76b08d`,
  snapshot SHA-256 `3b710e1b26722a62feeed1a5ecec5f7edcdb65659a9afb6516f69c158b19ecf3`.
- Each case directory records its exact binary and catalogue-input hashes,
  pre/postflight, telemetry, raw samples, resource inventory and outcome.

Re-run on each target GPU profile before changing a production default. Reopen
`PERF-018` if an analytical composition disappears from the generated matrix,
outputs differ across geometry, local memory or spills appear, publication is
no longer isolated, or a measured end-to-end gain of at least 5% is repeatable.

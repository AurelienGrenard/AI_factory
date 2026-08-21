# Model-sample dataset generation

This document fixes the common contract for the generative-training datasets
under `catalog/model/<asset_class>/<model>/samples/`.

## Dataset shape

Every model exposes two production recipes of exactly 3,000,000 terminal
samples:

```text
samples_01:    12,000 model parameter rows * 250 paths = 3,000,000 samples
samples_02: 3,000,000 model parameter rows *   1 path  = 3,000,000 samples
```

The two recipes use independent parameter, maturity, and sample seeds. They
share the same plausible parameter laws so that their difference is purely
the conditional sampling layout.

Every row has its own maturity `T = maturity_days / 360`, where
`maturity_days` is sampled uniformly from all 631 integer days in `[90, 720]`.
Thus the support is exactly `{90/360, ..., 720/360}`.

The final JSON is flat. Every entry in `samples` is an autonomous training row
containing `parameters`, `maturity_days`, `T`, and scalar terminal `values`.
`samples_01` intentionally repeats each parameter object 250 times. There is
no `models` table, `model_id` join, or standalone intermediate parameter JSON.

## Source of truth

`generator.cpp` is executable source of truth. It performs these operations in
order:

1. generate plausible parameters directly in one contiguous typed vector;
2. generate one discrete-uniform maturity per flattened sample;
3. group rows temporarily by day and invoke the fixed-`T` CUDA kernel;
4. scatter terminal outputs back into their original random order;
5. stream the complete JSON without constructing a 3M-row DOM;
6. write the adjacent YAML describing the recipe that was executed.

The YAML never drives parameter generation. Its parameter bounds are the
plausible core bounds used for the ordinary 90% of the corresponding pricing
parameter dataset; the 10% stress tail is intentionally excluded from GAN
training data.

Parameter and maturity generation use independent Philox-4x32-10 streams on
the host. The integer maturity draw uses rejection sampling, so the 631 days
are discrete-uniform without modulo bias. CUDA sampling uses a distinct key
per flattened training row. Fixed seeds make all three stages reproducible
independently of persistent-grid geometry.

## CUDA launch contract

Every model provides `sample.cuh` and `sample.cu` with terminal and regular
calendar launchers. The signatures retain this common order:

```text
device models, M, P, time/calendar, optional target_dt,
sample offset, launch sample count, threads, blocks, seed, outputs
```

Exact Markov and Levy models use exact transitions directly to `T` and
therefore have no `target_dt`. Heston, Bates, CEV, and Schobel-Zhu use
`target_dt = 1/360`. Rough Bergomi uses the same public sampling surface with a
caller-owned history workspace and the hybrid scheme at `1/360`.

Ordinary exact and time-stepped kernels use a capped persistent thread grid:
each physical thread handles one or more flattened samples through a
grid-stride loop. Rough Bergomi uses a capped persistent block grid: one block
prepares a model's shared hybrid coefficients while its threads cover that
model's conditional paths. In both cases the cap is 16 blocks per
multiprocessor and
logical sample indices are independent of launch geometry.

Rows are bucketed by their integer maturity only while launching the fixed-`T`
kernels. Outputs are scattered back to the original Philox-defined row order
before the JSON is streamed.

## Memory and smoke tests

The generator estimates contiguous model and observable storage before any
large allocation. It refuses a plan above 70% of currently available host RAM
or 85% of available device memory. This guard is diagnostic protection, not a
batching policy; the intended 3M rows fit in one in-memory execution on the
target GPU, with one fixed-`T` launch for each non-empty maturity bucket.

Every generator accepts `--smoke-test`. That mode preserves the production
paths-per-model layout (4 * 250 for `samples_01`, 1,000 * 1 for `samples_02`),
writes exactly 1,000 rows below
`/tmp/ai_factory_sample_smoke/`, rejects non-finite outputs, then parses the
JSON again and verifies the flat-row schema, maturity bounds, identity
`T = maturity_days / 360`, dimensions, and finite outputs. The YAML still
documents the production 3M recipe.

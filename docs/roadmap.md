# AI Factory Roadmap

This roadmap tracks the next engineering steps for reproducible quantitative
finance datasets. Historical implementation notes live in `docs/journal.md`.

## Current Baseline

The active reference cases are production-audit datasets using the `01`
database convention, for example:

```text
heston_01 x lookback_fixed_calls_01
heston_01 x asian_arithmetic_calls_01
rough_bergomi_01 x lookback_fixed_calls_01
```

with four price engines:

- Python CPU / PyTorch RNG;
- Python GPU / PyTorch RNG;
- C++ CPU / project Philox-4x32-10;
- C++ CUDA / project Philox-4x32-10.

and four price-plus-delta engines using central finite differences with common
random numbers.

Production databases live under `registry/production`; validation slices and
four-engine audits live under `registry/validation`.

Current guarantees:

- C++ CPU and C++ CUDA Philox engines match to floating-point precision.
- Python engines are statistical references, not pathwise replicas of C++.
- Heston uses QE-M for the active result databases.
- Result YAML files own engine, path count, time-grid rule, source files,
  references, and output definitions.

## Near-Term Priorities

1. Repeat the Heston/lookback pattern for the next payoff.
2. Keep one compact validation notebook under
   `notebooks/validation/<model_family>` per active `(model, product, output)`
   pair.
3. Add tests before expanding a result pattern to many payoffs.
4. Keep C++ Philox CPU/GPU pathwise checks as the core reproducibility guard.

## Large Dataset Runner

Million-row production databases need a dedicated runner rather than a single
in-memory result generation call.

Required behavior:

- assign row ids and seeds before any chunking;
- compute `step_count` from the YAML `time_grid` rule;
- group rows by exact `step_count`;
- split each group into GPU-memory-safe chunks;
- append outputs without changing row order or seed mapping;
- record timings per engine clearly enough to separate cold wall time, hot wall
  time, and CUDA kernel time where useful.

Exact grouping and chunking are execution mechanics. They do not need extra YAML
fields because they do not change the numerical grid.

If a future run bucketizes or caps `step_count`, that changes the numerical grid
and must be declared explicitly in the result YAML as a different `time_grid`
rule.

## Performance Work

Avoid micro-tuning unless a profile points to a clear bottleneck.

High-impact candidates:

- reusable CUDA workspace for Heston/lookback result generation;
- large-dataset chunked runner;
- optional benchmark harness for cold wall, hot wall, and kernel timings;
- memory-peak reporting for large runs.
- production notebooks under `notebooks/production` for large-run audits.

Not currently planned:

- mixed precision for Heston QE-M;
- aggressive PyTorch compilation/tuning;
- bucketed time grids hidden inside runners;
- changing the C++ Philox counter layout.

## Model And Product Growth

Each new payoff/model pair should include:

- model and product databases with compact YAML specs;
- four result databases when both Python and C++ paths exist;
- C++ CPU/GPU reproducibility tests when Philox engines exist;
- Python/C++ statistical consistency checks;
- a short validation notebook using the same visual style as the Heston
  lookback notebooks.

Rough Bergomi should add the hybrid-scheme paper reference when its active
result databases are promoted to the cleaned registry pattern.

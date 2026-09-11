# Volterra exploratory performance fixtures

These CSV files preserve the measurements that informed the first Volterra
engine choices on an NVIDIA GeForce RTX 4090 Laptop GPU (SM89). They are
supporting experiment records, not the current performance baseline and not a
portable tuning recommendation.

Use these files only to revisit the corresponding design question:

| File | Question covered |
|---|---|
| `rtx4090_laptop_primary.csv` | Relative throughput across the original Heston and rough engines |
| `rtx4090_laptop_dt_impact.csv` | Cost sensitivity to the tested time-step counts |
| `rtx4090_laptop_fft_boundaries.csv` | Runtime discontinuities at FFT-size boundaries |
| `rtx4090_laptop_saturation.csv` | Batch and chunk sizes needed to saturate the tested GPU |

The measurements predate performance protocol v3. In particular, they do not
have its three-campaign journal, environmental preflight, resource evidence or
blocking noise rules. Do not use their timings or time-grid choices as current
acceptance criteria.

For authoritative information, use:

- [`tests/performance/baseline_sm89_v3.json`](../../baseline_sm89_v3.json) for
  the current SM89 workload, observations and budgets;
- [`docs/performance-regression-protocol.md`](../../../../docs/performance-regression-protocol.md)
  for measurement and retuning rules;
- [`docs/model-sample-dataset-generation.md`](../../../../docs/model-sample-dataset-generation.md)
  for current sample-grid ownership;
- a new architecture-specific baseline before drawing conclusions for another
  GPU or toolchain.

The CSV files are immutable evidence. A new experiment belongs in a separately
named fixture set with its protocol and hardware scope recorded explicitly.

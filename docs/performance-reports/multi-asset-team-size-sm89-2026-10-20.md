# Multi asset TeamSize, SM89 — 2026-10-20

## Configuration

- GPU: NVIDIA GeForce RTX 4090 Laptop, compute capability 8.9.
- Build: Release, CUDA 12.9.41, architecture 89.
- One CUDA block processes 128 joint paths in the cross asset scan. The same Philox key, path, step and factor group are used for every team size.
- Times below are medians of the three CUDA event measurements within each test. They exclude CPU preparation and host transfers. The rough row includes volatility FFT preparation, joint scan and finalization.

| Workload | Paths | Steps | TeamSize 1 | TeamSize 2 | TeamSize 4 | TeamSize 8 | TeamSize 16 | TeamSize 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 Black–Scholes assets, fixed rate, worst of | 8,192 | 21 | 17.20 ms | — | 7.89 ms | — | — | — |
| 30 Black–Scholes assets, Vasicek rate, worst of | 8,192 | 21 | 0.356 ms | **0.177 ms** | 0.198 ms | 0.256 ms | — | — |
| 160 Black–Scholes assets, OU rate, worst of | 32,768 | 21 | 25.58 ms | 26.46 ms | 17.01 ms | 18.95 ms | **14.61 ms** | 19.14 ms |
| 30 Black–Scholes assets, rough Heston and Vasicek, worst of | 2,048 | 32 | 0.572 ms | — | **0.485 ms** | 0.700 ms | — | — |

Every row measures **one price**, not throughput across a dataset of prices. These measurements predate the batched cross-asset runner and use the then-default full split of paths across 128-thread blocks. The new default preserves full path splitting for one isolated price and chooses a bounded block count for batched prices. Use explicit geometry to reproduce a particular launch. The pure Markov and cross asset kernels have different launch geometries, so their absolute times should not be compared to each other. TeamSize comparisons **within a row** use identical inputs, seed and geometry. These measurements do not choose between one block per price and multiple blocks per price for a large dataset.

## Implementation and correctness

Each lane generates its Philox normal groups and calculates its assigned rows of the packed Cholesky product. When shared memory permits, a team stores the independent vector there, computes its assigned rows, then reuses the same buffer for the correlated vector. Each lane reads the factors of its assigned assets directly. The rate is advanced by lane 0, which also advances its assigned assets; its one-step integral is broadcast within the team. The payoff is reduced once per joint path.

The launcher selects shared or local state storage according to the GPU's per-block shared memory capacity. If the correlated vector does not fit, it uses the synchronized factor access path. This avoids making a team size invalid merely because the matrix is large.

- A direct 128-path rough Heston + Black–Scholes + Vasicek test compares every payoff for TeamSize 1, 2, 4 and 8 **exactly**.
- Worst-of and basket barrier tests compare price and standard error across team sizes; the largest test covers 160 factors and 32,768 paths.
- The 30-asset pure Markov test preserves exact price and standard error across its tested team sizes.

These timings are specific to SM89. The default remains TeamSize 1; choose a larger specialization for a measured workload.

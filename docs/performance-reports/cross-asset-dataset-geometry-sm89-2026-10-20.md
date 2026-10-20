# Cross asset dataset geometry, SM89 — 2026-10-20

## Question and setup

Compare one block per price with several blocks per price **at dataset scale**. Each launch batches price rows. `blockIdx.y` selects the price within a batch and `blockIdx.x` selects a group of Monte Carlo paths. A block has 128, 256 or 512 threads. Each path uses one, four or sixteen threads as indicated by `TeamSize`.

GPU: NVIDIA GeForce RTX 4090 Laptop, SM89. Release build, CUDA 12.9.41. The opt-in local benchmark (`maintainer/tests/performance/cross_asset_dataset_geometry.cu`, ignored by Git) uses Black–Scholes baskets with an OU short rate, worst-of-call products, eight scenarios, four maturities/strikes and two correlation matrices. Price rows rotate through these parameters. Correlations and model inputs are identical across geometries. A reference output is compared with **every** price and standard error from every geometry; the maximum observed difference was zero after the float outputs. Each listed time is the median of three CUDA event measurements and covers the simulation and any required finalization kernels. CPU preparation, allocations and transfers are excluded.

When a price uses one block, that block writes price and standard error directly. Multiple blocks write partial moments and use a batched finalization kernel. Thus the one-block strategy is not charged for an unnecessary second launch.

## Results

| Prices | Assets | Paths/price | TeamSize | Threads/block | Blocks/price | Rows/launch | GPU time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 2 | 128 | 1 | 128 | 1 | 256 | 11.69 ms |
| 100,000 | 2 | 128 | 1 | 128 | 1 | 1,024 | **7.80 ms** |
| 100,000 | 2 | 128 | 1 | 256 | 1 | 1,024 | 9.08 ms |
| 100,000 | 2 | 512 | 1 | 128 | 1 | 4,096 | **12.99 ms** |
| 100,000 | 2 | 512 | 1 | 128 | 2 | 4,096 | 14.33 ms |
| 10,000 | 2 | 1,024 | 1 | 128 | 1 | 256 | 4.97 ms |
| 10,000 | 2 | 1,024 | 1 | 256 | 2 | 256 | **4.34 ms** |
| 2,000 | 30 | 1,024 | 1 | 128 | 1 | 256 | 14.95 ms |
| 2,000 | 30 | 1,024 | 1 | 128 | 8 | 256 | 9.25 ms |
| 2,000 | 30 | 1,024 | 4 | 128 | 1 | 256 | 11.79 ms |
| 2,000 | 30 | 1,024 | 4 | 128 | 4 | 256 | **8.79 ms** |
| 2,000, distinct matrices | 30 | 1,024 | 1 | 128 | 1 | 256 | 14.94 ms |
| 2,000, distinct matrices | 30 | 1,024 | 1 | 128 | 8 | 256 | **9.26 ms** |
| 200 | 160 | 2,048 | 16 | 512 | 1 | 128 | 72.50 ms |
| 200 | 160 | 2,048 | 16 | 512 | 2 | 128 | **62.27 ms** |

Full measurements, including all variants and wall times, are in the ignored `project_journal/evidence/performance/cross-asset-dataset-geometry-2026-10-20/` directory.

Launching 10,000 two-asset prices **one price per launch** took 159.28 ms of GPU event time at 128 paths per price. Batching 1,024 prices per launch, still with **one block per price**, took 0.71 ms. The gain comes primarily from removing thousands of launches, not from splitting a price across blocks.

## Decision and limits

- Batch multiple prices in one launch. The markovian cross asset runner now does this; the former per-price launch was inappropriate for large datasets.
- Keep both one and multiple blocks per price. With many small prices, one block is best or close. For the measured 30- and 160-asset cases, additional blocks still help even though the price count can occupy the GPU.
- Set `TeamSize`, block size and blocks per price from the workload, not from the price count alone. `TeamSize` remains a compile-time template parameter because it controls lane ownership. Blocks per price and threads per block are runtime launch parameters; the default uses a bounded heuristic for batches, preserves full path splitting for one isolated price, and explicit geometry remains available for qualification.
- Keep the two-dimensional grid: `blockIdx.x` selects the path block and `blockIdx.y` the price row. With one block per price, `blockIdx.x` is always zero; this allocates no idle blocks or threads. A flattened one-dimensional grid has no demonstrated advantage here.
- Rough volatility preparation still uses its existing per-price FFT path. This study does not qualify rough dataset throughput or a batched rough FFT engine.

The main synthetic datasets reuse eight model scenarios and two matrices. The separate 2,000-row check uses 2,000 distinct positive-definite correlation matrices and preserves the multi-block ranking. No measurement here covers a catalogue with 100,000 distinct model and correlation rows. Requalify on such a catalogue before treating the measured milliseconds as production throughput.

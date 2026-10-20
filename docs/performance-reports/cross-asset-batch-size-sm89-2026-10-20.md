# Cross asset batch size, SM89 — 2026-10-20

## Method

The local benchmark `maintainer/tests/performance/cross_asset_dataset_geometry.cu` varies only rows per launch within each topology. It uses an RTX 4090 Laptop, CUDA 12.9.41, a Release SM89 build, Black–Scholes baskets, an OU short rate and worst-of-call products. GPU times include the pricing kernel and its finalizer when needed; host preparation, allocations and transfers are excluded. Every price and standard error matched the reference exactly after float conversion.

For the 65,535-row, two-asset case, each configuration was measured in three independent runs in the listed order and three more in reverse order. Each run reports the median of three CUDA event timings. The table reports the median of those six values. The 30-asset cases use three independent runs. Raw CSV files are in the ignored `project_journal/evidence/performance/cross-asset-batch-scaling-2026-10-20/` directory.

## 65,535 prices, two assets, 512 paths per price

TeamSize is 1 throughout. The columns correspond to 16, 4 and 1 pricing launches respectively.

| Threads/block | Blocks/price | 4,096 rows | 16,384 rows | 65,535 rows |
|---:|---:|---:|---:|---:|
| 128 | 1 | 8.55 ms | 7.99 ms | 7.89 ms |
| 128 | 2 | 9.77 ms | 9.23 ms | 9.29 ms |
| 256 | 1 | 9.22 ms | 8.93 ms | 8.76 ms |

Within one topology, increasing a lot from 4,096 to 16,384 rows saves about 3–7%. Moving from 16,384 to one 65,535-row launch is small and not consistently favorable across topologies.

## 65,535 prices, 30 assets, 512 paths per price

TeamSize is 1 and each block has 128 threads.

| Blocks/price | 4,096 rows | 16,384 rows | 65,535 rows |
|---:|---:|---:|---:|
| 1 | 132.15 ms | **127.71 ms** | 128.35 ms |
| 4 | 133.24 ms | **131.88 ms** | 132.21 ms |

The difference between four launches and one is below 1% here. Sixteen launches are 1–3% slower. Thus a full-grid launch has no established performance advantage over four large launches for this more expensive workload.

## Smaller dataset: 2,000 prices, 30 assets, 1,024 paths per price

Every block has 128 threads. Here the batch size also controls how many blocks the GPU can schedule from a launch.

| TeamSize | Blocks/price | 256 rows | 1,000 rows | 2,000 rows |
|---:|---:|---:|---:|---:|
| 1 | 1 | 13.37 ms | 8.70 ms | **7.83 ms** |
| 1 | 4 | 9.18 ms | 7.66 ms | **7.34 ms** |
| 4 | 4 | 9.01 ms | 7.59 ms | **7.42 ms** |

Small batches can matter much more when they expose too few blocks at once. These measurements do not isolate block availability from launch overhead, but they show that the batch size cannot be chosen independently of blocks per price and total price count.

## Choice

Keep the current two-dimensional mapping: `blockIdx.x` selects a path block and `blockIdx.y` a price row. A batch of at most 65,535 rows fits in one pricing launch without changing it. The current 4,096-row default is conservative. Around 16,384 rows is a good performance candidate for these workloads while retaining four progress points over 65,535 prices. The moment buffer for multi-block pricing grows as `rows_per_launch × blocks_per_price`; any global default increase should retain a device-memory bound for larger path counts. No production default is changed by this benchmark.

The cross-asset runner does not currently publish a per-batch progress log, although its batched loop could do so. The mono-asset MC runner already uses batches for progress and checkpoints.

## High path count: 262,144 and 1,048,576 paths per price

A follow-up holds `TeamSize=1`, 128 threads/block and the blocks-per-price choice fixed within each row. The 512-price experiment uses 128 blocks per price, so even its 32-row batches launch 4,096 blocks. The 32-price experiment uses 512 blocks per price; it checks what happens when a nominally large path count is split into batches of only two prices. Every configuration again produced identical float prices and standard errors.

| Prices | Paths/price | Blocks/price | 16 launches | 4 launches | 1 launch |
|---:|---:|---:|---:|---:|---:|
| 32 | 262,144 | 512 | 2.945 ms | 2.312 ms | 2.216 ms |
| 32 | 1,048,576 | 512 | 9.009 ms | 7.793 ms | 7.611 ms |
| 512 | 262,144 | 128 | 38.346 ms | 37.552 ms | 37.111 ms |
| 512 | 1,048,576 | 128 | 142.079 ms | 137.584 ms | 136.895 ms |

The 512-price, million-path row combines three forward-order and three reverse-order independent runs; other rows use three runs. Every run reports the median of three CUDA event timings, and the table reports the median across runs. GPU clock variation is visible in the raw CSV, so differences below roughly 1% should not drive a launch-policy change.

For 512 prices, four launches and one differ by about 0.5–1.2%; even sixteen launches add only about 3–4%. This supports keeping a modest number of large batches when many paths are priced. More paths per price alone do not erase the cost of *tiny* batches: the 32-price case still pays appreciably for sixteen launches containing just two prices each. These runs remain synthetic and do not substitute for production-catalogue qualification at thousands of prices and hundreds of thousands of paths.

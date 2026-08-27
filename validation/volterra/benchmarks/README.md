# Volterra CUDA kernel timing report

## Protocol

Measurements were taken on an NVIDIA GeForce RTX 4090 Laptop GPU with 76 SMs,
compute capability 8.9 and 16 GiB of VRAM. Every reported price uses 1,048,576
paths.
CUDA events surround only the production launcher calls; host preparation,
allocation and host/device copies are excluded. Each configuration is warmed
up and the median of three measured launches is retained.

The throughput configurations are deliberately model-specific:

- Heston QE-M: 128 concurrent prices, 1,024 threads per price block;
- rough Heston N=7: 64 concurrent prices, 256 threads per price block;
- rough Bergomi/rough SABR cuFFTDx: one price, 65,536 paths per FFT chunk.

Heston and rough Heston use one persistent block per price and need a price
batch to fill the GPU. Rough Bergomi distributes one price over many FFT and
payoff blocks, so a single price already fills the device. Consequently,
isolated one-price latency is not a fair production-throughput comparison.

Build the isolated harness against the already built production libraries:

```bash
nvcc -ccbin=/usr/bin/g++-14 -std=c++23 -O3 -arch=sm_89 \
  -I. -Isrc \
  -I/path/to/mathdx/include \
  -I/path/to/mathdx/external/cutlass/include \
  -c validation/volterra/kernel_benchmark.cu \
  -o /tmp/volterra_kernel_benchmark.o

g++-14 -O3 /tmp/volterra_kernel_benchmark.o \
  build-dev/libai_factory_equity_heston_european_option.a \
  build-dev/libai_factory_equity_rough_heston_european_option.a \
  build-dev/libai_factory_equity_rough_bergomi_european_option.a \
  build-dev/libai_factory_equity_rough_sabr_european_option.a \
  build-dev/libai_factory_runtime.a \
  -L/usr/local/cuda/lib64 \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
  -o /tmp/volterra_kernel_benchmark
```

The executable syntax is:

```text
kernel_benchmark MODEL MATURITY_DAYS STEPS_PER_DAY PRICE_COUNT \
  TUNING REPETITIONS [PATH_COUNT]
```

`TUNING` is the thread count for Heston models and the path chunk size for
rough Bergomi and rough SABR. The default path count is 1,048,576.

The contractual calendar always uses 252 business days per year. The primary
policy uses two Markovian Heston steps per day (`dt=1/504`) but one step per day
for rough Heston and rough Bergomi (`dt=1/252`).

## Primary policy

Median kernel milliseconds per price from the complete maturity sweep:

| Maturity | Heston, 1/504 | Rough Heston N=7, 1/252 | Rough Bergomi FFT, 1/252 |
|---:|---:|---:|---:|
| 1 month | 1.03 | 1.06 | 1.00 |
| 3 months | 3.11 | 3.16 | 2.57 |
| 6 months | 7.08 | 7.74 | 4.94 |
| 1 year | 15.66 | 16.18 | 10.96 |
| 2 years | 34.16 | 32.33 | 25.99 |
| 5 years | 88.49 | 80.81 | 81.78 |

Relative to Heston in this sweep, rough Heston ranges from 0.91x to 1.09x and
rough Bergomi from 0.70x to 0.97x. The rough models are therefore not four
times slower in the production-throughput regime.

Laptop boost clocks introduce visible run-to-run variation in the persistent
one-block-per-price kernels. A back-to-back five-repetition check at one year
gave 13.14 ms for Heston, 12.70 ms for rough Heston and 11.01 ms for rough
Bergomi. Across the two campaigns, the useful one-year factor estimate is
0.97x--1.03x for rough Heston and 0.70x--0.84x for rough Bergomi relative to
Heston under the proposed policy.

## Effect of `dt` at one year

| Model | `dt=1/252` | `dt=1/504` | Cost multiplier |
|---|---:|---:|---:|
| Heston | 6.50 ms | 13.14 ms | 2.02x |
| Rough Heston N=7 | 12.70 ms | 26.70 ms | 2.10x |
| Rough Bergomi FFT | 11.01 ms | 25.51 ms | 2.32x |

At equal `dt=1/252`, rough Heston costs 1.95x Heston and rough Bergomi costs
1.70x Heston. At equal `dt=1/504`, the factors are 2.03x and 1.94x. The primary
policy hides most of that structural cost because it intentionally gives
ordinary Heston twice as many time steps.

On this one-year parameter point, changing from daily to twice-daily steps
changed the raw Monte Carlo prices by `-4.42e-5` (Heston), `+1.26e-4` (rough
Heston) and `-1.20e-4` (rough Bergomi). These are only timing-run diagnostics:
the same random seed makes the estimates correlated, so a dedicated common-
random-number convergence test is still required before turning daily rough
steps into a global accuracy policy.

The slight super-linearity of rough Bergomi comes from zero-padding. Moving
from 256 to 257 time steps changes the FFT length from 512 to 1,024 and raises
the observed cost from 9.37 to 14.71 ms per price. Exact boundary measurements
are stored in `rtx4090_laptop_fft_boundaries.csv`.

## Interpretation

The proposed convention is coherent for the current accuracy target:

- retain `day_fraction=1/252` for every equity model;
- use `steps_per_day=2` for ordinary Markovian Heston when desired;
- start rough Heston N=7 and rough Bergomi at one Brownian cell per business
day, hence `steps_per_day=1` and `dt=1/252`;
- refine a rough model only when the independent price validation shows that
  daily resolution is insufficient.

This conclusion applies to batched price generation. For one isolated price,
Heston and rough Heston expose poor GPU occupancy by construction: at one year
their optimized measured latencies are both about 0.71 s, whereas rough
Bergomi FFT is about 9.3 ms because it parallelizes within the price.

## Generic-engine extension check

After factorizing the FFT, schedule, model path and product policies, the
one-year one-price check with 1,048,576 paths and a 65,536-path chunk measured:

| Model | Median kernel time | Price | Standard error |
|---|---:|---:|---:|
| Rough Bergomi | 10.369 ms | 0.0737597 | 9.28e-5 |
| Rough SABR (`beta=0.7`) | 10.455 ms | 0.0742636 | 9.08e-5 |

One measured launch after warm-up at every requested dispatch length gave:

| Time steps `N` | FFT length `L` | Rough Bergomi | Rough SABR (`beta=0.7`) |
|---:|---:|---:|---:|
| 90 | 256 | 4.02 ms | 3.97 ms |
| 180 | 512 | 8.03 ms | 7.94 ms |
| 360 | 1,024 | 19.41 ms | 17.66 ms |
| 720 | 2,048 | 43.32 ms | 37.56 ms |
| 1,440 | 4,096 | 95.87 ms | 85.06 ms |
| 1,800 | 4,096 | 113.12 ms | 99.70 ms |
| 2,520 | 8,192 | 262.10 ms | 248.12 ms |

The model-policy layer therefore adds no measurable structural penalty. A
fully in-block IFFT-to-spot fusion was also tested and rejected: it retained
the FFT block resources while only two lanes per transform could perform the
sequential spot recursion, measuring about 52 ms. Bounded global staging uses
about 66 MiB at this configuration and restores the ~10 ms throughput while
remaining independent of the total path count.

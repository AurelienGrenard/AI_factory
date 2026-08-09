# Abandoned work

This file records ideas that were investigated and deliberately rejected. It
is not a backlog. Keeping the evidence here prevents the same optimization
from being repeatedly reimplemented without new information.

An abandoned item should be reopened only when its stated assumptions have
materially changed or new measurements contradict the recorded conclusion.

## Sorting early-exercise rows by exercise count

### Investigated strategies

The current American and Bermudan batch plan preserves dataset order and runs
the backward loop up to the largest exercise count in each batch. Rows whose
backward work is already complete return immediately from the regression and
cashflow kernels, although their blocks are still launched.

Two alternative schedules were implemented in an isolated benchmark:

1. keep the current batch boundaries, sort rows by decreasing exercise count
   inside each batch, and restrict `grid.y` to the active prefix;
2. sort all rows globally by decreasing exercise count before constructing the
   memory-aware batch plan, then scatter prices back to dataset order.

Both alternatives preserved the original result index, Philox key, path
indices, block geometry for each active price, and numerical reduction order.
All tested prices and standard errors matched the baseline bit for bit.

### Measurements

The benchmark covered 354 CUDA executions and 42 configurations on an RTX
4090 Laptop GPU. It varied Heston, Bates, Variance Gamma, and NIG; 64, 256, and
1,000 result rows; 65,536 to 1,048,576 paths per price; and homogeneous,
narrow, bimodal, wide, and catalog exercise-count distributions.

Median GPU times for the real 1,000-row catalog at 1,048,576 paths per price
were:

| Model | Current | Sort inside batch | Change | Global sort | Change |
|---|---:|---:|---:|---:|---:|
| Variance Gamma | 5.810 s | 5.465 s | -5.9% | 5.606 s | -3.5% |
| NIG | 3.240 s | 3.098 s | -4.4% | 3.142 s | -3.0% |
| Heston | 13.147 s | 12.882 s | -2.0% | 13.421 s | +2.1% |
| Bates | 24.259 s | 23.396 s | -3.6% | 24.166 s | -0.4% |

Across all configurations, sorting inside a batch improved the median runtime
by only about 1.3%. Homogeneous and narrow distributions showed essentially no
gain. Global sorting reduced the number of backward launches substantially but
did not turn that reduction into a reliable runtime improvement because the
skipped blocks were already cheap and the new packing produced less regular
batches.

### Decision

Abandon both sorting strategies. A typical 1-5% gain does not justify execution
permutations, original-index mapping, extra host/device index storage,
active-prefix logic in every early-exercise launcher, and the resulting loss of
readability.

Reopen only if early-exercise pricing becomes a measured production bottleneck
and new hardware or a materially different exercise-count distribution changes
the result. In that case, benchmark the within-batch active-prefix strategy
first; it was simpler and more consistent than global sorting.

## Secondary sorting by simulation step count

A secondary key based on the number of Heston or Bates QE-M sub-steps was
considered for rows with the same exercise count. It might reduce a simulation
tail when maturities or time-step counts vary widely.

This idea was not retained. Each block already owns one price, so sorting does
not remove intra-warp divergence, and no profile identified simulation-tail
imbalance as a material bottleneck. Combining another key with an optimization
that already produced only a small gain would add complexity without adequate
evidence.

Reopen only if profiling on the target production workload isolates this tail
as a significant share of total runtime.

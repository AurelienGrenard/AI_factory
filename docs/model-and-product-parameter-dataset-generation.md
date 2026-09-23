# Model and product parameter dataset generation

Every versioned model and product dataset contains exactly 1,000 rows split
into two ordered regimes:

- rows 1–900 are the representative **core** regime;
- rows 901–1,000 are the wider **stress** regime.

The split is deterministic and never shuffled. The canonical `recipe.yaml`
records the seed, proposal bounds or grids, conditional reconstruction,
rejection rule, row ordering, and payoff or calendar constraints. The
executable `generator.cpp` implements that recipe; `generation.yaml` records
only the completed materialization and timing. A generator rejects an invalid candidate
rather than clipping it silently.

Every core regime that contains a risk-free rate, an initial short rate, or an
initial forward curve uses a strictly positive 10-basis-point floor. Negative
rates remain available in the model stress regimes; the standalone core curve
bases are strictly positive.

## Model rows

Independent uniform sampling is used only for parameters whose bounds do not
depend on another parameter. The following constrained constructions keep the
draws meaningful:

- Heston and Bates draw `kappa` and `theta` first, then draw `gamma`
  conditionally from bounds proportional to `sqrt(kappa * theta)`. This controls
  the range of the Feller ratio without imposing it as a hard calibration rule.
- CIR draws `mean_reversion` and `long_term_mean` first, then draws `volatility`
  conditionally from bounds proportional to
  `sqrt(mean_reversion * long_term_mean)`. The core Feller ratio
  `2 * mean_reversion * long_term_mean / volatility^2` lies in `[1/6, 10]`;
  stress widens it to `[1/10, 16]`. Both accessible- and inaccessible-zero
  regimes are therefore represented while the exact transition remains valid.
- Bates jump intensity, log-jump mean, and log-jump volatility use wider uniform
  bounds in the stress regime. The catalog records the jump law and martingale
  compensator.
- Variance-Gamma uses uniform proposals and requires both
  `1 - theta*nu - sigma^2*nu/2 > 0.05` and
  `1 - 2*theta*nu - 2*sigma^2*nu > 0.05`. The first condition preserves the
  martingale moment; the second gives discounted vanilla payoffs finite
  variance, which is required for meaningful Monte Carlo error estimates.
- Normal-Inverse-Gaussian draws `alpha`, the latent ratio `beta/alpha`, and a
  target annual volatility. It reconstructs `beta` and `delta`, then rejects
  rows unless
  `alpha > max(abs(beta + 1), abs(beta + 2)) + 0.05`. The added margin
  guarantees the martingale moment and finite vanilla-payoff variance without
  evaluating either square root on the FP32 boundary.
- OU, Hull-White, and Vasicek draw a stationary standard deviation and
  reconstruct instantaneous volatility as `sigma_stationary * sqrt(2*a)`.
- G2 and G2++ draw the first mean reversion and a positive gap to the second;
  both factor volatilities are reconstructed from stationary dispersions.

## Product rows

Plain equity terms use a maturity-dependent exponential strike grid. The core
contains 45 maturities with 20 log-spaced strikes each; stress contains 10
maturities with 10 wider log-spaced strikes each. Derived terms such as barriers,
gap strikes, reset dates, and exercise schedules are then constructed from the
same row, with their exact formula recorded in `recipe.yaml`.

Autocalls, cliquets, and range accruals use seeded uniform and categorical
draws with explicit ordering, cap/floor, and calendar constraints. Fixed-income
products use a representative Cartesian grid followed by a sparse stress grid
covering short and long dates, tenors, accruals, and strikes.

Contractual dates are positive integer business-day counts under
`business_day / 252`. Pricing and validation convert them to model years once
at their input boundary. Coupon accruals with an independent day-count
convention are stored directly as FP32 year fractions.

Regular European swaptions store a payment interval, payment count, and
accrual fraction. Explicit schedules store parallel payment dates and accrual
fractions. Co-terminal Bermudan schedules add the first exercise date and
exercise count while preserving one final maturity. Numerical grids such as
`1 / 504` belong to the pricing scheme, not the product contract.

Price datasets remain distinct for call and put payoffs even when they share a
single product-parameter dataset.

## Versioned Philox domains

Every production recipe that reaches Philox owns one versioned reservation in
`tools/codegen/pricing_bindings/capability_manifest.py`. `RngDomainSpec` is the
only source of dataset seeds: catalogue generators must not invent local bases.
The reservation scheme allocates one `2^32`-key domain per canonical dataset
generator and one
`2^30`-key half-open interval per named stream. Sample recipes reserve distinct
`parameters`, `schedule`, and `dynamics` streams; stochastic price recipes
reserve `dynamics`. Analytical recipes reserve no Philox domain. Version 2
preserves all 588 version-1 ordinals and seeds, and appends the six G2/G2++
European-swaption MC domains. Version 3 additionally appends six CIR++ domains,
again without rekeying the existing recipes. Frozen legacy digests test these invariants.

The catalogue checker rejects missing seed literals, duplicate reservations,
overlaps, uint64 overflow, and undeclared stream names. Intentional common
random numbers require an explicit pair in
`RNG_COMMON_RANDOM_NUMBER_ALLOWLIST`. Price-delta recipes are explicit aliases
of their source price recipe's existing version-3 dynamics reservation, not
new independent domains. Their declared pair is the only permitted overlap;
the added alias does not reorder, resize or rekey any reservation. Adding,
reordering, or resizing a domain changes the declared random construction and
therefore requires a domain-version increment plus regeneration. A recipe must
also keep its largest `base_seed + row_index` strictly inside the reserved
stream interval.

Batch boundaries, grid dimensions, and block dimensions never enter the key or
counter derivation. Tests must replay a dataset across at least two batch or
launch geometries and compare outputs exactly.

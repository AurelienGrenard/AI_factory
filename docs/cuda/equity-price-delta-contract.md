# Equity price and spot-delta contract

This extension computes the central price and a centered finite-difference
delta with respect to the model's initial spot. Price-only launchers remain
separate. Model dynamics, product payoffs, calendars and Philox ownership do
not move into generated bindings.

## Current implementation boundary

The integration covers the 261 existing Markovian equity model/product
bindings: 244 MC, eight Black-Scholes closed formulas and nine American LSM.
`PRICE_DELTA_BINDING_SPECS` owns the generated launchers;
`PRICE_DELTA_DATASET_SPECS` derives recipes from the existing price catalogue.
This is implementation coverage, not catalogue-wide numerical, singularity
or performance certification. Rough FFT/N-factor remain out of scope.
No existing dataset or price-only launch profile is changed.

## Spot perturbation

`SpotBumpConfiguration::relative_width` is the **full** relative width.
The initial candidate is 0.01: each side is displaced by 0.005 times S0.
`prepare_spot_bump` constructs FP32 endpoints; the denominator is their actual
represented difference, not an idealized decimal epsilon. Host validation
requires finite, positive, distinct endpoints around a finite positive S0.
There is no silent clamp, one-sided fallback or per-path adaptive bump.

Every other model parameter and contractual product parameter stays fixed.
Product preparation is nevertheless repeated with each initial spot: a
contract explicitly expressed in S(t)/S(0) must retain that convention.
The bump is not a parameter recalibration.

## Path factorization

- `MultiplicativeSpotPath` runs the existing central dynamics and schedule
  unchanged. Perturbed spots are scaled central observations; native log-spot
  observations use an additive log-ratio. Volatility and jump generation are
  executed once. The model **and implemented scheme** must be homogeneous in S0.
- `CoupledSpotPaths` adapts a model-owned coupled transition. CEV draws one
  normal at each step and invokes its existing absorbed Milstein transition
  on the three spots. It does not replace Milstein by Euler or rescale CEV.
  SABR shares both Gaussian innovations and uses the original transition for
  each scenario. Its public `initial_volatility` is relative volatility:
  dimensional initial alpha is recomputed at each bumped S0, not held fixed.
- Share a volatility state or jump realization only if its law is unaffected
  by the spot bump. Common underlying innovations alone do not prove that the
  resulting states coincide. State-dependent rejection consumption requires
  an explicit coupling, not three loosely synchronized RNG contexts.

`PathProductPriceDeltaPolicy` owns three existing product handlers, not three
copies of the payoff equations. Each handler freezes its terminal observation
when it terminates. The common schedule continues while any scenario remains
active. A stopped central barrier must not stop the bumped surviving paths.
No full spot history is allocated in ordinary MC.

The MC kernel only distributes paths and reduces price/delta moments. Its
specializations select path construction at compile time, with no runtime
strategy switch or `ComputeDelta` flag in the price-only engine. Closed-form
bumping likewise calls the same pricing policy at the three spots.

## Numerical and launch contract

The row key remains `make_key(base_seed + result_index)` and counters remain
`(path_index, local_group_index)`. Path assignments and central moment reduction
order match price-only execution. The target is bitwise central price **and
standard error** at matching geometry/toolchain. Equal seeds alone do not
guarantee this after changing schedule partitioning, arithmetic or reductions.

For each path the policy returns its central discounted payoff and
`(upper_payoff - lower_payoff) / represented_width`. The delta standard error
is computed from these paired differences, not from independent price errors.
States, observations, payoff evaluation and subtraction remain FP32; only the
long moment accumulations and final statistics reuse the established FP64
Monte Carlo contract. No FP64 dynamics or transcendental functions are added.

The ordinary MC price and paired kernels pass their actual sequential summation
length to the common FP64 cancellation guard. A fixed 64-epsilon tolerance
incorrectly rejected a constant payoff at 2^20 paths with 128/256 threads.
The length-aware rounding bound changes only treatment of tiny negative
centered moments, not the sums, prices, reduction order or simulation.
`constant_payoff_moments_cuda` covers the reproduction and still rejects
materially inconsistent moments. See the shared [pricing contract](closed-form-and-monte-carlo-pricing-contract.md).

Launchers require truthful host mirrors of device model/product inputs,
validate bump/calendar/batch/geometry, and inspect the exact specialization.
Output arrays are distinct, non-overlapping caller-owned allocations. Prepared
rows retain the existing shared/thread budgets. The schedule observer borrows
three product states; each scenario has an explicit observation-handler size
bound. This does not make their combined register cost disappear: rich products
must be inspected as three live payoff states, not as a 16-byte adapter alone.

Existing price launch settings are candidates, not qualified delta settings.
`make_equity_price_delta_launch_plan` bounds ordinary MC to at most 256 threads
while retaining the price profile's price/batch distribution. On SM89 the
Bates autocall delta uses 140–142 registers/thread: blindly retaining 512
threads can exceed the register budget. This conservative bound is not a
performance optimum or a portable resource guarantee; native occupancy guards
still apply. Closed-form and LSM candidates are unchanged. Central bitwise
parity requires running price-only at the same chosen geometry, not comparing
against an old dataset produced with a different block size.
Record registers, local memory/spills, shared memory and bounded timings before
selecting production settings. Production recipes will use 2^20 paths; small
unit-test counts do not change that requirement.

## Recipes, artifacts and reuse

`catalog/model/equity/markovian/<model>/price_delta/<variant>/<id>/`
contains generated `generator.cpp` and `recipe.yaml`. The latter describes
planned inputs, method, full bump, CRN seed, fixed time grid when applicable
and 2^20 paths (zero for closed form). It is not a claim that data were generated.
Execution writes the mirrored JSON under `datasets/.../price_delta/` and an
adjacent `dataset.yaml` describing the actual outputs and geometry.

Run these through `tools/datasets/generate_catalog.py --kind price_delta`.
The controller freezes both recipe files and inputs, checks paired outputs
against the declared sensitivity/grid/seed, and attaches the existing generation
provenance before optional publication. Validation stays `pending`, `verified:
false`; no fictitious validator or certificate is added. Changing the bump or
method changes the semantic specification used by the compatibility checker.

CRN aliases explicitly reuse the source price recipe's dynamics seed and row
reservation. Existing reservations are not rekeyed. No independent stream is
claimed between a price and its paired sensitivity. The common launch planner
supplies inherited candidates with the MC thread bound; LSM's actual trace-aware batch sizes
and launch count are recorded after execution. Warmup uses one production-path
row, excluded from kernel timing but included in native wall time.

Outputs include price, delta, the two sampling errors for stochastic methods,
and each row's represented bump endpoints/denominator. Closed formulas have
no fabricated sampling error. Finite outputs do not certify finite-difference
bias, rare-crossing accuracy or approximate LSM stopping-policy bias.

## Discontinuous products

Digital and barrier finite differences require a separate bump/noise study.
Qualification must compare several bump widths, paired standard errors,
threshold-crossing counts and independent repetitions when crossings are rare.
No crossing and a zero empirical standard error do not prove a precise zero
delta. The current digital smoke establishes implementation parity only.
No smoothing or likelihood-ratio estimator is silently substituted.

## Frozen-exercise LSM

Run one central LSM solve, retain the realized exercise date per path, and
evaluate the bumped payoffs at that same date with CRN. Neither refit regressions
nor re-evaluate the stopping rule for bumped scenarios. Initial exercise at
time zero and maturity must also be represented explicitly.

`AmericanOptionPriceDeltaPolicy` composes the existing American policy with
one of two frozen-path strategies. `MultiplicativeFrozenExercise` scales the
recorded central exercise spot for multiplicative models, without another volatility path.
`CoupledFrozenExercise` replays the existing coupled CEV dynamics with the same
row key/path counter and original schedule prefix, stopping at the recorded
observation. Neither strategy applies the stopping rule to bumped states.

The shared LSM orchestration and regression remain unique. Optional static
exercise-output hooks record decisions in the price-delta specialization;
price-only policies have no trace allocation or additional launches. The
update uses the same FP64 exercise predicate, not equality of cashflows as a
proxy for the decision. The central time-zero comparison records its exact
branch before the workspace is recycled.

The trace costs eight bytes per path (exercise index and central spot), plus
one decision byte per row and a larger prepared row. These optional path/row
fields participate in the existing VRAM batch planner, even for zero stored
observations. There are no additional regression matrices or bumped histories.
Two dedicated kernels consume the finished trace and reuse the central moment
scratch: paired delta partials, then finalization. Their time and launches are
included in `LaunchResult`, before the batch workspace is reused.

Each bumped payoff receives the same sequence of FP32 interval discounts and
initial-stub discount as the central cashflow. Do not replace this by a newly
rounded `exp(-r*t)` or differentiate the discount with respect to S0. The
initial-exercise branch uses bumped immediate payoffs and zero sampling error.
Fatal regression diagnostics or invalid central moment statistics invalidate
the delta and its error as well. The ordinary American calendar guards remain
unchanged; this extension does not introduce a new maturity-only public product.
All nine bindings preserve their price-only continuation coordinates and
regression refinement, including Bates/Schobel-Zhu's second state and Kou's
normal-residual refinement.

The published reference is Pietersz and Pelsser (2010),
[A comparison of single factor Markov-functional and multi factor market models](https://doi.org/10.1007/s11147-009-9050-5),
Review of Derivatives Research 13, 245–272, which uses the constant exercise
method for Bermudan swaption sensitivities. Its use for this equity LSM requires
bounded comparison against full refitting; finite-basis approximation can bias
the frozen-date delta. Monte Carlo error bars do not certify that bias.

On the bounded 4,096-path fixtures, the largest absolute frozen/refit delta gap
was about 0.0322 (CEV call); a CEV put differed by about 0.0301. These diagnostics
do not establish a global bias bound or an equivalence to refitting. Production
qualification must keep this limitation visible, not interpret the paired
standard error as the error of the approximate stopping policy.

## Bounded checks

`price_delta_central_parity_cuda` checks BS exact MC, Heston, CEV, SABR and near-barrier
stopping, three rows at 4,096 paths, 128/256 threads, batches and central pathwise
bits. Bumped resimulation comparisons allow bounded FP32 scaling differences;
CEV uses identical innovations and requires exact equality in its pilot cases.
SABR also compares three independently prepared original transitions, preserving
the normalized-volatility parameter convention, with exact paired equality.
`price_delta_closed_form_cuda` checks public Black-Scholes call/put launchers,
three non-unit spots and three bumps against the analytic delta.
These tests are not an exhaustive dataset, singularity or performance campaign.

`price_delta_frozen_exercise_cuda` checks the three public American bindings,
call/put, four rows, 4,096 paths, 128/256 threads and full bumps 0.01/0.005.
It requires bitwise central price/error parity, frozen time-zero exercise,
finite paired statistics, and exactly two additional passes per batch. A few
full refits are printed as diagnostics; their discrepancies are not silently
treated as numerical parity. Logged execution times include cold-start effects
and are not a warmed-up performance comparison.
`price_delta_frozen_path_replay_cuda` checks 512 CEV paths at intermediate dates
and maturity against independently orchestrated original transitions, including
absorption; it also checks the trace memory budget and invalid-result propagation.

`price_delta_catalogue_parity_cuda` exercises public European launchers across
the 12 models, all eight closed-form products, richer Bates monitoring/calendar
handlers and the nine American bindings. BS/Heston/CEV American fixtures use
2^20 paths, two rows, 128 threads and 128 blocks per price on a short exercise
calendar. Other stochastic fixtures use 4,096 paths for bounded structural
coverage only; none chooses or certifies production tuning.

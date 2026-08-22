# Deferred work

This file lists the projects that are intentionally postponed but remain part
of the intended development roadmap. It contains planned work only. Ideas that
were tested and rejected belong in `abandoned-work.md`; permanent engineering
rules belong in the relevant implementation contract or extension workflow.

## Persistent equity-reference migration

Black-Scholes now uses the persistent 1,000-row contract documented in
`independent-price-validation-pipeline.md`. Migrate the remaining equity models
one family at a time to the same architecture:

- independently reference both the 900-row core and 100-row stress regimes;
- persist prices, standard errors, semantic fingerprints, backend versions,
  row provenance, tolerances, and verification under
  `validation/datasets/price/equity`;
- preserve the Premia, QuantLib specialized, QuantLib Monte Carlo hierarchy and
  fall through only on technical row failures;
- use proven continuous/discrete relations where applicable and explain every
  accepted systematic bias;
- replace adjacent reports and notebooks with the compact YAML cache link;
- make routine CTest cache-only and independent of Premia, Wine, and QuantLib.

Do not reuse stale legacy reports as reference prices. Regenerate each model
after its backend mapping has been audited. Kou in particular has no generic
QuantLib process, so products not covered by a reliable Premia method remain
unverified until a genuinely independent reference is implemented.

## Compiled Black-Scholes path-reference engine

Replace the scalar Python/QuantLib path loop used by the Black-Scholes
Athena-autocall, cliquet, Phoenix-autocall, Phoenix-memory-autocall,
double-knock-out, and up-no-touch references with an independently
reproducible compiled or genuinely vectorized engine. The latter families
normally use Premia first, but even a single technically unsupported row falls
through to the same slow path engine.

The current implementation covers all 1,000 rows with 1,024 antithetic pairs
per row and refines suspicious rows with 8,192 pairs. It is numerically useful
but performs every path generation, payoff observation, and `ql.Path` access
through the Python-to-QuantLib boundary. A single dataset consequently takes
tens of minutes. Athena was regenerated and verified under the
`business_day / 252` convention, but cliquet and double-knock-out-call
regeneration were intentionally interrupted. The two Phoenix families,
double-knock-out put, and up-no-touch were not started. Their persistent
references must therefore be regenerated after this performance work.

Preserve the existing independent-reference contract: cover the 900-row core
and 100-row stress regimes, retain deterministic per-row seeds and antithetic
sampling, persist reference standard errors and provenance, and keep the
refinement rule. Do not replace the external reference with the CUDA pricing
kernel or Philox stream being validated. Add equivalence tests against the
current QuantLib implementation on a small deterministic row selection before
switching the full regeneration pipeline.

## CIR++ model and CIR joint dynamics

The standalone CIR state dynamics, affine analytics, caplet/floorlet and
zero-coupon-bond-option datasets are implemented. Complete the shifted CIR++
form and the CIR joint rate/integral transition required by path-discounted
products.

The implemented state transition uses the exact non-central-chi-square law via
the common Poisson--Gamma sampler. Products requiring the integrated short
rate or the complete path still need a separately justified simulation
strategy and a finer time grid where exact joint sampling is not available.

CIR++ must reuse the CIR stochastic factor and add only the deterministic shift
needed to fit the initial term structure. Keep raw dynamics, curve fitting,
analytics, and product pricing in their existing project layers. Add core and
stress model datasets, independent QuantLib validation, catalog entries, and
website integration.

## European and early-exercise swaptions

European payer/receiver launchers and reusable Jamshidian analytics are
implemented for standalone Ornstein-Uhlenbeck, Vasicek, CIR, and Hull-White
one-factor fitted to either supported parametric curve. Their payer and receiver
datasets are independently certified on all 900 core and 100 stress rows.
Vasicek, centered OU, and Hull-White use Premia closed forms where the exact
contract is supported and reliable, with specialized QuantLib Jamshidian
fallbacks for the remaining rows. CIR uses QuantLib for all rows after both
Premia finite-difference methods failed the numerical audit. The persistent
caches include semantic and policy fingerprints and run cache-only in CI.

G2/G2++ European implementations and all early-exercise swaption launchers
remain deferred.

European swaptions should use the best deterministic model-specific formula
available. Add the G2 and G2++ European implementations only after selecting
and documenting their non-Jamshidian numerical method. The current one-factor
contract identifies exercise with the underlying swap start. Exact spot lags,
distinct forward swap starts, and mid-curve swaptions require the corresponding
generalized decomposition and remain deferred. American or Bermudan
swaptions should reuse the existing
early-exercise architecture: prepared rows, model-specific state simulation,
Longstaff-Schwartz regression, backward cashflow updates, moment reduction, and
memory-aware batch planning. Do not introduce a second early-exercise engine
for fixed income unless the swap state creates a demonstrated incompatibility.

The extension includes product and price datasets, independent QuantLib
validation for every supported model/curve combination, tests, catalog
metadata, and website equations and entries.

## Rough volatility models

Implement these models as separate extensions:

- rough Bergomi;
- rough Heston;
- quadratic rough Heston.

Their Volterra memory makes them structurally different from the current
Markov dynamics. Define and validate the discretization, covariance
construction, memory layout, and random-consumption strategy before adding
product kernels. Rough Bergomi should begin with a validated hybrid scheme.
Rough Heston and quadratic rough Heston require their own documented numerical
schemes rather than superficial adaptations of the Heston QE implementation.

For each model, first validate simulated moments, covariance structure, and
convergence on CPU and CUDA. Add pricing datasets only after the dynamics and
their discretization bias have independent references.

## LIBOR Market Model

Implement a tenor-based LIBOR Market Model with explicit forward-rate state,
initial-curve construction, volatility parametrization, correlation model, and
chosen pricing measure.

The design must specify the factor reduction, drift computation, tenor-date
layout, numeraire and discounting conventions, and the distinction between
terminal-only and path-dependent simulation. Start with caplet and European
swaption validation before using the model for Bermudan swaptions. Add the
model, curve composition, datasets, CUDA tests, independent QuantLib
validation, catalog entries, and website documentation as one coherent model
extension.

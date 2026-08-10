# Deferred work

This file lists the projects that are intentionally postponed but remain part
of the intended development roadmap. It contains planned work only. Ideas that
were tested and rejected belong in `abandoned-work.md`; permanent engineering
rules belong in the relevant implementation contract or extension workflow.

## Core-only independent price validation migration

Migrate every price validator to the certification policy documented in
`independent-price-validation-pipeline.md`. This work is intentionally paused:
do not resume the long Premia runs while implementing unrelated numerical or
CUDA features.

The target policy is unambiguous:

- certify only rows 1--900 (`core`) with external reference engines;
- retain rows 901--1,000 (`stress`) in every dataset, but run only internal
  robustness checks on them; stress never changes the public validation status;
- publish a successful result as `900/900 core`, never as `1000/1000`, and use
  the standard conclusion: "Dataset valide independamment sur son domaine
  core. Les lignes stress testent la robustesse numerique et ne sont pas
  couvertes par la certification externe.";
- make the YAML status and `verified` flag depend exclusively on the core and
  add the explicit scope `core (900 rows)`;
- give the JSON report separate external-core and internal-stress sections, and
  adapt the common notebook renderer, report fingerprinting, schema tests and
  YAML synchronization together.

For every `(model, product)` pair, first rebuild the exhaustive Premia method
inventory. Rank compatible methods globally on a fixed core pilot by contract
fidelity, robustness and then runtime. Validate each core row through this
strict hierarchy:

1. preferred Premia method;
2. every other compatible Premia method, in declared order, but only after a
   technical failure of the preceding method;
3. specialized QuantLib pricer;
4. independent QuantLib Monte Carlo when QuantLib exposes a simulable process;
5. `none` when no independent engine produces a comparable price.

A backend error, non-finite result, invalid standard error, documented domain
violation or finite price violating a no-arbitrage bound is a technical engine
failure. It must retain its row-level diagnostic and fall through; it never
invalidates the CUDA price. A finite, financially admissible price outside the
tolerance is a comparison failure. Other methods may diagnose it, but the
pipeline must not cherry-pick the closest reference. A core row left without an
independent reference is `unvalidated`, not `failed pricing`; nevertheless the
dataset cannot be marked independently verified until all 900 core rows are
objectively resolved.

Resume the work model by model and product by product, with bounded per-engine
timeouts and persisted progress so an interrupted slow Premia run does not
discard completed reports. Regenerate stale reports after any repricing before
publishing YAML or notebooks. In particular, the interrupted Kou experiment
left a mixture of fresh and stale reports after its move to 1,048,576 paths;
none of those partial artifacts should be treated as the completed migration.
Kou has no generic QuantLib process, so products not covered by a reliable
Premia method will remain explicitly unvalidated unless a genuinely independent
reference is added.

## CIR and CIR++ models

Implement the Cox-Ingersoll-Ross short-rate model and its shifted CIR++ form.

Use exact CIR transitions rather than reusing the Heston QE scheme. The
noncentral chi-square transition can be generated through the Poisson-Gamma
mixture already supported by the common random layer. Boundary-only products
should draw one exact transition per required date; products requiring the
integrated short rate or the complete path need a separately justified
simulation strategy and finer time grid where exact joint sampling is not
available.

CIR++ must reuse the CIR stochastic factor and add only the deterministic shift
needed to fit the initial term structure. Keep raw dynamics, curve fitting,
analytics, and product pricing in their existing project layers. Add core and
stress model datasets, independent QuantLib validation, catalog entries, and
website integration.

## European and early-exercise swaptions

Implement a common swaption product definition covering the underlying swap,
exercise schedule, settlement convention, payer/receiver side, strike, and
notional.

European swaptions should use the best deterministic model-specific formula
available. American or Bermudan swaptions should reuse the existing
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

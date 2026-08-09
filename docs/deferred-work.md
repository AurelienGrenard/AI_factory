# Deferred work

This file lists the projects that are intentionally postponed but remain part
of the intended development roadmap. It contains planned work only. Ideas that
were tested and rejected belong in `abandoned-work.md`; permanent engineering
rules belong in the relevant implementation contract or extension workflow.

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

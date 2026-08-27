# Deferred work

This file lists general projects that are intentionally postponed but remain
part of the intended development roadmap. It is not an audit registry: audit
findings, including postponed ones, belong in `audit/response.md`; closed audit
findings belong in `audit/closed.md`. Permanent engineering rules belong in
the relevant implementation contract or extension workflow.

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

## CIR++ model

The standalone CIR state dynamics, affine analytics, caplet/floorlet and
zero-coupon-bond-option datasets are implemented. Its joint Monte Carlo state
uses exact non-central-chi-square endpoints and accumulates the short-rate
integral by a fixed-step trapezoidal rule. Complete the shifted CIR++ form.

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

G2/G2++ European implementations remain deferred. Co-terminal regular
Bermudan payer/receiver launchers are implemented for OU, Vasicek, CIR, G2,
Hull-White with either parametric curve, and G2++ with either parametric curve.
They reuse the common multi-block Longstaff-Schwartz engine, pathwise rate
integrals, a degree-three one-factor Hermite basis or a degree-two two-factor
Hermite basis, and FP64 normal-equation reductions.

European swaptions should use the best deterministic model-specific formula
available. Add the G2 and G2++ European implementations only after selecting
and documenting their non-Jamshidian numerical method. The current one-factor
contract identifies exercise with the underlying swap start. Exact spot lags,
distinct forward swap starts, and mid-curve swaptions require the corresponding
generalized decomposition and remain deferred.

Full independent Bermudan certification remains urgent. QuantLib tree/PDE
cross-checks currently cover representative central and short-stress rows, and
all one-factor rows are checked against the maximum analytical European
exercise value. OU payer stress row `000957` remains unresolved: 65,536 paths
produce a zero estimate while the analytical European lower bound is about
`1.80e-6`. Premia exposes regular Hull-White Bermudan contracts, but its fixed
exercise schedule and fitted-curve adapter did not provide a stable reference;
do not run bulk Premia Bermudan validation. CIR also lacks a reliable QuantLib
Bermudan engine in the tested binding, so its analytical European lower-bound
check must remain visible rather than being presented as full certification.

Still deferred are irregular exercise calendars, distinct exercise and swap
start dates, independent full-row references, and website entries.

## Rough volatility models

Rough Bergomi, log-modulated rough Bergomi, rough SABR and rough Stein--Stein
now share the factored Gaussian-Volterra hybrid engine: one block-cooperative
cuFFTDx convolution, bounded chunk staging, model path policies and four
schedule families. All non-American equity products use the same canonical
path-product policy in this engine. Rough Heston and quadratic rough Heston
consume those policies through generic positive exponential-kernel
representations, host-prepared fixed-factor lifts, and documented weak
splitting schemes for 2, 3 and 7 factors. They cannot use the linear Gaussian
FFT engine because their Volterra integrands depend on the evolving state.

Still deferred are full independent production-price certification campaigns
and price datasets/catalog entries for the rough families, together with an
exact published BL2 quadrature implementation. The isolated Volterra
validation suite checks driver identities, published limiting cases and the
separation between lift error and time-scheme error, but is not a substitute
for 1,000-row price certification. Add pricing datasets only after the
numerical bias has an independent reference.

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

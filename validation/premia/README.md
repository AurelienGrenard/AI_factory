# Premia Validation

This folder contains independent CPU references backed by the Windows Premia
library distributed in `validation/premia/premia-19-win64`. Model-specific
parameter conversion is centralized in the C runner; Python modules only load
catalogue rows, build a typed batch, and apply the common accuracy and bias
checks.

Premia is stateful and its binary interface is Windows-only. A small C runner is
therefore cross-compiled with MinGW-w64 and executed under a private 64-bit Wine
prefix. One runner process handles a complete dataset through a tab-separated
stdin/stdout protocol; Wine and DLL startup are never repeated per row. Build
artifacts and the Wine prefix live below `build/` and are not versioned.

Install the two runtime dependencies once on Ubuntu/WSL:

```bash
sudo apt install --no-install-recommends \
    wine wine64 gcc-mingw-w64-x86-64-posix
```

Build the runner explicitly when desired:

```bash
python -m validation.premia.build_runner
```

The validators also build it automatically when it is absent or stale. Direct
Premia entry points are backend diagnostics:

```bash
python -m validation.premia.model.equity.validate DATASET MODEL PRODUCT
python -m validation.premia.model.fixed_income.validate DATASET MODEL PRODUCT
```

Catalog publication always uses the corresponding unified module under
`validation/model/...`. All fixed-income and Black-Scholes modules take
`DATASET REFERENCE_DATASET`: their default path checks the immutable cache,
while `--generate` explicitly reruns Premia and synchronizes the compact YAML.
Other equity families keep their legacy interface only until they are migrated
to the same persistent-reference contract. A direct Premia command never marks
a catalog entry as validated on its own.

Each registered batch uses one runner process. Datasets following the 90/10
convention are audited separately on their 900 core and 100 stress rows. The
runner retains successful prices when an individual row returns an error, so a
later engine can validate only the failed row. Premia discovery is exhaustive
and precedes engine selection: enumerate every method registered for the exact
model-product pair across all Premia asset menus, inspect direct contracts and
exact Premia-based decompositions, and record every compatible candidate. A
closed form, approximation, tree, finite-difference method, or Monte Carlo
method is equally eligible at this discovery stage.

When several candidates exist, benchmark representative core and stress rows.
Choose the fastest engine among the compatible, robust candidates as the
primary engine, and retain the remaining Premia methods as ordered row-level
fallbacks. A technical failure therefore tries every compatible Premia method
before QuantLib. A successful finite and comparable Premia price that disagrees
with the generated price remains a comparison failure: another method may be
used diagnostically, but must not be selected after the fact merely because it
is closer.

The current catalogue includes the following direct Premia comparisons:

- Black-Scholes: European calls/puts, straddles, cash digitals,
  asset-or-nothing options, gap options, forward-start and geometric-Asian
  reductions, range-accrual digital decompositions, arithmetic Asians, and
  continuous-monitoring bounds for barriers, fixed lookbacks, and touch
  products;
- Heston: European calls/puts and straddles;
- Bates: European calls and straddles;
- Merton: European calls/puts, cash digitals, asset-or-nothing puts, and gap
  puts;
- Kou: European calls/puts and straddles, using `AP_Carr_Kou`;
- Vasicek and centered Ornstein-Uhlenbeck: European calls/puts on zero-coupon
  bonds, plus caplets/floorlets through their exact scaled bond-option
  identities;
- CIR: the bond-option formulas and finite-difference alternatives were
  inventoried and benchmarked, but the bundled methods are not used for
  certification for the reasons below;
- Hull-White and G2++ fitted to Nelson-Siegel or Svensson: the same four
  one-period and bond-option families through Premia HW1D/HW2D formulas.

The primitive engines are `CF_Call`, `CF_Put`, and `CF_Digit` for
Black-Scholes; `CF_Call_Heston`/`CF_Put_Heston`; `CF_Call_MerHes` for the
passing Bates call; `CF_Call_Merton`/`CF_Put_Merton` plus `MC_Merton` for the
Merton cash digital; `AP_Carr_Kou`; and
`CF_Vasicek1d_ZBCallEuro`/`CF_Vasicek1d_ZBPutEuro` for both Vasicek mappings.
The same two methods validate one-period floorlets/caplets after the exact
strike and notional transformation. Premia's `Cap` and `Floor` contracts are
multi-reset instruments and are deliberately not treated as caplets.

Merton arithmetic Asians use `AP_Asian_FMM_Mer` on its documented 52-date
grid. Puts are reconstructed from the call using the expectation of that same
grid; mixing the 52-date call with the CUDA daily-grid expectation is invalid.
The complete batch uses 1,024 integration points. A row that violates an
analytic price bound or the declared comparison tolerance is deterministically
recomputed at 4,096 points, and that refined result always replaces the coarse
one. This recovers the small deep-OTM put tails without selecting whichever
answer happens to be closer. The Merton Asian-put core then passes 900/900.

Premia exposes `CF_Cir1d_ZBCallEuro`, `CF_Cir1d_ZBPutEuro`,
`FD_Explicit_Cir1d_ZBO`, and `FD_Gauss_Cir1d_ZBO`. The bundled closed-form
source replaces `h = sqrt(k^2 + 2 sigma^2)` by `2*h` only inside the
non-central-chi-square parameters, and representative rows confirm the
resulting finite pricing bias. The Gaussian finite-difference engine is
independent but remains outside the catalogue tolerance on representative core
rows even after a 1024-by-1024 refinement; the explicit scheme also has a
restricted boundary regime. These methods therefore remain callable but
unreliable candidates, not eligible references. The complete closed-form core audit gives
100/900 passing caplets, 101/900 floorlets, 363/900 zero-coupon calls, and
321/900 zero-coupon puts. Each persistent CIR reference database records only
`status: available but not reliable` for Premia, then immediately describes
the QuantLib specialized formula actually used. The detailed failed audit
remains here; no divergent Premia price is selected or presented as a fallback.

Premia HW1D/HW2D expects an initial-curve filename instead of curve
coefficients. The runner writes one process-private temporary curve file per
row, with locally bracketing nodes at the two contract dates and discounts
evaluated from the catalogue Nelson-Siegel or Svensson formula. This preserves
Premia as an independent pricer while making its internal linear interpolation
numerically invisible. G2++ is converted exactly to Premia's HW2D state:
the short-rate noise combines the two G2++ Brownian shocks and the second
Premia factor is the rescaled second OU factor. QuantLib remains the row-wise
specialized fallback.

Straddles and the other terminal claims are exact static combinations of the
model's compatible vanilla and cash-digital Premia engines. When one primitive
is Monte Carlo, its reported standard error is propagated into the comparison.

The existence of a compatible method makes Premia available for the pair; it
does not by itself validate the dataset. If a candidate such as `AP_Kou_Eu`, a
Schobel-Zhu approximation, a VG/NIG approximation, or a stress-domain method
fails technically on some rows, the runner must try every other compatible
Premia candidate on those rows before considering QuantLib. If instead a
method returns a finite comparable price outside tolerance, the discrepancy is
reported and investigated rather than hidden by switching references.

Premia also exposes the approximate CEV method `AP_BGM_Cev`. It must therefore
appear in the CEV engine inventory and be tested before QuantLib. If it fails
technically for a row, another Premia CEV method is tried when available; only
after the Premia list is exhausted may QuantLib be used. A finite discrepancy
remains a visible validation failure and is not replaced by the exact QuantLib
price simply because that reference is more accurate.

Premia availability is determined by the actual model-product pair and a
compatible pricing method, not by whether AI_factory uses the same numerical
scheme. Premia's standard barriers, Asians, touch products, and lookbacks are
continuously monitored, whereas the corresponding AI_factory contracts use
explicit daily grids. The continuous engines remain the primary independent
references: barriers, touches, and lookbacks use proven directional bounds;
arithmetic Asians use the exact Black-Scholes L2 bound between the continuous
average and the equally weighted daily grid. The report records the resulting
signed bias instead of pretending that the contracts are identical.

QuantLib is primary for a Black-Scholes product only when Premia exposes no
compatible engine for that model-product pair. This is currently limited to
Athena, Phoenix, Phoenix Memory, and Cliquet. QuantLib can still resolve an
isolated technical Premia failure, including a finite point estimate that
violates a no-arbitrage bound, but never replaces a valid divergent Premia
price.

Standalone G2 is not mapped to Premia's Hull-White 2D object. That object
requires an externally supplied fitted initial curve and does not directly
encode both raw G2 initial factors; QuantLib G2 is therefore the specialized
reference for the current standalone contract.

The adapter converts continuous AI_factory rates to Premia's annual-effective
percentage convention before every call. Model-specific ordering and
transformations, such as Merton's jump standard deviation to Premia's jump
variance, remain explicit at the C boundary.

See [MODEL_PRODUCT_MAP.md](MODEL_PRODUCT_MAP.md) for the complete catalogue of
models, product families, and available numerical-strategy classes.
See
[`../../docs/independent-price-validation-pipeline.md`](../../docs/independent-price-validation-pipeline.md)
for row-level fallback and failure-reporting rules.

The Wine warning about a missing 32-bit installation can be ignored: Premia and
the runner are both PE32+ x86-64 executables. Wine Gecko is not required by the
console runner.

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
`validation/model/...` with the `DATASET VALIDATION_REPORT` arguments. That
module owns backend fallback, the canonical JSON report, and YAML
synchronization. A direct Premia command never marks a catalog entry as
validated on its own.

Each registered batch uses one runner process. Datasets following the 90/10
convention are audited separately on their 900 core and 100 stress rows. The
runner retains successful prices when an individual row returns an error, so a
later backend can validate only the failed row. A successful finite and
comparable Premia price that disagrees with the generated price remains a
comparison failure and does not trigger fallback.

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

The audit deliberately rejects a method that merely exists. `AP_Kou_Eu`, both
Schobel-Zhu approximations, the tested VG/NIG approximations, and several stress
cases of the Bates, Merton, and CEV methods do not pass every current row. Those
datasets therefore fall through to QuantLib by regime or row when a compatible
engine passes, or remain explicitly unverified.

Premia also exposes the approximate CEV method `AP_BGM_Cev`. The catalogue
uses QuantLib's exact analytic CEV engine instead, because the validation
hierarchy applies only to a compatible backend that passes the complete
dataset; a Premia approximation is not preferred over an exact specialized
reference merely because it is present in the package.

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

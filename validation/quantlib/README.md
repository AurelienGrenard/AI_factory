# QuantLib Validation

This folder contains independent CPU references for generated price datasets.
Its hierarchy mirrors `src/model`: every model, optional curve, and product owns
one thin validator exposing `validation_from_quantlib(price_dataset_path)`.
JSON resolution, row metrics, bias checks, term structures, and product identities
are shared so validators differ only where their QuantLib pricing recipe differs.

Install the official Python binding and SciPy once:

```bash
pip install QuantLib scipy
```

Direct backend modules can be used for diagnostics, for example:

```bash
python -m validation.quantlib.model.fixed_income.ornstein_uhlenbeck.zero_coupon_bond_call \
  datasets/price/fixed_income/ornstein_uhlenbeck/zero_coupon_bond_calls/\
ornstein_uhlenbeck_01__zero_coupon_bond_calls_01__01.json
```

Catalog publication instead runs the corresponding unified validator under
`validation/model/...` with `DATASET VALIDATION_REPORT`. It selects QuantLib
only after any higher-priority compatible Premia engine is unavailable or
fails technically, then writes the canonical report and synchronizes the YAML.
Direct QuantLib modules do neither.

The common report checks every absolute-plus-relative row tolerance and records
signed errors, reported Monte-Carlo uncertainty, RMSE, worst rows, directional
counts, and aggregate bias. Monte Carlo rows use a five-standard-error gate;
this controls spurious row failures when thousands of independent estimates are
tested together, while the separate signed-bias check remains at four standard
errors.

For ordered 90/10 datasets, validators accept separate `core` and `stress`
regimes. They also accept explicit row identifiers when QuantLib is the
row-level fallback after a Premia technical failure. QuantLib is not used to
replace a finite Premia result that failed the numerical comparison. The full
selection and reporting rules are documented in
[`../../docs/independent-price-validation-pipeline.md`](../../docs/independent-price-validation-pipeline.md).

Supported datasets are:

- OU, Vasicek, G2, and Hull-White or G2++ fitted to Nelson-Siegel or Svensson:
  bond calls, bond puts, caplets, and floorlets;
- Heston terminal-payoff families through specialized analytic engines;
- Heston arithmetic and geometric Asians, discrete barriers, touches, double
  knock-outs, Athena, Phoenix, Cliquet, and Range Accrual products through
  independent antithetic QuantLib paths;
- Heston maturity-anchored American call/put adapters through a finite-
  difference engine, retained for diagnostics but not reported as successful
  validation because the current Longstaff-Schwartz datasets do not pass every
  row;
- Heston forward-start adapters through independent QuantLib Monte Carlo,
  retained for diagnostics but reported as `none` because a few complete-
  dataset rows remain outside the statistical threshold;
- Bates European options through the analytic Bates engine;
- Bates path products through independent `BatesProcess` paths;
- Bates maturity-anchored American adapters through the finite-difference Bates
  engine, retained for diagnostics but reported as `none`: the current
  Longstaff-Schwartz datasets fail 51 call rows and 46 put rows out of 1,000.
- Variance-Gamma European, digital, asset-or-nothing, gap, and straddle
  payoffs through QuantLib's conditional `BlackCalculator` representation and
  an independent logarithmic Gamma quadrature. This is the same mixture formula
  as `VarianceGammaEngine`, with the zero-density singularity removed for small
  `maturity / nu`.
- CEV European calls and puts through `AnalyticCEVEngine`, after the exact
  deterministic time change that removes carry from the CEV state variable.

QuantLib 1.43 exposes no Normal-Inverse-Gaussian process or pricing engine in
its Python binding. NIG datasets are therefore published with explicit `none`
metadata until an independent reference exists; the CUDA implementation is not
reused as a purported validator. The binding's Variance-Gamma process also does
not implement path evolution, so VG path-dependent and early-exercise datasets
are likewise not presented as QuantLib-validated.

Likewise, QuantLib's `Merton76Process` is an analytic-engine input and does not
implement the stochastic-process drift/evolution interface required by its path
generators. QuantLib 1.43 exposes no compatible Kou or Schobel-Zhu process.
Their non-European datasets therefore remain explicitly `none` unless a
different independent backend validates the complete 1,000-row dataset.

The standard CTest suite runs all analytical references. Slow Heston/Bates path
and early-exercise references are opt-in:

```bash
cmake -S . -B build -DAI_FACTORY_QUANTLIB_EXOTIC_VALIDATION=ON
```

Independent Bates path references use 512 antithetic pairs per row and propagate
their own standard errors into the row and bias checks. High-activity Bates
American diagnostic rows use `FdmBatesOp` on a jump-aware log-spot mesh because
QuantLib's default diffusion-sized domain can exclude material jump tails. The
American reports intentionally retain the tighter numerical tolerance: they
expose the expected low bias of the Longstaff-Schwartz estimator instead of
hiding it behind a broad method-level threshold. They are therefore not
registered as passing CTest validations. Lookback options are excluded because
the installed QuantLib binding has no Heston/Bates lookback engine.

The Heston geometric-Asian validator uses a cheap deterministic first pass on
every row. A row more than four combined standard errors away is recomputed
with a fresh deterministic stream and a larger sample. This adaptive allocation
resolves rare deep-OTM payoffs without weakening the row tolerance or
multiplying the cost of all 1,000 references.

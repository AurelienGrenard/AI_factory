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

Validate any dataset through its model/product module, for example:

```bash
python -m validation.quantlib.model.fixed_income.ornstein_uhlenbeck.zero_coupon_bond_call \
  datasets/price/fixed_income/ornstein_uhlenbeck/zero_coupon_bond_calls/\
ornstein_uhlenbeck_01__zero_coupon_bond_calls_01__01.json
```

The common report checks every absolute-plus-relative row tolerance and records
signed errors, reported Monte-Carlo uncertainty, RMSE, worst rows, directional
counts, and aggregate bias. Monte Carlo rows use a five-standard-error gate;
this controls spurious row failures when thousands of independent estimates are
tested together, while the separate signed-bias check remains at four standard
errors.

Supported datasets are:

- OU, Vasicek, Hull-White/Nelson-Siegel, G2, and G2++/Nelson-Siegel bond calls,
  bond puts, caplets, and floorlets;
- Heston European calls through the analytic Heston engine;
- Heston arithmetic Asian calls through independent Monte Carlo;
- Heston Athena, Phoenix, Cliquet, and Range Accrual products through antithetic paths;
- Heston maturity-anchored American puts through a finite-difference engine;
- Bates European options through the analytic Bates engine;
- Bates path products through independent `BatesProcess` paths;
- Bates maturity-anchored American options through the finite-difference Bates
  engine.
- Variance-Gamma European, digital, asset-or-nothing, gap, and straddle
  payoffs through QuantLib's conditional `BlackCalculator` representation and
  an independent logarithmic Gamma quadrature. This is the same mixture formula
  as `VarianceGammaEngine`, with the zero-density singularity removed for small
  `maturity / nu`.

QuantLib 1.43 exposes no Normal-Inverse-Gaussian process or pricing engine in
its Python binding. NIG publication therefore remains blocked until an
independent QuantLib reference exists; the CUDA implementation is not reused as
a purported validator. The binding's Variance-Gamma process also does not
implement path evolution, so VG path-dependent and early-exercise datasets are
likewise not presented as QuantLib-validated.

The standard CTest suite runs all analytical references. Slow Heston/Bates path
and early-exercise references are opt-in:

```bash
cmake -S . -B build -DAI_FACTORY_QUANTLIB_EXOTIC_VALIDATION=ON
```

Independent Bates path references use 512 antithetic pairs per row and propagate
their own standard errors into the row and bias checks. High-activity Bates
American rows use `FdmBatesOp` on a jump-aware log-spot mesh because QuantLib's
default diffusion-sized domain can exclude material jump tails. The American
reports intentionally retain the tighter numerical tolerance: they expose the
expected low bias of the Longstaff-Schwartz estimator instead of hiding it behind
a broad method-level threshold. Lookback options are excluded because the
installed QuantLib binding has no Heston/Bates lookback engine.

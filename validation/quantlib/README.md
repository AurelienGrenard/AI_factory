# QuantLib Validation

This folder contains independent CPU references for generated price datasets.
Its hierarchy mirrors `src/model`: every model, optional curve, and product owns
one thin validator exposing `validation_from_quantlib(price_dataset_path)`.
JSON resolution, row metrics, bias checks, term structures, and product identities
are shared so validators differ only where their QuantLib pricing recipe differs.

Install the official Python binding once:

```bash
pip install QuantLib
```

Validate any dataset through its model/product module, for example:

```bash
python -m validation.quantlib.model.ornstein_uhlenbeck.zero_coupon_bond_call \
  datasets/price/ornstein_uhlenbeck/zero_coupon_bond_calls/\
ornstein_uhlenbeck_01__zero_coupon_bond_calls_01__01.json
```

The common report checks every absolute-plus-relative row tolerance and records
signed errors, reported Monte-Carlo uncertainty, RMSE, worst rows, directional
counts, and aggregate bias.

Supported datasets are:

- OU, Vasicek, Hull-White/Nelson-Siegel, G2, and G2++/Nelson-Siegel bond calls,
  bond puts, caplets, and floorlets;
- Heston European calls through the analytic Heston engine;
- Heston arithmetic Asian calls through independent Monte Carlo;
- Heston Athena, Phoenix, Cliquet, and Range Accrual products through antithetic paths;
- Heston maturity-anchored American puts through a finite-difference engine.

The standard CTest suite runs all analytical references. Slow Heston path and
early-exercise references are opt-in:

```bash
cmake -S . -B build -DAI_FACTORY_QUANTLIB_EXOTIC_VALIDATION=ON
```

The American-put report intentionally retains the tighter numerical tolerance:
it currently exposes the expected low bias of the Longstaff-Schwartz estimator
instead of hiding it behind a broad method-level threshold. Lookback options are
excluded because the installed QuantLib binding has no Heston lookback engine.

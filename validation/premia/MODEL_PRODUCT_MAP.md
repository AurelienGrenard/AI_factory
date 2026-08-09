# Premia model and product map

This document describes the catalogue embedded in
`premia-19-win64/lib/libpremia.dll`. It is generated conceptually from Premia's
exported `Model`, `Family`, `Option`, `Pricing`, and `PricingMethod` objects, not
from PDF filenames. The inspected binary exposes 99 models, 22 product
families, 161 individual products, 144 model-family pricing registrations, and
673 pricing methods.

A model-family registration means that Premia contains pricing code for at
least part of that family. It does not imply that every numerical method in the
registration accepts every product: each `PricingMethod` owns a `CheckOpt`
function that applies the final product-level restriction. The validation
adapter must therefore record both the product and the exact selected method.

Availability is decided by the complete model-product-method triple, not by a
family registration alone and not by AI_factory's numerical approximation.
Once a compatible Premia engine exists for the financial contract, Premia is
the primary backend. A discrete CUDA implementation may therefore be checked
against a continuous Premia engine when the report uses a proved directional
ordering or a quantitative contract-difference bound and documents the bias.
Core and stress coverage may differ, and only a technical row failure can be
delegated explicitly to the next backend. A valid finite Premia divergence is
never hidden by selecting QuantLib instead.

## Product families

### Equity and generic products

- `STD`: European and American calls, puts, call spreads, and digitals, plus a
  European butterfly.
- `LIM`: European and American call/put barriers for every up/down and in/out
  direction, plus European Parisian call/put barriers for every direction.
- `LIMDISC`: discretely monitored European down-and-out call.
- `DOUBLIM`: European and American double-barrier calls and puts, double
  Parisian calls, and two-double-step put-outs.
- `PAD`: fixed- and floating-strike Asian calls and puts, fixed- and
  floating-strike lookbacks, American moving-average options, cliquets, and
  the `EuroI` Asian variants.
- `VOL`: variance and volatility swaps, correlation swaps, timer options,
  volatility indices, and European calls/puts on realized variance, VIX, and
  volatility.
- `STD2D`: two-asset best-of, maximum/minimum, exchange, and basket options in
  European or American form where provided.
- `STDND`: multi-asset basket, geometric basket, maximum/minimum, and
  multi-spread options.
- `CALLABLE`: no-call, standard-call, path-dependent,
  highly-path-dependent, and intermittent-call structures.
- `STDa`: equity-linked surrender endowment, gap option, GLWB, GMWB, GMMB,
  GMDB, GMIB, and iCPPI.
- `STDz`: GMDB-risk and GMMB-risk products.

### Fixed income, credit, inflation, energy, risk, FX, and trading

- `STDi`: zero-coupon bond; European and American calls/puts on a zero-coupon
  bond; European coupon-bearing bond call; cap, floor, caplet; payer and
  receiver European or Bermudan swaption.
- `EXOi`: callable capped floater, inverse floater, range accrual, and CMS
  spread.
- `STDc`: credit-default swap.
- `STDNDc`: CDO and CDO hedging.
- `STDf`: inflation cap, inflation caplet, and year-on-year inflation swap.
- `STDg`: call/put on a future and swing option.
- `STD2Dg`: European two-dimensional call/put spread.
- `STDr`: VaR, credit VaR, CVA, and European/American CVA calls and puts.
- `STDNDr`: multi-asset CVA basket and geometric-basket calls and puts.
- `STDx`: FX call.
- `STDt`: optimal execution.

## Models and registered product families

The identifiers below are the exact Premia model and family identifiers used
by the C API.

### Equity diffusion, local-volatility, and multi-asset models

- `ACDP1D`: `STD`
- `BS1D`: `STD`, `LIM`, `LIMDISC`, `DOUBLIM`, `PAD`, `STDa`, `STDz`
- `BSDISDIV1D`: `STD`
- `BS2D`: `STD2D`, `STD2Dg`
- `BSND`: `STDND`
- `BSHW1D`: `STD`, `STDa`
- `BSCIR2D`: `STD`, `STDa`
- `CEV1D`: `STD`
- `DUP1D`: `STD`
- `LOCAL_VOL`: `CALLABLE`
- `LOCVOLHW1D`: `STD`
- `MRC30D`: `STDND`
- `UVM1D`: `STD`, `PAD`

### Equity jump and Lévy models

- `MER1D`: `STD`, `LIM`, `PAD`, `STDa`
- `MER2D`: `STD2D`
- `MER1DLOCVOL`: `STD`
- `PUREJUMP1D`: `PAD`
- `VARIANCEGAMMA1D`: `STD`, `LIM`, `PAD`, `STDa`
- `NIG1D`: `STD`, `LIM`, `PAD`, `STDa`
- `KOU1D`: `STD`, `LIM`, `PAD`, `STDa`
- `RSKOU1D`: `STD`, `LIM`
- `TEMPEREDSTABLE1D`: `STD`, `LIM`, `VOL`
- `RSTEMPEREDSTABLE1D`: `STD`, `LIM`
- `CGMY1D`: `STD`, `LIM`, `PAD`, `STDg`, `STDr`
- `BMSPECTRALY_NEGATIVE1D`: `STD`

### Stochastic-volatility and hybrid equity models

- `HES1D`: `STD`, `LIM`, `PAD`, `STDa`, `VOL`
- `HES1D_SLV`: `STD`
- `HES1D_MULTIFACTOR`: `STD`
- `HES4over2`: `STD`
- `TIMEHES1D`: `STD`
- `DOUBLEHES1D`: `STD`, `VOL`
- `SCOTT1D`: `STD`
- `STEIN1D`: `STD`
- `ALSABR11D`: `STD`
- `ALSABR21D`: `STD`
- `SABR1D`: `STD`
- `GARCH1D`: `STD`
- `FPS1D`: `STD`
- `FPS2D`: `STD`
- `HESCIR1D`: `STD`
- `HESHW1D`: `STD`
- `HESHW2D`: `STD`
- `HESVASICEK1D`: `STD`
- `HESVASICEK2D`: `STD`
- `WISHART2D`: `STD2D`, `VOL`
- `MERHES1D` (Bates): `STD`, `LIM`, `PAD`, `VOL`
- `BNS`: `STD`
- `DPS`: `STD`
- `NONPAR1D`: `VOL`
- `VARSWAP3D`: `STD`
- `BERGOMI2D`: `STD`
- `BERGOMIREV2D`: `STD`, `VOL`
- `GUYON1D`: `STD`
- `GUYON2D`: `STD2D`
- `ROUGHBERGOMI2D`: `STD`
- `TWOHYPERGEOMETRIC1D`: `STD`
- `HW1D`: `STD`, `VOL`

### Short-rate, term-structure, and LIBOR models

- `Vasicek1D`: `STDi`
- `Cir1D`: `STDi`, `PAD`
- `CirPP1D`: `STDi`
- `Cir2D`: `STDi`
- `HullWhite1D`: `STDi`
- `HullWhite1DGeneralized`: `STDi`
- `HullWhite2D`: `STDi`
- `BlackKarasinski1D`: `STDi`
- `SG1D`: `STDi`
- `LRSHJM1D`: `STDi`
- `BharChiarella1D`: `STDi`
- `HK1D`: `STDi`
- `QTSM2D`: `STDi`
- `Affine3D`: `STDi`
- `SCHWARTZ`: `STDi`
- `LMM1D`: `STDi`, `EXOi`
- `LMM_HESTON1D`: `STDi`
- `LMM_JUMP1D`: `STDi`
- `LMM1D_CGMY`: `STDi`
- `LMM_STOCHVOL_PITERBARG`: `STDi`
- `LIBOR_AFFINE_CIR1D`: `STDi`
- `LIBOR_AFFINE_GOU1D`: `STDi`

### Credit, inflation, energy, insurance, risk, FX, and trading models

- `BLACK_COX_EXTENDED`: `STDc`
- `CirPP2D`: `STDc`
- `COPULA`: `STDNDc`
- `DYNAMIC`: `STDc`, `STDNDc`
- `HAWKES_INTENSITY`: `STDNDc`
- `SLI`: `STDNDc`
- `JarrowYildirim1D`: `STDf`
- `INFLATION_LMM_HESTON1D`: `STDf`
- `HHW4D`: `STDf`, `STDx`
- `JUMP1D`: `STDg`
- `NIG1FACT1D`: `STD`
- `OU1D`: `STD`
- `VARIANCEGAMMA2D`: `STD2Dg`
- `SCHWARTZTROLLE`: `STDg`
- `STATIC_MERTON`: `STDr`
- `BS1D_DEFAULT`: `STDr`
- `BSND_DEFAULT`: `STDNDr`
- `HES1D_DEFAULT`: `STDr`
- `MERHES1D_DEFAULT`: `STDr`
- `HAWKES_TRADING`: `STDt`

Several model objects are reused in more than one asset menu. For example,
`CGMY1D` appears in equity, energy, and risk contexts, while `HHW4D` appears in
inflation and FX. The family and method selected by the adapter, rather than
the menu in which the model appears, determine the actual pricing contract.

## Pricing and simulation methods

Premia prefixes its pricing methods by numerical strategy. This binary exports:

- 98 `CF` methods: closed formulas;
- 193 `AP` methods: analytical approximations, Fourier/cosine methods, and
  asymptotics;
- 122 `FD` methods: finite differences or finite elements;
- 75 `TR` methods: trees, quantization, and related backward schemes;
- 185 `MC` methods: Monte Carlo pricing methods.

The `MC` methods make Premia usable when no deterministic formula exists. They
encapsulate their own path simulation, payoff evaluation, and reduction and
usually return a price together with an error estimate or confidence bounds.
The Premia backend can therefore combine that independent standard error with
the dataset's reported Monte Carlo standard error in the same row checks used
for other independent simulation references.

Premia does not expose one uniform public process API equivalent to a generic
`simulate_path(model, dates, rng)` function. Its `DynamicTest::Simul` entry
points are primarily pricing/hedging tests, and the lower-level simulators vary
by numerical method. Validation should consequently call a compatible Premia
`MC` pricing method directly rather than attempt to extract and standardize raw
paths. If a model-product pair has neither a deterministic method nor a usable
`MC` method, it is not independently validated by Premia.

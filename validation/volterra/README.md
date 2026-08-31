# Volterra price references

This directory is an offline validation boundary for rough-volatility prices.
It deliberately imports neither `src` nor CUDA code and is not linked into any
runtime target.

## What is certified

For rough Bergomi, the package separates:

1. the direct kappa=1 hybrid convolution from an independently padded NumPy
   FFT of the same weights;
2. the hybrid Gaussian driver from the exact finite-grid joint Gaussian law of
   `(delta_W, Y)` built in FP64 and sampled through Cholesky;
3. both driver constructions from the orthogonal stock Brownian noise by using
   a conditional Black payoff;
4. Monte-Carlo uncertainty from the remaining grid bias using antithetic-pair
   standard errors and several step counts.

For rough Heston, the package separates:

1. the CUDA weak simulation error from the exact affine price of the supplied
   exponential lift, computed through its multidimensional Riccati ODE;
2. the lift/kernel error from the continuous rough-Heston price, computed with
   an independent fractional Adams PECE solver and Lewis inversion;
3. Fourier and fractional-time discretization indicators from the actual
   model difference.

For rough SABR, the independent FP64 NumPy path reference verifies:

1. direct and padded-FFT evaluations of the hybrid driver path by path;
2. the exact `beta=1` reduction to the rough-Bergomi log-spot recursion;
3. the `eta=0, beta=1` reduction to Black-Scholes within Monte Carlo error;
4. the Lamperti-spot convention used for `beta<1`, including invariance
   of the initial log-return variance when `S0` changes.

The production extension uses the risk-neutral spot Lamperti coordinate when
`r-q` is non-zero. The Fukasawa--Gatheral paper case has `r=q=0`, so this
extension leaves the cited forward-martingale benchmark unchanged.

For the three new rough families, the offline checks additionally separate:

- log-modulated rough Bergomi: unit one-year kernel normalization, the
  integrable `H=0` logarithmic boundary, and direct versus padded-FFT hybrid
  paths, plus the zero-`eta` Black--Scholes limit;
- rough Stein--Stein: the Mittag--Leffler resolvent equation, direct versus
  padded-FFT convolution for both the raw resolvent and the production hybrid
  driver, the zero-vol-of-vol Black--Scholes limit, and the
  long-maturity/high-mean-reversion tail;
- quadratic rough Heston: the factor recurrence versus the exact dense
  convolution of the same exponential kernel, followed separately by the
  seven-factor versus fractional-kernel path and paired-call-price indicators.

The QRH equations are validated against the restricted specification in
[*Quadratic rough Heston*](https://arxiv.org/abs/2001.01789).  The
[authors' repository](https://github.com/jgatheral/QuadraticRoughHeston) is
linked for research traceability, but its current code uses a later
gamma/resolvent-kernel parameterization.  It is therefore not imported or
treated as a numerical oracle for the original-paper fixture used here.
The log-modulated kernel follows Bayer, Harang and Pigato,
[*Log-modulated rough stochastic volatility models*](https://arxiv.org/abs/2008.03204).
The Gaussian-Volterra rough Stein--Stein equation and its fractional-kernel
specialization follow Bayer, Hall and Tempone,
[*Weak error rates for option pricing under linear rough volatility*](https://arxiv.org/abs/2009.01219).
No companion GitHub implementation is linked by that paper; the checks here
therefore use independent FP64 quadrature and direct convolution rather than
duplicating an upstream code path.

The Fukasawa--Gatheral boundary additionally verifies the production CUDA
launcher against equations (3.4), (5.2) and (6.1) of
[*A rough SABR formula*](https://doi.org/10.3934/fmf.2021003). It reproduces
their Figure 6.4 parameters:

```text
S0 = 1, r = q = 0, xi0 = 0.04, eta = 1,
H = 0.05, rho = -0.9, beta(s) = sqrt(s).
```

There is no eta conversion: the paper specifies `d xi / xi` with `eta` and
then `alpha = sqrt(xi)`, exactly as AI_factory does. In particular, the paper's
`eta=1` is the production model's `eta=1`.

The variance convention is exactly

```text
V_t = V_0 + integral K_H(t-s)
      [(variance_drift - mean_reversion * V_s) ds
       + volatility_of_variance * sqrt(V_s) dW_s].
```

Thus `variance_drift` is the constant theta, not a long-run variance.

## Commands

Run the isolated tests:

```bash
python -m unittest discover -s validation/volterra/tests -v
```

Price rough Heston and compare the current seven-factor fixture with the
continuous fractional model:

```bash
python -m validation.volterra rough-heston \
  --kernel-json \
  validation/volterra/fixtures/rough_heston_h010_t1_dt360_n7.json \
  --riccati-time-steps 1024 \
  --fourier-cutoff 80 \
  --fourier-points 1601
```

Add `--generated-price` and `--generated-standard-error` to obtain two explicit
four-sigma decisions: CUDA versus the exact lift, and CUDA versus continuous
rough Heston. These decisions must not be merged: the first diagnoses the weak
time scheme, while the second also contains the kernel approximation bias.
`--generated-json` reads the selected `--side` directly from a CUDA-probe JSON.

The standalone CUDA probe in this directory calls the production launcher but
retains both moments, unlike a smoke test that only prints the price. Build it
outside the source tree after the rough-Heston library has been built:

```bash
nvcc -ccbin=/usr/bin/g++-14 -std=c++23 -O3 -arch=sm_89 \
  -I. -Isrc \
  -c \
  validation/volterra/rough_heston_cuda_probe.cu \
  -o /tmp/rough_heston_cuda_probe.o

g++-14 -O3 /tmp/rough_heston_cuda_probe.o \
  build-dev/libai_factory_equity_rough_heston_european_option.a \
  build-dev/libai_factory_runtime.a \
  -L/usr/local/cuda/lib64 \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
  -o /tmp/rough_heston_cuda_probe

/tmp/rough_heston_cuda_probe 1048576 912000001 1
```

The architecture flag must match the configured CUDA build. Feed each emitted
price and standard error back to `rough-heston` with `--side call` or
`--side put`; this produces separate lift and continuous-rough decisions.
The optional final argument is the number of simulation steps per contractual
day. Values 1, 2 and 4 refine the weak scheme while retaining the same fitted
kernel, so the exact-lift target remains unchanged.

For example, reproduce the one-million-path call certification stored here:

```bash
python -m validation.volterra rough-heston \
  --kernel-json \
  validation/volterra/fixtures/rough_heston_h010_t1_dt360_n7.json \
  --generated-json \
  validation/volterra/fixtures/rough_heston_cuda_h010_t1_dt360_n7_1m.json \
  --side call \
  --riccati-time-steps 1024 \
  --fourier-cutoff 80 \
  --fourier-points 1601
```

Compare the direct hybrid, independently transformed hybrid and exact-grid
Gaussian rough-Bergomi references:

```bash
python -m validation.volterra rough-bergomi \
  --steps 90,180,360 \
  --antithetic-pairs 16384
```

The Cholesky reference is intentionally expensive. It is suitable for explicit
reference regeneration and convergence campaigns, not routine CUDA pricing.

The rough-SABR path and formula checks are part of the isolated unit suite:

```bash
python -m unittest validation.volterra.tests.test_rough_sabr -v
```

They validate the implemented scheme, limiting cases, both analytical endpoint
solutions used by the paper, Black--Scholes inversion and the locked CUDA
campaign described below. A full 900/100 parameter-dataset certification is
still required before publishing production price datasets.

Build the isolated Figure 6.4 CUDA probe against the production launcher:

```bash
nvcc -ccbin=/usr/bin/g++-14 -std=c++23 -O3 -arch=sm_89 \
  -I. -Isrc \
  -I/path/to/mathdx/include \
  -I/path/to/mathdx/external/cutlass/include \
  -c validation/volterra/rough_sabr_cuda_probe.cu \
  -o /tmp/rough_sabr_cuda_probe.o

g++-14 -O3 /tmp/rough_sabr_cuda_probe.o \
  build-dev/libai_factory_equity_rough_sabr_european_option.a \
  build-dev/libai_factory_runtime.a \
  -L/usr/local/cuda/lib64 \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
  -o /tmp/rough_sabr_cuda_probe
```

Generate and analyze the one-million-path campaign:

```bash
/tmp/rough_sabr_cuda_probe 1048576 913000001 \
  validation/volterra/fixtures/rough_sabr_fukasawa_gatheral_figure_6_4_1m.json

python -m validation.volterra rough-sabr-fukasawa-gatheral \
  --cuda-json \
  validation/volterra/fixtures/rough_sabr_fukasawa_gatheral_figure_6_4_1m.json \
  --output-json \
  validation/volterra/fixtures/rough_sabr_fukasawa_gatheral_figure_6_4_1m_report.json
```

The probe uses out-of-the-money puts below spot and calls at or above spot,
then compares normalized Black--Scholes implied volatilities. The finest level
uses the paper's 4,096 time steps. The 1,024 and 2,048 levels measure the last
time-refinement movement. The acceptance envelope is

```text
abs(cuda normalized IV - formula normalized IV)
    <= 0.02 + last refinement indicator + 4 * normalized IV standard error.
```

The fixed `0.02` term is an analytical-formula allowance, not Monte Carlo
noise: Fukasawa--Gatheral equation (6.1) is a short-time approximation. The
report keeps this term separate from statistical and time-grid uncertainty.

Validate selected rows directly from the three aligned AI_factory datasets:

```bash
python -m validation.volterra rough-bergomi-dataset \
  --price-json "$ROUGH_BERGOMI_CALL_PRICES" \
  --model-json datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json \
  --product-json datasets/product/european_option/european_options_01.json \
  --side call \
  --row-offset 0 \
  --limit 4 \
  --antithetic-pairs 16384 \
  --exact-grid
```

The adapter reproduces the production rule
`step_count = round(maturity / (1/360))`. It uses an independent NumPy FFT
hybrid implementation at every maturity. `--exact-grid` additionally uses the
FP64 joint-Gaussian Cholesky reference when `step_count <= 360`; the threshold
can be changed with `--exact-max-steps`. The default selection is one row so an
accidental command never starts a 1,000-price CPU campaign.

## Acceptance

`certify_price` accepts a comparison when

```text
abs(generated - reference)
    <= numerical_allowance
       + 4 * hypot(generated_standard_error, reference_standard_error).
```

The CLI reports the signed gap, allowance and z-score. Kernel bias is always
reported as a price difference and is never hidden inside the Monte-Carlo
allowance.

Before publishing a 1,000-row price dataset, run a core/stress campaign over
several maturities, log-moneyness values, Hurst exponents, correlations and
vol-of-vol levels. Persist generated references through the existing cache-only
price-reference workflow only after the refinement indicators have reached the
chosen business tolerance.

## References

- Bennedsen, Lunde and Pakkanen, *Hybrid scheme for Brownian semistationary
  processes*, Finance and Stochastics 21 (2017), arXiv:1507.03004.
- McCrickerd and Pakkanen, *Turbocharging Monte Carlo pricing for the rough
  Bergomi model*, Quantitative Finance 18 (2018), arXiv:1708.02563.
- El Euch and Rosenbaum, *The characteristic function of rough Heston models*,
  Mathematical Finance 29 (2019), arXiv:1609.02108.
- Abi Jaber and El Euch, *Multi-factor approximation of rough volatility
  models*, SIAM Journal on Financial Mathematics 10 (2019), arXiv:1801.10359.
- Bayer and Breneis, *Efficient option pricing in the rough Heston model using
  weak simulation schemes*, Quantitative Finance 25 (2025), arXiv:2310.04146.
- Fukasawa and Gatheral, *A rough SABR formula*, Frontiers of Mathematical
  Finance 1 (2022), arXiv:2105.05359.
- Bayer, Harang and Pigato, *Log-modulated rough stochastic volatility
  models*, SIAM Journal on Financial Mathematics 12 (2021), arXiv:2008.03204.
- Gatheral, Jusselin and Rosenbaum, *The quadratic rough Heston model and the
  joint S&P 500/VIX smile calibration problem*, Risks 8 (2020),
  arXiv:2001.01789.

The fixture stores the N=7 coefficients currently emitted by the CUDA host
preparation for `H=0.10`, `T=1` and `dt=1/360`. It is test evidence, not a
universal quadrature rule.

## Locked rough-Heston baseline

For the parameters used by the CUDA probe (`S0=K=1`, `r=0.02`, `q=0.01`,
`V0=0.04`, `lambda=0.30`, `variance_drift=0.02`, `nu=0.30`, `H=0.10`,
`rho=-0.70`), the one-million-path N=7 run gives:

| Side | CUDA price | CUDA SE | Exact lift | Continuous rough | z vs lift | z vs rough |
|---|---:|---:|---:|---:|---:|---:|
| call | 0.0791683644 | 0.0000959103 | 0.0791148145 | 0.0790546576 | 0.56 | 1.19 |
| put | 0.0691591129 | 0.0001216475 | 0.0692636540 | 0.0692034971 | -0.86 | -0.36 |

All four comparisons pass the four-sigma rule. The N=7 kernel price bias is
`6.02e-5`; the fine-grid Fourier and fractional-Riccati indicators are
respectively `9.90e-8` and `1.85e-7`.

The Lewis phase is regression-tested with an asymmetric gamma distribution.
This is essential: a Black-Scholes-only test cannot detect exchanging
`log(F/K)` and `log(K/F)`, because the Gaussian log-return is symmetric.

## Locked Fukasawa--Gatheral rough-SABR baseline

The production CUDA campaign uses 1,048,576 paths per point and seven scaled
log-moneyness values from `-0.45` to `0.45`. All 28 finest-grid comparisons
pass:

| Maturity | Finest steps | Max normalized-IV gap | Max 2048-to-4096 movement |
|---:|---:|---:|---:|
| 32 business days | 4,096 | 0.01335 | 0.00197 |
| 3 months | 4,096 | 0.01384 | 0.00198 |
| 6 months | 4,096 | 0.01441 | 0.00199 |
| 12 months | 4,096 | 0.01523 | 0.00202 |

The maximum gap is therefore about 1.52% of ATM implied volatility, while the
last refinement indicator is below 0.202%. The one-year case has expansion
parameter `eta * T^H = 1`, the boundary of the strict small-time regime stated
in the paper; shorter cases remain below one.

## Kernel benchmarks

The isolated production-launcher harness and the raw RTX 4090 Laptop results
are documented in [`benchmarks/README.md`](benchmarks/README.md). Timings use
CUDA events and exclude allocations, copies and host preparation.

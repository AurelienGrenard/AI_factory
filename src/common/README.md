# Shared runtime primitives

`src/common` owns reusable implementation that is independent of a concrete
model-product pair. A helper belongs here only when its contract and numerical
meaning are shared; moving code here merely because it is duplicated is not
sufficient.

| Directory | Responsibility |
|---|---|
| `closed_form/` | Generic closed-form execution |
| `equity/` | Equity observation and Monte Carlo adapters |
| `fixed_income/` | Rate analytics concepts and shared numerical identities |
| `longstaff_schwartz/` | Early-exercise execution and regression |
| `monte_carlo/` | Generic pricing kernels and moment reduction |
| `payoff/` | Product-independent payoff primitives |
| `sample/` | Generic model-sample execution |
| `simulation/` | Dynamics concepts, schedules and path loops |
| `volterra/` | Gaussian-Volterra kernels, FFT execution and concepts |

Top-level files own cross-cutting runtime services such as Philox, CUDA launch
checks and diagnostics. Their public names must describe that responsibility;
model-specific equations and product-specific rules remain in their owners.

Use the [CUDA documentation map](../../docs/cuda/README.md) for normative
interfaces. Shared mathematical notes are indexed in the
[model and curve reference index](../../docs/model-and-curve-reference-index.md).

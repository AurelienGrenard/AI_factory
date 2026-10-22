# Source modules

`src` contains concrete models, products and curves alongside reusable
simulation and pricing modules. Code is placed with the component that owns
its contract; sharing alone does not create a separate directory level.

| Directory | Responsibility |
|---|---|
| `model/` | Concrete equity and fixed-income model policies and bindings |
| `product/` | Product parameters, observations and payoff policies |
| `curve/` | Curve parameterizations and datasets |
| `closed_form/` | Generic analytic pricing, including lognormal option identities |
| `correlation/` | Host correlation datasets, packed Cholesky, correlated Brownian steps |
| `cuda/` | CUDA errors and kernel diagnostics |
| `datasets/` | Structural validation of input datasets |
| `dynamics/` | Model-state contracts and generic path advancement |
| `equity/` | Shared equity observables and single-asset pricing adapters |
| `fixed_income/` | Shared rate analytics and numerical identities |
| `longstaff_schwartz/` | Early-exercise execution and regression |
| `monte_carlo/` | Generic pricing kernels and moment reduction |
| `multi_asset/` | Correlated model packs, basket observation and joint pricing kernels |
| `payoff/` | Product-independent payoff primitives |
| `philox/` | Counter-based RNG, independent domains and random laws |
| `price_gradients/` | Sensitivity plans and reconstruction shared by engines |
| `pricing_rows/` | Compact dataset joins and device-row reconstruction |
| `probability/` | Deterministic probability functions and CDFs |
| `sample/` | Generic model-sample execution |
| `schedule/` | Time grids and contractual observation calendars |
| `volterra/` | Gaussian-Volterra kernels and FFT execution |

`multi_asset/` uses the shared dynamics contracts and Monte Carlo moment
reduction. Volterra FFT code depends on `correlation/` for Brownian cells,
without depending on the joint pricing engine. Files defining `__global__`
functions have `kernel` in their filename.

`sample.cuh` is the public sampling facade. The small cross-cutting primitives
`compensated_sum.cuh` and `workspace_layout.hpp` remain at the root.

Use the [CUDA documentation map](../project_journal/reference/cuda/README.md)
for normative interfaces. Shared mathematical notes are indexed in the
[model and curve reference index](../project_journal/reference/model-and-curve-reference-index.md).

# Schöbel–Zhu

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Under the risk-neutral measure,

```math
\frac{\mathrm dS_t}{S_t}
=(r-q)\,\mathrm dt+v_t\,\mathrm dW_t^S,
```

```math
\mathrm dv_t
=\kappa(\theta-v_t)\,\mathrm dt+\gamma\,\mathrm dW_t^v,
\qquad
\mathrm d\langle W^S,W^v\rangle_t=\rho\,\mathrm dt.
```

The volatility $`v_t`$ is a signed Gaussian Ornstein–Uhlenbeck state. Its exact
endpoint over an interval of length $`\Delta`$ is

```math
v_{t+\Delta}
=\theta+(v_t-\theta)e^{-\kappa\Delta}
+\gamma\sqrt{\frac{1-e^{-2\kappa\Delta}}{2\kappa}}\,Z_v,
\qquad Z_v\sim\mathcal N(0,1).
```

The implementation couples that endpoint to the interval Brownian increment
and advances $`\log S_t`$ by an Euler step using the left-end volatility.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`v_0`$ | `initial_volatility` | Initial volatility |
| $`\kappa`$ | `mean_reversion` | Volatility mean reversion |
| $`\theta`$ | `long_run_volatility` | Long-run volatility |
| $`\gamma`$ | `volatility_of_volatility` | Volatility diffusion scale |
| $`\rho`$ | `correlation` | Brownian correlation |

The model follows [Schöbel and Zhu (1999)](https://www.econstor.eu/handle/10419/104833).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

# Heston

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Under the risk-neutral measure,

```math
\frac{\mathrm dS_t}{S_t}
=(r-q)\,\mathrm dt+\sqrt{v_t}\,\mathrm dW_t^S,
```

```math
\mathrm dv_t
=\kappa(\theta-v_t)\,\mathrm dt
+\gamma\sqrt{v_t}\,\mathrm dW_t^v,
\qquad
\mathrm d\langle W^S,W^v\rangle_t=\rho\,\mathrm dt.
```

Here $`v_t`$ is the instantaneous variance, $`\kappa`$ its mean-reversion speed,
$`\theta`$ its long-run level, $`\gamma`$ its volatility, and $`\rho`$ the
instantaneous Brownian correlation. The state is $`(\log S_t,v_t)`$ and is
simulated with the QE-M scheme.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`v_0`$ | `initial_variance` | Initial variance |
| $`\kappa`$ | `kappa` | Variance mean reversion |
| $`\theta`$ | `theta` | Long-run variance |
| $`\gamma`$ | `gamma` | Volatility of variance |
| $`\rho`$ | `rho` | Brownian correlation |

The model follows [Heston (1993)](https://doi.org/10.1093/rfs/6.2.327); simulation uses the QE-M scheme of [Andersen (2008)](https://doi.org/10.21314/JCF.2008.189).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

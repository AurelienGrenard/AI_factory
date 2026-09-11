# Bates

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Bates combines the Heston variance process with independent lognormal jumps:

```math
\frac{\mathrm dS_t}{S_{t^-}}
=\left(r-q-\lambda\kappa_J\right)\mathrm dt
+\sqrt{v_t}\,\mathrm dW_t^S
+\left(e^Y-1\right)\mathrm dN_t,
```

```math
\mathrm dv_t
=\kappa(\theta-v_t)\,\mathrm dt
+\gamma\sqrt{v_t}\,\mathrm dW_t^v,
\qquad
\mathrm d\langle W^S,W^v\rangle_t=\rho\,\mathrm dt.
```

The Poisson process $`N`$ has intensity $`\lambda`$ and is independent of the
Brownian motions. Jump log-sizes satisfy

```math
Y\sim\mathcal N(\mu_J,\sigma_J^2),
\qquad
\kappa_J=\mathbb E[e^Y-1]
=e^{\mu_J+\sigma_J^2/2}-1.
```

Over an interval of length $`\Delta`$, the jump count satisfies
$`N_\Delta\sim\mathrm{Poisson}(\lambda\Delta)`$. Conditional on
$`N_\Delta=n`$, the total log jump is normal with mean $`n\mu_J`$ and variance
$`n\sigma_J^2`$. The Heston component uses QE-M; the jump sum is exact at
observation boundaries.

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
| $`\lambda`$ | `jump_intensity` | Jump intensity |
| $`\mu_J`$ | `jump_log_mean` | Mean log jump |
| $`\sigma_J`$ | `jump_log_volatility` | Log-jump volatility |

The model follows [Bates (1996)](https://doi.org/10.1093/rfs/9.1.69).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

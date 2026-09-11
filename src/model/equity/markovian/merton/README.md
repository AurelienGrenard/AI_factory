# Merton jump diffusion

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Under the risk-neutral measure,

```math
\frac{\mathrm dS_t}{S_{t^-}}
=\left(r-q-\lambda\kappa_J\right)\mathrm dt
+\sigma\,\mathrm dW_t
+\left(e^Y-1\right)\mathrm dN_t,
```

where $`W`$ is a Brownian motion, $`N`$ is an independent Poisson process with
intensity $`\lambda`$, and

```math
Y\sim\mathcal N(\mu_J,\sigma_J^2),
\qquad
\kappa_J=\mathbb E[e^Y-1]
=e^{\mu_J+\sigma_J^2/2}-1.
```

For an interval of length $`\Delta`$,

```math
\log S_{t+\Delta}
=\log S_t
+\left(r-q-\lambda\kappa_J-\frac{\sigma^2}{2}\right)\Delta
+\sigma\sqrt{\Delta}\,Z
+N_\Delta\mu_J+\sigma_J\sqrt{N_\Delta}\,Z_J,
```

where $`Z,Z_J\sim\mathcal N(0,1)`$ are independent and
$`N_\Delta\sim\mathrm{Poisson}(\lambda\Delta)`$. The interval transition is
exact and the state is $`\log S_t`$.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`\sigma`$ | `volatility` | Diffusion volatility |
| $`\lambda`$ | `jump_intensity` | Jump intensity |
| $`\mu_J`$ | `jump_log_mean` | Mean log jump |
| $`\sigma_J`$ | `jump_log_volatility` | Log-jump volatility |

The model follows [Merton (1976)](https://doi.org/10.1016/0304-405X(76)90022-2).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

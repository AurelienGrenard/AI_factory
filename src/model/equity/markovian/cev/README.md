# CEV

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Under the risk-neutral measure, the spot follows

```math
\mathrm dS_t=(r-q)S_t\,\mathrm dt+\sigma S_t^\beta\,\mathrm dW_t,
\qquad S_0\gt 0,
```

where $`W`$ is a standard Brownian motion, $`r`$ is the risk-free rate, $`q`$ is the
dividend yield, $`\sigma`$ is the local-volatility scale, and $`\beta`$ is the
elasticity exponent.

For the grid $`t_n=n\Delta`$, let $`Z_n\sim\mathcal N(0,1)`$ be independent. The
absorbed Milstein step is

```math
\widetilde S_{n+1}
=S_n+(r-q)S_n\Delta+\sigma S_n^\beta\sqrt{\Delta}\,Z_n
+\frac{\beta\sigma^2}{2}S_n^{2\beta-1}\Delta(Z_n^2-1),
```

```math
S_{n+1}=\max(\widetilde S_{n+1},0).
```

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`\sigma`$ | `sigma` | Local-volatility scale |
| $`\beta`$ | `beta` | Elasticity exponent |

The model follows [Cox and Ross (1976)](https://doi.org/10.1016/0304-405X(76)90023-4).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

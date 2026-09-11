# Normal Inverse Gaussian

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Let $`W`$ be a Brownian motion and $`G`$ an independent
inverse-Gaussian subordinator. Define

```math
\gamma=\sqrt{\alpha^2-\beta^2},
\qquad
G_\Delta\sim
\mathrm{IG}\!\left(
\frac{\delta\Delta}{\gamma},
(\delta\Delta)^2
\right),
```

where the two inverse-Gaussian arguments are its mean and shape. For
$`Z\sim\mathcal N(0,1)`$ independent of $`G_\Delta`$, the exact log-price
increment is

```math
\log S_{t+\Delta}
=\log S_t+(r-q+\omega)\Delta
+\beta G_\Delta+\sqrt{G_\Delta}\,Z,
```

with martingale correction

```math
\omega
=\delta\left(
\sqrt{\alpha^2-(\beta+1)^2}
-\sqrt{\alpha^2-\beta^2}
\right).
```

The admissibility condition is $`\alpha\gt |\beta+1|`$.

The state is $`\log S_t`$ and exact independent increments are sampled directly
between requested dates.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`\alpha`$ | `alpha` | Tail parameter |
| $`\beta`$ | `beta` | Asymmetry parameter |
| $`\delta`$ | `delta` | Scale parameter |

The model follows [Barndorff-Nielsen (1997)](https://doi.org/10.1111/1467-9469.00045).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

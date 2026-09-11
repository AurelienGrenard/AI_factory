# Variance Gamma

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Let $`W`$ be a Brownian motion and $`G`$ an independent
Gamma subordinator. Over an interval of length $`\Delta`$,

```math
G_\Delta\sim
\mathrm{Gamma}\!\left(\frac{\Delta}{\nu},\nu\right),
\qquad
\mathbb E[G_\Delta]=\Delta,
```

where the Gamma arguments are its shape and scale. For
$`Z\sim\mathcal N(0,1)`$ independent of $`G_\Delta`$,

```math
\log S_{t+\Delta}
=\log S_t+(r-q+\omega)\Delta
+\theta G_\Delta+\sigma\sqrt{G_\Delta}\,Z,
```

with martingale correction

```math
\omega
=\frac{1}{\nu}
\log\!\left(1-\theta\nu-\frac{\sigma^2\nu}{2}\right).
```

The admissibility condition is
$`1-\theta\nu-\sigma^2\nu/2\gt 0`$.

The state is $`\log S_t`$ and exact independent increments are sampled directly
between requested dates.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`\sigma`$ | `sigma` | Brownian scale |
| $`\nu`$ | `nu` | Gamma-clock variance rate |
| $`\theta`$ | `theta` | Gamma-clock drift |

The model follows [Madan, Carr, and Chang (1998)](https://doi.org/10.1023/A:1009703431535).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

# Black–Scholes

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Under the risk-neutral measure, the spot follows

```math
\frac{\mathrm dS_t}{S_t}=(r-q)\,\mathrm dt+\sigma\,\mathrm dW_t,
\qquad S_0\gt 0,
```

where $`W`$ is a standard Brownian motion, $`r`$ is the constant risk-free rate,
$`q`$ is the constant dividend yield, and $`\sigma`$ is the constant volatility.
Over an interval of length $`\Delta`$,

```math
\log S_{t+\Delta}
=\log S_t+\left(r-q-\frac{\sigma^2}{2}\right)\Delta
+\sigma\sqrt{\Delta}\,Z,
\qquad Z\sim\mathcal N(0,1).
```

The transition is exact at every requested observation date and the simulated
state is $`\log S_t`$.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`\sigma`$ | `volatility` | Volatility |

The model follows [Black and Scholes (1973)](https://doi.org/10.1086/260062).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

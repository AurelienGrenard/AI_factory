# Kou jump diffusion

[Dynamics](#dynamics) · [Related contracts](#related-contracts)

## Dynamics

Under the risk-neutral measure,

```math
\frac{\mathrm dS_t}{S_{t^-}}
=\left(r-q-\lambda\kappa_J\right)\mathrm dt
+\sigma\,\mathrm dW_t
+\left(e^Y-1\right)\mathrm dN_t.
```

Here $`W`$ is a Brownian motion and $`N`$ is an independent Poisson process with
intensity $`\lambda`$. Let $`\mathbf 1_{\{A\}}`$ equal one when condition $`A`$
holds and zero otherwise. The log-jump density is

```math
f_Y(y)
=p\eta_+e^{-\eta_+y}\mathbf 1_{\{y\geq0\}}
+(1-p)\eta_-e^{\eta_-y}\mathbf 1_{\{y\lt 0\}},
```

and its compensator is

```math
\kappa_J
=\mathbb E[e^Y-1]
=p\frac{\eta_+}{\eta_+-1}
+(1-p)\frac{\eta_-}{\eta_-+1}-1,
\qquad \eta_+\gt 1.
```

For an interval of length $`\Delta`$,

```math
\log S_{t+\Delta}
=\log S_t
+\left(r-q-\lambda\kappa_J-\frac{\sigma^2}{2}\right)\Delta
+\sigma\sqrt{\Delta}\,Z
+\sum_{k=1}^{N_\Delta}Y_k,
```

where $`Z\sim\mathcal N(0,1)`$,
$`N_\Delta\sim\mathrm{Poisson}(\lambda\Delta)`$, and the $`Y_k`$ are independent
with density $`f_Y`$. The interval transition is exact and the state is
$`\log S_t`$.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`S_0`$ | `spot` | Initial spot |
| $`r`$ | `risk_free_rate` | Risk-free rate |
| $`q`$ | `dividend_yield` | Dividend yield |
| $`\sigma`$ | `volatility` | Diffusion volatility |
| $`\lambda`$ | `jump_intensity` | Jump intensity |
| $`p`$ | `up_probability` | Up-jump probability |
| $`\eta_+`$ | `positive_jump_rate` | Positive-tail rate |
| $`\eta_-`$ | `negative_jump_rate` | Negative-tail rate |

The model follows [Kou (2002)](https://doi.org/10.1287/mnsc.48.8.1086.166).

## Related contracts

Shared product and engine contracts: [Markovian equity entry point](../README.md).

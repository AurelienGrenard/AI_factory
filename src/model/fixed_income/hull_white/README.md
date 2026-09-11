# Hull–White one-factor

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Related contracts](#related-contracts)

## Dynamics

The short rate is a centered Gaussian factor plus a deterministic shift:

```math
r_t=x_t+\phi(t),
\qquad
\mathrm dx_t=-a x_t\,\mathrm dt+\sigma\,\mathrm dW_t,
\qquad x_0=0.
```

Let $`P^M(0,T)`$ be the input market discount factor and define its instantaneous
forward rate by

```math
f^M(0,t)
=-\left.\frac{\partial}{\partial T}\log P^M(0,T)\right|_{T=t}.
```

The fitted shift is

```math
\phi(t)
=f^M(0,t)
+\frac{\sigma^2}{2a^2}\left(1-e^{-at}\right)^2,
```

which enforces $`P(0,T)=P^M(0,T)`$. Over an interval of length $`\Delta`$,

```math
x_{t+\Delta}
=e^{-a\Delta}x_t
+\sigma\sqrt{\frac{1-e^{-2a\Delta}}{2a}}\,Z,
\qquad Z\sim\mathcal N(0,1).
```

| Symbol | Dataset input | Meaning |
|---:|---|---|
| $`a`$ | `mean_reversion` | Mean-reversion speed |
| $`\sigma`$ | `volatility` | Factor volatility |
| $`P^M(0,T)`$ | Parametric curve | Initial discount curve |

The model follows [Hull and White (1990)](https://doi.org/10.1093/rfs/3.4.573).

## Core formulas

Let $`\tau=T-t`$. Define

```math
B(t,T)=\frac{1-e^{-a\tau}}{a},
```

and the conditional variance of the future factor integral

```math
v_I(t,T)
=\frac{\sigma^2}{a^2}
\left[
\tau-\frac{2(1-e^{-a\tau})}{a}
+\frac{1-e^{-2a\tau}}{2a}
\right].
```

The zero-coupon bond is

```math
P(t,T)=A(t,T)e^{-B(t,T)x_t},
```

```math
\log A(t,T)
=-\int_t^T\phi(u)\,\mathrm du+\frac{1}{2}v_I(t,T).
```

For $`I_t=\int_0^t x_u\,\mathrm du`$, define the full short-rate discount factor

```math
D(0,t)
=\exp\!\left(-I_t-\int_0^t\phi(u)\,\mathrm du\right).
```

Shared rate and swap identities: [fixed-income rate reference](../../../common/fixed_income/fixed-income-rate-identities-reference.md).

## Related contracts

Shared product and engine contracts: [fixed-income entry point](../README.md).

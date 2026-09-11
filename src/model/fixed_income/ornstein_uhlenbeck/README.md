# Ornstein–Uhlenbeck short-rate model

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Related contracts](#related-contracts)

## Dynamics

The short rate is the centered Gaussian state $`r_t=x_t`$, where

```math
\mathrm dx_t=-a x_t\,\mathrm dt+\sigma\,\mathrm dW_t,
\qquad x_0\in\mathbb R.
```

Here $`W`$ is a standard Brownian motion, $`a\gt 0`$ is the mean-reversion speed, and
$`\sigma\gt 0`$ is the volatility. Over an interval of length $`\Delta`$,

```math
x_{t+\Delta}
=e^{-a\Delta}x_t
+\sigma\sqrt{\frac{1-e^{-2a\Delta}}{2a}}\,Z,
\qquad Z\sim\mathcal N(0,1).
```

The endpoint and the joint law of the endpoint with the rate integral are
simulated exactly.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`x_0`$ | `initial_state` | Initial short rate |
| $`a`$ | `mean_reversion` | Mean-reversion speed |
| $`\sigma`$ | `volatility` | Volatility |

The process follows [Uhlenbeck and Ornstein (1930)](https://doi.org/10.1103/PhysRev.36.823).

## Core formulas

Let $`\tau=T-t`$. Define

```math
B(t,T)=\frac{1-e^{-a\tau}}{a},
```

and the conditional variance of the future rate integral

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
\qquad
\log A(t,T)=\frac{1}{2}v_I(t,T).
```

For the accumulated integral $`I_t=\int_0^t x_u\,\mathrm du`$, define the path
discount factor

```math
D(0,t)=e^{-I_t}.
```

Shared rate and swap identities: [fixed-income rate reference](../../../common/fixed_income/fixed-income-rate-identities-reference.md).

## Related contracts

Shared product and engine contracts: [fixed-income entry point](../README.md).

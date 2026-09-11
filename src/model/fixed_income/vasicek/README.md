# Vasicek

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Related contracts](#related-contracts)

## Dynamics

The short rate follows

```math
\mathrm dr_t=a(b-r_t)\,\mathrm dt+\sigma\,\mathrm dW_t,
\qquad r_0\in\mathbb R,
```

where $`W`$ is a standard Brownian motion, $`a\gt 0`$ is the mean-reversion speed,
$`b`$ is the long-run rate, and $`\sigma\gt 0`$ is the volatility. Over an interval
of length $`\Delta`$,

```math
r_{t+\Delta}
=b+(r_t-b)e^{-a\Delta}
+\sigma\sqrt{\frac{1-e^{-2a\Delta}}{2a}}\,Z,
\qquad Z\sim\mathcal N(0,1).
```

The endpoint and the joint law of the endpoint with the rate integral are
simulated exactly.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`r_0`$ | `initial_state` | Initial short rate |
| $`a`$ | `mean_reversion` | Mean-reversion speed |
| $`b`$ | `long_term_mean` | Long-run rate |
| $`\sigma`$ | `volatility` | Volatility |

The model follows [Vasicek (1977)](https://doi.org/10.1016/0304-405X(77)90016-2).

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
P(t,T)=A(t,T)e^{-B(t,T)r_t},
```

```math
\log A(t,T)
=\frac{1}{2}v_I(t,T)-b[\tau-B(t,T)].
```

For the accumulated integral $`I_t=\int_0^t r_u\,\mathrm du`$, define

```math
D(0,t)=e^{-I_t}.
```

Shared rate and swap identities: [fixed-income rate reference](../../../common/fixed_income/fixed-income-rate-identities-reference.md).

## Related contracts

Shared product and engine contracts: [fixed-income entry point](../README.md).

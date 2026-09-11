# G2

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Related contracts](#related-contracts)

## Dynamics

The standalone short rate is $`r_t=x_t+y_t`$, where

```math
\mathrm dx_t=-a x_t\,\mathrm dt+\sigma\,\mathrm dW_t^x,
\qquad
\mathrm dy_t=-b y_t\,\mathrm dt+\eta\,\mathrm dW_t^y,
```

```math
\mathrm d\langle W^x,W^y\rangle_t=\rho\,\mathrm dt.
```

Over an interval of length $`\Delta`$, the conditional endpoint is jointly
Gaussian. Its variances and covariance are

```math
\mathrm{Var}_t(x_{t+\Delta})
=\sigma^2\frac{1-e^{-2a\Delta}}{2a},
\qquad
\mathrm{Var}_t(y_{t+\Delta})
=\eta^2\frac{1-e^{-2b\Delta}}{2b},
```

```math
\mathrm{Cov}_t(x_{t+\Delta},y_{t+\Delta})
=\rho\sigma\eta\frac{1-e^{-(a+b)\Delta}}{a+b}.
```

The factor endpoints and their joint rate integral are simulated exactly.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`x_0`$ | `initial_state_x` | Initial first factor |
| $`y_0`$ | `initial_state_y` | Initial second factor |
| $`a`$ | `mean_reversion_x` | First mean reversion |
| $`\sigma`$ | `volatility_x` | First volatility |
| $`b`$ | `mean_reversion_y` | Second mean reversion |
| $`\eta`$ | `volatility_y` | Second volatility |
| $`\rho`$ | `correlation` | Brownian correlation |

## Core formulas

Let $`\tau=T-t`$. Define the factor loadings

```math
B_x(t,T)=\frac{1-e^{-a\tau}}{a},
\qquad
B_y(t,T)=\frac{1-e^{-b\tau}}{b}.
```

The conditional variance of the future short-rate integral is

```math
v_I(t,T)
=\frac{\sigma^2}{a^2}
\left[
\tau-\frac{2(1-e^{-a\tau})}{a}
+\frac{1-e^{-2a\tau}}{2a}
\right]
+\frac{\eta^2}{b^2}
\left[
\tau-\frac{2(1-e^{-b\tau})}{b}
+\frac{1-e^{-2b\tau}}{2b}
\right]
+2\frac{\rho\sigma\eta}{ab}
\left[
\tau-\frac{1-e^{-a\tau}}{a}
-\frac{1-e^{-b\tau}}{b}
+\frac{1-e^{-(a+b)\tau}}{a+b}
\right].
```

The zero-coupon bond is

```math
P(t,T)=A(t,T)e^{-B_x(t,T)x_t-B_y(t,T)y_t},
\qquad
\log A(t,T)=\frac{1}{2}v_I(t,T).
```

For $`I_t=\int_0^t(x_u+y_u)\,\mathrm du`$, define

```math
D(0,t)=e^{-I_t}.
```

Shared rate and swap identities: [fixed-income rate reference](../../../common/fixed_income/fixed-income-rate-identities-reference.md).

## Related contracts

European payer/receiver swaptions use terminal Monte Carlo with one exact
joint factor/integral transition under Q. Conditional bonds value the swap
at exercise; future cash-flow dates do not add simulated steps. Bermudans
retain the common LSM engine and exact joint transitions between exercises.

Shared product and engine contracts: [fixed-income entry point](../README.md).

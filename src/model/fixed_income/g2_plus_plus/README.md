# G2++

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Related contracts](#related-contracts)

## Dynamics

The short rate is

```math
r_t=x_t+y_t+\phi(t),
```

where the centered factors satisfy

```math
\mathrm dx_t=-a x_t\,\mathrm dt+\sigma\,\mathrm dW_t^x,
\qquad
\mathrm dy_t=-b y_t\,\mathrm dt+\eta\,\mathrm dW_t^y,
\qquad
\mathrm d\langle W^x,W^y\rangle_t=\rho\,\mathrm dt,
```

with $`x_0=y_0=0`$. Let $`P^M(0,T)`$ be the input market discount factor and

```math
f^M(0,t)
=-\left.\frac{\partial}{\partial T}\log P^M(0,T)\right|_{T=t}.
```

Define the time-zero integrated-factor variance

```math
\Gamma(t)
=\mathrm{Var}\!\left[\int_0^t(x_u+y_u)\,\mathrm du\right].
```

The fitted shift is

```math
\phi(t)
=f^M(0,t)+\frac{1}{2}\frac{\mathrm d}{\mathrm dt}\Gamma(t),
```

which enforces $`P(0,T)=P^M(0,T)`$. The exact G2 factor and factor-integral laws
are reused unchanged.

| Symbol | Dataset input | Meaning |
|---:|---|---|
| $`a`$ | `mean_reversion_x` | First mean reversion |
| $`\sigma`$ | `volatility_x` | First volatility |
| $`b`$ | `mean_reversion_y` | Second mean reversion |
| $`\eta`$ | `volatility_y` | Second volatility |
| $`\rho`$ | `correlation` | Brownian correlation |
| $`P^M(0,T)`$ | Parametric curve | Initial discount curve |

## Core formulas

Let $`\tau=T-t`$. Define

```math
B_x(t,T)=\frac{1-e^{-a\tau}}{a},
\qquad
B_y(t,T)=\frac{1-e^{-b\tau}}{b}.
```

The conditional variance of the future centered-factor integral is

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
```

```math
\log A(t,T)
=-\int_t^T\phi(u)\,\mathrm du+\frac{1}{2}v_I(t,T).
```

For $`I_t=\int_0^t(x_u+y_u)\,\mathrm du`$, define

```math
D(0,t)
=\exp\!\left(-I_t-\int_0^t\phi(u)\,\mathrm du\right).
```

Shared rate and swap identities: [fixed-income rate reference](../../../common/fixed_income/fixed-income-rate-identities-reference.md).

## Related contracts

European payer/receiver swaptions use the same terminal Monte Carlo policy as
G2, for both Nelson–Siegel and Svensson curves. One exact joint factor/integral
transition under Q supplies the terminal state; curve-fitted bond analytics
and the deterministic shift supply the payoff and discounting. No quadrature
or fine simulation grid is used. Bermudans retain exact-transition LSM.

Shared product and engine contracts: [fixed-income entry point](../README.md).

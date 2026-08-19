# G2

<details>
<summary>Implementation</summary>

```text
g2/
├── README.md
├── dataset.hpp
├── dataset.cpp
├── dynamics.cuh
├── dynamics.cu
├── analytics.cuh
├── analytics.cu
├── rate_option.cu
├── rate_option.cuh
├── zero_coupon_bond_option.cu
└── zero_coupon_bond_option.cuh
```

</details>

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Products](#products)

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

For an accrual period $`[T_1,T_2]`$ with contractual year fraction
$`\delta>0`$, define

```math
L(t,T_1,T_2)
=\frac{1}{\delta}
\left(\frac{P(t,T_1)}{P(t,T_2)}-1\right).
```

For swap dates $`T_0<T_1<\cdots<T_n`$ and accrual fractions
$`\delta_1,\ldots,\delta_n`$, define

```math
A_{\mathrm{swap}}(t)
=\sum_{i=1}^{n}\delta_iP(t,T_i),
```

```math
S(t;T_0,T_n)
=\frac{P(t,T_0)-P(t,T_n)}{A_{\mathrm{swap}}(t)}.
```

## Products

For every real number $`z`$, define $`[z]^+=\max(z,0)`$. The function $`\Phi`$
denotes the standard normal cumulative distribution function.

### Zero-coupon bond

**Pricing method:** Closed form.

Parameters: notional $`N`$ and maturity $`T`$.

```math
V_{\mathrm{ZCB}}(t)=NP(t,T).
```

### Option on a zero-coupon bond

**Pricing method:** Closed form.

Parameters: notional $`N`$, option expiry $`T_e`$, bond maturity $`T_b>T_e`$, bond
strike $`K_B`$, and side.

Set $`\Delta_e=T_e-t`$ and define the conditional factor moments

```math
V_x=\sigma^2\frac{1-e^{-2a\Delta_e}}{2a},
\qquad
V_y=\eta^2\frac{1-e^{-2b\Delta_e}}{2b},
```

```math
C_{xy}
=\rho\sigma\eta
\frac{1-e^{-(a+b)\Delta_e}}{a+b}.
```

The bond-forward log-variance is

```math
\nu_B^2
=B_x(T_e,T_b)^2V_x+B_y(T_e,T_b)^2V_y
+2B_x(T_e,T_b)B_y(T_e,T_b)C_{xy}.
```

Let $`\nu_B>0`$ be its positive square root and define

```math
d_1
=\frac{\log\!\left(P(t,T_b)/(K_BP(t,T_e))\right)+\nu_B^2/2}{\nu_B},
\qquad
d_2=d_1-\nu_B.
```

The unit-notional call and put prices are

```math
c_B(t;T_e,T_b,K_B)
=P(t,T_b)\Phi(d_1)-K_BP(t,T_e)\Phi(d_2),
```

```math
p_B(t;T_e,T_b,K_B)
=K_BP(t,T_e)\Phi(-d_2)-P(t,T_b)\Phi(-d_1).
```

The product value is $`Nc_B`$ for a call and $`Np_B`$ for a put.

### Caplet and floorlet

**Pricing method:** Closed form.

Parameters: notional $`N`$, fixing $`T_1`$, payment $`T_2`$, accrual $`\delta`$,
rate strike $`K`$, and side.

```math
H_{\mathrm{caplet}}(T_2)
=N\delta[L(T_1,T_1,T_2)-K]^+,
\qquad
H_{\mathrm{floorlet}}(T_2)
=N\delta[K-L(T_1,T_1,T_2)]^+.
```

Define

```math
K_B=\frac{1}{1+\delta K}.
```

Then

```math
V_{\mathrm{caplet}}(t)
=N(1+\delta K)p_B(t;T_1,T_2,K_B),
```

```math
V_{\mathrm{floorlet}}(t)
=N(1+\delta K)c_B(t;T_1,T_2,K_B).
```

### Swap and par swap rate

**Pricing method:** Closed form.

Parameters: notional $`N`$, fixed rate $`K`$, start $`T_0`$, payment dates
$`T_1,\ldots,T_n`$, and accruals $`\delta_1,\ldots,\delta_n`$.

For $`t\leq T_0`$,

```math
V_{\mathrm{float}}(t)
=N\sum_{i=1}^{n}
\delta_iL(t,T_{i-1},T_i)P(t,T_i)
=N[P(t,T_0)-P(t,T_n)],
```

```math
V_{\mathrm{fixed}}(t)=NKA_{\mathrm{swap}}(t),
```

```math
V_{\mathrm{payer}}(t)
=N[P(t,T_0)-P(t,T_n)-KA_{\mathrm{swap}}(t)]
=NA_{\mathrm{swap}}(t)[S(t;T_0,T_n)-K].
```

The payer receives floating and pays fixed. The contractual rate
$`K=S(0;T_0,T_n)`$ makes the swap worth zero at inception.

### European payer swaption

**Pricing method:** Semi-analytical — conditional Gaussian quadrature.

**Status:** Planned; the pricing launcher is not implemented.

Parameters: exercise and swap start $`T_0`$, notional $`N`$, fixed rate $`K`$,
payment dates $`T_1,\ldots,T_n`$, and accruals $`\delta_1,\ldots,\delta_n`$.

Define

```math
c_i=K\delta_i+\mathbf 1_{\{i=n\}},
\qquad i=1,\ldots,n,
```

where $`\mathbf 1_{\{i=n\}}`$ equals one for $`i=n`$ and zero otherwise. At
exercise, let $`X=x_{T_0}`$ and $`Y=y_{T_0}`$. Under the $`T_0`$-bond numeraire,
define their conditional Gaussian law by

```math
\begin{pmatrix}X\\Y\end{pmatrix}
\sim\mathcal N\!\left(
\begin{pmatrix}\mu_X\\\mu_Y\end{pmatrix},
\begin{pmatrix}V_X&C_{XY}\\C_{XY}&V_Y\end{pmatrix}
\right).
```

The five quantities $`\mu_X,\mu_Y,V_X,V_Y,C_{XY}`$ are the conditional moments
at $`T_0`$ produced by the exact two-factor Gaussian transition under that
numeraire. Define the marginal density of $`X`$ by

```math
f_X(x)
=\frac{1}{\sqrt{2\pi V_X}}
\exp\!\left[-\frac{(x-\mu_X)^2}{2V_X}\right],
```

and the conditional moments of $`Y`$ given $`X=x`$ by

```math
m(x)=\mu_Y+\frac{C_{XY}}{V_X}(x-\mu_X),
\qquad
s^2=V_Y-\frac{C_{XY}^2}{V_X}.
```

Let $`s>0`$ be the positive square root of $`s^2`$.

For each payment date, set

```math
A_i=A(T_0,T_i),
\qquad
B_{x,i}=B_x(T_0,T_i),
\qquad
B_{y,i}=B_y(T_0,T_i),
```

so that

```math
P(T_0,T_i;x,y)
=A_i e^{-B_{x,i}x-B_{y,i}y}.
```

For every $`x`$, define $`y^\star(x)`$ as the unique solution of

```math
\sum_{i=1}^{n}
c_iA_i e^{-B_{x,i}x-B_{y,i}y^\star(x)}=1.
```

Then define

```math
d_0(x)=\frac{m(x)-y^\star(x)}{s},
\qquad
d_i(x)=\frac{m(x)-B_{y,i}s^2-y^\star(x)}{s},
```

and

```math
g(x)
=\Phi(d_0(x))
-\sum_{i=1}^{n}
c_iA_i
e^{-B_{x,i}x-B_{y,i}m(x)+B_{y,i}^2s^2/2}
\Phi(d_i(x)).
```

The time-$`t`$ price is the deterministic one-dimensional integral

```math
V_{\mathrm{payer\ swaption}}(t)
=NP(t,T_0)
\int_{-\infty}^{\infty}g(x)f_X(x)\,\mathrm dx.
```

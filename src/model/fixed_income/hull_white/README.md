# Hull–White one-factor

<details>
<summary>Implementation</summary>

```text
hull_white/
├── README.md
├── parameters.hpp
├── dataset.hpp
├── dataset.cpp
├── nelson_siegel/analytics.cu
├── nelson_siegel/analytics.cuh
├── nelson_siegel/european_swaption.cu
├── nelson_siegel/european_swaption.cuh
├── nelson_siegel/rate_option.cu
├── nelson_siegel/rate_option.cuh
├── nelson_siegel/zero_coupon_bond_option.cu
├── nelson_siegel/zero_coupon_bond_option.cuh
├── svensson/analytics.cu
├── svensson/analytics.cuh
├── svensson/european_swaption.cu
├── svensson/european_swaption.cuh
├── svensson/rate_option.cu
├── svensson/rate_option.cuh
├── svensson/zero_coupon_bond_option.cu
└── svensson/zero_coupon_bond_option.cuh

../ornstein_uhlenbeck/
├── dynamics.cuh
└── dynamics.cu
```

</details>

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Products](#products)

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

For an accrual period $`[T_1,T_2]`$ with contractual year fraction
$`\delta\gt 0`$, the single-curve forward rate is

```math
L(t,T_1,T_2)
=\frac{1}{\delta}
\left(\frac{P(t,T_1)}{P(t,T_2)}-1\right).
```

For swap dates $`T_0\lt T_1\lt \cdots\lt T_n`$ and contractual accrual fractions
$`\delta_1,\ldots,\delta_n`$, define the swap annuity and par swap rate by

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

The bond pays $`N`$ at $`T`$, hence

```math
V_{\mathrm{ZCB}}(t)=NP(t,T).
```

### Option on a zero-coupon bond

**Pricing method:** Closed form.

Parameters: notional $`N`$, option expiry $`T_e`$, bond maturity $`T_b\gt T_e`$, bond
strike $`K_B`$, and side.

Let

```math
\nu_B^2
=B(T_e,T_b)^2\sigma^2
\frac{1-e^{-2a(T_e-t)}}{2a},
```

and let $`\nu_B\gt 0`$ be its positive square root. Define

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

Their payment-date payoffs are

```math
H_{\mathrm{caplet}}(T_2)
=N\delta[L(T_1,T_1,T_2)-K]^+,
```

```math
H_{\mathrm{floorlet}}(T_2)
=N\delta[K-L(T_1,T_1,T_2)]^+.
```

Define the equivalent bond strike

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

For $`t\leq T_0`$, the payer swap receives the floating leg and pays the fixed
leg. The floating leg is

```math
V_{\mathrm{float}}(t)
=N\sum_{i=1}^{n}
\delta_iL(t,T_{i-1},T_i)P(t,T_i).
```

Since

```math
\delta_iL(t,T_{i-1},T_i)P(t,T_i)
=P(t,T_{i-1})-P(t,T_i),
```

the sum telescopes:

```math
V_{\mathrm{float}}(t)
=N[P(t,T_0)-P(t,T_n)].
```

The fixed leg and payer-swap value are

```math
V_{\mathrm{fixed}}(t)=NKA_{\mathrm{swap}}(t),
```

```math
V_{\mathrm{payer}}(t)
=N[P(t,T_0)-P(t,T_n)-KA_{\mathrm{swap}}(t)]
=NA_{\mathrm{swap}}(t)[S(t;T_0,T_n)-K].
```

The contractual rate $`K=S(0;T_0,T_n)`$ makes the swap worth zero at inception.

### European payer swaption

**Pricing method:** Closed form — Jamshidian decomposition into zero-coupon bond puts.

**Implementation:**
`launch_hull_white_nelson_siegel_european_swaption_cuda` and
`launch_hull_white_svensson_european_swaption_cuda` compose each model with
its parametric curve. Their `SwaptionSide::payer` and
`SwaptionSide::receiver` specializations share the common one-factor
Jamshidian engine.

Parameters: exercise and swap start $`T_0`$, notional $`N`$, fixed rate $`K`$,
payment dates $`T_1,\ldots,T_n`$, and accruals $`\delta_1,\ldots,\delta_n`$.

The integer dates use the model clock; each $`\delta_i`$ is supplied directly
from the fixed leg's contractual day-count convention.

At exercise,

```math
H_{\mathrm{payer}}(T_0)
=N\left[
1-P(T_0,T_n)
-K\sum_{i=1}^{n}\delta_iP(T_0,T_i)
\right]^+.
```

Define

```math
c_i=K\delta_i+\mathbf 1_{\{i=n\}},
\qquad i=1,\ldots,n,
```

where $`\mathbf 1_{\{i=n\}}`$ equals one for $`i=n`$ and zero otherwise. The
notation $`P(T_0,T_i;x)`$ means that the bond formula is evaluated at
$`x_{T_0}=x`$. The unique exercise boundary $`x^\star`$ solves

```math
\sum_{i=1}^{n}c_iP(T_0,T_i;x^\star)=1.
```

Define the bond strikes

```math
K_i^\star=P(T_0,T_i;x^\star),
\qquad i=1,\ldots,n.
```

The time-$`t`$ price is

```math
V_{\mathrm{payer\ swaption}}(t)
=N\sum_{i=1}^{n}
c_i\,p_B(t;T_0,T_i,K_i^\star).
```

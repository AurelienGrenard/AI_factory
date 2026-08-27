# CIR

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Products](#products)

## Dynamics

The short rate follows the square-root diffusion

```math
\mathrm dr_t
=\kappa(\theta-r_t)\,\mathrm dt
+\sigma\sqrt{r_t}\,\mathrm dW_t,
\qquad r_0\geq0,
```

where $`W`$ is a standard Brownian motion, $`\kappa\gt 0`$ is the mean-reversion
speed, $`\theta\gt 0`$ is the long-run rate, and $`\sigma\gt 0`$ is the volatility.

For an interval of length $`\Delta`$, define

```math
c_\Delta
=\frac{\sigma^2(1-e^{-\kappa\Delta})}{4\kappa},
\qquad
d=\frac{4\kappa\theta}{\sigma^2},
\qquad
\lambda_\Delta
=\frac{e^{-\kappa\Delta}r_t}{c_\Delta}.
```

If $`\chi'^2_d(\lambda)`$ denotes a non-central chi-square variable with $`d`$
degrees of freedom and noncentrality $`\lambda`$, then

```math
r_{t+\Delta}
=c_\Delta X,
\qquad
X\sim\chi'^2_d(\lambda_\Delta).
```

This exact transition preserves non-negativity and remains valid on either
side of the Feller condition $`2\kappa\theta\geq\sigma^2`$.

For fixed-step Monte Carlo paths, define the accumulated short rate by

```math
I_t=\int_0^t r_u\,\mathrm du.
```

Each rate endpoint is simulated with the exact transition above, while the
integral is accumulated with the trapezoidal rule

```math
I_{t+\Delta}
\approx I_t+\frac{\Delta}{2}(r_t+r_{t+\Delta}).
```

Thus the rate transition is exact on every numerical step, whereas the joint
rate-integral transition converges as the fixed step is refined.

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`r_0`$ | `initial_state` | Initial short rate |
| $`\kappa`$ | `mean_reversion` | Mean-reversion speed |
| $`\theta`$ | `long_term_mean` | Long-run rate |
| $`\sigma`$ | `volatility` | Volatility |

The model follows [Cox, Ingersoll, and Ross (1985)](https://doi.org/10.2307/1911242).

## Core formulas

Let $`\tau=T-t`$ and define

```math
\gamma=\sqrt{\kappa^2+2\sigma^2}.
```

The zero-coupon bond is exponential-affine:

```math
P(t,T)=A(t,T)e^{-B(t,T)r_t},
```

with

```math
B(t,T)
=\frac{2(e^{\gamma\tau}-1)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma},
```

```math
A(t,T)
=\left[
\frac{2\gamma e^{(\kappa+\gamma)\tau/2}}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}
\right]^{2\kappa\theta/\sigma^2}.
```

For an accumulated short-rate integral
$`I_t=\int_0^t r_u\,\mathrm du`$, define

```math
D(0,t)=e^{-I_t}.
```

For an accrual period $`[T_1,T_2]`$ with contractual year fraction
$`\delta\gt 0`$, define

```math
L(t,T_1,T_2)
=\frac{1}{\delta}
\left(\frac{P(t,T_1)}{P(t,T_2)}-1\right).
```

For swap dates $`T_0\lt T_1\lt \cdots\lt T_n`$ and accrual fractions
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

For every real number $`z`$, define $`[z]^+=\max(z,0)`$.

### Zero-coupon bond

**Pricing method:** Closed form.

Parameters: notional $`N`$ and maturity $`T`$.

```math
V_{\mathrm{ZCB}}(t)=NP(t,T).
```

### Option on a zero-coupon bond

**Pricing method:** Closed form.

Parameters: notional $`N`$, option expiry $`T_e`$, bond maturity $`T_b\gt T_e`$, bond
strike $`K_B`$, and side.

Define the expiry-to-maturity bond coefficients and critical short rate by

```math
A_e=A(T_e,T_b),
\qquad
B_e=B(T_e,T_b),
\qquad
r^\star=\frac{\log A_e-\log K_B}{B_e}.
```

Let $`h=T_e-t`$ and define

```math
u=\frac{2\gamma e^{-\gamma h}}
{\sigma^2(1-e^{-\gamma h})},
\qquad
\psi=\frac{\kappa+\gamma}{\sigma^2},
```

```math
\ell
=\frac{8\gamma^2e^{-\gamma h}r_t}
{\sigma^4(1-e^{-\gamma h})^2},
\qquad
q_e=u+\psi,
\qquad
q_b=q_e+B_e.
```

Using the degrees of freedom $`d=4\kappa\theta/\sigma^2`$ already defined by the
transition, set

```math
\lambda_e=\frac{\ell}{q_e},
\qquad
z_e=2r^\star q_e,
\qquad
\lambda_b=\frac{\ell}{q_b},
\qquad
z_b=2r^\star q_b.
```

Let $`F_{d,\lambda}(z)`$ be the CDF of $`\chi'^2_d(\lambda)`$ at $`z`$ and define
its survival function by
$`\overline F_{d,\lambda}(z)=1-F_{d,\lambda}(z)`$. The unit-notional prices are

```math
c_B(t;T_e,T_b,K_B)
=P(t,T_b)F_{d,\lambda_b}(z_b)
-K_BP(t,T_e)F_{d,\lambda_e}(z_e),
```

```math
p_B(t;T_e,T_b,K_B)
=K_BP(t,T_e)\overline F_{d,\lambda_e}(z_e)
-P(t,T_b)\overline F_{d,\lambda_b}(z_b).
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
\delta_iL(t,T_{i-1},T_i)P(t,T_i).
```

Since

```math
\delta_iL(t,T_{i-1},T_i)P(t,T_i)
=P(t,T_{i-1})-P(t,T_i),
```

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

The payer receives floating and pays fixed. The contractual rate
$`K=S(0;T_0,T_n)`$ makes the swap worth zero at inception.

### European payer swaption

**Pricing method:** Closed form — Jamshidian decomposition into zero-coupon bond puts.

**Implementation:**
`launch_cir_european_swaption_cuda<SwaptionSide::payer>` and the
`SwaptionSide::receiver` specialization share the common one-factor
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
notation $`P(T_0,T_i;r)`$ means that the bond formula is evaluated at
$`r_{T_0}=r`$. The unique Jamshidian boundary $`r_J^\star`$ solves

```math
\sum_{i=1}^{n}c_iP(T_0,T_i;r_J^\star)=1.
```

Define

```math
K_i^\star=P(T_0,T_i;r_J^\star),
\qquad i=1,\ldots,n.
```

Then

```math
V_{\mathrm{payer\ swaption}}(t)
=N\sum_{i=1}^{n}
c_i\,p_B(t;T_0,T_i,K_i^\star).
```

### Bermudan swaption

**Pricing method:** Monte Carlo — Longstaff–Schwartz.

Parameters: notional $`N`$, fixed rate $`K`$, first exercise $`E_0`$, regular
payment interval $`\Delta`$, contractual accrual $`\delta`$, payment count $`n`$,
exercise count $`m\leq n`$, and side. Let

```math
E_j=E_0+j\Delta,\quad j=0,\ldots,m-1,
\qquad
T_i=E_0+i\Delta,\quad i=1,\ldots,n.
```

```math
H_j^{\mathrm{payer}}
=N\left[1-P(E_j,T_n)-K\delta\sum_{i=j+1}^{n}P(E_j,T_i)\right]^+,
\qquad
H_j^{\mathrm{receiver}}
=N\left[P(E_j,T_n)+K\delta\sum_{i=j+1}^{n}P(E_j,T_i)-1\right]^+.
```

With $`I_t=\int_0^t r_u\,\mathrm du`$ and $`D(s,t)=e^{-(I_t-I_s)}`$, a
degree-three Hermite regression of the standardized rate estimates

```math
C_j(r)=\mathbb E\!\left[D(E_j,E_{j+1})V_{j+1}\mid r_{E_j}=r\right].
```

Backward exercise uses $`H_j\geq\widehat C_j`$, and the price is
$`\mathbb E[D(0,E_0)V_0]`$. CIR endpoints are exact; $`I_t`$ uses a fine
trapezoidal grid with two simulation steps per business day.

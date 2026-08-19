# G2++

| At a glance | Value |
|---|---|
| Process | Two-factor Gaussian model fitted to an initial curve |
| Transition | Exact G2 states and state/integral laws |
| Path state | Centered factors `x`, `y`, plus deterministic `phi(t)` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements the fitted two-factor Gaussian short-rate model

```text
r_t = x_t + y_t + phi(t),
dx_t = -a x_t dt + sigma dW_t^x,
dy_t = -b y_t dt + eta dW_t^y,
d<W^x,W^y>_t = rho dt.
```

`W^x` and `W^y` are standard Brownian motions with instantaneous correlation
`rho`. `x` and `y` are centered Gaussian Ornstein–Uhlenbeck factors starting
at zero. `phi(t)` is deterministic and derived from the selected initial
curve.

The deterministic shift fits the selected initial curve exactly. This is the
two-factor counterpart of the Gaussian term-structure construction in
[Hull and White (1990)](https://doi.org/10.1093/rfs/3.4.573).

## Formula index

- [Analytics — fitted two-factor Gaussian bonds](#analytics)
- [Zero-coupon bond — fitted exponential-affine formula](#zero-coupon-bond)
- [Zero-coupon bond option — Gaussian bond-forward projection](#zero-coupon-bond-option)
- [Caplet / floorlet — scaled bond-option identity](#caplet--floorlet)
- [Swap and swap rate — discounted-leg formulas](#swap-and-swap-rate)
- [European payer swaption — conditional Gaussian quadrature](#european-payer-swaption)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load curve-independent two-factor rows.
- [`nelson_siegel/`](nelson_siegel/) and [`svensson/`](svensson/) contain curve-specific analytics and launchers.
- each curve folder provides `analytics.cuh/.cu`, `rate_option.cuh/.cu`, and
  `zero_coupon_bond_option.cuh/.cu`.

There is deliberately no local dynamics file: the exact centered process is
reused from [`../g2/dynamics.cuh`](../g2/dynamics.cuh).

## Dataset row

Initial factor states are zero; the initial term structure is supplied as a
separate curve row.

| Symbol | Dataset field |
|---|---|
| $a$ | `mean_reversion_x` |
| $\sigma$ | `volatility_x` |
| $b$ | `mean_reversion_y` |
| $\eta$ | `volatility_y` |
| $\rho$ | `correlation` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `process` | G2 process built from $a$, $\sigma$, $b$, $\eta$, and $\rho$ |
| `initial_curve` | One Nelson–Siegel or Svensson curve row |

Each curve namespace calls this structure `G2PlusPlusFittedParameters` and
builds it with `compose_model`. The stochastic state is
`G2State {state_x, state_y}`; `short_rate_shift` and `short_rate` reconstruct
the fitted short rate.

## Analytics

In both curve namespaces,

$$
P(t,T)=A(t,T)e^{-B_x(t,T)x_t-B_y(t,T)y_t},
$$

with the same factor loadings as standalone G2,

$$
B_x(t,T)=\frac{1-e^{-a(T-t)}}a,\qquad
B_y(t,T)=\frac{1-e^{-b(T-t)}}b,
$$

and

$$
\log A(t,T)=-\int_t^T\phi(s)ds
+\frac12\operatorname{Var}_t\!\left[\int_t^T(x_s+y_s)ds\right].
$$

`B` returns the shared `G2BondLoadings` type. At $t=0$, `log_A` reads the
fitted curve directly; otherwise `log_A`, $B_x$, and $B_y$ are produced from
one grouped coefficient calculation.

For $I_t=\int_0^t(x_u+y_u)\,du$,

$$
D(0,t)=\exp\!\left(-I_t-\int_0^t\phi(u)\,du\right).
$$

In the single-curve convention, the same $P(t,T)$ projects forwards and
discounts cashflows. The simple forward, swap annuity, and par swap rate are

$$
L(t,T_1,T_2)=\frac1\delta
\left(\frac{P(t,T_1)}{P(t,T_2)}-1\right),
$$

$$
\operatorname{Ann}(t)=\sum_{i=1}^n\delta_iP(t,T_i),
\qquad
S(t;T_0,T_n)=
\frac{P(t,T_0)-P(t,T_n)}{\operatorname{Ann}(t)}.
$$

## Zero-coupon bond

For notional $N$ paid at $T$,

$$
V_{\mathrm{ZCB}}(t)
=NP(t,T)
=NA(t,T)e^{-B_x(t,T)x_t-B_y(t,T)y_t}.
$$

The fitted shift fixes $P(0,T)$; the two-factor Gaussian transform supplies
the conditional affine price.

## Zero-coupon bond option

Let $S$ be the option expiry, $T>S$ the bond maturity, $K_B$ the strike, and
$\Delta=S-t$. Define

$$
V_x=\sigma^2\frac{1-e^{-2a\Delta}}{2a},
\quad
V_y=\eta^2\frac{1-e^{-2b\Delta}}{2b},
\quad
C_{xy}=\rho\sigma\eta\frac{1-e^{-(a+b)\Delta}}{a+b},
$$

$$
\nu^2=B_x(S,T)^2V_x+B_y(S,T)^2V_y
+2B_x(S,T)B_y(S,T)C_{xy}.
$$

With

$$
d_1=\frac{\log\!\left(P(t,T)/(K_BP(t,S))\right)+\nu^2/2}{\nu},
\qquad d_2=d_1-\nu,
$$

$$
C_{\mathrm{ZCB}}(t)=P(t,T)\Phi(d_1)-K_BP(t,S)\Phi(d_2),
$$

$$
P_{\mathrm{ZCB}}(t)=K_BP(t,S)\Phi(-d_2)-P(t,T)\Phi(-d_1).
$$

The deterministic shift changes bond levels, not the conditional Gaussian
bond-forward variance.

## Caplet / floorlet

For fixing $T_1$, payment $T_2$, accrual $\delta$, strike $K$, and notional
$N$,

$$
\Pi_{\mathrm{caplet}}(T_2)=N\delta[L(T_1,T_1,T_2)-K]^+,
\qquad
\Pi_{\mathrm{floorlet}}(T_2)=N\delta[K-L(T_1,T_1,T_2)]^+.
$$

Let $K_B=(1+\delta K)^{-1}$. Then

$$
V_{\mathrm{caplet}}(t)
=N(1+\delta K)P_{\mathrm{ZCB}}(t;T_1,T_2,K_B),
$$

$$
V_{\mathrm{floorlet}}(t)
=N(1+\delta K)C_{\mathrm{ZCB}}(t;T_1,T_2,K_B).
$$

## Swap and swap rate

$$
V_{\mathrm{float}}(t)
=N\sum_{i=1}^n\delta_iL(t,T_{i-1},T_i)P(t,T_i)
=N[P(t,T_0)-P(t,T_n)],
$$

$$
V_{\mathrm{fixed}}(t)=NK\operatorname{Ann}(t),
$$

$$
V_{\mathrm{payer}}(t)
=N[P(t,T_0)-P(t,T_n)-K\operatorname{Ann}(t)]
=N\operatorname{Ann}(t)[S(t;T_0,T_n)-K].
$$

Here $K$ is the contractual fixed rate. Setting $K=S(0;T_0,T_n)$ makes the
swap worth zero at inception; afterward $S(t)$ moves while $K$ stays fixed.

## European payer swaption

**Method: planned conditional Gaussian quadrature.** At exercise and swap
start $T_0$, set $c_i=K\delta_i+\mathbf 1_{\{i=n\}}$. Under the $T_0$-bond
numeraire,

$$
V(t)=NP(t,T_0)\,
\mathbb E_t^{T_0}\!\left[
\left(1-\sum_{i=1}^nc_iP(T_0,T_i;X,Y)\right)^+
\right].
$$

With $P(T_0,T_i;x,y)=A_i e^{-B_{x,i}x-B_{y,i}y}$, solve for every $x$

$$
\sum_{i=1}^nc_iA_i e^{-B_{x,i}x-B_{y,i}y^\star(x)}=1.
$$

If $Y\mid X=x\sim\mathcal N(m(x),s^2)$, set

$$
d_0(x)=\frac{m(x)-y^\star(x)}s,
\qquad
d_i(x)=\frac{m(x)-B_{y,i}s^2-y^\star(x)}s,
$$

$$
g(x)=\Phi(d_0(x))-
\sum_{i=1}^nc_iA_i
e^{-B_{x,i}x-B_{y,i}m(x)+B_{y,i}^2s^2/2}\Phi(d_i(x)).
$$

The selected deterministic price is

$$
V(t)=NP(t,T_0)\int_{-\infty}^{\infty}g(x)f_X(x)\,dx,
$$

evaluated by one-dimensional Gaussian quadrature. The swaption launcher is not
implemented yet.

## Dynamics interface

| Function | Role |
|---|---|
| `compose_model` | Combine one process row with one initial curve |
| `short_rate_shift`, `short_rate` | Evaluate $\phi(t)$ and reconstruct $r_t$ |
| `log_discount_factor`, `discount_factor` | Consume and exponentiate the scalar accumulated path integral |
| `zero_coupon_bond` | Evaluate $P(t,T)$ analytically |
| `zero_coupon_bond_call_price`, `zero_coupon_bond_put_price` | Price bond options analytically |
| `forward_rate`, `swap_rate` | Evaluate curve-derived rates |
| Reused G2 dynamics | Simulate exact state-only or state/integral laws |

Both the correlated factor law and its joint rate integral are exact over the
requested interval.

<details>
<summary>Exact analytics signatures</summary>

The declarations below omit CUDA attributes and use `CurveParameters` for
either supported curve type.

```cpp
G2PlusPlusFittedParameters compose_model(const G2PlusPlusModelParameters&, const CurveParameters&);
float short_rate_shift(const G2PlusPlusFittedParameters&, float time);
float short_rate(const G2PlusPlusFittedParameters&, const G2State&, float time);
float log_A(const G2PlusPlusFittedParameters&, float valuation_time, float maturity);
float A(const G2PlusPlusFittedParameters&, float valuation_time, float maturity);
G2BondLoadings B(const G2PlusPlusFittedParameters&, float valuation_time, float maturity);
float log_zero_coupon_bond(const G2PlusPlusFittedParameters&, const G2State&, float valuation_time, float maturity);
float log_discount_factor(const G2PlusPlusFittedParameters&, float state_integral, float time);
float discount_factor(const G2PlusPlusFittedParameters&, float state_integral, float time);
float zero_coupon_bond(const G2PlusPlusFittedParameters&, const G2State&, float valuation_time, float maturity);
float zero_coupon_bond_call_price(const G2PlusPlusFittedParameters&, const G2State&, float valuation_time, float expiry, float maturity, float strike);
float zero_coupon_bond_put_price(const G2PlusPlusFittedParameters&, const G2State&, float valuation_time, float expiry, float maturity, float strike);
float forward_rate(const G2PlusPlusFittedParameters&, const G2State&, float valuation_time, float start, float end, float accrual);
float swap_rate(const G2PlusPlusFittedParameters&, const G2State&, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

`CurveParameters` is `NelsonSiegelParameters` or `SvenssonParameters`; the
exact declarations are in each curve subfolder's `analytics.cuh`.

</details>

## Random-number strategy

Current pricing is closed form and consumes no random numbers. Future path
simulation inherits the G2 rule: one `philox::UniformSequence(key, path)` and
one normal cache for the complete path, with two normals for states or three
for states plus integral per interval.

## Pricing kernels

Both curve families price rate and zero-coupon-bond options with one thread per
`(model, curve, product)` result. Files follow `PreparedRow` →
`evaluate_price<Side>` → launcher, with compile-time call/put specialization.

## Memory and numerical policy

The two-factor dynamics are reused rather than copied per curve. Analytical
curve evaluation avoids term-structure arrays, exact Gaussian transitions
avoid time stepping, and prepared Cholesky loadings avoid repeat work.
Fast-math is forbidden.

## American and Bermudan options

No G2++ American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/g2_plus_plus/),
[validation](../../../../validation/model/fixed_income/g2_plus_plus/),
[Nelson–Siegel curve](../../../curve/nelson_siegel/),
[Svensson curve](../../../curve/svensson/), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

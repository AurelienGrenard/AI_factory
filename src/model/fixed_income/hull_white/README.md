# Hull–White

| At a glance | Value |
|---|---|
| Process | One-factor Gaussian model fitted to an initial curve |
| Transition | Exact OU state and state/integral laws |
| Path state | Centered factor `x`, plus deterministic `phi(t)` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements the one-factor Gaussian model fitted exactly to an
input discount curve through

```text
r_t = x_t + phi(t),    dx_t = -a x_t dt + sigma dW_t.
```

`W` is a standard Brownian motion, `x` is a centered Gaussian
Ornstein–Uhlenbeck factor starting at zero, and `phi(t)` is a deterministic
function derived from the selected initial curve.

The deterministic shift `phi` makes the model reproduce the selected initial
curve. See [Hull and White (1990)](https://doi.org/10.1093/rfs/3.4.573).

## Formula index

- [Analytics — fitted Gaussian state and affine bonds](#analytics)
- [Zero-coupon bond — fitted exponential-affine formula](#zero-coupon-bond)
- [Zero-coupon bond option — expiry-bond numeraire](#zero-coupon-bond-option)
- [Caplet / floorlet — scaled bond-option identity](#caplet--floorlet)
- [Swap and swap rate — discounted-leg formulas](#swap-and-swap-rate)
- [European payer swaption — Jamshidian decomposition](#european-payer-swaption)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load curve-independent `(a, sigma)` rows.
- [`nelson_siegel/`](nelson_siegel/) and [`svensson/`](svensson/) contain curve-specific analytics and launchers.
- each curve folder provides `analytics.cuh/.cu`, `rate_option.cuh/.cu`, and
  `zero_coupon_bond_option.cuh/.cu`.

There is deliberately no local dynamics pair: the stochastic factor reuses
[`../ornstein_uhlenbeck/dynamics.cuh`](../ornstein_uhlenbeck/dynamics.cuh).

## Dataset row

The initial curve is a separate input so the same process row can be combined
with Nelson–Siegel or Svensson without duplicating model datasets.

| Symbol | Dataset field |
|---|---|
| $a$ | `mean_reversion` |
| $\sigma$ | `volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `process` | OU process built from $a$ and $\sigma$ |
| `initial_curve` | One Nelson–Siegel or Svensson curve row |

Each curve namespace calls this structure `HullWhiteFittedParameters` and
builds it with `compose_model`. The stochastic state is one centered OU
`float`; `short_rate_shift` and `short_rate` reconstruct the fitted rate.

## Analytics

In both curve namespaces,

$$
P(t,T)=A(t,T)e^{-B(t,T)x_t},\qquad
B(t,T)=\frac{1-e^{-a(T-t)}}a,
$$

and

$$
\log A(t,T)=-\int_t^T\phi(s)ds
+\frac12\operatorname{Var}_t\!\left[\int_t^T x_sds\right].
$$

At $t=0$, `log_A` reads the fitted curve discount directly, avoiding
cancellation between shift and convexity terms. Each ZCB computes `log_A` and
`B` together once.

For a simulated centered-factor integral $I_t=\int_0^t x_u\,du$,

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
V_{\mathrm{ZCB}}(t)=NP(t,T)=NA(t,T)e^{-B(t,T)x_t}.
$$

The deterministic shift fixes the initial bond levels; the conditional
Gaussian transform supplies the stochastic affine factor.

## Zero-coupon bond option

Let $S$ be the option expiry, $T>S$ the bond maturity, and $K_B$ the strike.
Under the $S$-bond numeraire,

$$
\nu^2=B(S,T)^2\sigma^2
\frac{1-e^{-2a(S-t)}}{2a},
$$

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

The shift changes the bond levels but not the Gaussian bond-forward variance.
Both fitted-curve implementations use one analytical CUDA thread per price.

## Caplet / floorlet

For fixing $T_1$, payment $T_2$, accrual $\delta$, strike $K$, and notional
$N$,

$$
\Pi_{\mathrm{caplet}}(T_2)=N\delta[L(T_1,T_1,T_2)-K]^+,
\qquad
\Pi_{\mathrm{floorlet}}(T_2)=N\delta[K-L(T_1,T_1,T_2)]^+.
$$

Let

$$
K_B=\frac1{1+\delta K}.
$$

Then

$$
V_{\mathrm{caplet}}(t)
=N(1+\delta K)P_{\mathrm{ZCB}}(t;T_1,T_2,K_B),
$$

$$
V_{\mathrm{floorlet}}(t)
=N(1+\delta K)C_{\mathrm{ZCB}}(t;T_1,T_2,K_B).
$$

## Swap and swap rate

For a swap starting at $T_0$ with payments $T_1,\ldots,T_n$,

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

The first equality telescopes after substituting the zero-coupon definition of
each forward rate.

## European payer swaption

**Method: planned Jamshidian decomposition.** At exercise and swap start $T_0$,

$$
\Pi_{\mathrm{payer}}(T_0)=N\left[
1-P(T_0,T_n)-K\sum_{i=1}^n\delta_iP(T_0,T_i)
\right]^+.
$$

Set $c_i=K\delta_i+\mathbf 1_{\{i=n\}}$ and solve

$$
\sum_{i=1}^nc_iP(T_0,T_i;x^\star)=1.
$$

With $K_i^\star=P(T_0,T_i;x^\star)$,

$$
V_{\mathrm{payer\ swaption}}(t)
=N\sum_{i=1}^nc_iP_{\mathrm{ZCB}}(t;T_0,T_i,K_i^\star).
$$

The swaption launcher is not implemented yet.

## Dynamics interface

| Function | Role |
|---|---|
| `compose_model` | Combine one process row with one initial curve |
| `short_rate_shift`, `short_rate` | Evaluate $\phi(t)$ and reconstruct $r_t$ |
| `log_discount_factor`, `discount_factor` | Consume and exponentiate the scalar accumulated path integral |
| `zero_coupon_bond` | Evaluate $P(t,T)$ analytically |
| `zero_coupon_bond_call_price`, `zero_coupon_bond_put_price` | Price bond options analytically |
| `forward_rate`, `swap_rate` | Evaluate curve-derived rates |
| Reused OU dynamics | Simulate exact state-only or state/integral laws |

State and state/integral transitions are exact over the requested interval; a
fine artificial grid is not introduced.

<details>
<summary>Exact analytics signatures</summary>

The declarations below omit CUDA attributes and use `CurveParameters` for
either supported curve type.

```cpp
HullWhiteFittedParameters compose_model(const HullWhiteModelParameters&, const CurveParameters&);
float short_rate_shift(const HullWhiteFittedParameters&, float time);
float short_rate(const HullWhiteFittedParameters&, float state, float time);
float log_A(const HullWhiteFittedParameters&, float valuation_time, float maturity);
float A(const HullWhiteFittedParameters&, float valuation_time, float maturity);
float B(const HullWhiteFittedParameters&, float valuation_time, float maturity);
float log_zero_coupon_bond(const HullWhiteFittedParameters&, float state, float valuation_time, float maturity);
float log_discount_factor(const HullWhiteFittedParameters&, float state_integral, float time);
float discount_factor(const HullWhiteFittedParameters&, float state_integral, float time);
float zero_coupon_bond(const HullWhiteFittedParameters&, float state, float valuation_time, float maturity);
float zero_coupon_bond_call_price(const HullWhiteFittedParameters&, float state, float valuation_time, float expiry, float maturity, float strike);
float zero_coupon_bond_put_price(const HullWhiteFittedParameters&, float state, float valuation_time, float expiry, float maturity, float strike);
float forward_rate(const HullWhiteFittedParameters&, float state, float valuation_time, float start, float end, float accrual);
float swap_rate(const HullWhiteFittedParameters&, float state, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

`CurveParameters` is `NelsonSiegelParameters` or `SvenssonParameters`; the
exact declarations are in each curve subfolder's `analytics.cuh`.

</details>

## Random-number strategy

Current pricing is closed form and consumes no random numbers. Future path
simulation inherits the OU rule: one `philox::UniformSequence(key, path)` and
one normal cache for the complete regular-grid path.

## Pricing kernels

Both curve families implement rate and zero-coupon-bond options with one
thread per `(model, curve, product)` result. Files follow `PreparedRow` →
`evaluate_price<Side>` → launcher. Call/put is compile-time `OptionSide`.

## Memory and numerical policy

The stochastic model is not duplicated per curve. One compact fitted
parameter structure and one scalar state are sufficient. Analytical curve
functions avoid stored term-structure grids, and exact Gaussian moments avoid
time stepping. Fast-math is forbidden.

## American and Bermudan options

No Hull–White American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/hull_white/),
[validation](../../../../validation/model/fixed_income/hull_white/),
[Nelson–Siegel curve](../../../curve/nelson_siegel/),
[Svensson curve](../../../curve/svensson/), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

# G2

| At a glance | Value |
|---|---|
| Process | Two-factor correlated Gaussian short rate |
| Transition | Exact states and state/integral laws |
| Path state | `state_x`, `state_y`, optionally `state_integral` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements a standalone two-factor Gaussian short-rate model:

```text
dx_t = -a x_t dt + sigma dW_t^x
dy_t = -b y_t dt + eta dW_t^y
d<W^x,W^y>_t = rho dt,    r_t = x_t + y_t.
```

`W^x` and `W^y` are standard Brownian motions with instantaneous correlation
`rho`. `x` and `y` are correlated centered Gaussian Ornstein–Uhlenbeck
factors. Their sum is the standalone short rate.

It is the unshifted two-factor Gaussian process also reused by G2++. The
mean-reverting factors are exact Ornstein–Uhlenbeck processes; the fitted
time-dependent extension follows the Gaussian term-structure framework of
[Hull and White (1990)](https://doi.org/10.1093/rfs/3.4.573).

## Formula index

- [Analytics — correlated Gaussian factors and affine bonds](#analytics)
- [Zero-coupon bond — two-factor exponential-affine formula](#zero-coupon-bond)
- [Zero-coupon bond option — Gaussian bond-forward projection](#zero-coupon-bond-option)
- [Caplet / floorlet — scaled bond-option identity](#caplet--floorlet)
- [Swap and swap rate — discounted-leg formulas](#swap-and-swap-rate)
- [European payer swaption — conditional Gaussian quadrature](#european-payer-swaption)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load both factors plus initial state.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact correlated state and state/integral laws.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) provide short-rate, discount, bond, forward, swap, and option formulas.
- [`rate_option.cuh`](rate_option.cuh) / [`rate_option.cu`](rate_option.cu) and [`zero_coupon_bond_option.cuh`](zero_coupon_bond_option.cuh) / [`zero_coupon_bond_option.cu`](zero_coupon_bond_option.cu) own launchers.

## Dataset row

`G2ModelParameters` combines `G2ProcessParameters` with both initial factor
states.

| Symbol | Dataset field |
|---|---|
| $x_0$ | `initial_state_x` |
| $y_0$ | `initial_state_y` |
| $a$ | `mean_reversion_x` |
| $\sigma$ | `volatility_x` |
| $b$ | `mean_reversion_y` |
| $\eta$ | `volatility_y` |
| $\rho$ | `correlation` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `decay_x`, `decay_y` | $e^{-a\Delta t}$, $e^{-b\Delta t}$ |
| `state_x_standard_deviation` | Exact variance of $x_{t+\Delta t}$ |
| `state_y_x_normal_loading`, `state_y_independent_standard_deviation` | Cholesky loadings of the correlated endpoints |
| `integral_state_x_loading`, `integral_state_y_loading` | Conditional integral mean |
| `integral_x_normal_loading`, `integral_y_normal_loading`, `integral_independent_standard_deviation` | Joint state/integral Cholesky loadings |

The integral fields belong to the `joint` transition. The joint state adds one
accumulated integral to the two factor values.

## Dynamics interface

| Function | Role |
|---|---|
| `integral_moments` | Return both factor loadings and exact integral variance |
| `prepare_model` | Precompute an exact correlated two-factor transition |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Apply one prepared transition directly |
| `simulate_on_regular_grid` | Store pre-terminal date-major factor states |
| `joint::*` variants | Evolve both factors and their accumulated integral |

All laws are exact over the requested interval. Products needing only rates
sample `(x,y)` at observation dates. Products needing discounting use the
exact three-dimensional joint law instead of a finer time grid.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
G2IntegralMoments integral_moments(const G2ProcessParameters&, float interval);
G2ExactTransition prepare_model(const G2ProcessParameters&, float interval);
void one_step_transition(const G2ExactTransition&, float x_normal, float y_normal, G2State&);
G2State simulate_terminal_state(const G2ExactTransition&, G2State initial_state, float x_normal, float y_normal);
G2State simulate_on_regular_grid(const G2ExactTransition& stub, const G2ExactTransition& regular, G2State initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states_x, float* states_y);
joint::G2JointExactTransition joint::prepare_model(const G2ProcessParameters&, float interval);
void joint::one_step_transition(const joint::G2JointExactTransition&, float x_normal, float y_normal, float integral_normal, joint::G2JointState&);
joint::G2JointState joint::simulate_terminal_state(const joint::G2JointExactTransition&, G2State initial_state, float x_normal, float y_normal, float integral_normal);
joint::G2JointState joint::simulate_on_regular_grid(const joint::G2JointExactTransition& stub, const joint::G2JointExactTransition& regular, G2State initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states_x, float* states_y, float* integrals);
```

</details>

## Random-number strategy

Direct terminal helpers receive normals from the caller. Grid helpers keep one
`philox::UniformSequence(key, path)` and one normal cache alive. The state law
uses two normals per interval and the joint state/integral law uses three.

## Analytics

For $\tau=T-t$,

$$
P(t,T)=A(t,T)e^{-B_x(t,T)x_t-B_y(t,T)y_t},
$$

where

$$
B_x(t,T)=\frac{1-e^{-a\tau}}a,\qquad
B_y(t,T)=\frac{1-e^{-b\tau}}b,
$$

and $\log A(t,T)=\tfrac12\operatorname{Var}_t[\int_t^T(x_s+y_s)ds]$.
`B` returns `G2BondLoadings {state_x, state_y}`. Bond evaluation obtains
`log_A`, $B_x$, and $B_y$ from one shared integral-moment calculation.

For $I_t=\int_0^t(x_u+y_u)\,du$, $D(0,t)=e^{-I_t}$. In the single-curve
convention, the same $P(t,T)$ projects forwards and discounts cashflows. The
simple forward, swap annuity, and par swap rate are

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

This is the conditional Gaussian transform of the integrated two-factor
short rate.

## Zero-coupon bond option

Let $S$ be the option expiry, $T>S$ the bond maturity, $K_B$ the strike, and
$\Delta=S-t$. Define

$$
V_x=\sigma^2\frac{1-e^{-2a\Delta}}{2a},
\quad
V_y=\eta^2\frac{1-e^{-2b\Delta}}{2b},
\quad
C_{xy}=\rho\sigma\eta\frac{1-e^{-(a+b)\Delta}}{a+b}.
$$

The bond-forward log-variance is

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

A single bond log-price is a linear projection of the Gaussian state, so its
forward remains lognormal under the expiry-bond numeraire.

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

Write $P(T_0,T_i;x,y)=A_i e^{-B_{x,i}x-B_{y,i}y}$. For every $x$, solve

$$
\sum_{i=1}^nc_iA_i e^{-B_{x,i}x-B_{y,i}y^\star(x)}=1.
$$

If $Y\mid X=x\sim\mathcal N(m(x),s^2)$, define

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

## Pricing kernels

Current rate options and zero-coupon-bond options are closed form and use one
thread per result. Files follow `PreparedRow` → `evaluate_price<Side>` →
launcher, with compile-time call/put specialization.

<details>
<summary>Exact analytics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
struct G2BondLoadings { float state_x; float state_y; };
float log_A(const G2ModelParameters&, float valuation_time, float maturity);
float A(const G2ModelParameters&, float valuation_time, float maturity);
G2BondLoadings B(const G2ModelParameters&, float valuation_time, float maturity);
float log_zero_coupon_bond(const G2ModelParameters&, const G2State&, float valuation_time, float maturity);
float short_rate(const G2State&);
float log_discount_factor(float state_integral);
float discount_factor(float state_integral);
float zero_coupon_bond(const G2ModelParameters&, const G2State&, float valuation_time, float maturity);
float zero_coupon_bond_call_price(const G2ModelParameters&, const G2State&, float valuation_time, float expiry, float maturity, float strike);
float zero_coupon_bond_put_price(const G2ModelParameters&, const G2State&, float valuation_time, float expiry, float maturity, float strike);
float forward_rate(const G2ModelParameters&, const G2State&, float valuation_time, float start, float end, float accrual);
float swap_rate(const G2ModelParameters&, const G2State&, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

</details>

## Memory and numerical policy

Correlated Gaussian loadings are prepared once. Each OU factor reuses the
stable formulas in `../common/mean_reverting_gaussian.cuh`. State, joint state,
and analytics remain register-sized; optional grids use separate date-major
arrays for `x`, `y`, and their integral. Fast-math is forbidden.

## American and Bermudan options

No G2 American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/g2/),
[validation](../../../../validation/model/fixed_income/g2/),
[fixed-income Gaussian helpers](../common/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

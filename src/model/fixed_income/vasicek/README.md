# Vasicek

| At a glance | Value |
|---|---|
| Process | One-factor affine Gaussian short rate |
| Transition | Exact state and state/integral laws |
| Path state | `state`, optionally `state_integral` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements the affine Gaussian short-rate model

```text
dr_t = a(b - r_t) dt + sigma dW_t.
```

`W` is a standard Brownian motion. `r` is a Gaussian Markov short rate and
`b` is its constant mean-reversion level.

See [Vasicek (1977)](https://doi.org/10.1016/0304-405X(77)90016-2).

## Formula index

- [Analytics — Gaussian state, discounting, and affine bonds](#analytics)
- [Zero-coupon bond — exponential-affine formula](#zero-coupon-bond)
- [Zero-coupon bond option — expiry-bond numeraire](#zero-coupon-bond-option)
- [Caplet / floorlet — scaled bond-option identity](#caplet--floorlet)
- [Swap and swap rate — discounted-leg formulas](#swap-and-swap-rate)
- [European payer swaption — Jamshidian decomposition](#european-payer-swaption)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the process plus initial rate.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact state and joint state/integral transitions.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) provide discount, bond, forward, swap, and option formulas.
- [`rate_option.cuh`](rate_option.cuh) / [`rate_option.cu`](rate_option.cu) and [`zero_coupon_bond_option.cuh`](zero_coupon_bond_option.cuh) / [`zero_coupon_bond_option.cu`](zero_coupon_bond_option.cu) own launchers.

## Dataset row

`VasicekModelParameters` combines the process parameters with its initial
short rate.

| Symbol | Dataset field |
|---|---|
| $r_0$ | `initial_state` |
| $a$ | `mean_reversion` |
| $b$ | `long_term_mean` |
| $\sigma$ | `volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `decay` | $e^{-a\Delta t}$ |
| `state_mean_increment` | $b(1-e^{-a\Delta t})$ |
| `state_standard_deviation` | Exact Gaussian endpoint variance |
| `integral_state_loading`, `integral_mean_increment` | Conditional integral mean |
| `integral_state_normal_loading` | Exact state/integral covariance |
| `integral_independent_standard_deviation` | Residual integral variance |

The state-only transition names its deterministic field `mean_increment`; the
joint transition uses `state_mean_increment`.

The state-only API uses one `float`; `joint::VasicekJointState` adds only
`state_integral` for discounting.

## Dynamics interface

| Function | Role |
|---|---|
| `integral_state_loading` | Load the current rate into its future integral mean |
| `integral_variance`, `integral_moments` | Return exact conditional integral moments |
| `prepare_model` | Precompute an exact state transition |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Apply one prepared transition directly |
| `simulate_on_regular_grid` | Store pre-terminal date-major states |
| `joint::*` variants | Evolve the rate and accumulated integral together |

The state and joint state/integral transitions are exact. A payoff needing only
dated rates samples directly at those dates; one needing path discounting uses
the exact joint Gaussian law. No fine numerical time grid is required.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float integral_state_loading(float mean_reversion, float interval);
float integral_variance(const VasicekProcessParameters&, float interval);
VasicekIntegralMoments integral_moments(const VasicekProcessParameters&, float interval);
VasicekExactTransition prepare_model(const VasicekProcessParameters&, float interval);
void one_step_transition(const VasicekExactTransition&, float normal, float& state);
float simulate_terminal_state(const VasicekExactTransition&, float initial_state, float normal);
float simulate_on_regular_grid(const VasicekExactTransition& stub, const VasicekExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states);
joint::VasicekJointExactTransition joint::prepare_model(const VasicekProcessParameters&, float interval);
void joint::one_step_transition(const joint::VasicekJointExactTransition&, float state_normal, float integral_normal, joint::VasicekJointState&);
joint::VasicekJointState joint::simulate_terminal_state(const joint::VasicekJointExactTransition&, float initial_state, float state_normal, float integral_normal);
joint::VasicekJointState joint::simulate_on_regular_grid(const joint::VasicekJointExactTransition& stub, const joint::VasicekJointExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states, float* integrals);
```

</details>

## Random-number strategy

Direct terminal helpers accept caller-provided normals. A regular-grid helper
owns one `philox::UniformSequence(key, path)` and one normal cache for the whole
path. The state-only transition consumes one normal and the joint transition
two.

## Analytics

For $\tau=T-t$,

$$
B(t,T)=\frac{1-e^{-a\tau}}a,
\qquad
V_I(t,T)=\operatorname{Var}_t\!\left[\int_t^T r_u\,du\right],
$$

$$
\log A(t,T)=\frac12V_I(t,T)-b[\tau-B(t,T)],
\qquad
P(t,T)=A(t,T)e^{-B(t,T)r_t}.
$$

For $I_t=\int_0^t r_u\,du$,

$$
D(0,t)=e^{-I_t}.
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

These quantities are exposed by `analytics.cuh/.cu` and feed the product
formulas below.

## Zero-coupon bond

For notional $N$ paid at $T$,

$$
V_{\mathrm{ZCB}}(t)=NP(t,T)=NA(t,T)e^{-B(t,T)r_t}.
$$

The formula is the conditional Laplace transform of the Gaussian future
short-rate integral.

## Zero-coupon bond option

Let $S$ be the option expiry, $T>S$ the bond maturity, and $K_B$ the strike.
Under the $S$-bond numeraire,

$$
\nu^2=B(S,T)^2\sigma^2
\frac{1-e^{-2a(S-t)}}{2a},
$$

$$
d_1=\frac{\log\!\left(P(t,T)/(K_BP(t,S))\right)+\nu^2/2}{\nu},
\qquad d_2=d_1-\nu.
$$

The closed-form prices are

$$
C_{\mathrm{ZCB}}(t)=P(t,T)\Phi(d_1)-K_BP(t,S)\Phi(d_2),
$$

$$
P_{\mathrm{ZCB}}(t)=K_BP(t,S)\Phi(-d_2)-P(t,T)\Phi(-d_1).
$$

The expiry-bond numeraire makes the bond forward lognormal. The implementation
uses one analytical CUDA thread per price.

## Caplet / floorlet

For fixing $T_1$, payment $T_2$, accrual $\delta$, strike $K$, and notional
$N$,

$$
\Pi_{\mathrm{caplet}}(T_2)=N\delta[L(T_1,T_1,T_2)-K]^+,
\qquad
\Pi_{\mathrm{floorlet}}(T_2)=N\delta[K-L(T_1,T_1,T_2)]^+.
$$

Define

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

These exact identities follow from $L(T_1,T_1,T_2)$.

## Swap and swap rate

For a swap starting at $T_0$ with payments $T_1,\ldots,T_n$,

$$
V_{\mathrm{float}}(t)
=N\sum_{i=1}^n\delta_iL(t,T_{i-1},T_i)P(t,T_i).
$$

Using

$$
\delta_iL(t,T_{i-1},T_i)P(t,T_i)
=P(t,T_{i-1})-P(t,T_i),
$$

gives

$$
V_{\mathrm{float}}(t)=N[P(t,T_0)-P(t,T_n)].
$$

For fixed rate $K$,

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

**Method: planned Jamshidian decomposition.** At exercise and swap start $T_0$,

$$
\Pi_{\mathrm{payer}}(T_0)=N\left[
1-P(T_0,T_n)-K\sum_{i=1}^n\delta_iP(T_0,T_i)
\right]^+.
$$

Set

$$
c_i=K\delta_i+\mathbf 1_{\{i=n\}},
$$

and solve

$$
\sum_{i=1}^nc_iP(T_0,T_i;r^\star)=1.
$$

With $K_i^\star=P(T_0,T_i;r^\star)$,

$$
\Pi_{\mathrm{payer}}(T_0)
=N\sum_{i=1}^nc_i[K_i^\star-P(T_0,T_i)]^+,
$$

and therefore

$$
V_{\mathrm{payer\ swaption}}(t)
=N\sum_{i=1}^nc_iP_{\mathrm{ZCB}}(t;T_0,T_i,K_i^\star).
$$

The common one-dimensional rate orders every bond in the same direction. The
swaption launcher is not implemented yet.

## Memory and numerical policy

Prepared exact coefficients and the shared formulas in
`../common/mean_reverting_gaussian.cuh` avoid repeated transcendental work and
small-time cancellation. Grids write separate date-major state and integral
arrays only when needed. Fast-math is forbidden.

## American and Bermudan options

No Vasicek American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/vasicek/),
[validation](../../../../validation/model/fixed_income/vasicek/),
[fixed-income Gaussian helpers](../common/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

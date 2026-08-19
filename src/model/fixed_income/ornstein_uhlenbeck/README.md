# Ornstein–Uhlenbeck short-rate model

| At a glance | Value |
|---|---|
| Process | One-factor centered Gaussian short rate |
| Transition | Exact state and state/integral laws |
| Path state | `state`, optionally `state_integral` |
| Random laws | Normal when simulation is requested |
| Pricing | Closed form, one thread per price |
| Early exercise | Not implemented |

## Role and reference

This directory uses a centered Ornstein–Uhlenbeck process directly as the
short rate:

```text
dX_t = -a X_t dt + sigma dW_t,    r_t = X_t.
```

`W` is a standard Brownian motion. `X` is a centered Gaussian Markov process;
the standalone model identifies that state directly with the short rate.

The Gaussian mean-reverting process originates in
[Uhlenbeck and Ornstein (1930)](https://doi.org/10.1103/PhysRev.36.823).

## Formula index

- [Analytics — Gaussian state, discounting, and affine bonds](#analytics)
- [Zero-coupon bond — exponential-affine formula](#zero-coupon-bond)
- [Zero-coupon bond option — expiry-bond numeraire](#zero-coupon-bond-option)
- [Caplet / floorlet — scaled bond-option identity](#caplet--floorlet)
- [Swap and swap rate — discounted-leg formulas](#swap-and-swap-rate)
- [European payer swaption — Jamshidian decomposition](#european-payer-swaption)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the process plus initial state.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact state and joint state/integral transitions.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) provide discount, bond, forward, swap, and option formulas.
- [`rate_option.cuh`](rate_option.cuh) / [`rate_option.cu`](rate_option.cu) and [`zero_coupon_bond_option.cuh`](zero_coupon_bond_option.cuh) / [`zero_coupon_bond_option.cu`](zero_coupon_bond_option.cu) own launchers.

## Dataset row

`OrnsteinUhlenbeckModelParameters` combines the process parameters with its
initial state.

| Symbol | Dataset field |
|---|---|
| $X_0$ | `initial_state` |
| $a$ | `mean_reversion` |
| $\sigma$ | `volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `decay` | $e^{-a\Delta t}$ |
| `state_standard_deviation` | $\sigma\sqrt{(1-e^{-2a\Delta t})/(2a)}$ |
| `integral_state_loading` | Conditional mean loading of $\int X_s ds$ |
| `integral_state_normal_loading` | Exact state/integral covariance |
| `integral_independent_standard_deviation` | Residual integral variance |

The last three fields belong to the `joint` transition.

The state-only API uses one `float`. `joint::OrnsteinUhlenbeckJointState` adds
only the accumulated integral needed for path discounting.

## Dynamics interface

| Function | Role |
|---|---|
| `integral_state_loading` | Load the current state into its future integral mean |
| `integral_variance`, `integral_moments` | Return exact conditional integral moments |
| `prepare_model` | Precompute an exact state transition |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Apply one prepared transition directly |
| `simulate_on_regular_grid` | Store pre-terminal date-major states |
| `joint::*` variants | Evolve the state and accumulated integral together |

All transitions are exact over the requested interval. Code needing only a
short rate samples the state directly at observation dates. Code needing a
discount integral uses the exact joint law; neither case requires a finer
artificial `dt`.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float integral_state_loading(float mean_reversion, float interval);
float integral_variance(const OrnsteinUhlenbeckProcessParameters&, float interval);
OrnsteinUhlenbeckIntegralMoments integral_moments(const OrnsteinUhlenbeckProcessParameters&, float interval);
OrnsteinUhlenbeckExactTransition prepare_model(const OrnsteinUhlenbeckProcessParameters&, float interval);
void one_step_transition(const OrnsteinUhlenbeckExactTransition&, float normal, float& state);
float simulate_terminal_state(const OrnsteinUhlenbeckExactTransition&, float initial_state, float normal);
float simulate_on_regular_grid(const OrnsteinUhlenbeckExactTransition& stub, const OrnsteinUhlenbeckExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states);
joint::OrnsteinUhlenbeckJointExactTransition joint::prepare_model(const OrnsteinUhlenbeckProcessParameters&, float interval);
void joint::one_step_transition(const joint::OrnsteinUhlenbeckJointExactTransition&, float state_normal, float integral_normal, joint::OrnsteinUhlenbeckJointState&);
joint::OrnsteinUhlenbeckJointState joint::simulate_terminal_state(const joint::OrnsteinUhlenbeckJointExactTransition&, float initial_state, float state_normal, float integral_normal);
joint::OrnsteinUhlenbeckJointState joint::simulate_on_regular_grid(const joint::OrnsteinUhlenbeckJointExactTransition& stub, const joint::OrnsteinUhlenbeckJointExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states, float* integrals);
```

</details>

## Random-number strategy

Direct terminal helpers receive their normal variates from the caller.
Regular-grid helpers create one `philox::UniformSequence(key, path)` and one
normal cache for the complete path. The joint law consumes two normals per
interval; the state-only law consumes one.

## Analytics

For $\tau=T-t$, the conditional rate integral is Gaussian and

$$
B(t,T)=\frac{1-e^{-a\tau}}a,
\qquad
V_I(t,T)=\operatorname{Var}_t\!\left[\int_t^T X_u\,du\right].
$$

Consequently,

$$
\log A(t,T)=\frac12V_I(t,T),
\qquad
P(t,T)=A(t,T)e^{-B(t,T)X_t}.
$$

For a simulated accumulated state integral $I_t=\int_0^tX_u\,du$,

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

These quantities are exposed by `analytics.cuh/.cu` and are the inputs to the
product formulas below.

## Zero-coupon bond

For notional $N$ paid at $T$,

$$
V_{\mathrm{ZCB}}(t)=NP(t,T)=NA(t,T)e^{-B(t,T)X_t}.
$$

The formula is the conditional Laplace transform of the Gaussian future
short-rate integral.

## Zero-coupon bond option

Let $S$ be the option expiry, $T>S$ the bond maturity, and $K_B$ the bond
strike. Under the $S$-bond numeraire,

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

The exact bond-option identities are

$$
V_{\mathrm{caplet}}(t)
=N(1+\delta K)P_{\mathrm{ZCB}}(t;T_1,T_2,K_B),
$$

$$
V_{\mathrm{floorlet}}(t)
=N(1+\delta K)C_{\mathrm{ZCB}}(t;T_1,T_2,K_B).
$$

They follow directly from the definition of
$L(T_1,T_1,T_2)$ and reuse the analytical bond-option launcher.

## Swap and swap rate

For a swap starting at $T_0$ with payments $T_1,\ldots,T_n$,

$$
V_{\mathrm{float}}(t)
=N\sum_{i=1}^n\delta_iL(t,T_{i-1},T_i)P(t,T_i).
$$

Since

$$
\delta_iL(t,T_{i-1},T_i)P(t,T_i)
=P(t,T_{i-1})-P(t,T_i),
$$

the sum telescopes:

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

**Method: planned Jamshidian decomposition.** Exercise $T_0$ is also the swap
start. Its payoff is

$$
\Pi_{\mathrm{payer}}(T_0)=N\left[
1-P(T_0,T_n)-K\sum_{i=1}^n\delta_iP(T_0,T_i)
\right]^+.
$$

Set

$$
c_i=K\delta_i+\mathbf 1_{\{i=n\}},
$$

and solve the unique scalar boundary

$$
\sum_{i=1}^nc_iP(T_0,T_i;x^\star)=1.
$$

With $K_i^\star=P(T_0,T_i;x^\star)$,

$$
\Pi_{\mathrm{payer}}(T_0)
=N\sum_{i=1}^nc_i[K_i^\star-P(T_0,T_i)]^+,
$$

so the time-$t$ price is

$$
V_{\mathrm{payer\ swaption}}(t)
=N\sum_{i=1}^nc_i
P_{\mathrm{ZCB}}(t;T_0,T_i,K_i^\star).
$$

All bond prices decrease with the same one-dimensional state, which makes the
payoff decomposition exact. The swaption launcher is not implemented yet.

## Memory and numerical policy

The exact Gaussian coefficients are prepared once and shared mathematical
moments come from `../common/mean_reverting_gaussian.cuh`. Small-time formulas
avoid cancellation. Simulation grids use separate date-major state and
integral arrays only when requested. Fast-math is forbidden.

## American and Bermudan options

No Ornstein–Uhlenbeck American/Bermudan launcher is currently present.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/ornstein_uhlenbeck/),
[validation](../../../../validation/model/fixed_income/ornstein_uhlenbeck/),
[fixed-income Gaussian helpers](../common/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

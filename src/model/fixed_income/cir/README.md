# CIR

| At a glance | Value |
|---|---|
| Process | One-factor affine square-root short rate |
| Transition | Exact state law at requested dates |
| Path state | `state` |
| Random laws | Adaptive Poisson plus Gamma |
| Pricing | ZCB, caplet/floorlet, bond-option, forward, and swap analytics |
| Early exercise | Not implemented |

## Role and reference

This directory implements the Cox-Ingersoll-Ross short-rate model

```text
dr_t = kappa(theta - r_t) dt + sigma sqrt(r_t) dW_t.
```

The exact transition preserves non-negativity. The Feller condition
`2*kappa*theta >= sigma^2` is not required by the implementation: when it is
violated, zero is accessible but the exact transition law remains valid.

See [Cox, Ingersoll, and Ross (1985)](https://doi.org/10.2307/1911242).

## Formula index

- [Analytics — CIR transition, discounting, and affine bonds](#analytics)
- [Zero-coupon bond — exponential-affine formula](#zero-coupon-bond)
- [Zero-coupon bond option — non-central chi-square formula](#zero-coupon-bond-option)
- [Caplet / floorlet — scaled bond-option identity](#caplet--floorlet)
- [Swap and swap rate — discounted-leg formulas](#swap-and-swap-rate)
- [European payer swaption — Jamshidian decomposition](#european-payer-swaption)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load
  the process plus initial rate.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement the
  exact endpoint transition.
- [`analytics.cuh`](analytics.cuh) / [`analytics.cu`](analytics.cu) expose the
  affine bond coefficients, path discount, zero-coupon, bond-option, forward,
  and swap formulas.
- [`rate_option.cuh`](rate_option.cuh) / [`rate_option.cu`](rate_option.cu)
  price caplets and floorlets through the exact scaled bond-option identity.
- [`zero_coupon_bond_option.cuh`](zero_coupon_bond_option.cuh) /
  [`zero_coupon_bond_option.cu`](zero_coupon_bond_option.cu) price European
  calls and puts on zero-coupon bonds.

## Dataset row

`CirModelParameters` combines the process parameters with its initial short
rate.

| Symbol | Dataset field |
|---|---|
| $r_0$ | `initial_state` |
| $\kappa$ | `mean_reversion` |
| $\theta$ | `long_term_mean` |
| $\sigma$ | `volatility` |

All process parameters are positive and the initial rate is non-negative.

## Analytics

For $\tau=T-t$ and

$$
\gamma=\sqrt{\kappa^2+2\sigma^2},
$$

the model is affine:

$$
P(t,T)=A(t,T)\exp\{-B(t,T)r_t\},
$$

with

$$
B(t,T)=\frac{2(e^{\gamma\tau}-1)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma},
$$

and

$$
A(t,T)=\left[
\frac{2\gamma e^{(\kappa+\gamma)\tau/2}}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}
\right]^{2\kappa\theta/\sigma^2}.
$$

The implementation evaluates equivalent formulas in terms of
`exp(-gamma*tau)` to avoid overflow. `A`, `B`, and `log_A` are public device
functions; `log_zero_coupon_bond` uses `log_A-B*r` semantics without an
`exp`/`log` round trip.

For an accumulated rate integral $I_t=\int_0^t r_u\,du$,

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

## Zero-coupon bond

For notional $N$ paid at $T$,

$$
V_{\mathrm{ZCB}}(t)=NP(t,T)=NA(t,T)e^{-B(t,T)r_t}.
$$

The coefficients solve the affine Riccati equations associated with the CIR
conditional transform.

## Exact dynamics

Over one interval $\Delta$, define

$$
c=\frac{\sigma^2(1-e^{-\kappa\Delta})}{4\kappa},\qquad
\nu=\frac{4\kappa\theta}{\sigma^2},\qquad
\lambda=\frac{e^{-\kappa\Delta}r_t}{c}.
$$

Then

$$
r_{t+\Delta}=cX,\qquad X\sim\chi'^2_\nu(\lambda).
$$

The device sampler uses the exact mixture

```text
N ~ Poisson(lambda / 2)
r_next ~ Gamma(nu / 2 + N, 2 * c)
```

so `c` is applied directly as the Gamma scale. Small Poisson means use
inversion; large means use Hoermann PTRS. One `UniformSequence(key, path)` and
one `NormalPairCache` live for the complete path.

<details>
<summary>Dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
CirExactTransition prepare_model(const CirProcessParameters&, float interval);
void one_step_transition(const CirExactTransition&, philox::UniformSequence&, philox::NormalPairCache&, float& state);
float simulate_terminal_state(const CirExactTransition&, float initial_state, philox::PhiloxKey, std::size_t path);
float simulate_on_regular_grid(const CirExactTransition& stub, const CirExactTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states);

namespace joint {
struct CirJointTransition;
struct CirJointState { float state; float state_integral; };
CirJointTransition prepare_model(const CirProcessParameters&, float interval);
void one_step_transition(const CirJointTransition&, philox::UniformSequence&, philox::NormalPairCache&, CirJointState&);
CirJointState simulate_terminal_state(const CirJointTransition&, float initial_state, philox::PhiloxKey, std::size_t path);
CirJointState simulate_on_regular_grid(const CirJointTransition& stub, const CirJointTransition& regular, float initial_state, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* states, float* integrated_states);
}
```

</details>

## Analytics interface

```cpp
float log_A(const CirModelParameters&, float valuation_time, float maturity);
float A(const CirModelParameters&, float valuation_time, float maturity);
float B(const CirModelParameters&, float valuation_time, float maturity);
float log_zero_coupon_bond(const CirModelParameters&, float state, float valuation_time, float maturity);
float log_discount_factor(float state_integral);
float discount_factor(float state_integral);
float zero_coupon_bond(const CirModelParameters&, float state, float valuation_time, float maturity);
float zero_coupon_bond_call_price(const CirModelParameters&, float state, float valuation_time, float option_expiry, float bond_maturity, float strike);
float zero_coupon_bond_put_price(const CirModelParameters&, float state, float valuation_time, float option_expiry, float bond_maturity, float strike);
float forward_rate(const CirModelParameters&, float state, float valuation_time, float start, float end, float accrual);
float swap_rate(const CirModelParameters&, float state, float valuation_time, float start, const float* payment_times, const float* accruals, std::size_t payments);
```

## Zero-coupon bond option

The CIR bond option uses the standard two-forward-measure formula. If `S` is
the option expiry, `T` the bond maturity, and `r_star` solves
`A(S,T) exp(-B(S,T) r_star) = strike`, the call combines two non-central
chi-square probabilities. The put uses the two survival probabilities directly
instead of subtracting FP32 CDFs from one.

With bond strike $K_B$, define

$$
r^\star=\frac{\log A(S,T)-\log K_B}{B(S,T)},
\qquad
\nu=\frac{4\kappa\theta}{\sigma^2},
$$

and let $F_{\chi'^2_\nu(\lambda)}(z)$ and
$\bar F_{\chi'^2_\nu(\lambda)}(z)$ denote the non-central chi-square CDF and
survival function. The implementation constructs the two forward-measure
parameter pairs $(\lambda_T,z_T)$ and $(\lambda_S,z_S)$ from
$(\kappa,\theta,\sigma,r_t,t,S,T,r^\star)$. It then evaluates

$$
C_{\mathrm{ZCB}}(t)
=P(t,T)F_{\chi'^2_\nu(\lambda_T)}(z_T)
-K_BP(t,S)F_{\chi'^2_\nu(\lambda_S)}(z_S),
$$

$$
P_{\mathrm{ZCB}}(t)
=K_BP(t,S)\bar F_{\chi'^2_\nu(\lambda_S)}(z_S)
-P(t,T)\bar F_{\chi'^2_\nu(\lambda_T)}(z_T).
$$

The shared device primitive in
[`common/noncentral_chi_square.cuh`](../../../common/noncentral_chi_square.cuh)
returns both tails. It uses a centered Poisson--Gamma mixture over the ordinary
range and a Lugannani--Rice saddlepoint evaluation when a very large
noncentrality would make the mixture impractical. Gamma probabilities use a
series or modified-Lentz continued fraction. All internal CDF work is FP64;
only the public probabilities are returned as FP32.

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

**Method: planned Jamshidian decomposition.** At exercise and swap start $T_0$,

$$
\Pi_{\mathrm{payer}}(T_0)=N\left[
1-P(T_0,T_n)-K\sum_{i=1}^n\delta_iP(T_0,T_i)
\right]^+.
$$

Set $c_i=K\delta_i+\mathbf 1_{\{i=n\}}$ and solve on the CIR state domain

$$
\sum_{i=1}^nc_iP(T_0,T_i;r^\star)=1.
$$

With $K_i^\star=P(T_0,T_i;r^\star)$,

$$
V_{\mathrm{payer\ swaption}}(t)
=N\sum_{i=1}^nc_iP_{\mathrm{ZCB}}(t;T_0,T_i,K_i^\star).
$$

The swaption launcher is not implemented yet.

The complete Premia core audit does not certify any of the four datasets:
caplets pass 100/900 rows, floorlets 101/900, zero-coupon calls 363/900, and
zero-coupon puts 321/900 at the catalogue tolerance. Their respective maximum
absolute gaps are `0.004156287`, `0.004156335`, `0.006731222`, and
`0.006731215`. Premia is recorded only as
`status: available but not reliable`; this detailed audit remains in the model
documentation rather than being repeated in every reference-price database.

QuantLib's `CoxIngersollRoss.discountBondOption` agrees numerically when the
out-of-the-money tail and zero-coupon parity are used for deep in-the-money
stability. It is the reliable primary reference: caplets, floorlets, and both
zero-coupon option sides pass 900/900 core and 100/100 stress rows. The four
1,000-row reference databases are cached below
`validation/datasets/price/fixed_income/cir`, and the catalogues are therefore
`available` and `verified: true`.

Each cached database fingerprints the semantic source prices and model/product
parameters, records QuantLib's exact version and `row_priced`, and persists the
core/stress tolerance and comparison evidence in a separate `verification`
block. Routine validation is cache-only and fail-closed; QuantLib is imported
only for an explicit `--generate` refresh.

## Path discounting and early exercise

The exact endpoint transition does not jointly sample
$\int_t^{t+\Delta}r_s\,ds$. The common joint types and four function
signatures are exposed so CIR follows the fixed-income interface, but their
definitions remain deliberately unavailable until the integration scheme is
chosen and validated. `log_discount_factor(state_integral)` and
`discount_factor(state_integral)` are already exposed and evaluate
`-state_integral` and `exp(-state_integral)`. They deliberately accept only the
integral, not a `CirJointState`, so callers that need only `r_t` do not carry an
unused joint state. Boundary-only formulas use the direct conditional bond
value. A future
Monte-Carlo Bermudan swaption must specify and validate its integration scheme
or use a lattice/PDE approach; exact exercise-date rates alone are not enough
for pathwise discounting.

Related navigation: [model catalog](../../../../catalog/model/fixed_income/cir/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

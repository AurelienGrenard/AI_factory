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

## Dynamics interface

| Function | Role |
|---|---|
| `compose_model` | Combine one process row with one initial curve |
| `short_rate_shift`, `short_rate` | Evaluate $\phi(t)$ and reconstruct $r_t$ |
| `log_discount_factor`, `discount_factor` | Convert joint state into path discounting |
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
float log_discount_factor(const G2PlusPlusFittedParameters&, const joint::G2JointState&, float time);
float discount_factor(const G2PlusPlusFittedParameters&, const joint::G2JointState&, float time);
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

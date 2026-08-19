# Schöbel–Zhu

| At a glance | Value |
|---|---|
| Process | Gaussian stochastic volatility |
| Transition | Exact OU endpoint + log-spot Euler step |
| Path state | `log_spot`, `volatility` |
| Random laws | Normal |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements stochastic volatility driven by an
Ornstein–Uhlenbeck process:

```text
dS_t / S_t = (r - q) dt + v_t dW_t^S
dv_t       = kappa(theta - v_t) dt + gamma dW_t^v
d<W^S,W^v>_t = rho dt.
```

`W^S` and `W^v` are standard Brownian motions with instantaneous correlation
`rho`. `v_t` is a Gaussian Ornstein–Uhlenbeck volatility process, not a CIR
variance process; it is therefore a signed state in the implementation.

See Schöbel and Zhu, *Stochastic Volatility With an Ornstein–Uhlenbeck
Process: An Extension* (1999), available from the authors' institution as a
[primary working-paper version](https://www.econstor.eu/handle/10419/104833).

## Formula index

- [Dynamics and simulation](#dynamics-interface)
- [Pricing convention](#pricing-convention)
- [Terminal and two-time payoffs](#terminal-and-two-time-payoffs)
- [Averages and extrema](#averages-and-extrema)
- [Barrier and touch products](#barrier-and-touch-products)
- [Structured coupons](#structured-coupons)


## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement the volatility/spot scheme and path summaries.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $v_0$ | `initial_volatility` |
| $\kappa$ | `mean_reversion` |
| $\theta$ | `long_run_volatility` |
| $\gamma$ | `volatility_of_volatility` |
| $\rho$ | `correlation` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot`, `initial_volatility` | $\log S_0$, $v_0$ |
| `long_run_volatility` | $\theta$ |
| `exp_mean_reversion_dt` | $e^{-\kappa\Delta t}$ |
| `ou_std` | Exact OU endpoint variance |
| `endpoint_increment_correlation`, `endpoint_increment_residual` | Exact OU endpoint/Brownian coupling |
| `drift_dt`, `sqrt_dt` | $(r-q)\Delta t$, $\sqrt{\Delta t}$ |
| `correlation`, `correlation_residual` | $\rho$, $\sqrt{1-\rho^2}$ |

`SchobelZhuState` contains `log_spot` and the signed OU `volatility`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one OU/Euler step |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from supplied normals |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

The volatility endpoint is sampled exactly with its correct joint Gaussian
coupling to the interval Brownian increment. The log spot is advanced with the
left-end volatility, so terminal simulation still follows a numerical grid.
Each step uses three independent normals.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
SchobelZhuPreparedParameters prepare_model(const SchobelZhuModelParameters&, float maturity, std::size_t steps);
SchobelZhuState initial_state(const SchobelZhuPreparedParameters&);
void one_step_transition(const SchobelZhuPreparedParameters&, float ou_normal, float increment_residual_normal, float asset_residual_normal, SchobelZhuState&);
SchobelZhuState simulate_terminal_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuMeanPathResult simulate_mean_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuGeometricMeanPathResult simulate_geometric_mean_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuTwoTimePathResult simulate_at_two_times(const SchobelZhuPreparedParameters& first, const SchobelZhuPreparedParameters& second, philox::PhiloxKey, std::size_t path, std::size_t first_steps, std::size_t second_steps);
SchobelZhuMaximumPathResult simulate_maximum_state(const SchobelZhuPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
SchobelZhuState simulate_on_regular_grid(const SchobelZhuPreparedParameters& stub, const SchobelZhuPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t stub_steps, std::uint32_t steps_per_exercise, std::uint32_t exercise_count, std::size_t path_count, float* spots, float* volatilities);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
The three step normals are consumed in a fixed order and the sequence is never
restarted.

## Pricing convention

Let \(\varepsilon=+1\) for a call and \(\varepsilon=-1\) for a put. For
maturity \(T\), the risk-neutral price of a payoff \(H\) is

$$
V_0=\mathbb E^{\mathbb Q}[e^{-rT}H].
$$

A Monte Carlo launcher evaluates

$$
\widehat V_0=\frac1M\sum_{m=1}^M e^{-rT}H^{(m)},
$$

and returns both \(\widehat V_0\) and its sampling standard error. Products
with intermediate payments discount each cashflow at its own payment date.
The monitoring grid is \(0=t_0<t_1<\cdots<t_J=T\); \(\Delta_o\) denotes
the product observation interval.

## Terminal and two-time payoffs

| Product | Product parameters | Payoff \(H\) | Pricing |
|---|---|---|---|
| European option | strike \(K\), maturity \(T\) | \([\varepsilon(S_T-K)]^+\) | Monte Carlo |
| Digital option | strike \(K\), maturity \(T\), cash payoff \(Q\) | \(Q\mathbf 1_{\{\varepsilon(S_T-K)>0\}}\) | Monte Carlo |
| Asset-or-nothing option | strike \(K\), maturity \(T\) | \(S_T\mathbf 1_{\{\varepsilon(S_T-K)>0\}}\) | Monte Carlo |
| Straddle | strike \(K\), maturity \(T\) | \(|S_T-K|\) | Monte Carlo |
| Gap option | trigger \(K_1\), payoff strike \(K_2\), maturity \(T\) | \(\varepsilon(S_T-K_2)\mathbf 1_{\{\varepsilon(S_T-K_1)>0\}}\) | Monte Carlo |
| Forward-start option | moneyness \(m\), reset \(T_r\), maturity \(T\) | \([\varepsilon(S_T-mS_{T_r})]^+\) | Monte Carlo |

## Averages and extrema

Define

$$
\bar S_A=\frac1{J+1}\sum_{j=0}^J S_{t_j},
\qquad
\bar S_G=\exp\!\left(\frac1{J+1}\sum_{j=0}^J\log S_{t_j}\right),
\qquad
M_T=\max_{0\le j\le J}S_{t_j}.
$$

| Product | Product parameters | Payoff \(H\) | Pricing |
|---|---|---|---|
| Arithmetic Asian option | strike \(K\), maturity \(T\) | \([\varepsilon(\bar S_A-K)]^+\) | Monte Carlo |
| Geometric Asian option | strike \(K\), maturity \(T\) | \([\varepsilon(\bar S_G-K)]^+\) | Monte Carlo |
| Fixed-strike lookback call | strike \(K\), maturity \(T\) | \([M_T-K]^+\) | Monte Carlo |

Both averages include issuance \(S_0\) and maturity \(S_T\). The observation
count is derived from \(T\) and the numerical monitoring step.

## Barrier and touch products

Let

$$
I_D(B)=\mathbf 1_{\{\min_jS_{t_j}\le B\}},
\qquad
I_U(B)=\mathbf 1_{\{\max_jS_{t_j}\ge B\}}.
$$

| Product | Product parameters | Payoff \(H\) | Pricing |
|---|---|---|---|
| Down-and-in option | strike \(K\), barrier \(B\), maturity \(T\) | \([\varepsilon(S_T-K)]^+I_D(B)\) | Monte Carlo |
| Down-and-out option | strike \(K\), barrier \(B\), maturity \(T\) | \([\varepsilon(S_T-K)]^+[1-I_D(B)]\) | Monte Carlo |
| Up-and-in option | strike \(K\), barrier \(B\), maturity \(T\) | \([\varepsilon(S_T-K)]^+I_U(B)\) | Monte Carlo |
| Up-and-out option | strike \(K\), barrier \(B\), maturity \(T\) | \([\varepsilon(S_T-K)]^+[1-I_U(B)]\) | Monte Carlo |
| Double-knock-out option | strike \(K\), lower \(B_L\), upper \(B_U\), maturity \(T\) | \([\varepsilon(S_T-K)]^+\mathbf 1_{\{B_L<S_{t_j}<B_U,\ \forall j\}}\) | Monte Carlo |
| Up no-touch | barrier \(B\), cash payoff \(Q\), maturity \(T\) | \(Q[1-I_U(B)]\) paid at \(T\) | Monte Carlo |
| Up one-touch | barrier \(B\), cash payoff \(Q\), maturity \(T\) | \(QI_U(B)\) paid at \(T\) | Monte Carlo |

Barriers are monitored on the simulation grid, including issuance and
maturity; no continuous-barrier correction is applied.

## Structured coupons

For a cliquet with participation \(p\), local bounds
\([f_\ell,c_\ell]\), and global bounds \([f_g,c_g]\),

$$
R_j=\frac{S_{t_j}}{S_{t_{j-1}}}-1,
\qquad
R_{\mathrm{cliquet}}=
\operatorname{clamp}\!\left(
\sum_{j=1}^J\operatorname{clamp}(pR_j,f_\ell,c_\ell),
f_g,c_g
\right),
$$

$$
H_{\mathrm{cliquet}}=1+R_{\mathrm{cliquet}}.
$$

For a range accrual with barriers \(B_L,B_U\) and annual coupon rate \(c\),

$$
H_{\mathrm{range}}=
1+c\Delta_o\sum_{j=1}^J
\mathbf 1_{\{B_L\le S_{t_j}\le B_U\}}.
$$

| Product | Product parameters | Redemption rule | Pricing |
|---|---|---|---|
| Cliquet | \(T,\Delta_o,p,f_\ell,c_\ell,f_g,c_g\) | \(1+R_{\mathrm{cliquet}}\) at \(T\) | Monte Carlo |
| Range accrual | \(T,\Delta_o,B_L,B_U,c\) | \(H_{\mathrm{range}}\) at \(T\) | Monte Carlo |
| Athena autocall | \(T,\Delta_o,B_A,B_P,c\) | first \(t_j<T\) with \(S_{t_j}\ge B_A\): pay \(1+ct_j\); at \(T\): pay \(1+cT\) if \(S_T\ge B_A\), \(1\) if \(B_P\le S_T<B_A\), otherwise \(S_T\) | Monte Carlo |
| Phoenix autocall | \(T,\Delta_o,B_A,B_C,B_P,c\) | pay \(c\Delta_o\) when \(S_{t_j}\ge B_C\); before \(T\), \(S_{t_j}\ge B_A\) also redeems \(1\); at \(T\), redeem \(1\) if \(S_T\ge B_P\), otherwise \(S_T\) | Monte Carlo |
| Phoenix memory autocall | \(T,\Delta_o,B_A,B_C,B_P,c\) | missed coupons accumulate and are released when \(S_{t_j}\ge B_C\); autocall and final capital follow the Phoenix rule | Monte Carlo |

The autocall catalogues use normalized nominal and issuance spot equal to one.

## Pricing kernels

All current products use the standard one-block-per-price Monte Carlo kernel:
one prepared row, strided paths, FP64 moments, block reduction, and FP32
outputs. Call/put behavior is a compile-time `OptionSide`.

## Memory and numerical policy

Ordinary payoffs keep `(log_spot, volatility)` in registers and retain only
their payoff statistic. The regular-grid helper exposes separate date-major
spot and volatility arrays when both are requested. Prepared Gaussian
loadings avoid repeated exponentials in hot loops. Fast-math is forbidden.

## American and Bermudan options

No Schöbel–Zhu American/Bermudan launcher is currently present in this
directory.

Related navigation: [model catalog](../../../../catalog/model/equity/schobel_zhu/),
[Premia validation](../../../../validation/premia/model/equity/schobel_zhu/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

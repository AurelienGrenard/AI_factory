# Merton jump diffusion

| At a glance | Value |
|---|---|
| Process | Lognormal diffusion + lognormal jumps |
| Transition | Exact over each observation interval |
| Path state | `log_spot` |
| Random laws | Normal + Poisson |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Not implemented |

## Role and reference

This directory implements geometric Brownian diffusion with independent
compound-Poisson Gaussian log jumps:

```text
dS_t / S_(t-) = (r - q - lambda E[J-1]) dt + sigma dW_t + (J-1) dN_t.
```

`W` is a standard Brownian motion. `N` is an independent Poisson process with
intensity `lambda`. At each event, the spot is multiplied by `J = exp(Y)`,
where the jump logs are independent variables
`Y ~ Normal(mu_J, sigma_J^2)`. The term `lambda E[J-1]`, with
`E[J] = exp(mu_J + sigma_J^2/2)`, is the martingale compensator.

See [Merton (1976)](https://doi.org/10.1016/0304-405X(76)90022-2).

## Formula index

- [Dynamics and simulation](#dynamics-interface)
- [Pricing convention](#pricing-convention)
- [Terminal and two-time payoffs](#terminal-and-two-time-payoffs)
- [Averages and extrema](#averages-and-extrema)
- [Barrier and touch products](#barrier-and-touch-products)
- [Structured coupons](#structured-coupons)


## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact finite-interval increments.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\sigma$ | `volatility` |
| $\lambda$ | `jump_intensity` |
| $\mu_J$ | `jump_log_mean` |
| $\sigma_J$ | `jump_log_volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot` | $\log S_0$ |
| `drift_dt` | $(r-q-\lambda E[J-1]-\sigma^2/2)\Delta t$ |
| `diffusion_std` | $\sigma\sqrt{\Delta t}$ |
| `poisson_mean` | $\lambda\Delta t$ |
| `zero_jump_probability` | $e^{-\lambda\Delta t}$ |
| `jump_log_mean` | $\mu_J$ |
| `jump_log_volatility` | $\sigma_J$ |

`MertonState` stores only `log_spot`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one exact interval law |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from caller-supplied variates |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

Every interval is exact. One Poisson draw gives the jump count; conditional on
that count, the sum of Gaussian jump logs is itself Gaussian. Direct terminal
and two-time helpers therefore do not use an artificial `dt`; monitored
payoffs use exact increments between observation dates.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
MertonPreparedParameters prepare_model(const MertonModelParameters&, float interval);
MertonPreparedParameters prepare_model(const MertonModelParameters&, float maturity, std::size_t steps);
MertonState initial_state(const MertonPreparedParameters&);
void one_step_transition(const MertonPreparedParameters&, std::uint32_t jump_count, float diffusion_normal, float jump_normal, MertonState&);
MertonState simulate_terminal_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path);
MertonMeanPathResult simulate_mean_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
MertonGeometricMeanPathResult simulate_geometric_mean_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
MertonTwoTimePathResult simulate_at_two_times(const MertonPreparedParameters& first, const MertonPreparedParameters& second, philox::PhiloxKey, std::size_t path);
MertonMaximumPathResult simulate_maximum_state(const MertonPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
MertonState simulate_on_regular_grid(const MertonPreparedParameters& stub, const MertonPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t observations, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
An interval consumes one Poisson uniform, one diffusion normal, and a jump
normal only when the count is nonzero. No helper creates a second sequence.

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

All current products use one block per price. Threads evaluate strided paths,
accumulate payoff and squared-payoff sums in FP64, and reduce to an FP32 price
and standard error. Call/put behavior is a compile-time `OptionSide`.

## Memory and numerical policy

The evolving state is one register scalar. Path summaries retain only their
requested statistic; regular-grid storage contains spots only. Exact interval
simulation minimizes both work and discretization state. Fast-math is
forbidden.

## American and Bermudan options

No Merton American/Bermudan launcher is currently present in this directory.

Related navigation: [model catalog](../../../../catalog/model/equity/merton/),
[validation](../../../../validation/model/equity/merton/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[pricing contract](../../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

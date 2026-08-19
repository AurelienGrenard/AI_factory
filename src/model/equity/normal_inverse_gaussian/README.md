# Normal Inverse Gaussian

| At a glance | Value |
|---|---|
| Process | Pure-jump Lévy process through an inverse-Gaussian clock |
| Transition | Exact over each observation interval |
| Path state | `log_spot` |
| Random laws | Inverse Gaussian + normal |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Longstaff–Schwartz |

## Role and reference

This directory implements the exponential Normal-Inverse-Gaussian model as a
Brownian motion with drift evaluated on an inverse-Gaussian clock:

```text
X_t = beta G_t + W_(G_t),
S_t = S_0 exp((r - q + omega)t + X_t).
```

`W` is a standard Brownian motion. `G` is an independent inverse-Gaussian
subordinator. With `gamma = sqrt(alpha^2-beta^2)`, the implementation uses
`G_t ~ IG(mean=delta t/gamma, shape=(delta t)^2)`. Conditional on `G_t`,
`X_t` is Gaussian. `omega` is the deterministic martingale correction.

The input uses the standard NIG `alpha`, `beta`, and `delta` parameters and
derives the risk-neutral location correction. See
[Barndorff-Nielsen (1997)](https://doi.org/10.1111/1467-9469.00045).

## Formula index

- [Dynamics and simulation](#dynamics-interface)
- [Pricing convention](#pricing-convention)
- [Terminal and two-time payoffs](#terminal-and-two-time-payoffs)
- [Averages and extrema](#averages-and-extrema)
- [Barrier and touch products](#barrier-and-touch-products)
- [Structured coupons](#structured-coupons)
- [American option — Longstaff–Schwartz](#american-option)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load the model rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) implement exact Lévy increments and path summaries.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

There is no redundant free location parameter in the dataset.

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $\alpha$ | `alpha` |
| $\beta$ | `beta` |
| $\delta$ | `delta` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `initial_log_spot` | $\log S_0$ |
| `inverse_gaussian_mean` | $\delta\Delta t/\sqrt{\alpha^2-\beta^2}$ |
| `inverse_gaussian_shape` | $(\delta\Delta t)^2$ |
| `beta` | $\beta$ |
| `drift_dt` | $(r-q+\omega)\Delta t$ |

The state contains only `log_spot`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute one exact interval law |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from a clock draw and normal |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

Every interval increment is exact. Terminal and two-time products use direct
interval laws; monitored products advance through exact independent increments
at observation dates. The transition receives one inverse-Gaussian clock
increment and one normal.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
NormalInverseGaussianPreparedParameters prepare_model(const NormalInverseGaussianModelParameters&, float interval);
NormalInverseGaussianPreparedParameters prepare_model(const NormalInverseGaussianModelParameters&, float maturity, std::size_t steps);
NormalInverseGaussianState initial_state(const NormalInverseGaussianPreparedParameters&);
void one_step_transition(const NormalInverseGaussianPreparedParameters&, float inverse_gaussian_increment, float normal, NormalInverseGaussianState&);
NormalInverseGaussianState simulate_terminal_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path);
NormalInverseGaussianMeanPathResult simulate_mean_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
NormalInverseGaussianGeometricMeanPathResult simulate_geometric_mean_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
NormalInverseGaussianTwoTimePathResult simulate_at_two_times(const NormalInverseGaussianPreparedParameters& first, const NormalInverseGaussianPreparedParameters& second, philox::PhiloxKey, std::size_t path);
NormalInverseGaussianMaximumPathResult simulate_maximum_state(const NormalInverseGaussianPreparedParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
NormalInverseGaussianState simulate_on_regular_grid(const NormalInverseGaussianPreparedParameters& stub, const NormalInverseGaussianPreparedParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t exercise_count, std::size_t path_count, float* spots);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
The clock uses the Michael–Schucany–Haas generator
([Michael, Schucany, and Haas, 1976](https://doi.org/10.1080/00031305.1976.10479147))
on that same stream; the subordinated Brownian draw follows from the same
normal cache.

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

## American option

For strike \(K\), maturity \(T\), exercise interval \(\Delta_e\), and
exercise dates \(\mathcal E=\{T-n\Delta_e,\ldots,T\}\),

$$
h(t,S_t)=[\varepsilon(S_t-K)]^+,
\qquad t\in\mathcal E.
$$

Longstaff–Schwartz backward induction regresses continuation values on the
model state at each date and compares them with \(h(t,S_t)\). This is a
Bermudan approximation to continuous American exercise.


## Pricing kernels

All current products use the standard one-block-per-price Monte Carlo kernel:
one prepared row, strided paths, FP64 moment accumulation, block reduction,
and FP32 outputs. Call/put is a compile-time `OptionSide`.

## Memory and numerical policy

The evolving state is one scalar. Path summaries retain only their requested
statistic and early-exercise grids store spots only. Exact interval laws avoid
artificial `dt` for terminal claims. Fast-math is forbidden.

## American and Bermudan options

`american_option.cuh/.cu` use the shared Longstaff–Schwartz pipeline and a
spot-only exercise grid.

Related navigation: [model catalog](../../../../catalog/model/equity/normal_inverse_gaussian/),
[validation infrastructure](../../../../validation/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[American/Bermudan contract](../../../../docs/cuda-american-and-bermudan-pricing-contract.md).

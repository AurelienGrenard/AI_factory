# Bates

| At a glance | Value |
|---|---|
| Process | Heston stochastic variance + lognormal jumps |
| Transition | Heston QE-M + exact compound-Poisson interval sum |
| Path state | `log_spot`, `variance` |
| Random laws | Normal + uniform + Poisson |
| Pricing | Monte Carlo, one block per price |
| Early exercise | Longstaff–Schwartz |

## Role and reference

This directory extends Heston with independent compound-Poisson lognormal
jumps:

```text
dS_t / S_(t-) = (r - q - lambda E[J-1]) dt
                + sqrt(v_t) dW_t^S + (J-1) dN_t,
```

while `v_t` follows the Heston CIR variance process. See
[Bates (1996)](https://doi.org/10.1093/rfs/9.1.69). The diffusion part uses
Andersen QE-M.

`W^S` and `W^v` are the correlated Brownian motions defined by Heston. `N` is
a Poisson process with intensity `lambda`. At a jump, the spot is multiplied
by `J = exp(Y)`, where the independent jump logs are
`Y ~ Normal(mu_J, sigma_J^2)`. The Poisson process, jump sizes, and Heston
Brownian motions are mutually independent. The drift subtracts
`lambda E[J-1]` to preserve the discounted martingale.

## Formula index

- [Dynamics and simulation](#dynamics-interface)
- [Pricing convention](#pricing-convention)
- [Terminal and two-time payoffs](#terminal-and-two-time-payoffs)
- [Averages and extrema](#averages-and-extrema)
- [Barrier and touch products](#barrier-and-touch-products)
- [Structured coupons](#structured-coupons)
- [American option — Longstaff–Schwartz](#american-option)

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load Heston-plus-jump rows.
- [`dynamics.cuh`](dynamics.cuh) / [`dynamics.cu`](dynamics.cu) combine Heston QE-M with exact compound-Poisson increments.
- each other `<product>.cuh/.cu` pair owns one Monte Carlo launcher.

## Dataset row

| Symbol | Dataset field |
|---|---|
| $S_0$ | `spot` |
| $r$ | `risk_free_rate` |
| $q$ | `dividend_yield` |
| $v_0$ | `initial_variance` |
| $\kappa$ | `kappa` |
| $\theta$ | `theta` |
| $\gamma$ | `gamma` |
| $\rho$ | `rho` |
| $\lambda$ | `jump_intensity` |
| $\mu_J$ | `jump_log_mean` |
| $\sigma_J$ | `jump_log_volatility` |

## Prepared parameters and state

| Prepared field | Derived from |
|---|---|
| `heston` | Prepared Heston QE-M parameters |
| `poisson_mean` | $\lambda\Delta t$ |
| `poisson_zero_probability` | $e^{-\lambda\Delta t}$ |
| `jump_log_mean` | $\mu_J$ |
| `jump_log_volatility` | $\sigma_J$ |
| `jump_compensator` | $\lambda E[J-1]\Delta t$ |

`BatesState` is exactly `HestonState`: `log_spot` and `variance`.

## Dynamics interface

| Function | Role |
|---|---|
| `prepare_model` | Precompute Heston and jump laws for one step |
| `initial_state` | Build the time-zero register state |
| `one_step_transition` | Apply one transition from caller-supplied variates |
| `simulate_terminal_state` | Return only the maturity state |
| `simulate_mean_state` | Return only the arithmetic mean |
| `simulate_geometric_mean_state` | Return only the geometric mean |
| `simulate_at_two_times` | Return only two requested boundary spots |
| `simulate_maximum_state` | Return only the monitored maximum |
| `simulate_on_regular_grid` | Store only requested dated state fields |

For a payoff that observes only an interval boundary, the code simulates all
Heston QE steps and draws one Poisson count for the whole interval. Conditional
on that count, the sum of lognormal jump logs is one Gaussian draw. Fully
pathwise payoffs retain the one-step transition. This removes unnecessary
Poisson simulations without changing the boundary law.

<details>
<summary>Exact dynamics signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
BatesQeParameters prepare_model(const BatesModelParameters&, float maturity, std::size_t steps);
BatesState initial_state(const BatesQeParameters&);
void one_step_transition(const BatesQeParameters&, float variance_normal, float variance_uniform, float stock_normal, std::uint32_t jump_count, float jump_normal, BatesState&);
BatesState simulate_terminal_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesMeanPathResult simulate_mean_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesGeometricMeanPathResult simulate_geometric_mean_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesTwoTimePathResult simulate_at_two_times(const BatesQeParameters& first, const BatesQeParameters& second, philox::PhiloxKey, std::size_t path, std::size_t first_steps, std::size_t second_steps);
BatesMaximumPathResult simulate_maximum_state(const BatesQeParameters&, philox::PhiloxKey, std::size_t path, std::size_t steps);
BatesState simulate_on_regular_grid(const BatesQeParameters& stub, const BatesQeParameters& regular, philox::PhiloxKey, std::size_t path, std::uint32_t stub_steps, std::uint32_t steps_per_exercise, std::uint32_t exercise_count, std::size_t path_count, float* spots, float* variances);
```

</details>

## Random-number strategy

Each path owns one `philox::UniformSequence(key, path)` and one normal cache.
The Poisson count is obtained by CDF inversion; the jump normal is drawn only
when the count is nonzero. Variable consumption remains local to the path.

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
one `PreparedRow`, strided paths, FP64 payoff moments, block reduction, and
FP32 price/standard error. Call and put use compile-time `OptionSide`.

## Memory and numerical policy

Ordinary products retain only the two-scalar state and their payoff statistic.
Early-exercise grids use separate date-major spot and variance regions.
Prepared interval laws avoid repeated exponentials in hot loops. Fast-math is
forbidden.

## American and Bermudan options

`american_option.cuh/.cu` use the shared Longstaff–Schwartz pipeline and retain
both spot and variance for the two-factor regression basis.

Related navigation: [model catalog](../../../../catalog/model/equity/bates/),
[validation](../../../../validation/model/equity/bates/),
[dynamics contract](../../../../docs/cuda-model-dynamics-contract.md), and
[American/Bermudan contract](../../../../docs/cuda-american-and-bermudan-pricing-contract.md).

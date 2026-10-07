# Zero-coupon bond up-and-out call

The declared compositions cover all fixed-income models in the capability
manifest: standalone CIR, G2, Ornstein–Uhlenbeck and Vasicek, plus CIR++,
G2++ and Hull–White fitted to flat, Nelson–Siegel and Svensson curves (13
model/curve variants). Pricing uses each model's risk-neutral money-market
measure. One zero-coupon bond pays one currency unit at `U`; `P(t,U)` is its
ex-dividend price per unit of face value at `t`. The option pays at `T`, with
`0 < T < U`:

```text
N * max(P(T,U) - K, 0) * 1{P(t,U) < B for every monitored t in [0,T]}.
```

`N > 0` is the number of unit-face bonds, `K > 0` and `B > 0` are bond prices
in currency units per unit face, and `T` and `U` are business-day counts from
pricing time with 252 days per year. The barrier is up-and-out, has no rebate,
and is **breached at equality**: `P(t,U) >= B` extinguishes the option, including
at `t=0` or `t=T`. The loader permits an already breached contract so its
value is exactly zero. Negative OU short rates can make `P(t,U) > 1`.

Each launcher uses the model's joint factor/integral dynamics and affine bond
analytics to evaluate `P(t,U)` at `t=0` and every grid point
`j / (252 * simulation_steps_per_day)` through `T`. At `T`, it multiplies the
surviving payoff by the path discount `exp(-integral_0^T r_s ds)`. The fitted
models add the deterministic curve shift to both bond analytics and the rate
integral. OU, Vasicek, G2, Hull–White and G2++ use exact Gaussian joint
factor/integral increments. CIR and CIR++ use exact noncentral chi-square
factor transitions with a trapezoidal approximation to the stochastic rate
integral. All catalogue recipes use one observation per business day and
65,536 Philox paths per row; the CIR-family prices therefore have an additional
rate-integral time-step bias.

**The catalogue price is an approximation to the continuously monitored
barrier contract.** Exact Gaussian transitions do not reveal barrier crossings
between observations. For fixed dynamics and a fixed barrier, the grid price
is an upper approximation in expectation. A matched one-factor test at
`T=252`, `U=504`, `K=0.90`, `B≈0.9726724`, `a=0.5`, `sigma=0.02`, `r0=0.03`
measured `0.00286426 ± 0.00001791` at one step per day,
`0.00270299 ± 0.00001738` at two, and `0.00263437 ± 0.00001716` at four
(524,288 paths each). The one-versus-four-grid difference is `0.00022989`,
about nine combined Monte Carlo standard errors. A separate FP64 CPU joint
Gaussian implementation gave `0.00280483 ± 0.00002505` for the daily grid
(262,144 independent paths). These numbers validate the stated discrete
monitoring method; they are not a continuous first-passage correction.

See `tests/model/fixed_income/ornstein_uhlenbeck_bond_barrier_cuda_test.cpp` and
`ornstein_uhlenbeck_bond_barrier_reference.py` for the executable checks.

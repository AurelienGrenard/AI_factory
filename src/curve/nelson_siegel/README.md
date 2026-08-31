# Nelson–Siegel curve

| At a glance | Value |
|---|---|
| Representation | Four-parameter analytical zero curve |
| Compounding | Continuous |
| Device evaluation | Direct analytical functions |
| Stored maturity grid | None |
| Fitted stochastic models | Hull–White and G2++ |

## Role and reference

This directory implements the analytical continuously compounded
Nelson–Siegel zero curve. See
[Nelson and Siegel (1987)](https://doi.org/10.1086/296409).

## Dataset row

`NelsonSiegelParameters` rows are trivially copyable and transferred as one
contiguous FP32 array.

| Symbol | Dataset field |
|---|---|
| $\beta_0$ | `beta0` |
| $\beta_1$ | `beta1` |
| $\beta_2$ | `beta2` |
| $\tau$ | `tau` |

## Parameterization

For `x = T/tau`, the continuously compounded zero rate is

```text
z(0,T) = beta0
       + beta1 (1-exp(-x))/x
       + beta2 ((1-exp(-x))/x - exp(-x)).
```

The discount factor is `P(0,T) = exp(-T z(0,T))`. The instantaneous forward is
derived analytically rather than numerically differentiated.

## Device interface

| Function | Role |
|---|---|
| `zero_rate` | Evaluate $z(0,T)$ |
| `log_discount_factor` | Evaluate $\log P(0,T)$ without exponentiating |
| `discount_factor` | Evaluate $P(0,T)$ |
| `instantaneous_forward` | Evaluate $f(0,T)$ analytically |
| `forward_derivative` | Differentiate $f(0,T)$ with respect to maturity |
| `forward_rate` | Derive the continuous forward over `[start_years,end_years]` |

The exact device declarations are owned by
[`term_structure.cuh`](term_structure.cuh); their included definitions are in
[`term_structure_impl.cuh`](term_structure_impl.cuh).

## Consumers

The curve can be read directly by fixed-income products. Hull–White and G2++
Nelson–Siegel analytics combine one curve row with one stochastic-model row and
derive the deterministic short-rate shift needed to fit this initial curve.

## Memory and numerical policy

No term-structure grid is stored. Near `T=0`, `(1-exp(-x))/x` uses a short
series; elsewhere it uses `expm1f` to avoid cancellation. Log discounts are
computed before exponentiation so ratios remain stable. All curve values are
FP32 and fast-math is forbidden.

Related navigation: [curve catalog](../../../catalog/curve/nelson_siegel/),
[Hull–White consumer](../../model/fixed_income/hull_white/),
[G2++ consumer](../../model/fixed_income/g2_plus_plus/), and
[pricing contract](../../../docs/cuda/closed-form-and-monte-carlo-pricing-contract.md).

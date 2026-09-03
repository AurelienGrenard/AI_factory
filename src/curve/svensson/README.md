# Svensson curve

| At a glance | Value |
|---|---|
| Representation | Six-parameter analytical zero curve |
| Compounding | Continuous |
| Device evaluation | Direct analytical functions |
| Stored maturity grid | None |
| Fitted stochastic models | Hull–White and G2++ |

## Role and reference

This directory implements the analytical continuously compounded Svensson
extension of Nelson–Siegel. See
[Svensson (1994)](https://doi.org/10.5089/9781451853759.001).

## Files

- [`dataset.hpp`](dataset.hpp) / [`dataset.cpp`](dataset.cpp) define and load compact FP32 curve rows.
- [`term_structure.cuh`](term_structure.cuh) / [`term_structure_impl.cuh`](term_structure_impl.cuh) expose the analytical device functions.

There is no curve kernel here: pricing kernels include these device functions
and evaluate the curve directly for their own result row.

## Dataset row

`SvenssonParameters` rows are trivially copyable and transferred as one
contiguous FP32 array.

| Symbol | Dataset field |
|---|---|
| $\beta_0$ | `beta0` |
| $\beta_1$ | `beta1` |
| $\beta_2$ | `beta2` |
| $\beta_3$ | `beta3` |
| $\tau_1$ | `tau1` |
| $\tau_2$ | `tau2` |

## Parameterization

For `x1=T/tau1` and `x2=T/tau2`, the zero rate is

```text
z(0,T) = beta0
       + beta1 (1-exp(-x1))/x1
       + beta2 ((1-exp(-x1))/x1 - exp(-x1))
       + beta3 ((1-exp(-x2))/x2 - exp(-x2)).
```

The second curvature term permits an additional hump or trough. The discount
factor remains `P(0,T) = exp(-T z(0,T))`.

## Device interface

| Function | Role |
|---|---|
| `zero_rate` | Evaluate $z(0,T)$ |
| `log_discount_factor` | Evaluate $\log P(0,T)$ without exponentiating |
| `discount_factor` | Evaluate $P(0,T)$ |
| `instantaneous_forward` | Evaluate $f(0,T)$ analytically |
| `forward_derivative` | Differentiate $f(0,T)$ with respect to maturity |
| `forward_rate` | Derive the continuous forward over `[start,end]` |

<details>
<summary>Exact term-structure signatures</summary>

The declarations below omit CUDA attributes for readability.

```cpp
float zero_rate(const SvenssonParameters&, float maturity);
float log_discount_factor(const SvenssonParameters&, float maturity);
float discount_factor(const SvenssonParameters&, float maturity);
float instantaneous_forward(const SvenssonParameters&, float maturity);
float forward_derivative(const SvenssonParameters&, float maturity);
float forward_rate(const SvenssonParameters&, float start, float end);
```

</details>

## Consumers

The curve can be read directly by fixed-income products. Hull–White and G2++
Svensson analytics combine one curve row with one stochastic-model row and
derive the deterministic short-rate shift needed to fit this initial curve.

## Memory and numerical policy

No term-structure grid is stored. Both `(1-exp(-x))/x` loadings use a short
series near zero and `expm1f` elsewhere. Log discounts are evaluated before
exponentiation for stable ratios. All curve values are FP32 and fast-math is
forbidden.

Related navigation: [curve catalog](../../../catalog/curve/svensson/),
[Hull–White consumer](../../model/fixed_income/hull_white/),
[G2++ consumer](../../model/fixed_income/g2_plus_plus/), and
[pricing contract](../../../docs/cuda-closed-form-and-monte-carlo-pricing-contract.md).

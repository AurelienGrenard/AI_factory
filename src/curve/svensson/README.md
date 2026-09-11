# Svensson curve

| At a glance | Value |
|---|---|
| Representation | Six-parameter analytical zero curve |
| Compounding | Continuous |
| Device evaluation | Direct analytical functions |
| Stored maturity grid | None |

## Role and reference

This directory implements the analytical continuously compounded Svensson
extension of Nelson–Siegel. See
[Svensson (1994)](https://doi.org/10.5089/9781451853759.001).

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

## Implementation boundary

Implementation: [`term_structure.cuh`](term_structure.cuh) · [`term_structure_impl.cuh`](term_structure_impl.cuh) · [family contract](../README.md).

## Memory and numerical policy

No term-structure grid is stored. Both `(1-exp(-x))/x` loadings use a short
series near zero and `expm1f` elsewhere. Log discounts are evaluated before
exponentiation for stable ratios. All curve values are FP32 and fast-math is
forbidden.

Related navigation: [curve catalog](../../../catalog/curve/svensson/),
[analytics contract](../../../docs/cuda/model-analytics-contract.md), and
[capability manifest](../../../tools/codegen/pricing_bindings/capability_manifest.py).

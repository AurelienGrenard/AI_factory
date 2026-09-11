# Nelson–Siegel curve

| At a glance | Value |
|---|---|
| Representation | Four-parameter analytical zero curve |
| Compounding | Continuous |
| Device evaluation | Direct analytical functions |
| Stored maturity grid | None |

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

## Implementation boundary

Implementation: [`term_structure.cuh`](term_structure.cuh) · [`term_structure_impl.cuh`](term_structure_impl.cuh) · [family contract](../README.md).

## Memory and numerical policy

No term-structure grid is stored. Near `T=0`, `(1-exp(-x))/x` uses a short
series; elsewhere it uses `expm1f` to avoid cancellation. Log discounts are
computed before exponentiation so ratios remain stable. All curve values are
FP32 and fast-math is forbidden.

Related navigation: [curve catalog](../../../catalog/curve/nelson_siegel/),
[analytics contract](../../../docs/cuda/model-analytics-contract.md), and
[capability manifest](../../../tools/codegen/pricing_bindings/capability_manifest.py).

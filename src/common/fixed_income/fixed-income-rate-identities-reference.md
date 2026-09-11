# Shared fixed-income rate identities

This page defines the model-independent rate and swap quantities reused by
fixed-income analytics. Each model reference owns only its formula for the
zero-coupon bond $`P(t,T)`$.

For an accrual period $`[T_1,T_2]`$ with contractual year fraction
$`\delta\gt 0`$, the single-curve forward rate is

```math
L(t,T_1,T_2)
=\frac{1}{\delta}
\left(\frac{P(t,T_1)}{P(t,T_2)}-1\right).
```

For payment dates $`T_0\lt T_1\lt\cdots\lt T_n`$ and contractual accrual
fractions $`\delta_1,\ldots,\delta_n`$, the swap annuity is

```math
A_{\mathrm{swap}}(t)
=\sum_{i=1}^{n}\delta_iP(t,T_i).
```

The corresponding par swap rate is

```math
S(t;T_0,T_n)
=\frac{P(t,T_0)-P(t,T_n)}{A_{\mathrm{swap}}(t)}.
```

Product-specific schedules, sides, strikes and payoffs remain owned by
[`src/product`](../../product/README.md). The canonical device interfaces that
evaluate these quantities are defined by the
[analytics contract](../../../docs/cuda/model-analytics-contract.md).

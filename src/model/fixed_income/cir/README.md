# CIR

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Related contracts](#related-contracts)

## Dynamics

The short rate follows the square-root diffusion

```math
\mathrm dr_t
=\kappa(\theta-r_t)\,\mathrm dt
+\sigma\sqrt{r_t}\,\mathrm dW_t,
\qquad r_0\geq0,
```

where $`W`$ is a standard Brownian motion, $`\kappa\gt 0`$ is the mean-reversion
speed, $`\theta\gt 0`$ is the long-run rate, and $`\sigma\gt 0`$ is the volatility.

For an interval of length $`\Delta`$, define

```math
c_\Delta
=\frac{\sigma^2(1-e^{-\kappa\Delta})}{4\kappa},
\qquad
d=\frac{4\kappa\theta}{\sigma^2},
\qquad
\lambda_\Delta
=\frac{e^{-\kappa\Delta}r_t}{c_\Delta}.
```

If $`\chi'^2_d(\lambda)`$ denotes a non-central chi-square variable with $`d`$
degrees of freedom and noncentrality $`\lambda`$, then

```math
r_{t+\Delta}
=c_\Delta X,
\qquad
X\sim\chi'^2_d(\lambda_\Delta).
```

This exact transition preserves non-negativity and remains valid on either
side of the Feller condition $`2\kappa\theta\geq\sigma^2`$.

For fixed-step Monte Carlo paths, define the accumulated short rate by

```math
I_t=\int_0^t r_u\,\mathrm du.
```

Each rate endpoint is simulated with the exact transition above, while the
integral is accumulated with the trapezoidal rule

```math
I_{t+\Delta}
\approx I_t+\frac{\Delta}{2}(r_t+r_{t+\Delta}).
```

This joint approximation is not used by the Bermudan pricer. Finite-precision
sampling errors can accumulate over many short steps, so refinement alone does
not guarantee a more accurate GPU price.

For Bermudan swaptions, `forward_measure.cuh` supplies exact transitions under
the last-exercise bond measure. Only exercise dates are simulated; payoffs are
normalized by `P(0,T*) / P(t,T*)`. The product binding composes the shared LSM
engine and a scalar-rate continuation state, with no short-rate integral.
See the [early-exercise contract](../../../../docs/cuda/american-and-bermudan-pricing-contract.md#numéraire-terminal-pour-cir).

| Symbol | Dataset field | Meaning |
|---:|---|---|
| $`r_0`$ | `initial_state` | Initial short rate |
| $`\kappa`$ | `mean_reversion` | Mean-reversion speed |
| $`\theta`$ | `long_term_mean` | Long-run rate |
| $`\sigma`$ | `volatility` | Volatility |

The model follows [Cox, Ingersoll, and Ross (1985)](https://doi.org/10.2307/1911242).

## Core formulas

Let $`\tau=T-t`$ and define

```math
\gamma=\sqrt{\kappa^2+2\sigma^2}.
```

The zero-coupon bond is exponential-affine:

```math
P(t,T)=A(t,T)e^{-B(t,T)r_t},
```

with

```math
B(t,T)
=\frac{2(e^{\gamma\tau}-1)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma},
```

```math
A(t,T)
=\left[
\frac{2\gamma e^{(\kappa+\gamma)\tau/2}}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}
\right]^{2\kappa\theta/\sigma^2}.
```

For an accumulated short-rate integral
$`I_t=\int_0^t r_u\,\mathrm du`$, define

```math
D(0,t)=e^{-I_t}.
```

Shared rate and swap identities: [fixed-income rate reference](../../../common/fixed_income/fixed-income-rate-identities-reference.md).

## Related contracts

Shared product and engine contracts: [fixed-income entry point](../README.md).

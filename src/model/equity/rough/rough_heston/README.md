# Rough Heston

## Numerical model

The implemented variance convention is

```math
V_t=V_0+\int_0^t K_H(t-s)
\left[(\theta-\lambda V_s)\,\mathrm ds
+\nu\sqrt{V_s}\,\mathrm dW_s\right],
\qquad
K_H(t)=\frac{t^{H-1/2}}{\Gamma(H+1/2)}.
```

Consequently `variance_drift` is $`\theta`$ itself; the stationary level of
the drift is $`\theta/\lambda`$, not `variance_drift`.

The kernel is replaced on a caller-selected horizon by a positive sum of
exponentials,

```math
K_H(t)\simeq\sum_{i=1}^{N}w_i e^{-x_i t},
```

and the Volterra equation becomes a Markovian lift with `N` memory factors.
Those factors are not Gaussian: their common stochastic increment contains
`sqrt(V)`. `common/volterra/exponential_kernel.cuh` is generic; the weak
variance update in `dynamics_impl.cuh` remains rough-Heston-specific.

`fit_positive_fractional_kernel_l2<N>` starts from positive geometric cells in
the fractional kernel's Laplace measure, fits their truncation bounds, then
optimizes positive weights and ordered nodes in L2 on `[dt, horizon]`, the
interval actually resolved by the time grid. It
is intentionally named a bounded positive L2 rule, not BL2: replacing it by a
published BL2 catalogue or optimizer does not change the CUDA dynamics
contract.

## Factorization

The implementation separates raw [parameters](parameters.hpp), host-side
[numerical preparation](numerics.hpp), the fixed-factor
[dynamics](dynamics.cuh), and the public sampling and pricing compositions.
Dataset loading remains host-only; product policies remain shared with the
other equity models.

`prepare_dynamics<N>` runs on the host once per model and `dt`. It stores the
positive nodes/weights, the lifted initial state, and the matrix exponential
and affine shift of one ODE half-step. The caller uploads one
`PreparedDynamics<N>` per model. It is selected with the same model index as
the raw parameters, so Cartesian model/product pricing does not duplicate
these coefficients per result.
Bulk preparation caches the fitted kernel by the exact FP32 Hurst exponent;
models sharing `H`, the approximation horizon and `dt` do not repeat the L2
fit.

The common fixed-step schedules then add the product maturity or sample
calendar and time configuration. Product handlers, pricing policies and sample
observations see the usual spot state; they do not know how many lifted factors
produced it.

`N` is a compile-time parameter. The public European and sampling launchers are
instantiated for `N=2`, `N=3`, and `N=7`, with host dispatch selecting the
desired accuracy. There is no runtime factor loop bound, dynamic allocation,
virtual call or device-side matrix exponential.

Sampling uses the canonical `dt = 1/504` and two transitions per business day.
The caller prepares one dynamics row per model with an approximation horizon
covering every requested maturity, then uploads those rows. The common Markov
sampler uses grid-stride execution for one path per parameter and shares one
prepared row per block for conditional packages such as `P = 250`.

## Weak step and random stream

One time step applies a Strang splitting:

1. orthogonal stock Brownian half-step;
2. variance ODE half-step;
3. three-point weak stochastic variance step;
4. variance ODE half-step;
5. correlated stock update reconstructed from the first lifted factor;
6. orthogonal stock Brownian half-step.

The path consumes one normal, one uniform, then the cached second Box-Muller
normal. Repeating a launch with the same row seed and path index is bitwise
reproducible. FP64 is reserved for the final price moments; states and the hot
path remain FP32.

The ODE/SDE/ODE variance split, three-point weak law and correlated-stock
reconstruction follow the reference implementation published with
[Bayer and Breneis' Markovian rough-volatility approximations](https://github.com/SimonBreneis/approximations_to_fractional_stochastic_volterra_equations).

## Current validation boundary

The host test checks positivity and ordering of the exponential rule, decreasing
L2 error from 2 to 3 to 7 factors, exact reconstruction of `V0`, and finite
matrix-exponential coefficients. The CUDA tests cover call/put finiteness,
sampling terminal/calendar layouts, conditional path diversity and bitwise
replay across launch geometries when a GPU is available. Independent
price/convergence datasets and additional products remain separate work.

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

## N-factor approximation

`prepare_dynamics<N>` runs on the host once per model and `dt`. It stores the
positive nodes/weights, the lifted initial state, and the matrix exponential
and affine shift of one ODE half-step. The caller uploads one
`PreparedDynamics<N>` per model. It is selected with the same model index as
the raw parameters, so Cartesian model/product pricing does not duplicate
these coefficients per result.
Bulk preparation caches the fitted kernel by the exact FP32 Hurst exponent;
models sharing `H`, the approximation horizon and `dt` do not repeat the L2
fit.

`N` is a compile-time parameter. Supported factor counts and public bindings
are declared by the
[capability manifest](../../../../../tools/codegen/pricing_bindings/capability_manifest.py).
There is no runtime factor loop bound, dynamic allocation, virtual call or
device-side matrix exponential.

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
reproducible. State arithmetic remains FP32.

The ODE/SDE/ODE variance split, three-point weak law and correlated-stock
reconstruction follow the reference implementation published with
[Bayer and Breneis' Markovian rough-volatility approximations](https://github.com/SimonBreneis/approximations_to_fractional_stochastic_volterra_equations).

## Verification

Host and CUDA tests cover the exponential rule, preparation coefficients,
sampling layouts, finite pricing results, path diversity, and deterministic
replay. Discover the current test and product inventory from CTest and the
capability manifest rather than from a copied list in this page.

Shared schedules, product policies and runtime composition are documented by
the [rough-family entry point](../README.md) and the
[pricing composition](../../../../../docs/cuda/pricing-policy-composition.md).

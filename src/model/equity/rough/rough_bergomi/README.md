# Rough Bergomi

## Model

For correlated Brownian motions `W` and `Z`, the implementation uses

```math
Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,\mathrm dW_s,
\qquad \operatorname{Var}(Y_t)=t^{2H},
```

```math
v_t=\xi_0\exp\!\left(\eta Y_t-\frac{\eta^2}{2}t^{2H}\right),
```

```math
\frac{\mathrm dS_t}{S_t}=(r-q)\,\mathrm dt+\sqrt{v_t}\,\mathrm dZ_t,
\qquad \mathrm d\langle Z,W\rangle_t=\rho\,\mathrm dt.
```

`dynamics.cuh` and `dynamics_impl.cuh` contain only the model-specific path
mapping: prepare invariant coefficients, create `(log_spot, variance)`,
transform `Y_i` into `v_i`, and advance `S_i`. It deliberately does not pretend
to be a Markovian `t -> t + dt` dynamics policy.

## Numerical scheme

The production implementation uses the kappa=1 hybrid scheme evaluated by the
common cuFFTDx engine. The singular current cell is evaluated directly; the
stationary contribution of earlier cells is a linear convolution evaluated by
FFT. The inverse convolution is consumed in bounded path chunks, so workspace
size is independent of the total Monte Carlo path count.

The model follows Bayer, Friz and Gatheral (2016). The simulation follows the
hybrid scheme of Bennedsen, Lunde and Pakkanen (2017).

Related contracts: [rough-family entry point](../README.md) ·
[pricing composition](../../../../../docs/cuda/pricing-policy-composition.md).

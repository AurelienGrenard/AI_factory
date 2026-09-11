# Rough SABR

## Model

The CUDA implementation uses the normalized Gaussian Volterra kernel

```math
Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,\mathrm dW_s,
\qquad \operatorname{Var}(Y_t)=t^{2H},
```

and the convention in which `xi_0` is the initial log-return variance at
`S_0`,

```math
\alpha_t=\sqrt{\xi_0}S_0^{1-\beta}\exp\!\left(
\frac{\eta}{2}Y_t-\frac{\eta^2}{4}t^{2H}
\right),
```

```math
\mathrm dS_t=(r-q)S_t\,\mathrm dt
              +\alpha_t S_t^\beta\,\mathrm dZ_t,
\qquad \mathrm d\langle Z,W\rangle_t=\rho\,\mathrm dt.
```

Here `eta` is the log-variance volatility convention. Consequently `beta=1`
is pathwise identical to rough Bergomi with the same parameter row. For
`beta<1`, `dynamics.cuh` and `dynamics_impl.cuh` step the Lamperti coordinate

```math
L_t=\frac{S_t^{1-\beta}}{1-\beta},\qquad
\mathrm dL_t=(1-\beta)(r-q)L_t\,\mathrm dt+\alpha_t\,\mathrm dZ_t
-\frac{\beta\alpha_t^2}{2(1-\beta)L_t}\,\mathrm dt.
```

The transformed state is floored before mapping it back to spot. This is an
explicit numerical convention near zero, not an exact treatment of CEV
absorption.

## Numerical scheme

The implementation uses the shared kappa=1 hybrid discretization. The singular
current cell is evaluated directly and the stationary history is convolved by
FFT. After reconstruction, only the rough-SABR transformation and Lamperti
step are model-specific. Bounded path chunks keep workspace independent of the
total Monte Carlo path count.

The model convention follows the rough-SABR formulation of Fukasawa and
Gatheral. There is no parameter conversion for `eta`: their log-variance
coefficient becomes `eta/2` in `alpha=sqrt(xi)`, which also preserves the
exact rough-Bergomi `beta=1` limit.

Related contracts: [rough-family entry point](../README.md) ·
[pricing composition](../../../../../docs/cuda/pricing-policy-composition.md).

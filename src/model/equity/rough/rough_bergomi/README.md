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

## Unique production path

There is one production implementation: the kappa=1 hybrid scheme evaluated
by the common cuFFTDx engine. The former direct `O(N^2)` and separate
`european_option_fft` entry points do not exist.

`volterra_fft_pricing.cuh` composes four independent policies:

1. `FractionalHybridKernelPolicy` owns the singular current cell and the
   stationary far-cell kernel;
2. `rough_bergomi::PathPolicy` owns `Y_i -> v_i -> S_i`;
3. a `SchedulePolicy` maps contractual integer days to terminal, dense,
   regular or static-calendar observations;
4. a `ProductPolicy` owns product preparation, its scalar spot handler and
   final payoff.

The engine computes the kernel spectrum once per price. One complex C2C FFT
packs two real Brownian paths. Inverse convolutions are written only for a
caller-bounded path chunk; a dense 256-thread kernel then replays the canonical
Philox stream, evolves the model and invokes the product handler. The buffer is
reused across chunks and result rows, so memory is independent of the total
Monte Carlo path count.

The attempted in-block fusion of IFFT and spot evolution is intentionally not
used: after a cooperative FFT it leaves only two sequential path consumers per
transform while retaining the block's FFT resources. On the reference RTX
4090 Laptop this measured about 52 ms at one year and one million paths,
against about 10.4 ms for the bounded staging design.

`european_option.cuh/.cu` is the concrete call/put binding.
`volterra_fft_pricing.cuh` is the generic product/schedule entry point. The GPU tests
instantiate terminal vanilla, dense barrier, dense Asian and two-date
forward-start policies through the same FFT implementation.

The model follows Bayer, Friz and Gatheral (2016). The simulation follows the
hybrid scheme of Bennedsen, Lunde and Pakkanen (2017).

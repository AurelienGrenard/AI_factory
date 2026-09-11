# CIR++

[Dynamics](#dynamics) · [Core formulas](#core-formulas) · [Integration](#integration)

## Dynamics

CIR++ fits an independent initial discount curve while retaining the exact
[CIR factor dynamics](../cir/README.md):

$$
r_t=y_t+\phi(t),\qquad
\mathrm{d}y_t=\kappa(\theta-y_t)\,\mathrm{d}t
              +\sigma\sqrt{y_t}\,\mathrm{d}W_t,\qquad y_0\geq0.
$$

Here `initial_state` is the factor `y0`, not the initial short rate.
The process parameters and domain checks are shared with CIR. The exact
noncentral-chi-square transition supports both Feller regimes. Samples contain
the nonnegative factor `y`; the total rate may be negative after shifting.

Nelson–Siegel and Svensson supply the initial market curve. Neither the
parameter dataset nor the factor sampler owns a curve.

## Core formulas

Let $P_{CIR}(0,t)$ use the model's own initial factor. With market curve
$P_M(0,t)$, define

$$
\phi(t)=f_M(0,t)-f_{CIR}(0,t),\qquad
D_\phi(t,T)=\exp\left(-\int_t^T\phi(u)\,\mathrm{d}u\right)
=\frac{P_M(0,T)P_{CIR}(0,t)}{P_M(0,t)P_{CIR}(0,T)}.
$$

Conditional bonds and unit bond options follow directly:

$$
P_{++}(t,T;y)=D_\phi(t,T)P_{CIR}(t,T;y),\qquad P_{++}(0,T;y_0)=P_M(0,T),
$$

$$
V_{++}(t,S,T,K;y)
=D_\phi(t,T)\,V_{CIR}\!\left(t,S,T,\frac{K}{D_\phi(S,T)};y\right).
$$

The same scaling applies to calls and puts. Caplets/floorlets reuse their
bond-option identities; European swaptions reuse the shared Jamshidian engine.
The affine loading `B` and CIR option context are not duplicated.

See Brigo–Mercurio,
[On deterministic-shift extensions of short-rate models](https://www.ma.imperial.ac.uk/~dbrigo/detshiftrep.pdf),
equations (5)–(12).

## Integration

Bermudan swaptions reuse CIR's exact terminal-forward transitions under the
last-exercise bond measure and the common LSM engine. A deterministic shift
cancels from the normalized change-of-measure density. Only fitted conditional
bonds, exercise values and numeraire values differ. Exercise coefficients use
absolute dates: a fitted bond cannot be replaced by a time-zero bond with the
same tenor. No fine grid or simulated rate integral is introduced.

The codegen owns both curves' product bindings, all 16 price recipes, both
sample recipes and CMake registration. Each Bermudan recipe uses `2^20`
trajectories per price. Parameters follow the ordered 900 core / 100 stress
policy. Launch profiles remain defaults to measure on the user's GPU.

Build `model_cir_plus_plus` and `test_cir_plus_plus_cuda` with CMake, then run
the `cir_plus_plus_cuda` CTest. An explicit independent diagnostic consumes
the test's JSON-lines output through `tools/cuda/check_cir_plus_plus.py`.
It checks conditional analytics against QuantLib CIR with deterministic-shift
scaling, and Bermudans against a refined risk-neutral PDE.

These checks are not catalogue certification. Price recipes retain
`verified: false` until all core and stress rows have persistent independent
references. Premia exposes `CirPP1D/STDi`: bond and swaption closed forms,
`FD_GaussZBO`, `TR_CAPFLOOR` and `TR_SwaptionCIRpp1D`. Their operational
qualification must precede publication. `CirPP2D`, a credit model, is not CIR++.

Shared contracts: [fixed-income entry point](../README.md).

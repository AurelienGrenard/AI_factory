# Shared mean-reverting Gaussian formulas

<details>
<summary>Implementation</summary>

```text
common/
├── README.md
└── mean_reverting_gaussian.cuh
```

</details>

[Core formulas](#core-formulas)

## Core formulas

For a centered Ornstein–Uhlenbeck factor

```math
\mathrm dx_t=-a x_t\,\mathrm dt+\sigma\,\mathrm dW_t,
```

where $`W`$ is a standard Brownian motion, $`a\gt 0`$, and $`\sigma\gt 0`$. Let
$`\mathbb E_t`$, $`\mathrm{Var}_t`$, and $`\mathrm{Cov}_t`$ denote moments
conditional on $`x_t`$. For one transition interval $`\Delta\gt 0`$, define

```math
q=e^{-a\Delta},
\qquad
B_\Delta=\frac{1-q}{a}.
```

The exact endpoint has conditional mean and variance

```math
\mathbb E_t[x_{t+\Delta}]=q x_t,
\qquad
V_x=\mathrm{Var}_t(x_{t+\Delta})
=\sigma^2\frac{1-q^2}{2a}.
```

For the interval integral

```math
J_\Delta=\int_t^{t+\Delta}x_s\,\mathrm ds,
```

the conditional moments are

```math
\mathbb E_t[J_\Delta]=B_\Delta x_t,
```

```math
V_J=\mathrm{Var}_t(J_\Delta)
=\frac{\sigma^2}{a^2}
\left[
\Delta-\frac{2(1-q)}{a}+\frac{1-q^2}{2a}
\right],
```

```math
C_{xJ}=\mathrm{Cov}_t(x_{t+\Delta},J_\Delta)
=\frac{\sigma^2}{2a^2}(1-q)^2.
```

These deterministic quantities are reused by Ornstein–Uhlenbeck, Vasicek,
Hull–White, G2, and G2++ to assemble exact Gaussian state and state-integral
transitions.

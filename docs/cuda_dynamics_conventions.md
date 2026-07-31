# CUDA Dynamics Conventions

Every simulated model uses the same small interface. A derived model may reuse
another model when that dependency reflects its mathematical construction.

## Layers

Fixed-income code uses three explicit layers:

- `src/model/<model>/dynamics.cuh/.cu` implements standalone model dynamics;
- `src/model/<model>/analytics.cuh/.cu` exposes reusable model formulas;
- `src/curve/<curve>/term_structure.cuh/.cu` implements one analytical curve
  behind generic names such as `discount_factor` and `forward_rate`;
- `src/model/<model>/<curve>/analytics.cuh/.cu` composes the process and curve
  into the analytical quantities of the calibrated model.

The raw model row and its `load_models` function remain in
`src/model/<model>/dataset.hpp/.cpp`. Curve formulas, payoff logic, pricing
helpers, and product kernels stay in their dedicated layers or implementation
files.

## Types

Use clearly separated raw, prepared, and path-state structures:

```cpp
struct ModelNameModelParameters;
struct ModelNameMethodParameters;
struct ModelNameState;
```

- `ModelParameters` contains the raw FP32 row loaded from the model dataset.
- `MethodParameters` contains coefficients precomputed for the selected
  numerical method, such as `HestonQeParameters` or
  `OrnsteinUhlenbeckExactParameters`.
- `State` contains only mutable values private to one simulated path.

Process prepared structures retain only transition coefficients. A fitted
full-model structure may additionally retain the raw model row and curve when
later pricing analytics need them. In particular, `dt` is computed during
preparation and stored only when a later transition reads it.

Product dataset rows use the corresponding `ProductNameParameters` convention.

## Process Interface

Declare and implement these functions in this exact order for every simulated
process and every self-contained model such as Heston:

```cpp
MethodParameters prepare_model(
    const ModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

State initial_state(/* prepared model only when mathematically required */);

void one_step_transition(
    const MethodParameters& model,
    /* model-specific random variates */,
    State& state
);

State simulate_terminal_state(
    const MethodParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

State simulate_on_regular_grid(
    const MethodParameters& initial_stub_model,
    const MethodParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    /* model-specific observed-state arrays */
);
```

`prepare_model` computes:

```cpp
const float dt = maturity / static_cast<float>(num_steps);
```

and precomputes every coefficient shared by paths using that row and time
step.

`initial_state` takes the prepared model only when the initial state genuinely
depends on model data. Do not add unused arguments merely to make signatures
look identical.

`one_step_transition` receives explicit model-specific random variates.
Differences in their number or names must follow from the numerical method.

`simulate_terminal_state` performs all time steps and returns the complete
terminal state. Payoffs decide which state components to transform or consume.

`simulate_on_regular_grid` stores pre-maturity states in date-major arrays and
returns the terminal state directly. For each date, consecutive threads must
write consecutive paths.

## Fitted Models

A derived model must not repeat another model's functions merely to preserve
naming. Hull-White/Nelson-Siegel therefore reuses
`model::ornstein_uhlenbeck` and exposes only its reconstruction formulas:

```cpp
hull_white::nelson_siegel::prepare_model(...);
hull_white::nelson_siegel::short_rate(...);
hull_white::nelson_siegel::discount_factor(...);
hull_white::nelson_siegel::zero_coupon_bond(...);
```

The stochastic state contains the OU factor `x` and its integral. Hull-White
reconstructs `r(t) = x(t) + phi(t)` only when a payoff needs a rate, discount,
or bond value. This avoids curve evaluations inside the simulation loop.

## Optional Path Summaries

A model may expose a specialized path summary when several products can reuse
it without introducing product parameters into the dynamics layer. Heston, for
example, provides arithmetic-mean and maximum-spot simulations.

Keep product-dependent stopping rules in the product kernel. A barrier option
should therefore stop its own path simulation as soon as the barrier is hit,
rather than pass barrier levels into the generic model dynamics.

## Naming And Loop Structure

Namespaces carry the process, model, and curve names, so reusable functions do
not repeat them:

```cpp
heston::prepare_model(...);
model::ornstein_uhlenbeck::prepare_model(...);
hull_white::nelson_siegel::prepare_model(...);
```

Use the same control names across models:

```cpp
step_index
exercise
output_index
uniform_count
groups_per_path
first_group
uniforms
state
```

Use `initial_stub_model` and `regular_model` for grids whose first interval
differs from subsequent intervals.

Model-specific names are justified only when they identify different
mathematics or different data, for example:

```cpp
observed_spots
observed_variances
observed_factors
observed_integrated_factors
```

## Design Rule

Uniformity is the default. A new structure, function, argument, or local name
must correspond to a real difference in mathematics, data layout, output, or
execution strategy.

Do not introduce templates or wrapper abstractions solely to force identical
syntax. Keep the common shape visible in straightforward C++/CUDA code, while
leaving justified model-specific calculations explicit.

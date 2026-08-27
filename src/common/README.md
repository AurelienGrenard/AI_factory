# Common CUDA infrastructure

[`check_cuda.cuh`](#check-cuda) ·
[`cuda_kernel_diagnostics.cuh/.cpp`](#cuda-kernel-diagnostics) ·
[`device_inputs.cuh`](#device-inputs) ·
[`time_configuration.cuh`](#time-configuration) ·
[`option_side.cuh`](#option-side) ·
[`closed_form/`](#closed-form) ·
[`simulation/`](#simulation) ·
[`volterra/`](#volterra) ·
[`monte_carlo/`](#monte-carlo) ·
[`payoff/`](#payoff) ·
[`equity/`](#equity) ·
[`fixed_income/`](#fixed-income) ·
[`result_index.cuh`](#result-index) ·
[`sample.cuh`](#sample) ·
[`philox.cuh`](#philox) ·
[`normal_distribution.cuh`](#normal-distribution) ·
[`lognormal_option.cuh`](#lognormal-option) ·
[`noncentral_chi_square.cuh`](#noncentral-chi-square) ·
[`reductions.cuh`](#reductions) ·
[`longstaff_schwartz/`](#longstaff-schwartz)

<a id="check-cuda"></a>
## [`check_cuda.cuh`](check_cuda.cuh)

| Function | Definition |
|---|---|
| `check_cuda(status, operation)` | Throws a C++ exception containing `operation` and the CUDA Runtime error represented by `status`. |
| `checked_workspace_product(left, right, message)` | Returns `left * right` after rejecting `std::size_t` overflow. |
| `bounded_block_count(result_count, block_count)` | Returns `min(result_count, block_count)` after rejecting zero dimensions. |
| `validate_block_count(result_count, block_count)` | Requires `1 <= block_count <= result_count`. |
| `validate_device_pointer(pointer, name)` | Requires a non-null pointer whose CUDA attributes report device memory. |
| `validate_model_product_construction(...)` | Validates aligned rows or the Cartesian count `model_count * product_count`. |
| `validate_model_curve_product_construction(...)` | Validates aligned rows or the Cartesian count `model_count * curve_count * product_count`. |
| `validate_monte_carlo_path_count(paths_per_result)` | Requires at least two paths so a sample variance exists. |
| `validate_day_fraction(day_fraction)` | Requires a positive finite contractual day fraction. |
| `validate_time_step(dt)` | Requires a positive finite numerical time step. |
| `validate_monte_carlo_parameters(paths_per_result, dt)` | Validates the path count and the positive finite simulation step `dt`. |
| `validate_simulation_steps_per_day(steps)` | Requires at least one simulation step per contractual day. |
| `validate_cuda_block_size(threads_per_block)` | Checks the positive block size against the active device limit. |
| `validate_reduction_block_size(threads_per_block)` | Additionally requires a whole number of 32-thread warps. |
| `validate_grid_x_size(block_count)` | Checks `block_count` against the active device `gridDim.x` limit. |
| `validate_row_seed_range(result_count, base_seed)` | Requires `base_seed + result_count - 1` to fit in `std::uint64_t`. |

<a id="cuda-kernel-diagnostics"></a>
## [`cuda_kernel_diagnostics.cuh`](cuda_kernel_diagnostics.cuh) / [`cuda_kernel_diagnostics.cpp`](cuda_kernel_diagnostics.cpp)

For `B` active blocks per streaming multiprocessor, `W_B` warps per block and
`W_max` supported warps per streaming multiprocessor, the reported theoretical
occupancy is

```math
\mathrm{occupancy}=\frac{B W_B}{W_{\max}}.
```

| Function or type | Definition |
|---|---|
| `CudaKernelLaunchDiagnostics` | Device identity, launch geometry, kernel resources and theoretical occupancy. |
| `cuda_kernel_diagnostics_enabled()` | Reads the opt-in `AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS` flag. |
| `reserve_cuda_kernel_launch_diagnostics(...)` | Deduplicates one `(kernel, variant, grid, block, shared memory)` report. |
| `inspect_cuda_kernel_launch(kernel, grid, block, shared_bytes)` | Returns device limits, registers, local/shared memory, active blocks and theoretical occupancy for the exact specialization. |
| `emit_cuda_kernel_launch_diagnostics(...)` | Writes the reserved report as one JSON object. |
| `report_cuda_kernel_launch_if_enabled(...)` | Performs reservation, inspection and emission only when diagnostics are enabled. |
| `environment_flag_enabled(value)` | Internal parser accepting `1`, `true` or `on`. |
| `report_key(...)` | Internal stable key used to deduplicate reports. |

<a id="device-inputs"></a>
## [`device_inputs.cuh`](device_inputs.cuh)

| Type or function | Definition |
|---|---|
| `ModelProductDeviceInputs<Model, Product>` | Contiguous device views, row counts and aligned/Cartesian mapping for model-product prices. |
| `ModelCurveProductDeviceInputs<Model, Curve, Product>` | Same contract with one independent parametric-curve array. |
| `DeviceInputsWithContext<Inputs, Context>` | Adds a trivially-copyable device context such as an explicit schedule pool. |
| `PreparedModelProductDeviceInputs<Model, Product, Dynamics>` | Selects one host-prepared numerical dynamics row with the decoded model index, without result-wise duplication. |
| `make_model_product_device_inputs(...)` | Constructs the two-array input view without allocation. |
| `make_model_curve_product_device_inputs(...)` | Constructs the three-array input view without allocation. |
| `with_device_context(inputs, context)` | Adds one context to an existing primary input view. |

Each view owns no memory. `validate(result_count)` checks its pointers and row
construction on the host; `prepare_row<Pricing>(result_index, time)` decodes
the same mapping and calls the pricing policy on the device. Additional
trivially-copyable inputs are forwarded variadically to that policy.
`DeviceInputsWithContext` uses this delegation and never inspects a curve type
or the members of its primary input. Monte Carlo and closed-form kernels
therefore share one input contract without sharing an execution strategy.

<a id="volterra"></a>
## [`volterra/`](volterra)

The Volterra layer contains only kernel representations, schedules and shared
execution primitives. `FractionalPowerKernel` provides power-kernel cell
averages; `FractionalHybridDriverPolicy` adds the exact current-cell coupling;
`ExponentialKernel<N>` stores a fixed-size positive Markovian approximation;
and `fit_positive_fractional_kernel_l2<N>` prepares that approximation on the
host over `[dt, horizon]`. `execute_padded_linear_convolution` owns the padded
forward transform, spectrum product and inverse transform for one block FFT.

`hybrid_fft_pricer.cuh` composes one Gaussian driver, one model path mapping,
one observation schedule and one product policy. It packs two real paths into
each C2C transform, stages only a bounded chunk of convolution values, then
runs fully occupied path/product threads. Terminal, dense, regular and static
calendars reuse the same FFT code. Model mappings such as rough Bergomi and
rough SABR remain in their model directories; rough Heston's nonlinear
variance feedback instead uses its Markovian exponential lift.

The SM89 crossover experiment keeps cuFFTDx for every supported length: a
bounded direct convolution is already 35% slower at eight steps and degrades
quadratically thereafter. The 8192 transform uses 16 elements per thread;
against 32, this raises theoretical occupancy from 16.7% to 33.3%, lowers the
register count from 139 to 72 and improves the measured median by 20.4%. The
direct implementation remains confined to the performance experiment target.

<a id="time-configuration"></a>
## [`time_configuration.cuh`](time_configuration.cuh)

```math
t(d)=d\,\delta_{\mathrm{day}}.
```

| Type or function | Definition |
|---|---|
| `time::DayFractionTimeConfiguration` | Stores the year fraction represented by one contractual day. |
| `time::validate_time_configuration(...)` | Requires a positive finite day fraction. |
| `time::year_fraction(day_count, configuration)` | Converts an integer contractual day count to model time in FP32. |

<a id="option-side"></a>
## [`option_side.cuh`](option_side.cuh)

| Symbol | Definition |
|---|---|
| `OptionSide::call` | Call orientation. |
| `OptionSide::put` | Put orientation. |
| `option_side_name(side)` | Returns the stable host label `"call"` or `"put"`. |

<a id="closed-form"></a>
## [`closed_form/`](closed_form)

[`concepts.cuh`](#closed-form-concepts) ·
[`closed_form_kernels.cuh`](#closed-form-kernel) ·
[`cooperative_closed_form_kernels.cuh`](#cooperative-closed-form-kernel)

<a id="closed-form-concepts"></a>
### [`concepts.cuh`](closed_form/concepts.cuh)

| Concept | Contract |
|---|---|
| `ClosedFormPreparedRowPolicy` | Common device-input, time-configuration and prepared-row contract. |
| `ClosedFormPricingPolicy` | Trivially-copyable `DeviceInputs`, `TimeConfiguration` and `PreparedRow`; input-driven row preparation; scalar `evaluate_price(row)`. |
| `CooperativeClosedFormPricingPolicy` | Same prepared-row contract plus dynamic shared-memory sizing and block-cooperative `evaluate_price`. |

The concept constrains only the interface. `price_one` enforces the
`kMaximumThreadPreparedRowBytes = 256` per-thread storage budget and asks a
larger contract to use a compact view over device-resident data.

<a id="closed-form-kernel"></a>
### [`closed_form_kernels.cuh`](closed_form/closed_form_kernels.cuh)

| Function | Definition |
|---|---|
| `price_one<Pricing>(...)` | Prepares and evaluates one independent result row. |
| `closed_form_price_kernel<Pricing, false>(...)` | Direct specialization with one thread per price and no loop. |
| `closed_form_price_kernel<Pricing, true>(...)` | Grid-stride specialization used when the launch contains fewer threads than prices. |
| `validate_closed_form_launch<Pricing>(...)` | Validates inputs, time, result batch and CUDA geometry. |
| `launch_closed_form_cuda<Pricing>(...)` | Selects the direct or grid-stride specialization, reports diagnostics and launches it. |

<a id="cooperative-closed-form-kernel"></a>
### [`cooperative_closed_form_kernels.cuh`](closed_form/cooperative_closed_form_kernels.cuh)

| Function | Definition |
|---|---|
| `cooperative_closed_form_price_kernel<Pricing>(...)` | Stores one prepared row per block and evaluates one price cooperatively. |
| `validate_cooperative_closed_form_launch<Pricing>(...)` | Validates inputs, workspace capacity, result batch and CUDA geometry. |
| `launch_cooperative_closed_form_cuda<Pricing>(...)` | Checks dynamic shared-memory residency, reports diagnostics and launches the block-stride kernel. |

<a id="simulation"></a>
## [`simulation/`](simulation)

[`concepts.cuh`](#simulation-concepts) ·
[`adapted_dynamics.cuh`](#adapted-dynamics) ·
[`path_simulation.cuh`](#path-simulation) ·
[`schedule.cuh`](#simulation-schedules) ·
[`barrier_handlers.cuh`](#simulation-barrier-handlers)

<a id="simulation-concepts"></a>
### [`concepts.cuh`](simulation/concepts.cuh)

| Concept | Contract |
|---|---|
| `DynamicsPolicy` | Trivially-copyable parameters, random context and mutable state; no market-specific observable is imposed. |
| `FixedStepDynamicsPolicy` | Adds `PreparedDynamics`, homogeneous-step preparation, initialization and multi-step advancement. |
| `ExactTransitionDynamicsPolicy` | Adds invariant `PreparedModel`, interval-specific `PreparedTransition` and direct one-step simulation. |
| `ObservationHandlerFor` | Receives the initial state and observations and may stop a path. |
| `ScalarObservableFor` | Maps a model state to one scalar at inception and at every observation. |
| `SchedulePolicy` | Prepares a schedule from a calendar and a time configuration. |
| `ObservedSchedulePolicy` / `CountedObservedSchedulePolicy` | Adds observed path simulation and, when needed, its observation count. |
| `DenseSchedulePolicy` | Requires observation of every numerical transition. |
| `TerminalSchedulePolicy` | Returns a terminal state without an observation handler. |
| `TwoDateSchedulePolicy` | Requires a compile-time two-date calendar. |

Observation handlers remain trivially copyable and no larger than 128 bytes.
This is a measured local-state budget. The near-limit SM75/86/89 probes and
the compact-view alternative are recorded in the central pricing contract and
the versioned CUDA performance baseline.

<a id="adapted-dynamics"></a>
### [`adapted_dynamics.cuh`](simulation/adapted_dynamics.cuh)

| Type | Definition |
|---|---|
| `AdaptedExactTransitionDynamicsPolicy<OuterParameters, BaseDynamics, ParameterAdapter>` | Reuses one exact-transition dynamics under a fitted model's outer parameter row; only the compile-time parameter projection is model-specific. |

<a id="path-simulation"></a>
### [`path_simulation.cuh`](simulation/path_simulation.cuh)

| Function | Definition |
|---|---|
| `simulate_fixed_step_terminal(...)` | Advances a numerical scheme through homogeneous steps. |
| `simulate_exact_transition_terminal(...)` | Applies one direct transition over the horizon. |
| `simulate_fixed_step_dense_schedule(...)` | Observes the initial state and every numerical transition in one loop. |
| `simulate_fixed_step_regular_schedule(...)` / `simulate_exact_transition_regular_schedule(...)` | Applies one homogeneous interval between observations. |
| `simulate_fixed_step_stubbed_regular_schedule(...)` / `simulate_exact_transition_stubbed_regular_schedule(...)` | Adds a distinct first interval. |
| `simulate_fixed_step_calendar(...)` / `simulate_exact_transition_calendar(...)` | Reads one step count or prepared transition per irregular interval. |

Every function constructs one continuous path-local random context from
`(key, path)`.

<a id="simulation-schedules"></a>
### [`schedule.cuh`](simulation/schedule.cuh)

| Type or function | Definition |
|---|---|
| `RegularCalendar` / `MaturityCalendar` | Contractual dates expressed in days. |
| `FixedStepTimeConfiguration` | Elementary `dt` and numerical steps per contractual day. |
| `ExactTransitionTimeConfiguration` | Year fraction represented by one contractual day. |
| `validate_time_configuration(...)` | Validates the selected numerical time representation. |
| `day_count_year_fraction(...)` | Converts a contractual day count with the arithmetic of the simulation family. |
| `FixedStepTerminalSchedule` / `ExactTransitionTerminalSchedule` | Terminal-only schedule. |
| `FixedStepRegularSchedule` / `ExactTransitionRegularSchedule` | Homogeneous observation schedule. |
| `FixedStepDenseSchedule` | Observes every numerical transition. |
| `FixedStepCalendarSchedule<N>` / `ExactTransitionCalendarSchedule<N>` | Static irregular calendar of `N` intervals. |

<a id="simulation-barrier-handlers"></a>
### [`barrier_handlers.cuh`](simulation/barrier_handlers.cuh)

| Type | Definition |
|---|---|
| `KnockOutBarrierObservationHandler` | Stops when a scalar observable first breaches its oriented barrier. |
| `KnockInBarrierObservationHandler` | Records activation and continues to maturity. |
| `DoubleKnockOutObservationHandler` | Stops when a scalar observable leaves an open interval. |

<a id="monte-carlo"></a>
## [`monte_carlo/`](monte_carlo)

[`concepts.cuh`](#monte-carlo-concepts) ·
[`monte_carlo_kernel.cuh`](#monte-carlo-kernel)

<a id="monte-carlo-concepts"></a>
### [`concepts.cuh`](monte_carlo/concepts.cuh)

| Concept | Contract |
|---|---|
| `ScalarMonteCarloPricingPolicy` | Binds `DeviceInputs`, a schedule and a product to input-driven `PreparedRow` construction and `evaluate_path(row, key, path)`. |

`PreparedRow` remains trivially copyable. The kernel stores one row per block
in shared memory and enforces
`kMaximumSharedPreparedRowBytes = 2048`; a larger dynamic calendar must use a
compact schedule view over a device-resident pool.
The same placement rule and measured budget apply to Longstaff-Schwartz rows
and prepared sample inputs; their `static_assert` diagnostics name the compact
view or device-pool alternative.

<a id="monte-carlo-kernel"></a>
### [`monte_carlo_kernel.cuh`](monte_carlo/monte_carlo_kernel.cuh)

Let $`Y_1,\ldots,Y_M`$ be the discounted payoffs returned by a pricing policy.
The kernel reports

```math
\widehat V=\frac{1}{M}\sum_{j=1}^{M}Y_j,
\qquad
\widehat{\mathrm{se}}
=\sqrt{\frac{1}{M(M-1)}
\left(\sum_{j=1}^{M}Y_j^2-
\frac{(\sum_{j=1}^{M}Y_j)^2}{M}\right)}.
```

| Function | Definition |
|---|---|
| `monte_carlo_price_kernel<Pricing>(...)` | Uses persistent blocks over prices, one shared prepared row, one shared Philox key and an FP64 moment reduction. |
| `validate_monte_carlo_launch<Pricing>(...)` | Delegates input validation to `DeviceInputs`, then validates batch, numerical configuration, geometry and seed range. |
| `launch_monte_carlo_cuda<Pricing>(...)` | Checks occupancy, emits optional diagnostics, launches the specialized kernel and checks the launch status. |

<a id="payoff"></a>
## [`payoff/`](payoff)

[`vanilla_option.cuh`](#vanilla-option-payoff) ·
[`barrier.cuh`](#barrier-payoff)

<a id="vanilla-option-payoff"></a>
### [`vanilla_option.cuh`](payoff/vanilla_option.cuh)

For $`s=1`$ for a call and $`s=-1`$ for a put,

```math
\Pi_s(S,K)=\max\!\left(s(S-K),0\right).
```

| Function | Definition |
|---|---|
| `vanilla_option_payoff<Side>(underlying, strike)` | Evaluates the payoff with compile-time side dispatch. |

<a id="barrier-payoff"></a>
### [`barrier.cuh`](payoff/barrier.cuh)

| Type or function | Definition |
|---|---|
| `BarrierDirection` | Compile-time down/up orientation. |
| `barrier_breached<Direction>(value, barrier)` | Evaluates the inclusive oriented barrier condition. |

<a id="equity"></a>
## [`equity/`](equity)

[`concepts.cuh`](#equity-concepts) ·
[`observables.cuh`](#equity-observables) ·
[`handlers.cuh`](#equity-handlers) ·
[`discount.cuh`](#equity-discount) ·
[`barrier_pricing_policy.cuh`](#equity-barrier-pricing)

<a id="equity-concepts"></a>
### [`concepts.cuh`](equity/concepts.cuh)

| Concept | Contract |
|---|---|
| `SpotDynamicsPolicy` | Adds `spot(state)` to the market-neutral dynamics contract. |
| `LogSpotDynamicsPolicy` | Adds `log_spot(state)` and the `kNativeLogSpot` capability flag. |

<a id="equity-observables"></a>
### [`observables.cuh`](equity/observables.cuh)

| Type | Definition |
|---|---|
| `SpotObservable<Dynamics>` | Adapts `Dynamics::spot(state)` to the market-neutral scalar-observable interface. |

<a id="equity-handlers"></a>
### [`handlers.cuh`](equity/handlers.cuh)

| Type | Definition |
|---|---|
| `ArithmeticMeanObservationHandler` | Accumulates observed spots in FP64. |
| `GeometricMeanObservationHandler` | Accumulates log-spots in FP64. |
| `MaximumObservationHandler` | Retains the maximum observed spot. |
| `SpotObservationWriter` | Writes observed spots to a strided view. |
| `SpotAndStateObservationWriter` | Writes spots and one selected state member. |

<a id="equity-discount"></a>
### [`discount.cuh`](equity/discount.cuh)

```math
D(0,T)=\exp(-rT).
```

| Type or function | Definition |
|---|---|
| `ConstantRateParameters` | Requires a scalar risk-free-rate field. |
| `constant_rate_discount_factor(parameters, time)` | Evaluates the constant-rate discount factor. |

<a id="equity-barrier-pricing"></a>
### [`barrier_pricing_policy.cuh`](equity/barrier_pricing_policy.cuh)

| Type | Definition |
|---|---|
| `SingleBarrierOptionPricingPolicy` | Composes a dense schedule with an equity spot observable and vanilla payoff. |
| `UpTouchPricingPolicy` | Composes the same layers for one-touch and no-touch cash payoffs. |

<a id="fixed-income"></a>
## [`fixed_income/`](fixed_income)

[`one_factor_affine.cuh`](#one-factor-affine) ·
[`analytics_concepts.cuh`](#fixed-income-analytics-concepts) ·
[`cashflows.cuh`](#fixed-income-cashflows) ·
[`gaussian_bond_option.cuh`](#gaussian-bond-option) ·
[`swaption_side.cuh`](#swaption-side) ·
[`jamshidian.cuh`](#jamshidian) ·
[`jamshidian_cooperative.cuh`](#cooperative-jamshidian)

Product assembly is owned by the
[rate-option](../product/rate_option/pricing_policy.cuh),
[zero-coupon-bond-option](../product/zero_coupon_bond_option/pricing_policy.cuh),
and [European-swaption](../product/european_swaption/pricing_policy.cuh)
directories.

<a id="fixed-income-analytics-concepts"></a>
### [`analytics_concepts.cuh`](fixed_income/analytics_concepts.cuh)

| Concept | Required capability |
|---|---|
| `ZeroCouponBondProvider` | One `zero_coupon_bond(parameters, state, t, T)` evaluator. |
| `OneFactorAffineBondProvider` | A zero-coupon provider plus grouped `log_A` and `B` coefficients. |
| `BondOptionProvider` | A reusable expiry context and one signed bond-option evaluator. |
| `JamshidianAnalyticsProvider` | The one-factor affine and bond-option capabilities together. |
| `ParametricCurveProvider` | Curve log discount and instantaneous forward. |

The contracts are capability-based: G2 and G2++ are not required to expose
Jamshidian operations.

<a id="one-factor-affine"></a>
### [`one_factor_affine.cuh`](fixed_income/one_factor_affine.cuh)

For model-specific coefficients $`A(t,T)`$ and $`B(t,T)`$,

```math
\log P(t,T)=\log A(t,T)-B(t,T)x_t.
```

| Function or type | Definition |
|---|---|
| `OneFactorAffineBondCoefficients` | Carries `log_A` and `B` without an unnecessary `exp`/`log` round trip. |
| `log_zero_coupon_bond(...)` | Evaluates the log-affine bond formula supplied by a model provider. |
| `zero_coupon_bond(...)` | Returns the exponential of the common log-affine formula. |

<a id="fixed-income-cashflows"></a>
### [`cashflows.cuh`](fixed_income/cashflows.cuh)

For payment dates $`T_1,\ldots,T_n`$ and contractual accrual fractions
$`\delta_1,\ldots,\delta_n`$,

```math
L(t;T_0,T_1)
=\frac{P(t,T_0)/P(t,T_1)-1}{\delta_1},
```

```math
A_{\mathrm{swap}}(t)=\sum_{i=1}^{n}\delta_iP(t,T_i),
\qquad
S(t;T_0,T_n)
=\frac{P(t,T_0)-P(t,T_n)}{A_{\mathrm{swap}}(t)},
```

```math
\frac{V_{\mathrm{payer}}(t)}{N}
=P(t,T_0)-P(t,T_n)-K A_{\mathrm{swap}}(t).
```

| Function or type | Definition |
|---|---|
| `FixedLegScheduleView` | Adapts already-converted payment times and accrual fractions to the common schedule interface. |
| `forward_rate(...)` | Evaluates the simple forward rate from two zero-coupon bonds. |
| `fixed_leg_terms(...)` | Computes the annuity and final bond in one schedule pass. |
| `swap_rate(...)` | Evaluates the par swap rate. |
| `payer_swap_value(...)` | Evaluates the unit-notional receive-floating/pay-fixed swap value. |

<a id="gaussian-bond-option"></a>
### [`gaussian_bond_option.cuh`](fixed_income/gaussian_bond_option.cuh)

Let $`s=1`$ for a call, $`s=-1`$ for a put, $`P_e=P(t,T_e)`$,
$`P_i=P(t,T_i)`$, strike $`K_B`$, and total bond volatility $`\Sigma`$.

```math
d_1=\frac{\log(P_i/(K_BP_e))}{\Sigma}+\frac{\Sigma}{2},
\qquad d_2=d_1-\Sigma,
```

```math
V_B=s\left[P_i\Phi(sd_1)-K_BP_e\Phi(sd_2)\right].
```

| Function or type | Definition |
|---|---|
| `GaussianBondOptionDiscountContext` | Stores the expiry discount factor and its logarithm once per option strip. |
| `discounted_lognormal_bond_option_price(...)` | Adapts bond levels to the common discounted-lognormal primitive. |

The expression follows the forward option formula of
[Black (1976)](https://doi.org/10.1016/0304-405X%2876%2990024-6).

<a id="swaption-side"></a>
### [`swaption_side.cuh`](fixed_income/swaption_side.cuh)

| Symbol | Definition |
|---|---|
| `SwaptionSide::payer` | Right to receive floating and pay fixed. |
| `SwaptionSide::receiver` | Right to receive fixed and pay floating. |
| `swaption_side_name(side)` | Returns the stable host label `"payer"` or `"receiver"`. |

<a id="jamshidian"></a>
### [`jamshidian.cuh`](fixed_income/jamshidian.cuh)

With

```math
c_i=K\delta_i+\mathbf 1_{\{i=n\}},
```

the unique one-factor boundary $`x^\star`$ solves

```math
\sum_{i=1}^{n}c_iP(T_e,T_i;x^\star)=1,
\qquad
K_i^\star=P(T_e,T_i;x^\star).
```

The payer and receiver prices per unit notional are

```math
V_{\mathrm{payer}}(t)
=\sum_{i=1}^{n}c_i\,p_B(t;T_e,T_i,K_i^\star),
```

```math
V_{\mathrm{receiver}}(t)
=\sum_{i=1}^{n}c_i\,c_B(t;T_e,T_i,K_i^\star).
```

| Function | Definition |
|---|---|
| `jamshidian_cashflow_coefficient(...)` | Returns $`c_i=K\delta_i+\mathbf 1_{\{i=n\}}`$. |
| `jamshidian_state_boundary(...)` | Solves the monotone scalar equation with safeguarded Newton steps and returns NaN when the final residual is not certified. |
| `jamshidian_bond_strike(...)` | Evaluates one $`K_i^\star`$. |
| `european_swaption_price<Side>(...)` | Reuses one expiry context and sums bond puts or calls. |

Reference: [Jamshidian (1989)](https://doi.org/10.1111/j.1540-6261.1989.tb02413.x).

<a id="cooperative-jamshidian"></a>
### [`jamshidian_cooperative.cuh`](fixed_income/jamshidian_cooperative.cuh)

| Function or type | Definition |
|---|---|
| `CooperativeJamshidianWorkspace` | Views the shared `log_A`, `B` and bond-option arrays. |
| `cooperative_jamshidian_shared_memory_bytes(...)` | Returns the dynamic shared-memory requirement for a runtime payment capacity. |
| `jamshidian_state_boundary_from_coefficients(...)` | Applies safeguarded Newton iterations to the coefficients prepared by the block. |
| `cooperative_european_swaption_price<Side>(...)` | Distributes bond coefficients and bond options across the block, then accumulates cashflows in contractual order. |

### [`bermudan_swaption_continuation_state.cuh`](fixed_income/bermudan_swaption_continuation_state.cuh)

| Function or type | Definition |
|---|---|
| `OneFactorRateContinuationState<...>` | Stores one stochastic factor and the integrated short rate in SoA form, then applies the model's regression normalization. |
| `TwoFactorRateContinuationState<...>` | Stores two stochastic factors and the integrated short rate in SoA form, then builds the two-factor regression input. |

<a id="result-index"></a>
## [`result_index.cuh`](result_index.cuh)

Let `i` be a flattened result index and `P` the number of products.

| Function | Definition |
|---|---|
| `decode_model_product_result_index(i, P, construction)` | Returns `ModelProductIndices {model_index, product_index}` from the canonical `PriceConstruction` enum. |
| `decode_model_curve_product_result_index(i, C, P, construction)` | Returns `ModelCurveProductIndices {model_index, curve_index, product_index}`. |
| `decode_model_product_result_index_32(...)` | Uses the same mapping with validated `uint32_t` operands inside device kernels. |
| `decode_model_curve_product_result_index_32(...)` | Uses the three-input mapping with validated `uint32_t` operands inside device kernels. |

Aligned construction uses `(i,i)`. Cartesian construction lets products vary
fastest:

```math
\mathrm{model\_index}=\left\lfloor\frac{i}{P}\right\rfloor,
\qquad
\mathrm{product\_index}=i\bmod P.
```

For a model/curve/product Cartesian construction with `C` curves,

```math
\mathrm{model\_index}=\left\lfloor\frac{i}{CP}\right\rfloor,
\qquad
\mathrm{curve\_index}=\left\lfloor\frac{i\bmod(CP)}{P}\right\rfloor,
\qquad
\mathrm{product\_index}=i\bmod P.
```

Public cardinalities and addresses remain `std::size_t`. CUDA launch validation
rejects a logical result set above `uint32_max` and asks the caller to split
it before the device-only decoders are used. On SM89, this specialization cut
the indexing microbenchmark median from 0.270336 ms to 0.123904 ms while
preserving every decoded identifier; compile-time specialization of the enum
did not improve the `size_t` variant.

<a id="sample"></a>
## [`sample.cuh`](sample.cuh)

`sample.cuh` is the façade for the policy-based model-only sampling engine.
Parameter generation, calendar generation, observation and CUDA execution are
orthogonal compile-time policies. `SamplingSeeds` gives parameter, schedule
and dynamics streams independent public seeds; the parameter and schedule
sources additionally use separate Philox domains.

### Policies and sources

| Type | Definition |
|---|---|
| `UniformBounds` | Closed interval `[minimum, maximum]`. |
| `ModelSamplingPolicy<Schedule, Observation>` | Complete compile-time definition of one model-only sample. |
| `ExternallyPreparedSamplingPolicy<Policy>` | Markov sample whose numerical dynamics row is prepared on the host. |
| `VolterraFftModelSamplingPolicy<Driver, Path, Schedule, Observation>` | Gaussian-Volterra sample composition shared by rough Bergomi, log-modulated rough Bergomi, rough SABR and rough Stein--Stein. |
| `DeviceParameterSource<Parameters>` | Loads caller-materialized parameter rows. |
| `GeneratedParameterSource<Sampler>` | Samples a model policy deterministically from `(parameter_seed, parameter_index)`. |
| `ConstantCalendarSource<Calendar>` | Reuses a validated integer-day calendar. |
| `DeviceCalendarSource<Calendar>` | Loads one materialized calendar per logical sample. |
| `UniformMaturityCalendarSource` | Draws an unbiased integer maturity and optionally writes it to device output. |
| `RandomIncreasingCalendarSource<Calendar, N>` | Draws a feasible increasing calendar and optionally writes observation-major days. |
| `SpotSampleObservation<Dynamics>` | Writes the market spot observable. State and two-component observations are also provided. |

Calendar sources are day-native. Let `d_min` and `d_max` be the integer
business-day bounds. `UniformMaturityCalendarSource` draws

```math
d\sim\mathcal{U}\{d_{\min},\ldots,d_{\max}\},
\qquad
T=\frac{d}{252}.
```

For a calendar of `N` observations, `RandomIncreasingCalendarSource` draws every
integer day uniformly from its currently feasible interval. Let `g` be the
minimum gap, let `j` run from zero to `N-1`, and define

```math
\ell_j=
\begin{cases}
d_{\min}, & j=0,\\
d_{j-1}+g, & j>0,
\end{cases}
\qquad
u_j=d_{\max}-(N-j-1)g.
```

The next observation day is

```math
d_j\sim\mathcal{U}\{\ell_j,\ldots,u_j\}.
```

### Execution

| Function or type | Definition |
|---|---|
| `launch_samples_cuda<Policy>(...)` | Validates the complete composition and launches the selected generic kernel. |
| `launch_prepared_samples_cuda<Policy>(...)` | Uses the same thread/block strategies with caller-provided `PreparedInput` rows. |
| `SampleExecutionStrategy::thread_grid_stride` | One physical thread processes one or more flattened samples. |
| `SampleExecutionStrategy::parameter_block` | One block prepares invariant model coefficients once and its lanes process the conditional paths. |
| `SampleExecutionStrategy::automatic` | Chooses grid-stride for `P = 1`, parameter-block otherwise. |
| `volterra_fft::launch_samples_cuda<Policy>(...)` | Runs the block-cooperative cuFFTDx sampler with one reusable spectrum per parameter row. |
| `decode_sample(i, P)` | Returns `{i / P, i % P}`. |
| `total_sample_count(range)` | Returns `M P` after zero and overflow checks. |
| `validate_sample_launch(...)` | Validates sources, calendars, output views, launch slice, geometry, time convention and seed ranges. |

The canonical exact time configuration is `day_fraction = 1/252`. The
canonical discretized configuration is `dt = 1/504` with two transitions per
business day. Calendar outputs are written directly in time-major SoA order.
`sample.cu` files below models are intentionally thin bindings. Markov models
forward to the persistent thread/block engine. Rough Heston supplies its
host-prepared Markovian lift to that same engine. Rough Bergomi and rough SABR
forward to the separate FFT block engine while reusing the same parameter
sources, calendar sources, observations, flattened indices and seed domains.

<a id="philox"></a>
## [`philox.cuh`](philox.cuh)

The counter-based generator is Philox-4x32-10 from
[Salmon et al. (2011)](https://doi.org/10.1145/2063384.2063405). A 64-bit seed
defines the two-word key; a path index `p` and local group index `g` define the
four-word counter:

```math
K=(\mathrm{low}_{32}(\mathrm{seed}),\mathrm{high}_{32}(\mathrm{seed})),
```

```math
C=(\mathrm{low}_{32}(p),\mathrm{high}_{32}(p),
   \mathrm{low}_{32}(g),\mathrm{high}_{32}(g)).
```

### Counter and streams

| Function or type | Definition |
|---|---|
| `PhiloxCounter` | Four 32-bit counter/output words. |
| `PhiloxKey` | Two 32-bit key words. |
| `RandomQuad` | Four FP32 uniforms. |
| `NormalPair` | Two independent FP32 standard normals. |
| `make_key(seed)` | Splits the 64-bit seed into a `PhiloxKey`. |
| `philox4x32_10(key, counter)` | Applies ten Philox multiplication, permutation and key-bump rounds. |
| `random_bits(key, p, g)` | Evaluates Philox at the counter identified by path `p` and group `g`. |
| `uint32_to_uniform(x)` | Maps one 32-bit word to the midpoint `(x+1/2)2^{-32}` and clamps below one. |
| `uniform_quad(key, p, g)` | Converts the four Philox output words into four FP32 uniforms in `(0,1)`. |
| `UniformSequence(key, p)` | Initializes a scalar-uniform stream at group zero for path `p`. |
| `UniformSequence::next()` | Exposes the cached uniform groups as one ordered scalar stream. |
| `Uint32Sequence(key, p)` | Initializes a raw-integer stream at group zero for path `p`. |
| `Uint32Sequence::next()` | Exposes the cached Philox groups as one ordered raw-integer stream. |
| `bounded_uint32(integers, b)` | Draws exactly uniformly from `{0,...,b-1}` by rejecting the prefix of size `2^32 mod b`. |

### Distribution transforms

For independent uniforms `U_1,U_2` in `(0,1)`, `box_muller` returns two
independent standard normals:

```math
R=\sqrt{-2\log U_2},
\qquad
\Theta=2\pi U_1,
\qquad
(Z_1,Z_2)=R(\cos\Theta,\sin\Theta).
```

For a Gamma shape `alpha >= 1` and a standard normal variate `Z`, the
Marsaglia–Tsang core sets

```math
d=\alpha-\frac{1}{3},
\qquad
c=\frac{1}{\sqrt{9d}},
\qquad
V=(1+cZ)^3,
```

then accepts `d V` with the method's squeeze or logarithmic test. Let
`G_alpha` denote a unit-scale Gamma variate with shape `alpha`, and let `U` be
an independent uniform variate. For `0 < alpha < 1`, the method uses

```math
G_{\alpha}=G_{\alpha+1}U^{1/\alpha}.
```

Let `X` follow a non-central chi-square law with `nu` degrees of freedom and
noncentrality `lambda`, and let `s > 0` be the requested scale. With
`Gamma(alpha, theta)` denoting shape `alpha` and scale `theta`, the exact
Poisson–Gamma representation is

```math
N\sim\mathrm{Poisson}\!\left(\frac{\lambda}{2}\right),
\qquad
sX\mid N\sim\mathrm{Gamma}\!\left(\frac{\nu}{2}+N,2s\right),
```

For inverse-Gaussian mean `mu > 0` and shape `lambda > 0`, let `Z` be standard
normal and `U` uniform on `(0,1)`. The Michael–Schucany–Haas construction sets

```math
w=\frac{\mu Z^2}{2\lambda},
\qquad
r=1+w+\sqrt{w(2+w)},
```

then returns

```math
X=
\begin{cases}
\mu/r, & U\le r/(r+1),\\
\mu r, & U>r/(r+1).
\end{cases}
```

| Function or type | Definition |
|---|---|
| `poisson_from_uniform(U, lambda, p_0)` | Inverts the Poisson CDF from one uniform, with `p_0 = exp(-lambda)` supplied by the caller. |
| `poisson_from_uniform_sequence(uniforms, lambda)` | Uses inversion for `lambda < 10` and Hörmann PTRS otherwise. |
| `box_muller(U_1, U_2)` | Returns the pair `(Z_1,Z_2)` above. |
| `NormalPairCache` | Stores the unused second Box–Muller normal. |
| `next_normal(uniforms, cache)` | Returns the cached normal or consumes two uniforms to refill the cache. |
| `detail::marsaglia_tsang_gamma_shape_at_least_one(...)` | Draws a unit-scale Gamma variate for `alpha >= 1`. |
| `marsaglia_tsang_gamma(..., alpha, theta)` | Draws `Gamma(alpha, theta)` for every positive shape `alpha` and scale `theta`. |
| `scaled_noncentral_chi_square(..., nu, lambda, s)` | Draws the scaled non-central chi-square variate `sX` through the Poisson–Gamma mixture. |
| `michael_schucany_haas_inverse_gaussian(..., mu, lambda)` | Draws `IG(mu, lambda)` with the exact normal/uniform construction. |

References: [Box and Muller (1958)](https://doi.org/10.1214/aoms/1177706645),
[Hörmann (1993)](https://doi.org/10.1016/0167-6687%2893%2990997-4),
[Marsaglia and Tsang (2000)](https://doi.org/10.1145/358407.358414), and
[Michael, Schucany and Haas (1976)](https://doi.org/10.1080/00031305.1976.10479147).

<a id="normal-distribution"></a>
## [`normal_distribution.cuh`](normal_distribution.cuh)

Let `Z` be a standard normal random variable. `normal_cdf(z)` evaluates its
CDF directly in FP32.

| Function | Definition |
|---|---|
| `normal_cdf(z)` | Returns the standard-normal CDF at `z`. |

The implementation uses

```math
\Phi(z)=\mathbb{P}[Z\le z]
=\frac{1}{2}\,\mathrm{erfc}\!\left(-\frac{z}{\sqrt{2}}\right).
```

<a id="lognormal-option"></a>
## [`lognormal_option.cuh`](lognormal_option.cuh)

For discounted underlying $`U`$, discounted strike $`K_d`$, total volatility
$`\Sigma`$ and sign $`s\in\{-1,1\}`$,

```math
d_1=\frac{\log(U/K_d)}{\Sigma}+\frac{\Sigma}{2},
\qquad d_2=d_1-\Sigma,
```

```math
V=s\left[U\Phi(sd_1)-K_d\Phi(sd_2)\right].
```

| Function or type | Definition |
|---|---|
| `DiscountedLognormalOptionContext` | Stores the log underlying level and the strike discount in log and level form. |
| `LognormalOptionDValues` | Carries `d1` and `d2`. |
| `DiscountedLognormalOptionValues` | Carries discounted levels and their two normal arguments. |
| `lognormal_option_d_values(...)` | Computes `d1` and `d2` for positive total volatility. |
| `prepare_discounted_lognormal_option_values(...)` | Prepares levels and normal arguments, including the deterministic limit. |
| `discounted_lognormal_option_price(...)` | Prices either sign from a context or prepared values. |

<a id="noncentral-chi-square"></a>
## [`noncentral_chi_square.cuh`](noncentral_chi_square.cuh)

`DistributionProbabilities {cdf, survival}` carries both tails so callers do
not reconstruct a small probability by FP32 subtraction.

### Regularized Gamma law

For a shape `a > 0` and an argument `x >= 0`, with `Gamma(a)` denoting the
Gamma function, define

```math
P(a,x)=\frac{1}{\Gamma(a)}\int_0^x t^{a-1}e^{-t}\,\mathrm dt,
\qquad
Q(a,x)=\frac{1}{\Gamma(a)}\int_x^{\infty}t^{a-1}e^{-t}\,\mathrm dt.
```

| Function or type | Definition |
|---|---|
| `clamp_probability(x)` | Clamps a final probability to `[0,1]`. |
| `CompensatedSum(initial)` | Initializes a compensated FP32 sum. |
| `CompensatedSum::add(x)` | Adds `x` with Kahan compensation. |
| `log_one_plus_minus_argument(x)` | Evaluates `log(1+x)-x` by an 18-term local series when `abs(x) <= 1/4`. |
| `stirling_correction(a)` | Evaluates the Stirling remainder used for `log Gamma(a)` when `a >= 8`. |
| `gamma_log_scale(a, x)` | Evaluates `-x + a log(x) - log Gamma(a)` without subtracting large nearby terms. |
| `regularized_gamma_series(a, x)` | Evaluates `P(a,x)` directly on the left of the Gamma transition region. |
| `regularized_gamma_continued_fraction(a, x)` | Evaluates `Q(a,x)` directly on the right with the modified-Lentz fraction. |
| `regularized_gamma_probability_pair(a, x)` | Selects the numerically stable direct tail and returns both `(P,Q)`. |
| `regularized_gamma_probabilities(a, x)` | Public device interface returning `(P(a,x),Q(a,x))`. |

The common scale factor is

```math
e^{-x+a\log x-\log\Gamma(a)}.
```

For `a >= 8`, write `r=(x-a)/a` and let `C(a)` be
`stirling_correction(a)`. The same logarithm is evaluated as

```math
a\,[\log(1+r)-r]
+\frac{1}{2}\log\!\left(\frac{a}{2\pi}\right)-C(a),
```

### Non-central chi-square law

Let `X` follow a non-central chi-square law with degrees of freedom `nu > 0`
and noncentrality `lambda >= 0`. Its CDF at `x >= 0` is the Poisson mixture

```math
\mathbb{P}[X\le x]
=\sum_{k=0}^{\infty}
e^{-\lambda/2}\frac{(\lambda/2)^k}{k!}
P\!\left(\frac{\nu}{2}+k,\frac{x}{2}\right).
```

| Function | Definition |
|---|---|
| `poisson_mode_log_weight(lambda_over_two, mode)` | Evaluates the logarithm of the modal Poisson weight without large-term cancellation. |
| `poisson_gamma_mixture(nu, lambda, x)` | Sums both tails outward from the modal Poisson term with compensated FP32 sums. |
| `log_one_plus_minus_ratio(delta)` | Evaluates the stable deviance term `log(1+delta)-delta/(1+delta)`. |
| `saddlepoint_probabilities(nu, lambda, x)` | Applies the Lugannani–Rice saddlepoint approximation for large `lambda`. |
| `noncentral_chi_square_probabilities(nu, lambda, x)` | Uses the exact mixture for `lambda <= 1024` and the saddlepoint approximation above it. |

All device arithmetic in this file is FP32. The exact series implementation
follows the recurrence strategy of [Ding (1992)](https://doi.org/10.2307/2347584);
the large-noncentrality branch uses
[Lugannani and Rice (1980)](https://doi.org/10.2307/1426607), and the continued
fraction uses [Lentz (1976)](https://doi.org/10.1364/AO.15.000668).
The large gamma, Poisson-mixture and saddlepoint helpers are deliberately
`__noinline__`: on the SM89 CIR swaption workload this reduces the archive
from 3.40 MB to 1.64 MB, lowers the representative kernel from 64 to 56
registers and, for 16,384 results, improves its median from 1.158 ms to
1.063 ms. Both kernel coefficients of variation stay below 1.3%. The
force-inline variant remains a performance-only experiment target.

<a id="reductions"></a>
## [`reductions.cuh`](reductions.cuh)

Let `Y_1,...,Y_M` be `M >= 2` discounted Monte Carlo payoffs. `MomentSums`
stores the FP64 moments

```math
s_1=\sum_{i=1}^{M}Y_i,
\qquad
s_2=\sum_{i=1}^{M}Y_i^2.
```

| Function | Definition |
|---|---|
| `reduce_block(sum, sumsq)` | Deterministically reduces one pair `(s_1,s_2)` across a whole CUDA block. |
| `reduce_block_values<N>(values)` | Deterministically reduces `N` FP64 values into separate shared-memory totals. |
| `compute_statistics(total, M, price, standard_error)` | Validates the FP64 moments and sample count, then computes the mean and standard error with a bounded cancellation clamp. |

```math
\widehat V=\frac{s_1}{M},
\qquad
\widehat{\mathrm{Var}}(Y)
=\frac{s_2-M\widehat V^2}{M-1},
\qquad
\mathrm{SE}(\widehat V)
=\sqrt{\frac{\max(\widehat{\mathrm{Var}}(Y),0)}{M}}.
```

State evolution remains FP32; FP64 is used here for long reductions and final
statistics. A mixed-scale non-negative stress benchmark rejects lowering this
contract: compensated FP32 is faster but reaches `1.47e-2` absolute error,
while 32-value FP32 chunks reach `1.43e-1`; the FP64 reference remains exact
for the represented FP32 inputs.

<a id="longstaff-schwartz"></a>
## [`longstaff_schwartz/`](longstaff_schwartz)

[`basis/feature_vector.cuh`](longstaff_schwartz/basis/feature_vector.cuh) ·
[`basis/laguerre.cuh`](longstaff_schwartz/basis/laguerre.cuh) ·
[`basis/hermite.cuh`](longstaff_schwartz/basis/hermite.cuh) ·
[`basis/hinge.cuh`](longstaff_schwartz/basis/hinge.cuh) ·
[`concepts.cuh`](longstaff_schwartz/concepts.cuh) ·
[`exercise_decision.cuh`](longstaff_schwartz/exercise_decision.cuh) ·
[`execution_plan.cuh`](longstaff_schwartz/execution_plan.cuh) ·
[`launch.cuh`](longstaff_schwartz/launch.cuh) /
[`launch.cu`](longstaff_schwartz/launch.cu) ·
[`linear_solver.cuh`](longstaff_schwartz/linear_solver.cuh) ·
[`longstaff_schwartz_kernels.cuh`](longstaff_schwartz/longstaff_schwartz_kernels.cuh) ·
[`regression_status.cuh`](longstaff_schwartz/regression_status.cuh) ·
[`small_linear_regressor.cuh`](longstaff_schwartz/small_linear_regressor.cuh) ·
[`workspace.cuh`](longstaff_schwartz/workspace.cuh) /
[`workspace.cu`](longstaff_schwartz/workspace.cu)

For features `phi(X_i)` and discounted continuation targets `Y_i`, the small
linear regressor forms

```math
G=\sum_i\phi(X_i)\phi(X_i)^{\mathsf T},
\qquad
h=\sum_i\phi(X_i)Y_i,
```

then solves

```math
\left(G+10^{-10}\frac{\mathrm{tr}(G)}{p}I\right)\beta=h
```

in FP64 by Cholesky. Paths, continuation states, features and cashflows remain
FP32, while prediction and the immediate-exercise comparison stay FP64 until
the cashflow is selected. The compile-time basis may contain at most eight
features.

| File or type | Definition |
|---|---|
| `LaguerrePolynomialTwoFactorBasis` | Six-feature map used by all current American-option datasets. |
| `OneFactorLaguerreBasis<Degree>` | Compile-time Laguerre polynomial family. |
| `OneFactorHermiteBasis<Degree>` | Compile-time probabilists' Hermite family. |
| `CenteredHingeBasis<KnotSet>` | Intercept, linear term and compile-time ReLU hinges. |
| `NormalEquationRegressor<Basis>` | FP64 accumulation, deterministic reductions, ridge, solve and prediction for one small basis. |
| `RegressionStatus` | Distinguishes empty/underdetermined samples from non-finite statistics, Cholesky failure and non-finite coefficients. |
| `RegressionDiagnostics` | Counts every typed outcome for one result row across the backward dates. |
| `validate_regression_diagnostics(...)` | Rejects a host publication on a fatal regression cause and reports the first affected row. |
| `select_exercise_cashflow(...)` | Promotes the immediate FP32 payoff and compares it directly with the FP64 continuation estimate before selecting the FP32 cashflow. |
| `EarlyExerciseSchedulePolicy` | Requires a prepared exercise count and path simulation on that schedule. |
| `EarlyExercisePricingPolicy` | Requires row planning/preparation, continuation-state access, payoff and discount hooks. |
| `LongstaffSchwartzPolicy<Pricing, Regressor>` | Verifies the complete pricing/regression composition. |
| `make_execution_plan<Pricing, Regressor>` | Derives row storage and memory-aware batches from host product inputs. |
| `launch_longstaff_schwartz_cuda<Pricing, Regressor>` | Runs the seven shared multi-block kernels and returns timing, workspace and typed regression diagnostics. |
| `WorkspaceLayout` | Aligned views over one persistent device buffer containing SoA states, FP32 cashflows and FP64 regression data. |
| `plan_batches(...)` | Greedily forms consecutive batches under the available workspace budget. |
| `LaunchResources` | Owns the single device workspace and CUDA timing events. |

Maturity-aligned fixed-step and exact-transition exercise schedules live in
[`simulation/early_exercise_schedule.cuh`](simulation/early_exercise_schedule.cuh).
See
[Longstaff and Schwartz (2001)](https://doi.org/10.1093/rfs/14.1.113) and the
[`CUDA American and Bermudan pricing contract`](../../docs/cuda-american-and-bermudan-pricing-contract.md).

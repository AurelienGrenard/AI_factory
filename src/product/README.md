# Financial product contracts

This directory owns model-independent product parameters, schedules, path
state, and payoff policies. It never owns model dynamics or a concrete
model-product launcher.

## File roles

Each `src/product/<product>/` uses the smallest applicable subset of this
layout:

- `parameters.hpp` — compact product row transferred to CUDA;
- `dataset.hpp/.cpp` — host loader and row validation;
- `pricing_policy.cuh` — product preparation, path observation, and payoff;
- `schedule.cuh` — contractual dates when a generic schedule is insufficient;
- `continuation_state.cuh` — regression state for early exercise when needed.

The product path and file names identify the financial contract. Details that
depend on a model or curve belong under
`src/model/<asset-class>/<family>/<model>/product/[<curve>/]`.

## Multi-asset product contract

The multi-asset pricing kernels use one product policy for Markovian equities,
rough equities, and a stochastic discount rate. A new product supplies:

- `kTerminalCashflow = true` for the current cross-asset engine;
- `ProductParameters` with `maturity_days`, plus a host validator when it has
  extra calendar or domain constraints;
- `prepare(parameters, discount_factor)` and `make_handler(prepared)`;
- `is_observation_step(step, total_steps, steps_per_day, parameters)`;
- `observe<Pack, TeamSize>(states, prepared, lane)` for a Markovian basket;
- `observe_joint_group<MarkovPack, RoughPack, RoughCount, TeamSize>(...)`
  for a basket containing rough assets;
- a handler whose `on_observation(step, observation)` updates path state and
  returns whether simulation should continue, then `finalize(prepared, handler)`.

The observation type is chosen by the product policy: a scalar for these
contracts, or a compact struct when a payoff needs several basket statistics.
A terminal product observes only at maturity. A barrier product observes on its
monitoring calendar and can stop a path early. The pricing kernels do not need
a product-specific specialization. The worst-of, best-of, and basket calls share
[`terminal_basket_call_path_policy.cuh`](terminal_basket_call_path_policy.cuh);
the basket barrier uses the same arithmetic-mean observable.

Basket observations use `S_i(t) / S_i(0)`. Worst-of takes their minimum,
best-of their maximum, and basket call their equally weighted arithmetic
mean. All three start at one, so their `strike` is dimensionless. The mean
is over normalized assets, not over raw spots with different initial prices.
Price barriers use the same normalized basket units.

## Equity quote units at pricing boundaries

Equity product datasets quote strikes and price barriers as ratios of the
central model spot. The dataset pricing helpers resolve each quote once to an
absolute level before calling a model-product price kernel. Public gradient
preparation does the same before forming bumps: a spot bump keeps the original
absolute contract level fixed, while a product-quote derivative is reported per
unit of ratio. The common resolver is
[`src/common/equity/spot_relative_product.hpp`](../common/equity/spot_relative_product.hpp).

The low-level `launch_*_cuda` price and historical price-delta functions take
product levels that are **already absolute**. A direct caller comparing those
functions with gradient plans must resolve the same central contract first.
This distinction applies to terminal and path-dependent payoffs; fields that
are intrinsically ratios, such as forward-start moneyness and cliquet returns,
keep their native units.

## Find a supported composition

The authoritative model-product matrix is
[`tools/codegen/pricing_bindings/capability_manifest.py`](../../tools/codegen/pricing_bindings/capability_manifest.py).
Generated composition units use the templates documented in the
[code-generation entry point](../../tools/codegen/pricing_bindings/README.md).

Use the relevant contract for implementation details:

- [closed-form and Monte Carlo pricing](../../project_journal/reference/cuda/closed-form-and-monte-carlo-pricing-contract.md);
- [American and Bermudan pricing](../../project_journal/reference/cuda/american-and-bermudan-pricing-contract.md);
- [complete policy composition](../../project_journal/reference/cuda/pricing-policy-composition.md).

Product availability, model support, and generated target names are not copied
into this README because they are derived from the capability manifest.

For one product, `parameters.hpp` and `pricing_policy.cuh` are the executable
contract for its terms and payoff. Equation sheets in the separately
maintained website are a derived presentation and must remain consistent with
that source; the repository boundary is documented by the
[protected-download proposal](../../project_journal/reference/proposed-protected-dataset-download-design.md).

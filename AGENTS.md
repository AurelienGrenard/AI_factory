# Repository instructions

## Before changing code

- Read `README.md`, `docs/README.md`, the relevant file under `docs/`, and the
  corresponding CMake, test, generator, and validation entries.
- Read `docs/deferred-work.md` before starting a planned extension. Read
  `docs/abandoned-work.md` before reopening an optimization that may already
  have been measured and rejected. Never use the abandoned list as a backlog.
- Follow `docs/catalog-extension-and-validation-workflow.md` for every model,
  curve, product, price, validation, or website extension.
- For CUDA pricing work, read
  `docs/cuda-closed-form-and-monte-carlo-pricing-contract.md`; for early
  exercise work, also read
  `docs/cuda-american-and-bermudan-pricing-contract.md`; for launcher guards or
  resource inspection, read
  `docs/cuda-launch-validation-and-kernel-diagnostics.md`.
- For model simulation work, read `docs/cuda-model-dynamics-contract.md` before
  changing a `dynamics.cuh/.cu` interface or its Philox consumption.
- For model-parameter or product generation work, read
  `docs/model-and-product-parameter-dataset-generation.md` and preserve its
  ordered 90/10 policy and complete adjacent YAML recipe.
- Inspect the Git status first and preserve unrelated or pre-existing changes.
- Keep CUDA interfaces and implementations in focused `.cuh/.cu` pairs. Public
  headers should expose only declarations needed by callers.
- Keep the canonical function names and responsibilities documented under
  `docs/` consistent across model, curve, and product implementations. Update
  the associated document whenever the common contract changes.

## Project layout

- `src/common`: reusable CUDA/runtime, reduction, random-number, and numerical
  primitives.
- `src/model/<asset_class>/<model>`: model dynamics, analytics, datasets, and pricing kernels (`asset_class` is `equity` or `fixed_income`).
- `src/curve/<curve>`: curve datasets and term-structure analytics.
- `src/product/<product>`: product parameter rows and JSON loaders.
- `src/generative`: method-neutral generative-model source code.
- `catalog/model/<asset_class>/<model>`: model-owned `parameters`, `samples`,
  and `prices` recipes and generators.
- `catalog/curve` and `catalog/product`: curve and product recipes.
- `tests`: CPU/CUDA tests and performance benchmarks.
- `validation/model`: one unified validation pipeline per model/product pair,
  with an intermediate curve folder when the pricing contract uses one;
  `validation/premia` and `validation/quantlib`: reusable backend adapters.
- Migrated price validations persist 1,000 aligned independent prices below
  `validation/datasets/price` and publish only `status`, `verified`, and the
  cache path in catalogue YAML. Do not add adjacent validation reports or
  notebooks to a migrated dataset.
- External engines run only during explicit reference regeneration. Routine
  validation must be cache-only, verify semantic source fingerprints and row
  provenance, and import neither Premia nor QuantLib.
- Model/product validators declare the complete engine plan in priority order,
  name the exact native pricing method/function for every available slot,
  preserve all three hierarchy slots, and delegate row-wise fallback to
  `validation/hierarchy.py`; persistent-reference assembly and cache-only CLI
  behavior belong in the shared asset-class reference pipeline, not in a
  model-local `common.py`.
  Only technical backend exceptions descend through the ordered Premia
  candidates, then specialized QuantLib, then QuantLib Monte Carlo; finite
  comparison failures never fall through.
- Merton, Kou, Heston, and Bates validators reuse
  `validation/model/equity/stochastic_equity.py`. Keep model files declarative
  and product files as thin CLI wrappers; add model-specific pricing only in
  the Premia or QuantLib backend adapters.
- Audit Premia before selecting any reference engine: enumerate every method
  registered for the exact model-product pair across all Premia asset menus,
  including compatible exact decompositions. Only after this exhaustive
  inventory may the candidates be ranked by contractual compatibility,
  robustness, and measured runtime.
- Independent price certification covers the 900 core and 100 stress rows.
  Both regimes must be fully referenced and pass before publication may set
  `verified: true`.
- A technical failure of the preferred Premia method falls through to the next
  compatible Premia method for that row. QuantLib is considered only after all
  compatible Premia candidates have been exhausted. Continuous/discrete
  differences require a documented bound or bias criterion; they do not make
  Premia unavailable. A finite comparison failure is investigated and is never
  hidden by choosing whichever reference happens to be closer.
- `AI_factory_website`: website integration and equations.

## Numerical reproducibility

- For a discretized dataset, declare `time_grid.steps_per_year` globally and
  derive the fixed transition `dt = 1 / steps_per_year`. Store calendars as
  integer step counts; do not repeat the convention in every row. Exact
  transitions and contractual observation dates must not receive artificial
  sub-steps.
- Never enable CUDA fast math (`--use_fast_math`). The build intentionally
  exposes no fast-math option.
- Do not add `__launch_bounds__` without an explicitly approved,
  architecture-aware design and measurements on every targeted architecture.
- Preserve deterministic row-to-seed mappings, reduction order, launch geometry,
  and FP64 accumulation/linear algebra where currently used.
- Map Philox counters as `(path_index: uint64, local_group_index: uint64)` under
  the row key. Never restore flattened `groups_per_path` counter reservations.
- Do not add runtime branches, virtual calls, or indirect calls inside hot
  per-path kernels. Prefer compile-time specialization when it has a measured
  benefit and does not obscure the implementation.
- Select call/put payoff sides through the public launcher template
  (`launch_*_option_cuda<OptionSide::call/put>`). Keep its two explicit
  instantiations in the corresponding `.cu`; do not add a runtime side wrapper.
- Compare numerical outputs, kernel timings, registers per thread, spills, and
  shared memory on the same GPU architecture and toolchain before and after a
  performance-sensitive refactor.
- Use `AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1` to inspect the resources and
  theoretical occupancy of the exact launch geometry without changing the
  normal generator path.
- Run the relevant CUDA tests and independent Premia/QuantLib validations after
  numerical changes.

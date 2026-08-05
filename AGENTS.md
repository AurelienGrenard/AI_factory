# Repository instructions

## Before changing code

- Read `README.md`, the relevant files under `docs/` and `research_notes/`, and
  the corresponding CMake, test, generator, and validation entries.
- For CUDA pricing work, read `docs/cuda-pricing-kernel-api.md`; for early
  exercise work, also read `docs/early-exercise-pricing-api.md`; for launcher
  guards or resource inspection, read
  `docs/cuda-validation-and-diagnostics.md`.
- Inspect the Git status first and preserve unrelated or pre-existing changes.
- Keep CUDA interfaces and implementations in focused `.cuh/.cu` pairs. Public
  headers should expose only declarations needed by callers.
- Keep the canonical function names and responsibilities documented under
  `docs/` consistent across model, curve, and product implementations. Update
  the associated document whenever the common contract changes.

## Project layout

- `src/common`: reusable CUDA/runtime, reduction, random-number, and numerical
  primitives.
- `src/model/<model>`: model dynamics, analytics, datasets, and pricing kernels.
- `src/curve/<curve>`: curve datasets and term-structure analytics.
- `src/product/<product>`: product parameter rows and JSON loaders.
- `catalog`: product, model, curve, and price dataset recipes and generators.
- `tests`: CPU/CUDA tests and performance benchmarks.
- `validation`: QuantLib references and validation notebooks.
- `AI_factory_website`: website integration and equations.

## Numerical reproducibility

- Never enable CUDA fast math (`--use_fast_math`). The build intentionally
  exposes no fast-math option.
- Preserve deterministic row-to-seed mappings, reduction order, launch geometry,
  and FP64 accumulation/linear algebra where currently used.
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
- Run the relevant CUDA tests and QuantLib validations after numerical changes.

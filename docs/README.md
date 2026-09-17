# AI Factory documentation

Use this page to reach the authoritative document for a task. The
[repository README](../README.md) provides the shortest build-and-test path;
this index covers architecture, extension, datasets, validation, performance,
and project records.

## Choose a task

| Task | Start here | Continue with |
|---|---|---|
| Explore runtime ownership | [Source reference index](model-and-curve-reference-index.md) | [Shared primitives](../src/common/README.md) |
| Understand the CUDA architecture | [CUDA documentation](cuda/README.md) | [Pricing-policy composition](cuda/pricing-policy-composition.md) |
| Understand CMake and build a target | [CMake build guide](cmake-build-workflow.md) | [CMake ownership](../cmake/README.md) |
| Find local builds or historical evidence | [Local artifact layout](local-artifacts.md) | Active binaries, audit proofs and campaign records |
| Add a model, curve, product, or price | [Catalogue extension workflow](catalog-extension-and-validation-workflow.md) | Relevant [CUDA contract](cuda/README.md) |
| Generate model or product parameters | [Parameter-dataset contract](model-and-product-parameter-dataset-generation.md) | [Catalogue extension workflow](catalog-extension-and-validation-workflow.md) |
| Generate model-only training samples | [Model-sample contract](model-sample-dataset-generation.md) | [Code generation](../tools/codegen/pricing_bindings/README.md) |
| Read terminal samples for learning | [Learning guide](../learning/README.md) | [Model-sample contract](model-sample-dataset-generation.md) |
| Train and compare deep pricers | [Learning guide](../learning/README.md) | [Deep-pricing learning contract](deep-pricing-learning-contract.md) |
| Browse concrete model studies | [Experiments guide](../experiments/README.md) | Model-specific README and campaign configuration |
| Run or resume a price/sample campaign | [Dataset-generation workflow](dataset-generation-workflow.md) | [Catalogue extension workflow](catalog-extension-and-validation-workflow.md) |
| Keep a dataset after a refactor | [Provenance and reuse](dataset-provenance-contract.md) | Read-only compatibility checker and legacy-data rules |
| Explore recipes and tests | [Catalogue guide](../catalog/README.md) | [Test-suite guide](../tests/README.md) |
| Find an offline utility | [Tools directory guide](../tools/README.md) | Tool-specific README when present |
| Validate a generated price | [Independent price-validation pipeline](independent-price-validation-pipeline.md) | Separate [validation audit](validation/query.md) |
| Diagnose a CUDA launch | [Launch validation and kernel diagnostics](cuda/launch-validation-and-kernel-diagnostics.md) | [Performance protocol](performance-regression-protocol.md) |
| Qualify another GPU | [Performance protocol](performance-regression-protocol.md) | [SM89 hardware notebook](cuda/rtx4090-laptop-memory-map.ipynb) as a scoped example |
| Find model equations | [Model and curve reference index](model-and-curve-reference-index.md) | Source-local model or curve reference |
| Understand the protected-download boundary | [Protected-download proposal](proposed-protected-dataset-download-design.md) | The static website is maintained outside this repository |
| Audit the repository | [Main audit query](audit/query.md) | [Status](audit/status.md), [open findings](audit/response.md), and [closed findings](audit/closed.md) |

## Architecture contracts

- [Selected equity price gradients](cuda/equity-price-gradients-contract.md) — selectable parameters, CRN scenarios and initial European scope.
- [Equity price and spot delta](cuda/equity-price-delta-contract.md) — separate
  launchers, shared/coupled paths, bumping and bounded frozen-date LSM pilots.

- [CUDA documentation](cuda/README.md) — local map for CUDA composition,
  execution, model contracts, and performance.
- [Pricing-policy composition](cuda/pricing-policy-composition.md) — how model,
  schedule, product, pricing, sampling, and kernel policies compose.
- [Model dynamics contract](cuda/model-dynamics-contract.md) — state,
  transition, Philox, time-grid, and simulation interfaces.
- [Model analytics contract](cuda/model-analytics-contract.md) — canonical
  analytical APIs and providers.
- [Closed-form and Monte Carlo pricing contract](cuda/closed-form-and-monte-carlo-pricing-contract.md)
  — ordinary pricing policies, kernels, launchers, and numerical invariants.
- [American and Bermudan pricing contract](cuda/american-and-bermudan-pricing-contract.md)
  — Longstaff--Schwartz policies, regression, workspaces, and launch flow.
- [Launch validation and kernel diagnostics](cuda/launch-validation-and-kernel-diagnostics.md)
  — inspect catalogue launch plans, native guards, resources, and occupancy.

## Dataset and extension contracts

- [Catalogue extension workflow](catalog-extension-and-validation-workflow.md)
  — end-to-end checklist for code, recipe, test, validation, and publication.
- [Model and product parameter datasets](model-and-product-parameter-dataset-generation.md)
  — ordered core/stress rows and versioned Philox domains.
- [Model-sample datasets](model-sample-dataset-generation.md) — sample shapes,
  generated bindings, CUDA execution, memory guards, and smoke tests.
- [Independent price-validation pipeline](independent-price-validation-pipeline.md)
  — Premia/QuantLib hierarchy, cached references, fingerprints, and
  fail-closed publication.

## Performance and operations

- [CUDA performance regression protocol](performance-regression-protocol.md)
  — campaign preflight, timing scopes, resource budgets, rebaseline rules, and
  per-architecture evidence.
- [RTX 4090 Laptop memory notebook](cuda/rtx4090-laptop-memory-map.ipynb) —
  hardware-specific SM89 observations, never portable defaults.
- [Closed-form price-count scaling report](performance-reports/closed-form-price-count-scaling-sm89-2026-09-06.md)
  — 1/16/1,000-price geometry, resources and end-to-end SM89 evidence.
- [Jamshidian scalar/cooperative strategy](performance-reports/jamshidian-strategy-scaling-sm89-2026-09-08.md)
  — all one-factor rates models, 100 to 2²⁰ prices, launch geometry and
  calendar sensitivity; numerical and timing qualification limits.
- [Pricing workload scaling — ongoing](performance-reports/pricing-workload-scaling-sm89-2026-09-07.md)
  — 100/1,000/10,000 prices, per-price path counts and workload-specific geometry.
- [Dataset pricing runtime notebook](performance-reports/pricing-dataset-runtime-sm89.ipynb)
  — 1,000 actual catalogue prices per representative pair, 2²⁰ paths per
  Monte Carlo price; GPU time, generation phases and an editable campaign budget.
- [Catalogue generation readiness](performance-reports/catalogue-generation-readiness-sm89-2026-09-08.md)
  — bounded Kou regression correction, compiled resources and staged native pilots.
- [CIR forward-measure comparison](performance-reports/cir-forward-measure-comparison-sm89-2026-09-08.md)
  — integrated Bermudan method, row-wise comparison to the previous method,
  numerical evidence and remaining publication-certification limits.
- [G2/G2++ European-swaption Monte Carlo](performance-reports/g2-european-swaption-monte-carlo-sm89-2026-09-08.md)
  — shared-kernel integration, independent price diagnostics and real-generator timings.
- [Proposed protected-download design](proposed-protected-dataset-download-design.md)
  — Turnstile and signed-URL trust boundaries; it is not an implemented feature.

## Mathematical references

[Model and curve reference index](model-and-curve-reference-index.md) is the
complete list of mathematical notes kept beside source code. Those pages own
model equations, parameter meaning, and numerical schemes. They do not own
the model/product capability matrix, generated target lists, or file-tree
inventories.

The authoritative capability inventory is
[`tools/codegen/pricing_bindings/capability_manifest.py`](../tools/codegen/pricing_bindings/capability_manifest.py).

Source ownership is introduced by the
[Markovian equity](../src/model/equity/markovian/README.md),
[rough equity](../src/model/equity/rough/README.md),
[fixed-income](../src/model/fixed_income/README.md), and
[financial-product](../src/product/README.md) entry points.

## Project records

The main repository audit and the independent validation audit are separate:

- [Main audit query](audit/query.md), [status](audit/status.md),
  [open findings](audit/response.md), [closed findings](audit/closed.md).
- [Validation query](validation/query.md), [status](validation/status.md),
  [open findings](validation/response.md),
  [closed findings](validation/closed.md).

Audit and validation records preserve evidence. Ordinary documentation work
must not rewrite them to hide a contradiction or invalidate historical
provenance.

## Documentation conventions

- Follow the [writing and discovery rules](audit/query.md#documentation-et-parcours-de-découverte)
  for all documentation, including README files.
- `README.md` is a directory entry point, not a hidden contract or copied
  inventory.
- `*-contract.md` defines normative interfaces and invariants.
- `*-workflow.md` gives an ordered operational procedure.
- `*-protocol.md` defines a reproducible qualification or measurement.
- `*-reference.md` and `*-index.md` provide stable reference material.
- File names and code identifiers use English. A page may use English or
  French, but it remains in one language and preserves canonical code terms.
- One document owns each rule. Other pages summarize only enough context to
  link to that source.
- Lists derived from CMake or a typed manifest are discovered or generated,
  not copied into documentation.
- Hardware measurements always name their GPU, architecture, toolchain, and
  scope.

When documentation and implementation disagree, treat the implementation and
its tested manifest as evidence, then correct the owning document. Do not add
a second explanation beside the stale one.

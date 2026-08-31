# Documentation index

All maintained project documentation is indexed from this tree. Normative
contracts live under `docs`; mathematical model and curve references that are
kept beside their source are listed explicitly below. The main audit and the
deliberately separate validation audit use the symmetric `audit/` and
`validation/` folders documented below.

## Model and curve references

- [`model-and-curve-reference-index.md`](model-and-curve-reference-index.md):
  exhaustive index of the local mathematical references kept under `src`, with
  the boundary between descriptive equations and canonical capability data.

## Extension workflows

- [`catalog-extension-and-validation-workflow.md`](catalog-extension-and-validation-workflow.md):
  complete checklist for adding a model, curve, product, price dataset,
  independent validation, and website entry.
- [`model-and-product-parameter-dataset-generation.md`](model-and-product-parameter-dataset-generation.md):
  construction of ordered core and stress parameter rows and their YAML recipe.
- [`model-sample-dataset-generation.md`](model-sample-dataset-generation.md):
  availability matrix and contract for 3M-row generative-training datasets,
  in-memory Philox parameter generation, persistent CUDA sampling, streaming
  JSON, and smoke tests.
- [`independent-price-validation-pipeline.md`](independent-price-validation-pipeline.md):
  mandatory model-product-aware Premia-to-QuantLib hierarchy, row-level
  fallback, failure classification, persistent reference datasets,
  continuous/discrete bias handling, and cache-only publication checks.

## CUDA architecture and implementation contracts

- [`cuda/README.md`](cuda/README.md): local index for all CUDA-specific
  architecture guides, implementation contracts, hardware notes, and links to
  the performance protocol.
- [`cuda/pricing-policy-composition.md`](cuda/pricing-policy-composition.md):
  guide visuel des relations entre dynamique, calendrier, schedule, produit,
  handler, pricing policy et kernel pour chaque famille de pricing CUDA.
- [`cuda/model-dynamics-contract.md`](cuda/model-dynamics-contract.md): model
  dynamics layers, common device interface, state layout, Philox consumption,
  exact and discretized transitions, and naming conventions.
- [`cuda/model-analytics-contract.md`](cuda/model-analytics-contract.md):
  canonical analytics APIs, capability providers, fitted-model composition,
  shared lognormal primitives, and symmetric numerical tests.
- [`cuda/closed-form-and-monte-carlo-pricing-contract.md`](cuda/closed-form-and-monte-carlo-pricing-contract.md):
  required types, functions, kernels, launchers, and invariants for closed-form
  and standard Monte Carlo pricing.
- [`cuda/american-and-bermudan-pricing-contract.md`](cuda/american-and-bermudan-pricing-contract.md):
  early-exercise kernels, Longstaff-Schwartz responsibilities, workspace
  planning, launch interface, and memory layout.
- [`cuda/launch-validation-and-kernel-diagnostics.md`](cuda/launch-validation-and-kernel-diagnostics.md):
  common launch validation, CUDA error handling, resource inspection,
  theoretical occupancy, and diagnostics output.
- [`performance-regression-protocol.md`](performance-regression-protocol.md):
  versioned CUDA performance protocol, baselines, decision thresholds and
  reproduction commands.

## Operations

- [`website-protected-dataset-download-workflow.md`](website-protected-dataset-download-workflow.md):
  protected website download flow, Turnstile validation, temporary URLs, and
  server-side checks.

## Work tracking

- [`audit/query.md`](audit/query.md): stable audit protocol and checklists for
  numerical code, architecture, build, ownership, and CUDA performance.
- [`audit/status.md`](audit/status.md): revision, scope, exclusions, and
  evidence for the latest execution of the main audits.
- [`audit/response.md`](audit/response.md): actionable audit findings that
  remain unresolved, including findings explicitly postponed within the audit.
- [`audit/closed.md`](audit/closed.md): compact registry of corrected,
  disproved, merged, or inapplicable findings, with evidence and reopening
  conditions.
- [`validation/query.md`](validation/query.md): separate, explicitly
  triggered audit protocol for independent references, caches, provenance, and
  reproducibility of published datasets.
- [`validation/status.md`](validation/status.md): revision, scope,
  exclusions, and evidence for the latest validation audit only.
- [`validation/response.md`](validation/response.md): unresolved findings
  owned by the separate validation audit.
- [`validation/closed.md`](validation/closed.md): closed findings of the
  validation audit, kept separate from the main audit registry.

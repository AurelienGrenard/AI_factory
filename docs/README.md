# Documentation index

All maintained project documentation lives directly in this directory. File
names describe both the subject and the role of each document; separate
`research_notes` and nested planning folders are intentionally not used.

## Extension workflows

- [`catalog-extension-and-validation-workflow.md`](catalog-extension-and-validation-workflow.md):
  complete checklist for adding a model, curve, product, price dataset,
  independent validation, and website entry.
- [`model-and-product-parameter-dataset-generation.md`](model-and-product-parameter-dataset-generation.md):
  construction of ordered core and stress parameter rows and their YAML recipe.
- [`independent-price-validation-pipeline.md`](independent-price-validation-pipeline.md):
  mandatory model-product-aware Premia-to-QuantLib hierarchy, row-level
  fallback, failure classification, core-only external certification, and
  separate internal stress diagnostics.

## CUDA implementation contracts

- [`cuda-model-dynamics-contract.md`](cuda-model-dynamics-contract.md): model
  dynamics layers, common device interface, state layout, Philox consumption,
  exact and discretized transitions, and naming conventions.
- [`cuda-closed-form-and-monte-carlo-pricing-contract.md`](cuda-closed-form-and-monte-carlo-pricing-contract.md):
  required types, functions, kernels, launchers, and invariants for closed-form
  and standard Monte Carlo pricing.
- [`cuda-american-and-bermudan-pricing-contract.md`](cuda-american-and-bermudan-pricing-contract.md):
  early-exercise kernels, Longstaff-Schwartz responsibilities, workspace
  planning, launch interface, and memory layout.
- [`cuda-launch-validation-and-kernel-diagnostics.md`](cuda-launch-validation-and-kernel-diagnostics.md):
  common launch validation, CUDA error handling, resource inspection,
  theoretical occupancy, and diagnostics output.

## Operations and financial derivations

- [`website-protected-dataset-download-workflow.md`](website-protected-dataset-download-workflow.md):
  protected website download flow, Turnstile validation, temporary URLs, and
  server-side checks.
- [`zero-coupon-bond-call-pricing-derivation.md`](zero-coupon-bond-call-pricing-derivation.md):
  change-of-numeraire derivation of zero-coupon bond call prices under
  standalone OU and fitted Hull-White models.

## Work tracking

- [`deferred-work.md`](deferred-work.md): planned extensions that remain on the
  roadmap but are intentionally outside the current task.
- [`abandoned-work.md`](abandoned-work.md): measured or analyzed ideas that were
  rejected; this is evidence, not a backlog.

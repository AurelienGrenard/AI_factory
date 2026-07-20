# Notebook Tooling

This directory is reserved for helpers that build or standardize validation and
production notebooks.

Notebook helpers belong here when they format tables, assemble repeated audit
cells, or generate notebook templates. Pricing, simulation, and payoff logic
must remain in `src_cpp` or `src_python`.

Run `python tools/notebooks/build_validation_audits.py` after adding a production
result mode. It enforces the common notebook spine and delegates loading,
comparison metrics, and timing presentation to `tools.validation.audit`.

Run `python tools/notebooks/audit_timings.py` after regenerating validation
results. Add `--strict` in CI once every reported hierarchy warning has either
been corrected or represented by an explicit tested exception.

Run `python tools/notebooks/execute_validation_notebooks.py` to execute every
validation notebook in a fresh Jupyter kernel and store its outputs in place.
Fresh kernels prevent allocator, thread-pool, and CUDA state from leaking from
one model/product audit into another.

Validation notebooks contain no product-specific diagnostic section. Economic
diagnostics, barrier parity, and other specialized invariants belong in tests
or future production-analysis notebooks. Path reconstruction remains part of
the validation spine when the model exports reproducible Philox paths.

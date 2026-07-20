# Dataset Certification

Certification turns production-audit evidence into automated acceptance rules.
Notebooks present the report; tests own every pass/fail decision.

This document is authoritative for acceptance criteria. Read
`architecture.md`, `production_dataset.md`, and `validation_dataset.md` first.
Historical journal entries and roadmap items do not modify these rules.
CUDA source migrations additionally follow `cuda.md` after `code_guide.md`.

## Registry Contract

Curve, model, product, and result JSON/YAML documents have exact top-level key
sets and canonical key order. Nested financial blocks remain family-specific.
Tests also verify triplets, paths, row counts, ids, database references, finite
outputs, source ownership, and CMake registration.

Validation model, curve, and product databases are deterministic prefixes of
their production databases. Validation results are independent repricings and
are never copied production results.

## Numerical Certification

Every production result mode has one four-engine validation audit. The common
report in `tools.validation.audit` checks:

- stored production C++ CUDA versus regenerated C++ CUDA;
- C++ CPU versus C++ CUDA;
- PyTorch CPU versus PyTorch GPU;
- C++ CUDA versus PyTorch GPU;
- price-only versus price-and-gradient C++ reuse when gradients exist.

Visible metrics are maximum absolute error, maximum relative error above the
normalized-price threshold `1e-4`, retained relative row count, and maximum
Z-score. Native reproducibility uses floating-point tolerances. Longstaff-
Schwartz products have a regression tolerance. Independent PyTorch comparisons
use a maximum Z-score of four.

Product-specific invariants such as barrier in/out parity and autocall cashflow
logic remain dedicated automated tests rather than notebook sections.

## Notebook Contract

Run:

```bash
python tools/notebooks/build_validation_audits.py
python tools/notebooks/execute_validation_notebooks.py
```

Every notebook has versioned `ai_factory` metadata and the canonical sections
documented in `validation_dataset.md`. Every code cell is introduced by
Markdown. Timing engines use the fixed order C++ CUDA, PyTorch GPU, C++ CPU,
PyTorch CPU, and stored outputs contain no execution error.

## Timing Contract

Every validation timing records a representative warm-up, benchmark row count,
workload, repetition count, aggregation statistic, and hot wall time. C++ CUDA
also records kernel time. The four engines in an audit use the same workload.

Portable tests validate timing semantics, not hardware speed. Performance
hierarchy is certified on the reference GPU host with:

```bash
AI_FACTORY_PERFORMANCE_CERTIFICATION=1 pytest -q
```

This mode requires the timing audit to report no hierarchy warning. Absolute
times are never acceptance criteria.

Closed-form zero-coupon batches under CIR and Hull-White are explicit
launch-bound exceptions for the relative ordering of C++ CPU, PyTorch CPU, and
PyTorch GPU. Their arithmetic intensity is too low for that ordering to measure
implementation quality. The C++ CUDA kernel must still beat PyTorch GPU on the
same duplicated workload.

Shifted Black-76 caplets and European swaptions are also launch-bound analytic
exceptions for the middle-engine ordering. Their CUDA kernel must still beat
PyTorch GPU on the shared 100,000-row duplicated workload.

## Standard Test Run

```bash
pytest -q
python tools/notebooks/audit_timings.py --strict
```

The strict timing audit runs on the certification host. A normal developer test
run skips hardware hierarchy and CUDA path execution when no GPU is visible.

# Dataset provenance and reuse

This contract separates **what produced a dataset**, **whether it matches the
requested recipe**, and **whether its prices are independently certified**.
These are different questions. A changed Git commit or codegen fingerprint
does not, by itself, require recalculating prices or samples.

## Ownership and scope

[`generate_catalog.py`](../tools/datasets/generate_catalog.py) enriches the
standalone `generation.yaml` receipt after checking the native outputs and
before freezing the publication hashes. JSON bytes,
pricing kernels, sample factories, seeds and launch configurations are unchanged.
Publication and interrupted-pair recovery remain owned by `artifact_publication.py`.

[`dataset_provenance.py`](../tools/datasets/dataset_provenance.py) owns the
record and comparison rules. No generated recipe, manifest or model duplicates
them. New supported recipes inherit them through the campaign controller.
Direct native invocation and standalone parameter generators do not write this
record; absence means **unknown provenance**, not invalid data. Model, product
and curve parameters are fingerprinted as the actual inputs of a price job.

## What is recorded

The receipt contains schema version 1 and four distinct kinds of evidence:

| Evidence | Purpose |
|---|---|
| JSON SHA-256 and generation-record checksum | Detect missing or changed bytes and accidental record edits |
| Logical identity, shape, path count and named RNG stream seeds | Identify the requested dataset independently of its physical location |
| Parameter fingerprints by role | Preserve ordered rows and all non-presentation fields, without depending on input paths |
| Executable, recipe, launch plan, build hashes and source provenance | Identify the actual generation, without claiming mathematical equivalence |

The canonical `recipe.yaml` fingerprint includes time grids,
numerical-method metadata, output definitions, parameter laws, seeds and
execution settings when present. Validation is stored independently in
`validation.yaml`. Parameter fingerprints likewise exclude only top-level
presentation fields; unknown non-presentation fields remain significant.
Row ordering is never sorted away. JSON output integrity is deliberately
byte-exact, including its original envelope; do not edit a published JSON to
update its paths or timings.

Each new campaign retains its revision and worktree status, selected recipes,
inputs, executables, build files and a source archive covering maintained
`src`, `tools`, CMake modules and presets, including untracked source files.
The archive hash is provenance, **not a global invalidation key**. SDKs, system
libraries and drivers are not bundled; GPU observations are informative and
are retained with the generation. This is not a hermetic build or a promise
of bitwise replay on another GPU/toolchain. SHA-256 is not an authenticity
signature against a party able to replace both data and records.

A non-publishing pilot may freeze a dirty worktree because its outputs remain
inside `work/generation/`. A publishing campaign requires a clean worktree so
the recorded revision names the exact maintained sources.

Retain/export the campaign directory with the dataset release: a hash without
its source archive identifies missing evidence but cannot reconstruct it.

## Inspect reuse without generating data

```bash
python3 tools/datasets/check_dataset_compatibility.py \
  --recipe catalog/model/equity/markovian/heston/samples/samples_01/recipe.yaml \
  --generation catalog/model/equity/markovian/heston/samples/samples_01/generation.yaml \
  --dataset datasets/model/equity/markovian/heston/samples/samples_01.json \
  --target generate_heston_samples_01 --build build
```

The checker only reads the existing pair and current inputs, checks Ninja's
build freshness, and invokes the host launch inspector for prices. It never
executes a generator, writes metadata, publishes, or calls a reference engine.
The explicit dataset path may point to a relocated artifact. Its JSON result
includes the candidate descriptor and fingerprints for retaining a review.

| Status / exit code | Meaning and action |
|---|---|
| `compatible` / 0 | Recorded recipe, inputs, executable and build configuration match; no recalculation required by this comparison |
| `review_required` / 2 | Evidence is absent, prerequisites are stale/missing, or recipe/build/calculation evidence changed; investigate, do not discard the base |
| `regenerate` / 3 | Identity, ordered parameters, seeds, shape or requested path count differs; the old base does not fulfill this new request |
| `invalid` / 4 | Stored data, recipe metadata or generation record fails integrity; recover the original pair or investigate corruption |

Reorganizing codegen without changing generated code or the build does not
invalidate data. Relocating unchanged JSON inputs or outputs does not either.
Recompilation, changed includes, a new geometry, different floating-point
precision or a kernel change can require review. Neither stripping C++ comments
nor hashing a source tree proves numerical equivalence, so this checker does
not attempt such an inference. A changed build hash may reflect only directory
paths; it is a review signal, not an instruction to recalculate.

## After a calculation or implementation change

Review the affected model/payoff, time grid, parameter law, RNG consumption,
precision, reduction order and launch strategy. Use targeted before/after
comparisons with the same inputs; record bitwise equality where required, or
explicit numerical budgets and scope otherwise. Preserve the old artifact
hash, generation-record hash, candidate descriptor/hash, decision and evidence
in the audit/release record. Never rewrite the old YAML to pretend the new
binary generated it. No automatic equivalence allowlist is maintained.

An old base can remain useful even if it is not the output of the newest
implementation. Keep its original recipe and qualifications; do not label it
as satisfying a changed seed, path count, payoff or discretization.

## Legacy data, resume and certification

Existing datasets are not regenerated or backfilled by this change. Missing
provenance cannot be reconstructed from today's checkout alone. Recover the
original campaign evidence and review it; otherwise leave the origin unknown.

Campaign state version 4 freezes the provenance module, generator, canonical
recipe and source archive and
binds terminal Monte Carlo checkpoints to the exact executable, recipe, inputs,
shape, seeds and launch plan. It refuses changed frozen sources, recipes, build
files, inputs or executables. Version-1 and version-2 campaigns must finish with
their original frozen controller; do not migrate a half-published pair or relax
resume guards to use newer code. Older campaign versions are not accepted by
the current controller. Their progress records contain no calculated
prices and therefore cannot recover an unfinished numerical prefix.

Independent validation remains governed by the
[price-validation contract](independent-price-validation-pipeline.md).
Compatibility never sets `verified: true`, bypasses stale-cache checks or
changes tolerances. Its result always states `certification: not_assessed`.

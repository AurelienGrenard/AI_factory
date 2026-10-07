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
execution settings when present. Independent certification is currently deferred
and does not create a catalogue document. Parameter fingerprints exclude only top-level
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

## Supplementary releases and qualified historical data

The [release index](../work/catalog/releases/index.json) identifies immutable
supplementary datasets by a release-qualified ID. Its manifest maps each ID to
the published JSON and the **original receipt from the generator run**. The
JSON and `generation.yaml` retain their embedded logical IDs and source paths;
relocation changes neither their bytes nor their provenance. Resolve a
release-qualified ID through the manifest, then check the hashes and the
receipt. A release is not a replacement for the historical catalogue entry.

The [qualification index](../work/catalog/qualifications/index.json) identifies
metadata-only qualifications of historical datasets. A qualification manifest
records the original hashes, the generation receipts and the evidence for its
decision. A qualified copy does not acquire a new generator receipt: its
numerical rows are unchanged and the added `methodology` describes the scope
of the published statistic. An exclusion mask refers to original rows without
copying their JSON. Consumers should use the qualification manifest when
interpreting the data; the historical JSON remains an unmodified record of
what was originally published.

The [NUM-037 SABR qualification](../work/catalog/qualifications/num037-sabr-parity-20261007-v1/manifest.json)
is a metadata-only exclusion mask for 26 aligned European call/put pairs in the
NUM-029 release. It identifies 52 original price rows by ID and hash, marks
them as **not certified for price or standard-error reference use**, and links
the replay evidence. The immutable prices and generator receipts are unchanged.
Unlisted rows do not gain independent certification from this mask. Resolve the
mask through the qualification index before using that release as a price
reference. The [unresolved NUM-037 closure](audit/unresolved-closures.md) records the
SABR price exclusions. The [resolved NUM-036 entry](audit/closed.md) defines
the conditional scope of LSM standard errors.

The [NUM-030 log-modulated rough Bergomi qualification](../work/catalog/qualifications/num030-lmb-stress-20261007-v1/manifest.json)
is an exclusion mask for 42 positive-correlation stress parameter rows across
29 price datasets, plus one negative-correlation up-and-in call with unstable
Monte Carlo tails. It excludes 1,219 original price/SE rows from reference use;
the original JSON and receipts are unchanged. NUM-030 remains open because
stable independent prices are not yet available.

The [rough Bergomi domain qualification](../work/catalog/qualifications/rough-bergomi-martingale-domain-20261007-v1/manifest.json)
excludes 15 positive-correlation stress rows in each of 29 historical price
datasets (435 prices/SE) from ordinary martingale reference use. It is a
model-domain decision, not an observed price discrepancy. The
[dataset gate](../maintainer/tools/datasets/check_martingale_reference_domains.py)
recomputes necessary exclusions for both Bergomi models and checks the
`β=1, ρ>0` boundary for the two SABR models. It fails if a necessary
Bergomi mask or SABR domain decision is missing. The remaining models and
Monte Carlo tail quality are outside this gate
([scope and counts](audit/martingale-reference-datasets.md)).

The release and qualifications are checked with
[`verify_num029_release.py`](../maintainer/tools/datasets/verify_num029_release.py),
[`verify_num036_qualification.py`](../maintainer/tools/datasets/verify_num036_qualification.py),
[`verify_num037_qualification.py`](../maintainer/tools/datasets/verify_num037_qualification.py)
[`verify_num030_qualification.py`](../maintainer/tools/datasets/verify_num030_qualification.py)
and [`verify_rough_bergomi_domain_qualification.py`](../maintainer/tools/datasets/verify_rough_bergomi_domain_qualification.py).

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

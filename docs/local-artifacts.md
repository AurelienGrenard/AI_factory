# Local builds and historical artifacts

Keep build products separate from records that must survive a rebuild:

| Location | Purpose |
|---|---|
| `builds/local-cuda12-9-0-sm89/` | Active local RTX 4090 build: CUDA 12.9, C++20, SM89. Recreated with `cmake --preset local-sm89`. |
| `builds/ppti-gpu4/` | PPTI GPU 4 V100 build: CUDA 12.9, C++20, SM70. Recreated with `cmake --preset ppti`. |
| `builds/ppti-gpu1/` | PPTI GPU 1 RTX 5090 build: CUDA 13.2, C++20, SM120. Recreated with `cmake --preset ppti-gpu1`. |
| `builds/ppti-gpu3/` | Reserved PPTI GPU 3 SM120 build. Recreated with `cmake --preset ppti-gpu3` after its configuration is confirmed. |
| `builds/<other-name>/` | Named alternate or historical CMake build; identify its cache before use. |
| `artifacts/audit/` | Ignored local evidence cited by the main audit, including old test runs, source snapshots, logs, manifests and selected historical binaries. |
| `artifacts/performance/legacy-build-dev/` | Historical measurement campaigns, raw results, diagnostics and source/binary snapshots that used to sit inside `build-dev/`. |
| `artifacts/validation/` | Historical validation reports. |
| `artifacts/price_gradients/` | Historical bump and batching diagnostics cited by price-gradient work. |
| `artifacts/learning/campaigns/` | Frozen training campaigns and measured outputs; not a parser cache. |
| `artifacts/legacy_datasets/` | Relocated historical datasets retained when experiments or caches refer to their hashes. |
| `artifacts/tooling/premia/` | Reusable local Premia runner and Wine prefix; the validation bridge uses this location. |
| `work/generation/<campaign>/` | Resumable dataset-generation campaigns: frozen inputs, progress journals, logs and publication records. |
| `work/experiments/<asset-class>/<model>/<study>/` | Disposable studies, one-off generators, training runs, notebooks and reports. |
| `work/catalog/` | Ignored local-only extension recipes, their manifest, and historical archives, qualifications and releases; separate from the selected production leaves. |
| `work/swaption-lsm-vs-sobolev/` | Local reference and robustness study; keep its source and campaign evidence while the comparison is in use. |
| `datasets/` | Selected published datasets identified by `catalog/manifest.json`. |
| `work/datasets/` | Local-only materialized datasets from `work/catalog/manifest.json`, plus historical release and qualification payloads; do not mistake ungenerated recipes for missing publication. |

Only `build/` and `builds/<name>/` are CMake build locations.
`artifacts/`, `work/` and `datasets/` are ignored by Git and are not distributed by a
clone. The active local build is `builds/local-cuda12-9-0-sm89/`; use
`cmake --preset local-sm89` followed by `cmake --build --preset local-all`
or `cmake --build builds/local-cuda12-9-0-sm89 --target <target>`.
The local and PPTI presets share C++20 build options, but select their own
CUDA toolkit paths, SMs, tuning profiles and build directories. PPTI GPU 1 and
GPU 3 use the same CUDA 13.2/SM120 options while keeping separate build files. Do not point CMake at a historical evidence directory.

The historical CUDA 13-era caches `build/` and `builds/ppti/` were removed
on 2026-10-06 after checking their CMake caches, active processes and frozen
campaign manifests. A campaign resumed by `tools/datasets/generate_catalog.py`
uses its own copied binaries, inputs, source archive and build hashes under
`work/generation/<campaign>/`; it does not execute the old root build.
Removing these two caches released approximately 9 GiB. The additional
`work/builds/heston-samples-3m-current/` CMake cache (247 MiB) was removed
after verifying both frozen GAN campaigns contain their own executable and
source archive. The CUDA 12.9 build under `builds/local-cuda12-9-0-sm89/`
remains the active local build.

The obsolete CMake object trees inside preserved audit directories were removed;
the logs, manifests, configurations, source snapshots and explicit proof binaries
remain. `artifacts/build-cleanup-receipt.json` records the removed directories.

The 2026-10-06 audit also moved the compact truth-audit snapshot and CTest log
from `/tmp/ai_factory_full_audit_20261006/` to
`artifacts/audit/axis-truth-2026-10-06/`, then discarded the 3.5 GiB temporary
build. Its `receipt.json` hashes the preserved files. A 62 MiB NumPy cache
under `artifacts/learning/cache/` was removed only after its dataset and two
parameter inputs were found at relocated paths and all three SHA-256 values
matched the cache manifest.

## Finding paths recorded before the cleanup

Audit records and older performance reports retain the path names from the
original runs. Resolve them as follows:

| Recorded path | Current local path |
|---|---|
| `build-audit-v9-20260909/`, `build-price-delta/`, `build-rough-price-delta/`, `build-dataset-provenance-20260910-z8KotZ/`, `build-remediation-*/` | `artifacts/audit/<same directory>/` |
| `build-dev/<performance run or diagnostic file>` | `artifacts/performance/legacy-build-dev/<same name>` |
| `build-dev/validation-reports/` | `artifacts/validation/legacy-build-dev/validation-reports/` |
| `build-dev/catalog-generation-native-pilot-20260908-01/` and `-02/` | Raw local campaigns removed; retained summaries are under `maintainer/tests/performance/reports/generation-readiness-sm89-2026-09-08/`. |
| `build-rough-price-delta/campaigns/` | Interrupted raw campaigns removed; no completed dataset was preserved there. |
| `build/premia-wine/`, `build/validation/premia/premia_runner.exe` | `artifacts/tooling/premia/wine-prefix/`, `artifacts/tooling/premia/premia_runner.exe` |

The ignored local `artifacts/relocation.json` records the earlier moves,
including generation campaigns that were subsequently removed. Their old
absolute paths in reports are historical references, not resumable campaigns.
Start a new campaign with a new run directory when generating data.

## Retention rule

- A CMake build is a cache. Keep only a named active build or a build whose
  configuration is being checked; recreate it from its preset when needed.
- Keep `work/generation/<campaign>/` while a run can resume, a checkpoint is
  needed, or an audit cites its attempts. After publication and review, verify
  that the dataset and generation record exist before removing the campaign.
  Do not delete failed attempts needed to diagnose a numerical issue.
- Keep `artifacts/` evidence while a finding or historical comparison cites it.
  Its source, inputs and binary hashes determine whether it can support a new
  claim. Discard an obsolete cache inside it only after separating that cache
  from the logs and proof binaries.
- Keep published `datasets/` together with their catalog generation records.
  A repository clone does not contain these ignored datasets; their absence
  locally must not be mistaken for a codegen failure.

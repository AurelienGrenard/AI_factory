# Local builds and historical artifacts

Keep build products separate from records that must survive a rebuild:

| Location | Purpose |
|---|---|
| `build/` | Main CMake `dev` preset: executable targets, libraries and object files. CMake may rebuild or replace it. |
| `builds/<name>/` | Optional separate CMake build for another GPU, compiler or configuration. |
| `artifacts/audit/` | Ignored local evidence cited by the main audit, including old test runs, source snapshots, logs, manifests and selected historical binaries. |
| `artifacts/performance/legacy-build-dev/` | Historical measurement campaigns, raw results, diagnostics and source/binary snapshots that used to sit inside `build-dev/`. |
| `artifacts/validation/` | Historical validation reports. |
| `artifacts/tooling/premia/` | Reusable local Premia runner and Wine prefix; the validation bridge uses this location. |
| `work/generation/<campaign>/` | Resumable dataset-generation campaigns: frozen inputs, progress journals, logs and publication records. |
| `work/experiments/<asset-class>/<model>/<study>/` | Disposable studies, one-off generators, training runs, notebooks and reports. |
| `datasets/` | Published dataset artifacts only; never campaign state or experiment scratch data. |

Only `build/` and optional `builds/<name>/` are CMake build locations.
`artifacts/`, `work/` and `datasets/` are ignored by Git and are not distributed by a
clone. `build/` is the sole persistent main build; a named alternate build is
reserved for a different configuration or isolated experiment. To rebuild the
active targets, use `cmake --preset dev` followed by
`cmake --build build --target <target>`; do not point CMake at a historical
evidence directory.

The obsolete CMake object trees inside preserved audit directories were removed;
the logs, manifests, configurations, source snapshots and explicit proof binaries
remain. `artifacts/build-cleanup-receipt.json` records the removed directories.

## Finding paths recorded before the cleanup

Audit records and older performance reports retain the path names from the
original runs. Resolve them as follows:

| Recorded path | Current local path |
|---|---|
| `build-audit-v9-20260909/`, `build-price-delta/`, `build-rough-price-delta/`, `build-dataset-provenance-20260910-z8KotZ/`, `build-remediation-*/` | `artifacts/audit/<same directory>/` |
| `build-dev/<performance run or diagnostic file>` | `artifacts/performance/legacy-build-dev/<same name>` |
| `build-dev/validation-reports/` | `artifacts/validation/legacy-build-dev/validation-reports/` |
| `build-dev/catalog-generation-native-pilot-20260908-01/` and `-02/` | Raw local campaigns removed; retained summaries are under `tests/performance/reports/generation-readiness-sm89-2026-09-08/`. |
| `build-rough-price-delta/campaigns/` | Interrupted raw campaigns removed; no completed dataset was preserved there. |
| `build/premia-wine/`, `build/validation/premia/premia_runner.exe` | `artifacts/tooling/premia/wine-prefix/`, `artifacts/tooling/premia/premia_runner.exe` |

The ignored local `artifacts/relocation.json` records the earlier moves,
including generation campaigns that were subsequently removed. Their old
absolute paths in reports are historical references, not resumable campaigns.
Start a new campaign with a new run directory when generating data.

# CMake target ownership

For a first explanation of configuration, targets, compilation and the active local
`builds/local-cuda12-9-0-sm89/` directory, start with the [CMake build guide](../docs/cmake-build-workflow.md).

The root `CMakeLists.txt` owns project-wide configuration and CTest dashboard
targets, then includes these domain modules. Product target registration and
source inventories belong to the narrowest applicable module.

| Module | Responsibility |
|---|---|
| `AIFactoryRuntime.cmake` | CUDA tuning plus host dataset-validation and offline-stage libraries |
| `AIFactoryTargets.cmake` | Runtime, Longstaff--Schwartz, dataset-loader and CUDA launch targets, plus their construction helpers |
| `AIFactoryCatalog.cmake` | Catalogue generators and aggregate generation targets |
| `AIFactoryTests.cmake` | Host/CUDA test registration and test aggregates |
| `generated/` | Codegen-owned bindings and capability metadata |

Do not add a second monolithic target list to the root file or this README.
The root orchestration derives each target's `AI_FACTORY_OWNER_MODULE` property
from the target inventory created by each module and rejects unowned targets at
configure time. The configured build graph is authoritative. The local and PPTI presets
share the base build options and use separate SMs and toolchain paths. Inspect
the active local graph and its tests with:

```sh
cmake --build builds/local-cuda12-9-0-sm89 --target help
cmake --build --preset local-tests
ctest --test-dir builds/local-cuda12-9-0-sm89 -N
```

Generated CMake files are updated through the
[pricing and sampling code generator](../tools/codegen/pricing_bindings/README.md).
The [catalogue extension workflow](../docs/catalog-extension-and-validation-workflow.md)
identifies which module and aggregate targets an extension must update.

Maintainer-only benchmarks, audits, and independent references are absent from
the delivered build graph. In a development checkout containing `maintainer/`,
configure with `-DAI_FACTORY_ENABLE_MAINTAINER=ON` to register them.

Disposable targets under ignored `work/experiments/` are absent from the
normal build graph. Configure explicitly with
`-DAI_FACTORY_ENABLE_LOCAL_EXPERIMENTS=ON` to include their local
`CMakeLists.txt`. Even in that mode, those targets remain outside the permanent
`parameter_generators`, `price_generators`, `price_gradient_generators`, and `sample_generators` aggregates. Optional recipes under `work/catalog/` can
be built individually when present, but never join those aggregates; a clean
checkout and a local workspace therefore build the same permanent set.

Sources read at configuration time to infer link dependencies or test labels
must be registered in `CMAKE_CONFIGURE_DEPENDS`. Editing their includes must
update the graph on the next build, without a manual CMake reconfiguration.
Compiler include dependencies alone do not update these inferred CMake rules.

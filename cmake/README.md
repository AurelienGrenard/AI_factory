# CMake target ownership

For a first explanation of configuration, targets, compilation and the local
`build/` directory, start with the [CMake build guide](../docs/cmake-build-workflow.md).

The root `CMakeLists.txt` owns project-wide configuration and CTest dashboard
targets, then includes these domain modules. Product target registration and
source inventories belong to the narrowest applicable module.

| Module | Responsibility |
|---|---|
| `AIFactoryRuntime.cmake` | CUDA tuning plus host dataset-validation and offline-stage libraries |
| `AIFactoryTargets.cmake` | Runtime, Longstaff--Schwartz, dataset-loader and CUDA launch targets, plus their construction helpers |
| `AIFactoryCatalog.cmake` | Catalogue generators and aggregate generation targets |
| `AIFactoryTests.cmake` | Host/CUDA test registration and test aggregates |
| `AIFactoryPerformance.cmake` | Benchmarks, campaigns, profiles and regression gates |
| `AIFactoryValidation.cmake` | Independent validation test registration |
| `generated/` | Codegen-owned bindings and capability metadata |

Do not add a second monolithic target list to the root file or this README.
The root orchestration derives each target's `AI_FACTORY_OWNER_MODULE` property
from the target inventory created by each module and rejects unowned targets at
configure time. The configured build graph is authoritative. `help` lists
primary targets; Ninja's full list also includes individual generators:

```sh
cmake --build build --target help
ninja -C build -t targets all | rg '^generate_heston_'
ctest --test-dir build -N
```

Generated CMake files are updated through the
[pricing and sampling code generator](../tools/codegen/pricing_bindings/README.md).
The [catalogue extension workflow](../docs/catalog-extension-and-validation-workflow.md)
identifies which module and aggregate targets an extension must update.

Sources read at configuration time to infer link dependencies or test labels
must be registered in `CMAKE_CONFIGURE_DEPENDS`. Editing their includes must
update the graph on the next build, without a manual CMake reconfiguration.
Compiler include dependencies alone do not update these inferred CMake rules.

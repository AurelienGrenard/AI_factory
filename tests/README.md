# Test suites

`tests` contains host, CUDA, and architectural contract checks. Maintainer-only
performance campaigns, benchmarks, numerical studies, and independent price
references live outside this delivered tree.

Group checks by their primary responsibility, not by date or author. The root
contains this guide, not test sources. Use this map before adding a test:

| Directory | What to look for or add |
|---|---|
| `datasets/` | Parameter-row sampling, loaders, catalogue contracts, dataset assembly, provenance/reuse and publication |
| `sampling/` | Model-only trajectories, host memory guards, recipe metadata and replay |
| `model/equity/` | Equity dynamics contracts and model-specific pricing; rough engines under `rough/` |
| `model/fixed_income/` | Rate dynamics, analytics and closed-form/terminal pricing, including cross-model contracts |
| `longstaff_schwartz/` | Basis/solver checks and American/Bermudan engine compositions |
| `product/` | Shared payoff policies, accumulators and boundary behavior |
| `price_delta/` | Central-price parity, paired spot sensitivities and bump qualification, grouped by engine |
| `volterra/` | Shared kernel/path policies, observation schedules and FFT workspace contracts |
| `numerical/` | Random-number contracts, distributions, reductions and numerical primitives |
| `cuda/` | Launch planning, schedule guards, runner, resource diagnostics and policy budgets |
| `common/` | Helpers genuinely shared across test domains; no test entry points |

A test spanning several models stays with its shared contract: for example,
`longstaff_schwartz/equity_composition_cuda_test.cpp` covers Black-Scholes,
CEV, Kou, Merton and Schobel-Zhu together. Do not duplicate it in five model
folders. `sampling/model_sampling_contract_cuda_test.cu` uses Black-Scholes
to exercise the common sampler, not just that model's dynamics. Parameter-row
sampling belongs in `datasets/`, not with simulated trajectories.

Use explicit `<subject>_test.cpp` / `<subject>_cuda_test.cu` names (some CUDA
launcher tests are host `.cpp` callers). Start each file with a short summary
of its purpose and covered models when transverse. Keep fixtures in their
owner's `fixtures/` subfolder; helpers are not standalone tests. Add a shared
helper only for real consumers; do not keep alternative production engines
inside the test tree.
Performance fixtures and measured evidence belong under `maintainer/`;
temporary executables and generated datasets stay in a build or run directory.

Tests are registered centrally by `cmake/AIFactoryTests.cmake`; the configured
CTest inventory is authoritative. Public target/test names and labels remain
stable when files move; they need not repeat the physical filename. Discover
names and labels with:

```sh
ctest --test-dir build -N
ctest --test-dir build --print-labels
```

Use the narrowest label or test regex while developing, then the aggregate
preset appropriate to the change:

```sh
ctest --preset equity-tests
ctest --preset fixed-income-tests
ctest --preset tests
```

The `tests` preset selects the exact `main` ownership label added after test
registration. Maintainer-only references and benchmarks can be enabled with
`AI_FACTORY_ENABLE_MAINTAINER=ON` when that local tree is present.

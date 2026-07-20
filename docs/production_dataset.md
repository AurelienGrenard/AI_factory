# Production Dataset Build Protocol

This protocol applies within the dependency and ownership rules defined in
`architecture.md`. Read that document first when adding a new model/product
pair or reorganizing source and orchestration code.

Production datasets are source-first. If a production database prices a
`<product>` under a `<model>`, the pricing implementation must be visible in
`src`, not hidden only in registry tooling.

## Parameter Databases

Before coding, agree on how model, optional market inputs such as curves, and
product parameters are built. The usual
choices are:

- uniform random sampling with documented bounds and RNG;
- Cartesian products of documented grids.

Production input databases generally have the same row count `N`. Equity
results normally pair model row `i` with product row `i`. Fixed-income results
may additionally pair curve row `i`; each result row then stores `model_id`,
`curve_id`, and `product_id`. Curves are market inputs, not intrinsic model
parameters and not contractual product parameters.

Hull-White results therefore pair `model_id`, `curve_id`, and `product_id`.
Standalone affine models such as CIR imply their initial discount curve from
their own parameters and pair only `model_id` and `product_id`. Do not attach a
redundant external curve to CIR unless the model is explicitly extended with a
deterministic calibration shift.

For shifted curve-fitting models such as Hull-White, CIR++, and G2++, a
time-zero zero-coupon result checks direct reproduction of the input curve and
uses `result_construction.purpose: initial_curve_reproduction`. A time-zero
swap is implied entirely by that curve and uses `curve_implied_pricing`.
Swaptions and other optional claims depend on model dynamics and use
`model_dependent_pricing`.

For the current shifted-model references, CIR++ simulates its positive CIR
factor with the QE scheme on the declared `target_dt` grid. G2++ uses the exact
joint Gaussian transition of both OU factors and their time integrals between
contractual dates. European swaption kernels fuse simulation, payoff, and
reduction; Bermudan kernels additionally apply the shared Longstaff-Schwartz
policy. Do not replace these methods silently in orchestration code.

The caplet reference uses one shared product database with fixing time,
accrual period, strike, and notional. Hull-White and G2++ use exact joint
factor/integral transitions to the fixing date. CIR and CIR++ use QE on the
declared `target_dt: 1/52` grid. In every Monte Carlo engine, simulation,
discounting, caplet payoff, and reduction are fused at the pricing boundary.

Shifted Black-76 is an analytic market model and is deliberately limited to
caplets and European swaptions. Its model database stores volatility and
displacement; Nelson-Siegel supplies initial discount factors and forwards.
The displacement must keep every shifted forward and strike positive. Do not
attach zero-coupon, swap, or Bermudan result families to Black-76: they are not
model-dependent Black-76 claims.

Registry files are grouped by family. Keep database ids stable, but write files
under:

```text
registry/production/models/<model_family>/{data,specifications,generators}/...
registry/production/curves/<curve_family>/{data,specifications,generators}/...
registry/production/products/<product_family>/{data,specifications,generators}/...
registry/production/results/<model_family>/<product_family>/{data,specifications,generators}/...
```

## Required Source Layout

For a new `<model>/<product>` pair, add source files before generating the
registry result:

```text
src_cpp/ai_factory/cpu/<model>/<product>.hpp
src_cpp/ai_factory/cpu/<model>/<product>.cpp
src_cpp/ai_factory/cuda/<model>/<product>.cuh
src_cpp/ai_factory/cuda/<model>/<product>.cu
src_python/ai_factory/pytorch/<model>/<product>.py
```

Then expose the C++ entry points through:

```text
src_cpp/ai_factory/c_api/c_api.cpp
src_cpp/ai_factory/cuda/common/types.cuh
src_cpp/ai_factory/cuda/<model>/api.cuh
src_cpp/CMakeLists.txt
```

The registry generators in `registry/production` and `registry/validation`
must call these source implementations through the common tooling. They should
not contain the only implementation of a pricing algorithm.

`tools/registry` is orchestration code only. It may load model/product/result
JSON files, group rows by time-grid shape, call C++/CUDA/PyTorch engines, time
the calls, and write JSON/YAML outputs. It must not become a hidden second
implementation layer. In particular, optimized PyTorch pricing logic belongs in:

```text
src_python/ai_factory/pytorch/<model>/<product>.py
```

not in `tools/registry/result/<model>_<product>_pricing.py`.

## Implementation Standard

Read existing products in the same model before adding a new one. The new
product should follow the same structure, naming, timing fields, seed mapping,
and validation philosophy.

CUDA is the production target. The CUDA implementation must be product-specific
for the selected model and highly optimized. For pathwise products, simulate and
accumulate the payoff in the same kernel when possible. If the algorithm
requires several phases, such as Longstaff-Schwartz backward regression, the
source files must still be model/product-specific and must document the phases.

C++ CPU and PyTorch CPU/GPU implementations are validation backends. They must
still be solid implementations:

- C++ CPU uses shared Philox, tight loops, and OpenMP batch wrappers where
  appropriate;
- PyTorch uses one implementation selected by `device`, with tensorized work
  where practical;
- all engines use the same model, product, time-grid, and output conventions.

OpenMP batch parallelism is mandatory for independent C++ CPU result rows when
the library is built with OpenMP. A new product follows the established
`#pragma omp parallel for schedule(static)` batch pattern unless profiling
demonstrates a better portable strategy. Omitting this wrapper can make correct
C++ code appear slower than PyTorch CPU and is an implementation defect.

The expected broad timing hierarchy is:

```text
C++ CUDA << C++ CPU with OpenMP ~= PyTorch GPU << PyTorch CPU
```

C++ CPU and PyTorch GPU may exchange places depending on the model and
hardware. PyTorch CPU must not materially outperform optimized C++ CPU. A
validation timing that violates this rule triggers an implementation audit
before the dataset is accepted.

If the measured workload is too short to reveal this hierarchy after proper
warm-up, preserve the validation slice for correctness and benchmark one larger
batch formed by deterministic duplication of its rows. Record the benchmark
row count and representative hot execution time. Do not average repeated tiny
launches, and do not compare a CUDA kernel-only timer with complete CPU or
PyTorch calls.

Validation notebooks expose only end-to-end `wall seconds` for all four engines
and `kernel seconds` for C++ CUDA. Detailed simulation, payoff, transfer,
repetition, and throughput fields are internal metadata, not notebook columns.
The CUDA wall clock always covers the complete hot call, while CUDA events cover
device computation only. Do not optimize or reinterpret the metrics merely to
make `wall/kernel` ratios look alike across products.

Validation measurements follow the detailed timing contract in
`docs/validation_dataset.md`: representative warm-up, one complete hot Monte
Carlo call, symmetric prepared-input boundaries, explicit GPU
synchronization, and identical duplicated workloads across all four engines.
Selective duplication or mixing one-call and median timings in a single chart
is invalid.

Run each validation result generator as its own Python process. Reusing one
interpreter across C++ CUDA, PyTorch GPU, C++ CPU, and PyTorch CPU generators is
not a valid benchmark environment because runtime and allocator state leak
between engines.

Every validation engine must use a representative untimed warm-up before
starting the wall clock. This initializes CUDA runtimes and workspaces, PyTorch
allocators and worker pools, and OpenMP teams. GPU warm-ups must also exercise
the representative tensor shape, random-generation path, and lazily loaded
kernels. Startup cost may be reported separately, but it must not be mixed with
steady-state pricing time or compared with an already initialized engine.

A generic warm-up kernel is not sufficient. The untimed call must enter the
real model/product implementation and cover its representative tensor shapes,
dtype, random-number path when present, payoff, reduction, and output transfer.
Every new engine must expose this behavior before its timed call.
Validation may repeat hot calls and report their median, but production result
generation performs one pricing pass only; benchmark repetition must never
multiply the cost of a large production database.

## Rough Heston Convention

The library represents rough Heston with

```text
K(t) = t^(H-1/2) / Gamma(H+1/2),
V_t = v0 + integral K(t-s) [kappa(theta-V_s) ds + gamma sqrt(V_s) dW_s^V].
```

Production uses one eight-factor positive-exponential approximation of `K`,
geometric Laplace-measure quadrature, and explicit lifted Euler with full
truncation. CPU, CUDA, and PyTorch must use the same factor count, nodes,
weights, time grid, and update order. This is a reproducible numerical model;
validation across engines does not remove its time-discretization or kernel-
approximation error.

American-option regressions must use a state sufficient for the simulated
Markov model, even when the immediate payoff only depends on spot. Black-Scholes
may use spot alone. Heston uses `(S_t, V_t)`. The eight-factor rough Heston lift
uses `(S_t, Y_t^1, ..., Y_t^8)`; omitting the factors defines a different,
restricted exercise policy and is not accepted as the reference database.
The exact state, basis functions, normalization, and regression regularization
must be declared under `exercise_policy.basis` in every result YAML.

For Heston, the reference basis is `1`, `L1(S_t/K)`, `L2(S_t/K)`, `V_t/theta`,
`(V_t/theta)^2`, and `L1(S_t/K) V_t/theta`. For rough Heston, it is `1`,
`L1(S_t/K)`, `L2(S_t/K)`, and each normalized Markovian factor
`Y_t^i/theta`. Both use a relative ridge of `1e-10` on the normal matrix.

Primary references are Abi Jaber and El Euch (2019), *Multifactor
Approximation of Rough Volatility Models*, and Longstaff and Schwartz (2001)
for American exercise.

## Production Audit

After generating the production C++ GPU result database:

1. slice the first 100 model rows;
2. slice the first 100 curve rows when the result uses a curve database;
3. slice the first 100 product rows;
4. reprice those aligned rows with C++ GPU, C++ CPU, PyTorch GPU, and PyTorch CPU;
5. run the matching validation notebook, which reads the first 100 production
   result rows directly from the production JSON.

The notebook must match the standard validation format: stored production vs
regenerated C++ GPU, C++ CPU vs C++ GPU, PyTorch CPU vs PyTorch GPU, and C++ GPU
vs PyTorch GPU, plus timing and path reconstruction when available.

### Slicing And Repricing Ownership

Parameter slicing is independent of pricing. Validation model generators call
`tools.registry.model.slicing`, and validation product generators call
`tools.registry.product.slicing`. Both use the deterministic implementation in
`tools.registry.common.slicing`; neither may import a model/product result
module.

Each `tools.registry.result.<model>.<product>` module contains both production
pricing orchestration and validation repricing orchestration. Do not create a
parallel `<product>_audit.py` module. Do not copy production results into the
validation registry: the notebook reads the production head directly, while
the four validation result generators invoke the real C++/CUDA/PyTorch source
implementations on the sliced parameter databases.

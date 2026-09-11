# Model-sample dataset generation

This contract defines published model-only datasets used for generative-model
training. Availability, bindings, recipes, parameter laws, and observables are
declared by the
[typed capability manifest](../tools/codegen/pricing_bindings/capability_manifest.py),
not inferred from the presence of a dynamics file.

## Published shapes

Each available model has two independent recipes with three million terminal
samples:

```text
samples_01:      12,000 parameter rows x 250 paths
samples_02:   3,000,000 parameter rows x   1 path
```

The recipes use independent parameter, maturity, and dynamics seeds. They use
the same plausible core parameter law; the stress tail reserved for pricing
datasets is excluded from training samples.

Each row is autonomous and contains:

- the complete model parameters;
- `maturity_days`;
- `T = maturity_days / 252`;
- the declared terminal observables under `values`.

`maturity_days` is sampled uniformly from the integer business days 63 through
504. The JSON is flat: it contains no model table, join key, or intermediate
parameter dataset. `samples_01` intentionally repeats each parameter row for
its 250 conditional paths.

For fitted short-rate models, observables describe the underlying factors,
not curve-adjusted rates. In particular, CIR++ `state` is the nonnegative CIR
factor `y(t)`; `initial_state` is `y0`, and reconstructing `r(t)=y(t)+phi(t)`
requires an independent curve. The observable description records this distinction.

## Source of truth

Each generated `generator.cpp` performs this sequence:

1. generate plausible parameters directly in a contiguous typed vector;
2. generate one maturity for each flattened sample;
3. run the declared CUDA sampling engine;
4. stream the flat JSON without a three-million-row DOM;
5. write adjacent YAML describing the executed recipe.

The generator is authoritative. YAML records seeds, bounds, shape, numerical
method, observables, and output location; it never drives generation.
`parameter_sampling` records the ordered latent proposals, every conditional
draw and deterministic reconstruction, constants absent from the proposals,
and intermediate expressions used by acceptance. These are descriptions of
the existing factory, not a second executable parameter sampler. A rejected
proposal advances to the next proposal key; it is not clipped or partially redrawn.

Bindings, recipe helpers, and both recipes are generated as described in the
[code-generation entry point](../tools/codegen/pricing_bindings/README.md).
A recipe must not reimplement model dynamics or create an ad hoc binding.

## Randomness and time

Parameter, maturity, and dynamics generation use independent versioned
Philox-4x32-10 domains. Integer maturity draws use rejection sampling, avoiding
modulo bias. Dynamics use the parameter row as key context and the path index
as counter, so batch and launch geometry do not change logical samples.

Public launchers accept integer business days. Exact-transition models advance
directly to the requested date. Discretized models derive their fixed `dt`
from the globally declared `steps_per_year`; the production sample recipes use
two numerical steps per business day where the model contract requires it.

## CUDA engine contracts

The generated binding selects one engine from the capability manifest:

| Engine family | Composition |
|---|---|
| Exact or fixed-step Markovian | `ModelSamplingPolicy` with a terminal schedule and observation policy |
| Rough Markovian N-factor | Host-prepared fixed-factor dynamics with the common Markovian sampler |
| Gaussian-Volterra FFT | Volterra kernel policy, model path policy, terminal schedule, and observation policy |

Markovian engines use grid-stride execution for one path per parameter and a
parameter-block strategy for conditional path packages. A conditional block
prepares or loads its parameter-dependent row once, then covers all paths,
including nonmultiples of the block size.

N-factor fitting and matrix preparation run on the host. The GPU receives one
prepared row per parameter and performs no nonlinear fit or matrix exponential.
Model-specific approximation rules belong to the corresponding model
reference, not this common dataset contract.

The Volterra FFT engine uses one persistent block per parameter row, prepares
one kernel spectrum, and reuses it across packed path pairs. Its workspace is
bounded by the declared maximum calendar and chunk policy; it does not grow
with the total published sample count.

Output is sample-major. Multi-observation buffers, when present, use explicit
time-major structure-of-arrays indexing. Kernels do not allocate per-thread
arrays proportional to the full calendar.

## Memory guards

Before allocating, a generator estimates model and observable storage. It
refuses a plan above 70% of currently available host RAM or 85% of available
device memory. These limits detect an unsuitable run; they do not alter the
published shape.
On Linux, available RAM is `MemAvailable` from `/proc/meminfo`, including
reclaimable page cache; free pages are only a conservative fallback when that
field is unavailable. The guard runs before parameter/prepared-input allocation.
Do not flush system caches or remove the guard to admit a sample recipe.

If an implementation batches internally, `sample_offset` and
`launch_sample_count` preserve the original Philox indices. Repartitioning a
run must reproduce identical outputs.
For packed FFT samples, reconstruct both valid paths in each pair even when
the requested batch contains only one partner. Only publication is masked by
the batch range; zero-padding a valid neighbour changes floating-point
rounding. A missing partner beyond `paths_per_parameter` remains zero-padded.

## Required verification modes

Every generated sample recipe supports:

- `--smoke-test`: write 1,000 rows to
  `/tmp/ai_factory_sample_smoke/`, reload the JSON, and validate identity,
  maturity bounds, dimensions, and finite values;
- `--preflight`: execute the full production shape without publishing, replay
  with a second admissible geometry, and require identical maturities and
  observables. Markovian/N-factor engines vary threads per block. FFT keeps
  its compiled block dimensions and decreases the actual grid block count by
  one; a single-block case explicitly reports identical geometry instead.

The preflight prints `MODEL_SAMPLE_PREFLIGHT ` followed by one JSON object with
the shape, geometries, and wall/kernel timings. It qualifies a machine run; it
does not replace the versioned dataset.
`primary_launch` and `replay_launch` describe actual block dimensions, grid
counts and the covered row count; `replay_scope` identifies the dimension
really varied. FFT dimensions are queried from the compiled cuFFTDx type,
not inferred from the generic sample thread setting. Both FFT layouts use
the persistent parameter-block strategy. Smoke metadata describes the smoke
launch, not a hypothetical production grid. Existing published YAML is not
rewritten by code generation or this metadata correction.

## Performance and portability

Launch geometry comes from a named CMake tuning profile and is recorded in
execution metadata. The checked-in values and baseline describe the SM89
reference GPU only.

For another GPU or toolchain, build one native architecture, inspect registers,
spills, local/shared memory and occupancy, then run the complete
[performance regression protocol](performance-regression-protocol.md). Publish
a separate tuning profile and baseline while preserving the same parameter
keys, path counters, time grid, and numerical checks.

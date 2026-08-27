# Model-sample dataset generation

This document fixes the common contract for the generative-training datasets
under `catalog/model/<asset_class>/<model>/samples/`.

## Dataset shape

Every model exposes two production recipes of exactly 3,000,000 terminal
samples:

```text
samples_01:    12,000 model parameter rows * 250 paths = 3,000,000 samples
samples_02: 3,000,000 model parameter rows *   1 path  = 3,000,000 samples
```

The two recipes use independent parameter, maturity, and sample seeds. They
share the same plausible parameter laws so that their difference is purely
the conditional sampling layout.

Every row has its own maturity `T = maturity_days / 252`, where
`maturity_days` is sampled uniformly from all 442 integer business days in
`[63, 504]`. Thus the support is exactly `{63/252, ..., 504/252}`: one quarter
to two years under the repository-wide 252-day convention.

The final JSON is flat. Every entry in `samples` is an autonomous training row
containing `parameters`, `maturity_days`, `T`, and scalar terminal `values`.
`samples_01` intentionally repeats each parameter object 250 times. There is
no `models` table, `model_id` join, or standalone intermediate parameter JSON.

## Source of truth

`generator.cpp` is executable source of truth. It performs these operations in
order:

1. generate plausible parameters directly in one contiguous typed vector;
2. expose those rows through a `DeviceParameterSource` (or use a model
   `GeneratedParameterSource` when no host copy is required);
3. generate one discrete-uniform maturity per flattened sample in the common
   CUDA sampling engine;
4. simulate directly into the final output views;
5. stream the complete JSON without constructing a 3M-row DOM;
6. write the adjacent YAML describing the recipe that was executed.

The YAML never drives parameter generation. Its parameter bounds are the
plausible core bounds used for the ordinary 90% of the corresponding pricing
parameter dataset; the 10% stress tail is intentionally excluded from GAN
training data.

Parameter, calendar and dynamics generation use independent Philox-4x32-10
domains and independent public seeds. Integer calendar draws use rejection
sampling and therefore have no modulo bias. The dynamics key is derived from
the parameter row and the path index is its Philox counter, so fixed seeds are
reproducible independently of launch geometry and batch boundaries.

## CUDA launch contract

Every ordinary Markov model provides a thin `sample.cuh` / `sample.cu`
binding built from the common policies:

```cpp
using Schedule =
    simulation::ExactTransitionTerminalSchedule<DynamicsPolicy>;
using SamplingPolicy = sample::ModelSamplingPolicy<
    Schedule,
    sample::SpotSampleObservation<DynamicsPolicy>
>;
static_assert(sample::SamplingPolicy<SamplingPolicy>);
```

The public model launchers accept integer business days, never an arbitrary
floating-point `T` or caller-selected `dt`. Exact Markov and Levy models use
`day_count / 252` directly. Heston, Bates, CEV and Schobel-Zhu use exactly two
numerical transitions per business day, hence `dt = 1/504`. Gaussian-Volterra
models use the same convention and deterministic seed separation through a
second, block-cooperative cuFFTDx sampling engine.

`launch_samples_cuda<SamplingPolicy>` composes a parameter source, a calendar
source and an observation policy. Its automatic execution strategy uses a
persistent grid-stride kernel for one path per parameter and a parameter-block
kernel for conditional packages. In the latter, one block prepares the
parameter-dependent model coefficients once in shared memory, then its lanes
cover all paths, including non-multiples of the block size such as `P = 250`.
Both strategies preserve the same logical `(parameter_index, path_index)`.

Rough Heston uses the same Markov execution strategies after its fixed-factor
lift has been prepared on the host:

```cpp
using SamplingPolicy = sample::ModelSamplingPolicy<
    simulation::FixedStepTerminalSchedule<DynamicsPolicy<FactorCount>>,
    sample::SpotSampleObservation<DynamicsPolicy<FactorCount>>
>;
static_assert(sample::ExternallyPreparedSamplingPolicy<SamplingPolicy>);
```

`launch_prepared_samples_cuda<SamplingPolicy>` loads one
`PreparedDynamics<FactorCount>` per parameter row. For `P > 1`, the block
loads that row once into shared memory before simulating its conditional paths.
The host preparation uses `dt = 1/504` and an approximation horizon covering
the maximum generated maturity. Factor counts 2, 3 and 7 match the European
pricing interface; no kernel fit or matrix exponential runs on the device.

Rough Bergomi and rough SABR compose the corresponding engine as follows:

```cpp
using SamplingPolicy = sample::VolterraFftModelSamplingPolicy<
    volterra::FractionalHybridDriverPolicy,
    PathPolicy,
    volterra::TerminalHybridSchedule,
    sample::SpotSampleObservation<PathPolicy>
>;
static_assert(sample::VolterraFftSamplingPolicy<
    SamplingPolicy,
    volterra::HybridTimeConfiguration
>);
```

`sample::volterra_fft::launch_samples_cuda<SamplingPolicy>` always uses one
persistent block per parameter row. The block prepares the fractional driver,
model coefficients, driver variances and one FFT spectrum, then reuses them for
all packed pairs of conditional paths. Each path still loads its own calendar;
the maximum calendar horizon selects the FFT length. The canonical support up
to 504 business days requires at most 1,008 steps and a padded FFT length of
2,048, so spectra and convolution outputs remain block-local and no workspace
grows with the parameter count.

Terminal values are sample-major. Calendar observations are written directly
as time-major SoA, `values[observation * sample_count + sample]`; generated
observation days use the same layout. No kernel allocates per-thread parallel
calendar/time/value arrays.

## Memory and smoke tests

The generator estimates contiguous model and observable storage before any
large allocation. It refuses a plan above 70% of currently available host RAM
or 85% of available device memory. This guard is diagnostic protection, not a
batching policy; the intended 3M rows fit in one in-memory execution on the
target GPU. When batching is needed, `sample_offset` and
`launch_sample_count` retain the same logical Philox indices, so batch
boundaries do not change the dataset.

Every generator accepts `--smoke-test`. That mode preserves the production
paths-per-model layout (4 * 250 for `samples_01`, 1,000 * 1 for `samples_02`),
writes exactly 1,000 rows below
`/tmp/ai_factory_sample_smoke/`, rejects non-finite outputs, then parses the
JSON again and verifies the flat-row schema, maturity bounds, identity
`T = maturity_days / 252`, dimensions, and finite outputs. The YAML still
documents the production 3M recipe.

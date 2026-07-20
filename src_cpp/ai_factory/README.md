# C++ Source Layout

This tree separates the reproducible CPU reference engines from the optimized
CUDA engines while keeping the public C API stable for Python registry tools.

## CPU

- `cpu/common/`: CPU utilities shared by several engines. Philox lives here.
- `common/fixed_income/`: inline Nelson-Siegel, Hull-White, CIR, and swap
  formulas shared by CPU
  and CUDA without introducing extra kernel launches.
- `cpu/common/payoffs/`: small pure payoff formulas used by CPU engines. Keep these
  simple and contract-focused.
- `cpu/<model>/common.*`: model dynamics and path statistics for one model.
- `cpu/<model>/lookback_fixed_calls.*`: optimized CPU pricing for that
  model/product pair.

CPU model/product engines should call shared payoff helpers when that improves
auditability. For example, Heston and Rough Bergomi lookback engines should use
the same `lookback_fixed_calls` payoff formula.

## CUDA

- `cuda/common/api.cuh`: shared CUDA entry points that are not model-specific.
- `cuda/common/philox.cuh`: shared inline Philox RNG used by CUDA kernels.
- `cuda/common/reductions.cuh`: shared inline block-reduction helpers.
- `cuda/common/runtime.cuh`: shared CUDA error checking and workspace helpers.
- `cuda/common/types.cuh`: shared row, output, timing, and workspace types.
- `cuda/<model>/api.cuh`: model-specific CUDA entry points exposed to the C API.
- `cuda/heston/dynamics.cuh`: Heston device-side dynamics shared by pricing
  and path export.
- `cuda/heston/lookback_fixed_calls.cu`: Heston fixed-lookback pricing and
  delta kernels.
- `cuda/heston/paths.cu`: Heston path-export kernels.
- `cuda/rough_bergomi/lookback_fixed_calls.cu`: Rough Bergomi fixed-lookback
  pricing kernels.

CUDA kernels are allowed to specialize model and payoff logic in one pass for
performance. Shared CUDA code should stay limited to reusable mechanics such as
RNG, reductions, and launch/runtime mechanics.

The default build emits native code for `sm_70` and `sm_86`, covering V100 and
recent NVIDIA GPUs without silently inheriting a stale CMake architecture cache.
Override this deliberately when targeting another fleet:

```bash
cmake -S src_cpp -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAI_FACTORY_CUDA_ARCHITECTURES="70;86"
```

Pricing entry points reuse grow-only device buffers and CUDA timing events
across exact-step groups. Path-export entry points remain independent tools:
they reconstruct trajectories on demand and are not part of the fused pricing
hot path.

## C API

- `c_api/`: stable C ABI loaded from Python generation tools.

Registry scripts should depend on the C API and source paths, not on private
implementation details inside a kernel file.

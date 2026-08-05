// Fixed regression bases used by Longstaff-Schwartz continuation estimates.
#pragma once

#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz {

struct TwoFactorLaguerreBasis {
    static constexpr std::size_t kSize = 6U;
    static constexpr std::size_t kGramValueCount =
        kSize * (kSize + 1U) / 2U;
    static constexpr std::size_t kRegressionValueCount =
        kGramValueCount + kSize + 1U;

    float values[kSize];

    __device__ __forceinline__ static TwoFactorLaguerreBasis evaluate(
        float primary_state,
        float secondary_state
    );
};

__device__ __forceinline__ float laguerre_0(float value);
__device__ __forceinline__ float laguerre_1(float value);
__device__ __forceinline__ float laguerre_2(float value);

}  // namespace ai_factory::workbench::longstaff_schwartz

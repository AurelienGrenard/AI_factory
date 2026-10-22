// Compact host/device selection of source dataset rows for one result row.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench {

inline constexpr std::uint64_t kSampledCurveSeed = 20261014ULL;

enum class RowIndexMode : std::uint8_t {
    Strided,
    SampledCurve,
};

__host__ __device__ constexpr std::uint64_t sampled_curve_row(
    std::uint64_t result_index, std::uint64_t curve_count
) noexcept {
    std::uint64_t value = kSampledCurveSeed + result_index
        + 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value % curve_count;
}

struct RowIndexSelector {
    std::uint32_t stride = 1U;
    std::uint32_t source_count = 1U;
    RowIndexMode mode = RowIndexMode::Strided;

    __host__ __device__ constexpr std::uint32_t select(
        std::uint32_t result_index
    ) const noexcept {
        if (mode == RowIndexMode::SampledCurve)
            return static_cast<std::uint32_t>(
                sampled_curve_row(result_index, source_count));
        return (result_index / stride) % source_count;
    }
};

// Selectors are ordered by source slot, independently of the expression tree.
// The descriptor is copied once; no result-row mapping table is required.
struct RowIndexPlanView {
    const RowIndexSelector* selectors = nullptr;
    std::uint32_t source_count = 0U;
    std::uint32_t result_count = 0U;

    __host__ __device__ constexpr std::uint32_t select(
        std::uint32_t result_index, std::uint32_t source_slot
    ) const noexcept {
        return selectors[source_slot].select(result_index);
    }
};

}  // namespace ai_factory::workbench

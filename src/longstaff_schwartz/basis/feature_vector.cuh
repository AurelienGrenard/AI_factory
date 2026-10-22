// Fixed-size regression inputs and feature vectors for small device bases.
#pragma once

#include <cstddef>
#include <type_traits>

namespace ai_factory::workbench::longstaff_schwartz::basis {

struct TwoFactorInput {
    float primary;
    float secondary;
};

template<std::size_t Size>
requires (Size > 0U)
struct FeatureVector {
    static constexpr std::size_t kSize = Size;
    float values[Size];
};

static_assert(std::is_trivially_copyable_v<TwoFactorInput>);
static_assert(std::is_trivially_copyable_v<FeatureVector<1U>>);

}  // namespace ai_factory::workbench::longstaff_schwartz::basis

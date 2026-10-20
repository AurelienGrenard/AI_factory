// One result-row mapping shared by device pricing and offline provenance.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo {

struct MultiAssetPriceRow {
    std::uint32_t scenario_index;
    std::uint32_t product_index;
    std::uint32_t correlation_index;
    std::uint32_t curve_index;
    float discount_factor;
};

static_assert(std::is_trivially_copyable_v<MultiAssetPriceRow>);

}  // namespace ai_factory::workbench::monte_carlo

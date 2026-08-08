// Vasicek dataset row and host-side JSON loader.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::vasicek {

// Coefficients in dr_t = a * (b - r_t) dt + sigma dW_t.
struct VasicekProcessParameters {
    float mean_reversion;
    float long_term_mean;
    float volatility;
};

// Standalone Vasicek short-rate process together with r(0).
struct VasicekModelParameters {
    VasicekProcessParameters process;
    float initial_state;
};

static_assert(std::is_trivially_copyable_v<VasicekModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<VasicekModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::vasicek

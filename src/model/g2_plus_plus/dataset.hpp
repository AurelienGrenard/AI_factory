// G2++ dataset row and host-side JSON loader.
#pragma once

#include "model/g2/dataset.hpp"

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::model::g2_plus_plus {

// Curve-independent process parameters of one centered G2++ model.
struct G2PlusPlusModelParameters {
    model::g2::G2ProcessParameters process;
};

static_assert(std::is_trivially_copyable_v<G2PlusPlusModelParameters>);

// Load every model row from JSON into one contiguous FP32 vector.
std::vector<G2PlusPlusModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::model::g2_plus_plus

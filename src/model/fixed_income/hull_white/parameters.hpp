// Hull-White parameters shared by host loaders and curve-fitted analytics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::hull_white {

// Model parameters independent of the initial discount curve.
struct ModelParameters {
    float mean_reversion;
    float volatility;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::hull_white

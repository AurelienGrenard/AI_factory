// NIG model parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

// Risk-neutral exponential NIG inputs without a redundant location parameter.
struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float alpha;
    float beta;
    float delta;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian

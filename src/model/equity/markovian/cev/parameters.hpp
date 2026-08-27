// CEV model parameters shared by host loaders and CUDA dynamics.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::cev {

struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float sigma;
    float beta;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::cev

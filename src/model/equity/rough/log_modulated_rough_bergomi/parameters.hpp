#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float xi_0;
    float eta;
    float hurst_exponent;
    float rho;
    float log_modulation_scale;
    float log_modulation_power;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi

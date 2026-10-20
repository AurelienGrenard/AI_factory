// Deterministic flat short rate for the correlated scenario engine.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::model::fixed_income::constant_rate {

struct ModelParameters {
    float rate;
};
static_assert(std::is_trivially_copyable_v<ModelParameters>);

}  // namespace ai_factory::workbench::model::fixed_income::constant_rate

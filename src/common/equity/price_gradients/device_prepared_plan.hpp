// Equity compatibility names for the asset-agnostic compact host plan.
#pragma once

#include "common/price_gradients/device_prepared_plan.hpp"

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<typename Preparation>
using DevicePreparedSensitivityPlan =
    pg::DevicePreparedSensitivityPlan<Preparation>;

using pg::prepare_device_sensitivities;

}  // namespace ai_factory::workbench::equity::price_gradients

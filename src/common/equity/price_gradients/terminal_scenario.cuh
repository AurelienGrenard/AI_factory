// Row-local terminal scenario built from a central model/product pair.
#pragma once

#include "common/price_gradients/central_requirement.hpp"

#include <cstdint>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<typename Model, typename Product>
struct Scenario {
    Model model;
    Product product;
    float maturity_years;
    std::uint32_t step_count;
    // Homogeneous dynamics can keep the central simulated spot and rescale it.
    float simulation_spot;
    float spot_scale;
    bool reuse_central;
    pg::CentralRequirement central_requirement;
    // Normalized Brownian endpoint in a three-normal, central-first coupling.
    float normal_weights[3];
};

}  // namespace ai_factory::workbench::equity::price_gradients

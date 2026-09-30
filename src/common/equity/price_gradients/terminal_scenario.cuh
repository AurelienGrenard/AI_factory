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
    // Immutable central grid metadata lets analytical path policies move only
    // the terminal date while keeping every earlier contractual date fixed.
    float central_maturity_years;
    std::uint32_t central_step_count;
    // Homogeneous dynamics can keep the central simulated spot and rescale it.
    float simulation_spot;
    float spot_scale;
    bool reuse_central;
    pg::CentralRequirement central_requirement;
    // Normalized Brownian endpoint in a four-normal, central-first coupling.
    // Four normals cover the central plus the three nodes of a second-order
    // one-sided stencil. They are read only by exact Gaussian transitions.
    float normal_weights[4];
};

}  // namespace ai_factory::workbench::equity::price_gradients

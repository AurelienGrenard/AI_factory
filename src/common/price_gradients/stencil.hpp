// Host construction of represented centered or quadratic one-sided stencils.
#pragma once

#include "common/price_gradients/configuration.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::price_gradients {

struct Stencil {
    float central;
    float first;
    float second;
    float displacement;
    float represented_width;
    // One-sided derivative = first_weight*(Y1-Y0) + second_weight*(Y2-Y0).
    float first_weight;
    float second_weight;
    StencilKind kind;
};

template<typename Admissible, typename Endpoints>
Stencil prepare_stencil(float central, BumpConfiguration configuration, Admissible admissible, Endpoints endpoint) {
    if (!std::isfinite(central) || !admissible(central))
        throw std::invalid_argument("Central parameter row is outside its domain.");
    if (!std::isfinite(configuration.displacement) || !(configuration.displacement > 0.0f)
        || (configuration.scale != BumpScale::absolute && configuration.scale != BumpScale::relative)
        || (configuration.boundary != BoundaryRule::central_only
            && configuration.boundary != BoundaryRule::central_then_one_sided_order2))
        throw std::invalid_argument("Invalid bump configuration.");
    const float h = configuration.scale == BumpScale::relative
        ? static_cast<float>(configuration.displacement) * std::abs(central) : static_cast<float>(configuration.displacement);
    if (!std::isfinite(h) || !(h > 0.0f))
        throw std::invalid_argument("Bump displacement is zero or non-finite; use an absolute bump at zero.");
    const auto valid = [&](float value) { return std::isfinite(value) && admissible(value); };
    const float lower = endpoint(-1), upper = endpoint(1);
    if (!(lower < central) || !(central < upper))
        throw std::invalid_argument("Bump does not produce distinct represented FP32 endpoints.");
    const float width = upper - lower;
    if (valid(lower) && valid(upper) && std::isfinite(width))
        return {central, lower, upper, h, width, 0.0f, 0.0f, StencilKind::centered};
    if (configuration.boundary == BoundaryRule::central_only)
        throw std::invalid_argument("Centered bump leaves the parameter domain.");
    for (int direction : {1, -1}) {
        const float first = endpoint(direction), second = endpoint(2 * direction);
        if (!valid(first) || !valid(second) || first == second) continue;
        const double a = static_cast<double>(first) - central;
        const double b = static_cast<double>(second) - central;
        const float w1 = static_cast<float>(b / (a * (b - a)));
        const float w2 = static_cast<float>(-a / (b * (b - a)));
        if (!std::isfinite(w1) || !std::isfinite(w2)) continue;
        return {central, first, second, h, second - first, w1, w2,
                direction > 0 ? StencilKind::forward : StencilKind::backward};
    }
    throw std::invalid_argument("No admissible second-order stencil at the requested displacement.");
}

template<typename Admissible>
Stencil prepare_stencil(float central, BumpConfiguration configuration, Admissible admissible) {
    const float h = configuration.scale == BumpScale::relative
        ? static_cast<float>(configuration.displacement) * std::abs(central) : static_cast<float>(configuration.displacement);
    return prepare_stencil(central, configuration, admissible, [&](int multiple) {
        return static_cast<float>(static_cast<double>(central) + static_cast<double>(multiple) * h);
    });
}

}  // namespace ai_factory::workbench::price_gradients

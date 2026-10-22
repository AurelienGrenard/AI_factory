// Host selection of sensitivity coordinates and explicit finite-difference rules.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ai_factory::workbench::price_gradients {

enum class BumpScale { absolute, relative };
enum class BoundaryRule { central_only, central_then_one_sided_order2 };
enum class StencilKind { centered, forward, backward };

struct BumpConfiguration {
    // Displacement on EACH side, unlike the historical spot full width.
    double displacement;
    BumpScale scale = BumpScale::relative;
    BoundaryRule boundary = BoundaryRule::central_then_one_sided_order2;
};

struct Sensitivity {
    std::string parameter;
    BumpConfiguration bump;
};

struct PriceGradientConfiguration {
    // Order is public: output column i belongs to sensitivities[i].
    std::vector<Sensitivity> sensitivities;

    void validate(std::size_t maximum_count) const {
        if (sensitivities.size() > maximum_count)
            throw std::invalid_argument("Too many selected price sensitivities.");
        std::unordered_set<std::string> names;
        for (const auto& sensitivity : sensitivities) {
            const auto& bump = sensitivity.bump;
            if (sensitivity.parameter.empty() || !names.insert(sensitivity.parameter).second)
                throw std::invalid_argument("Empty or duplicate sensitivity parameter: " + sensitivity.parameter);
            if (!std::isfinite(bump.displacement) || !(bump.displacement > 0.0f))
                throw std::invalid_argument("Bump displacement must be finite and positive: " + sensitivity.parameter);
            if (bump.scale != BumpScale::absolute && bump.scale != BumpScale::relative)
                throw std::invalid_argument("Unknown bump scale.");
            if (bump.boundary != BoundaryRule::central_only
                && bump.boundary != BoundaryRule::central_then_one_sided_order2)
                throw std::invalid_argument("Unknown bump boundary rule.");
        }
    }
};

}  // namespace ai_factory::workbench::price_gradients

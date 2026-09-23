// American-option parameter-domain predicate and host validation diagnostics.
#pragma once

#include "product/american_option/parameters.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product::american_option {

inline const char* parameter_domain_error(
    const AmericanOptionParameters& parameters
) {
    if (!std::isfinite(parameters.strike) || !(parameters.strike > 0.0f))
        return "strike must be finite and positive.";
    if (parameters.maturity_days == 0U)
        return "maturity must be a positive business-day count.";
    if (parameters.exercise_interval_days == 0U
        || !(parameters.exercise_interval_days < parameters.maturity_days))
        return "exercise_interval must be positive and below maturity.";
    return nullptr;
}

inline bool valid_parameters(const AmericanOptionParameters& parameters) {
    return parameter_domain_error(parameters) == nullptr;
}

inline void validate_parameters(
    const AmericanOptionParameters& parameters,
    const std::string& prefix
) {
    if (const char* error = parameter_domain_error(parameters))
        throw std::invalid_argument(prefix + error);
}

}  // namespace ai_factory::workbench::product::american_option

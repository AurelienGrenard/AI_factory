// Exercise-schedule helpers shared by regular-grid early-exercise products.
#pragma once

#include <cstdint>

namespace ai_factory::workbench::longstaff_schwartz {

// Count the maturity-anchored dates T-(E-1)delta, ..., T.
std::uint32_t maturity_anchored_exercise_count(
    float maturity,
    float exercise_interval,
    const char* product_name
);

}  // namespace ai_factory::workbench::longstaff_schwartz

// Regular maturity-anchored exercise-schedule implementation.
#include "common/longstaff_schwartz/exercise_schedule.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::longstaff_schwartz {

std::uint32_t maturity_anchored_exercise_count(
    float maturity,
    float exercise_interval,
    const char* product_name
) {
    const float raw_count = maturity / exercise_interval;
    const float adjusted =
        raw_count - 8.0f * FLT_EPSILON * std::max(raw_count, 1.0f);
    const double count = std::ceil(static_cast<double>(adjusted));
    if (!(count >= 1.0)
        || count > static_cast<double>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error(
            std::string(product_name) + " exercise count exceeds uint32_t."
        );
    }
    return static_cast<std::uint32_t>(count);
}

}  // namespace ai_factory::workbench::longstaff_schwartz

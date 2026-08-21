// Regular maturity-anchored exercise-schedule implementation.
#include "common/longstaff_schwartz/exercise_schedule.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::longstaff_schwartz {

std::uint32_t maturity_anchored_exercise_count(
    std::uint32_t maturity,
    std::uint32_t exercise_interval,
    const char* product_name
) {
    if (maturity == 0U || exercise_interval == 0U) {
        throw std::invalid_argument(
            std::string(product_name)
            + " maturity and exercise_interval must be positive."
        );
    }
    return 1U + (maturity - 1U) / exercise_interval;
}

}  // namespace ai_factory::workbench::longstaff_schwartz

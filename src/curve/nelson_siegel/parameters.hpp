// Compact Nelson-Siegel curve parameters shared by host and device code.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::curve::nelson_siegel {

struct NelsonSiegelParameters {
    float beta0;
    float beta1;
    float beta2;
    float tau;
};

static_assert(std::is_trivially_copyable_v<NelsonSiegelParameters>);

}  // namespace ai_factory::workbench::curve::nelson_siegel

// Compact Svensson curve parameters shared by host and device code.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::curve::svensson {

struct SvenssonParameters {
    float beta0;
    float beta1;
    float beta2;
    float beta3;
    float tau1;
    float tau2;
};

static_assert(std::is_trivially_copyable_v<SvenssonParameters>);

}  // namespace ai_factory::workbench::curve::svensson

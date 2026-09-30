// Compact flat continuously compounded curve parameter shared by host and device code.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::curve::flat {

struct FlatCurveParameters {
    float rate;
};

static_assert(std::is_trivially_copyable_v<FlatCurveParameters>);

}  // namespace ai_factory::workbench::curve::flat

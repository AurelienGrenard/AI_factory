// Empty and future auxiliary device views used by pricing policies.
#pragma once

#include <type_traits>

namespace ai_factory::workbench::equity {

struct EmptyDeviceInputs {};

static_assert(std::is_trivially_copyable_v<EmptyDeviceInputs>);

}  // namespace ai_factory::workbench::equity

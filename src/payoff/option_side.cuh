// Compile-time orientation shared by call/put option families.
#pragma once

namespace ai_factory::workbench {

enum class OptionSide {
    call,
    put,
};

// Stable label used by host-side diagnostics and metadata.
constexpr const char* option_side_name(OptionSide side) noexcept {
    return side == OptionSide::call ? "call" : "put";
}

}  // namespace ai_factory::workbench

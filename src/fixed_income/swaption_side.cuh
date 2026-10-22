// Compile-time orientation of an option on a payer or receiver swap.
#pragma once

namespace ai_factory::workbench {

enum class SwaptionSide {
    payer,
    receiver,
};

constexpr const char* swaption_side_name(SwaptionSide side) noexcept {
    return side == SwaptionSide::payer ? "payer" : "receiver";
}

}  // namespace ai_factory::workbench

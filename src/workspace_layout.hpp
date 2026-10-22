// Checked byte-layout primitives shared by caller-owned workspaces.
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::workspace_layout {

inline std::size_t align_offset(
    std::size_t offset,
    std::size_t alignment,
    const char* overflow_message
) {
    const auto remainder = offset % alignment;
    if (remainder == 0U) return offset;
    const auto padding = alignment - remainder;
    if (offset > std::numeric_limits<std::size_t>::max() - padding) {
        throw std::overflow_error(overflow_message);
    }
    return offset + padding;
}

template<typename Value>
std::size_t append_array(
    std::size_t& offset,
    std::size_t count,
    const char* overflow_message
) {
    offset = align_offset(offset, alignof(Value), overflow_message);
    const auto begin = offset;
    const auto remaining =
        std::numeric_limits<std::size_t>::max() - offset;
    if (count > remaining / sizeof(Value)) {
        throw std::overflow_error(overflow_message);
    }
    offset += count * sizeof(Value);
    return begin;
}

}  // namespace ai_factory::workbench::workspace_layout

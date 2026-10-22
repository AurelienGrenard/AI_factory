// Compile nested aligned/Cartesian dataset joins into independent strides.
#pragma once

#include "common/row_index_plan.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ai_factory::workbench {

struct RowIndexExpression {
    enum class Kind : std::uint8_t { Source, Aligned, Cartesian, SampledCurve };

    Kind kind;
    std::size_t slot = 0U;
    std::size_t count = 0U;
    std::vector<RowIndexExpression> children;

    static RowIndexExpression source(std::size_t slot, std::size_t count) {
        return {Kind::Source, slot, count, {}};
    }
    static RowIndexExpression aligned(
        std::vector<RowIndexExpression> children
    ) {
        return {Kind::Aligned, 0U, 0U, std::move(children)};
    }
    static RowIndexExpression cartesian(
        std::vector<RowIndexExpression> children
    ) {
        return {Kind::Cartesian, 0U, 0U, std::move(children)};
    }
    // Select with the existing deterministic curve hash for each result row.
    static RowIndexExpression sampled_curve(
        std::size_t slot, std::size_t curve_count, std::size_t result_count
    ) {
        return {Kind::SampledCurve, slot, result_count,
                {source(slot, curve_count)}};
    }
};

struct CompiledRowIndexPlan {
    std::vector<RowIndexSelector> selectors;
    std::uint32_t result_count = 0U;

    RowIndexPlanView view() const noexcept {
        return {selectors.data(),
                static_cast<std::uint32_t>(selectors.size()), result_count};
    }
};

namespace detail {

inline std::size_t checked_product(std::size_t first, std::size_t second) {
    if (second && first > std::numeric_limits<std::uint32_t>::max() / second)
        throw std::overflow_error("Result row count exceeds uint32.");
    return first * second;
}

inline std::size_t cardinality(const RowIndexExpression& node) {
    using Kind = RowIndexExpression::Kind;
    if (node.kind == Kind::Source || node.kind == Kind::SampledCurve) {
        if (node.count == 0U || node.count >
            std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Invalid source cardinality.");
        if (node.kind == Kind::SampledCurve &&
            (node.children.size() != 1U
             || node.children[0].kind != Kind::Source
             || node.children[0].slot != node.slot
             || node.children[0].count == 0U
             || node.children[0].count >
                std::numeric_limits<std::uint32_t>::max()))
            throw std::invalid_argument("Invalid sampled curve source.");
        return node.count;
    }
    if (node.children.empty())
        throw std::invalid_argument("Row index group cannot be empty.");
    std::size_t size = node.kind == Kind::Cartesian ? 1U : 0U;
    for (const auto& child : node.children) {
        const auto child_size = cardinality(child);
        if (node.kind == Kind::Cartesian) {
            size = checked_product(size, child_size);
        } else if (size == 0U) {
            size = child_size;
        } else if (size != child_size) {
            throw std::invalid_argument(
                "Aligned datasets must have the same row count.");
        }
    }
    return size;
}

inline void flatten(
    const RowIndexExpression& node, std::size_t stride,
    std::vector<RowIndexSelector>& selectors,
    std::vector<bool>& seen
) {
    using Kind = RowIndexExpression::Kind;
    if (node.kind == Kind::Source || node.kind == Kind::SampledCurve) {
        const auto source_count = node.kind == Kind::Source
            ? node.count : node.children[0].count;
        if (node.slot >= selectors.size() || seen[node.slot]
            || stride > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Invalid or duplicate source slot.");
        seen[node.slot] = true;
        selectors[node.slot] = {
            static_cast<std::uint32_t>(stride),
            static_cast<std::uint32_t>(source_count),
            node.kind == Kind::SampledCurve
                ? RowIndexMode::SampledCurve : RowIndexMode::Strided
        };
        return;
    }
    if (node.kind == Kind::Aligned) {
        for (const auto& child : node.children)
            flatten(child, stride, selectors, seen);
        return;
    }
    std::size_t child_stride = stride;
    for (std::size_t i = node.children.size(); i-- > 0U;) {
        flatten(node.children[i], child_stride, selectors, seen);
        child_stride = checked_product(child_stride,
                                       cardinality(node.children[i]));
    }
}

}  // namespace detail

inline CompiledRowIndexPlan compile_row_index_plan(
    const RowIndexExpression& expression, std::size_t source_count
) {
    if (source_count == 0U || source_count >
        std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid source slot count.");
    const auto rows = detail::cardinality(expression);
    CompiledRowIndexPlan result{
        std::vector<RowIndexSelector>(source_count),
        static_cast<std::uint32_t>(rows)
    };
    std::vector<bool> seen(source_count, false);
    detail::flatten(expression, 1U, result.selectors, seen);
    for (bool value : seen)
        if (!value) throw std::invalid_argument("Missing source slot.");
    return result;
}

}  // namespace ai_factory::workbench

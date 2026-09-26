// Derivative orders and host selection of requested sensitivity outputs.
#pragma once

#include "common/price_gradients/sensitivity_pair.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ai_factory::workbench::price_gradients {

enum class SensitivityOrders : std::uint8_t {
    none = 0U,
    first = 1U,
    second = 2U,
    first_and_second = 3U,
};

struct SensitivityRequest {
    SensitivityOrders orders = SensitivityOrders::first;
    std::vector<std::uint16_t> first;
    std::vector<std::uint16_t> diagonal_second;
    std::vector<SensitivityPair> mixed_second;
    bool all_mixed_second = false;
    bool selected_outputs = false;

    SensitivityRequest() = default;

    SensitivityRequest(SensitivityOrders requested_orders)
        : orders(requested_orders) {}

    static SensitivityRequest selected(
        std::vector<std::uint16_t> first,
        std::vector<std::uint16_t> diagonal_second,
        std::vector<SensitivityPair> mixed_second
    ) {
        SensitivityRequest result{};
        result.first = std::move(first);
        result.diagonal_second = std::move(diagonal_second);
        result.mixed_second = std::move(mixed_second);
        result.selected_outputs = true;
        const bool has_first = !result.first.empty();
        const bool has_second = !result.diagonal_second.empty();
        result.orders = has_first && has_second
            ? SensitivityOrders::first_and_second
            : has_second
                ? SensitivityOrders::second
                : has_first || !result.mixed_second.empty()
                    ? SensitivityOrders::first
                    : SensitivityOrders::none;
        return result;
    }

    static SensitivityRequest full_hessian() {
        SensitivityRequest result{SensitivityOrders::first_and_second};
        result.all_mixed_second = true;
        return result;
    }

    bool has_selected_outputs() const noexcept {
        return selected_outputs;
    }
};

template<SensitivityOrders Orders>
inline constexpr bool requests_first_v =
    Orders == SensitivityOrders::first
    || Orders == SensitivityOrders::first_and_second;

template<SensitivityOrders Orders>
inline constexpr bool requests_second_v =
    Orders == SensitivityOrders::second
    || Orders == SensitivityOrders::first_and_second;

template<SensitivityOrders Orders>
struct SensitivityTraits;

template<>
struct SensitivityTraits<SensitivityOrders::first> {
    static constexpr std::size_t node_capacity = 3U;
};

template<>
struct SensitivityTraits<SensitivityOrders::second> {
    static constexpr std::size_t node_capacity = 4U;
};

template<>
struct SensitivityTraits<SensitivityOrders::first_and_second> {
    static constexpr std::size_t node_capacity = 4U;
};

}  // namespace ai_factory::workbench::price_gradients

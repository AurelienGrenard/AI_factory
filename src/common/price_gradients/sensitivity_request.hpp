// Compile-time derivative orders and their bounded finite-difference capacity.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::price_gradients {

enum class SensitivityOrders : std::uint8_t {
    none = 0U,
    first = 1U,
    second = 2U,
    first_and_second = 3U,
};

struct SensitivityRequest {
    SensitivityOrders orders = SensitivityOrders::first;
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

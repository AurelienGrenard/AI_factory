#include "common/row_index_plan.hpp"
#include "common/result_index.cuh"

#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    using Expression = RowIndexExpression;
    const auto tree = Expression::aligned({
        Expression::cartesian({
            Expression::aligned({
                Expression::source(0U, 3U),
                Expression::source(1U, 3U)
            }),
            Expression::source(2U, 4U)
        }),
        Expression::source(3U, 12U)
    });
    const auto plan = compile_row_index_plan(tree, 4U);
    if (plan.result_count != 12U) throw std::runtime_error("row count");
    for (std::uint32_t row = 0U; row < 12U; ++row) {
        const auto view = plan.view();
        if (view.select(row, 0U) != row / 4U
            || view.select(row, 1U) != row / 4U
            || view.select(row, 2U) != row % 4U
            || view.select(row, 3U) != row)
            throw std::runtime_error("nested aligned/Cartesian mapping");
    }
    const auto sampled = compile_row_index_plan(
        Expression::aligned({
            Expression::cartesian({
                Expression::source(0U, 3U),
                Expression::source(1U, 4U)
            }),
            Expression::sampled_curve(2U, 5U, 12U)
        }), 3U);
    for (std::uint32_t row = 0U; row < 12U; ++row) {
        if (sampled.view().select(row, 2U)
            != sampled_curve_index(row, 5U))
            throw std::runtime_error("sampled curve parity");
    }
    // Exact five-equity/CIR slot order: five models, rate, product, correlation.
    const auto large = compile_row_index_plan(
        Expression::cartesian({
            Expression::source(6U, 100U),
            Expression::source(7U, 10U),
            Expression::aligned({
                Expression::source(0U, 100U),
                Expression::source(1U, 100U),
                Expression::source(2U, 100U),
                Expression::source(3U, 100U),
                Expression::source(4U, 100U),
                Expression::source(5U, 100U)
            })
        }), 8U);
    if (large.result_count != 100000U
        || large.selectors.size() != 8U)
        throw std::runtime_error("large compact plan");
    for (std::uint32_t j = 0U; j < large.result_count; ++j) {
        const auto view = large.view();
        for (std::uint32_t model = 0U; model < 6U; ++model)
            if (view.select(j, model) != j % 100U)
                throw std::runtime_error("asset or rate row mapping");
        if (view.select(j, 6U) != j / 1000U
            || view.select(j, 7U) != (j / 100U) % 10U)
            throw std::runtime_error("product or correlation row mapping");
    }
    bool rejected = false;
    try {
        (void)compile_row_index_plan(Expression::aligned({
            Expression::source(0U, 3U),
            Expression::source(1U, 4U)
        }), 2U);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("unequal aligned group accepted");
    rejected = false;
    try {
        (void)compile_row_index_plan(Expression::aligned({
            Expression::source(0U, 2U),
            Expression::source(0U, 2U)
        }), 2U);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("duplicate source slot accepted");
    rejected = false;
    try {
        (void)compile_row_index_plan(Expression::cartesian({
            Expression::source(0U, 65536U),
            Expression::source(1U, 65536U)
        }), 2U);
    } catch (const std::overflow_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("result count overflow accepted");
}

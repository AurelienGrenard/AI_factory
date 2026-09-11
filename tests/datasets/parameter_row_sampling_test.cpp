// Verify deterministic uniform and Cartesian row sampling for dataset generators.
#include "tools/datasets/sampling.hpp"

#include <stdexcept>

int main() {
    using namespace ai_factory::workbench::datasets;
    const auto first = uniform_rows(10U, 42U, {{"x", -1.0f, 1.0f}});
    const auto second = uniform_rows(10U, 42U, {{"x", -1.0f, 1.0f}});
    if (first.rows != second.rows) {
        throw std::runtime_error("sampling is not deterministic");
    }
    const auto grid = cartesian_grid({
        {"x", {1.0f, 2.0f}},
        {"y", {3.0f, 4.0f, 5.0f}},
    });
    if (grid.rows.size() != 6U) {
        throw std::runtime_error("Cartesian cardinality is incorrect");
    }
    const auto regimes = core_stress_rows(
        uniform_rows(9U, 1U, {{"x", 0.0f, 1.0f}}),
        uniform_rows(1U, 2U, {{"x", 2.0f, 3.0f}})
    );
    if (regimes.rows.size() != 10U
        || regimes.construction.at("core_share") != 0.9) {
        throw std::runtime_error("90/10 regime assembly is incorrect");
    }
}

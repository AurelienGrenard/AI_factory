// Execution strategy for finite-difference Monte Carlo sensitivities.
#pragma once

#include <stdexcept>
#include <string_view>

namespace ai_factory::workbench::price_gradients {

enum class SensitivityStrategy {
    mono,
    node_graph,
};

constexpr std::string_view to_string(SensitivityStrategy strategy) {
    switch (strategy) {
    case SensitivityStrategy::mono:
        return "mono";
    case SensitivityStrategy::node_graph:
        return "node_graph";
    }
    throw std::invalid_argument("Unknown sensitivity strategy.");
}

inline SensitivityStrategy sensitivity_strategy(
    std::string_view name
) {
    if (name == "mono") return SensitivityStrategy::mono;
    if (name == "node_graph") {
        return SensitivityStrategy::node_graph;
    }
    throw std::invalid_argument(
        "Sensitivity strategy must be mono or node_graph."
    );
}

}  // namespace ai_factory::workbench::price_gradients

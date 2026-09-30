// Price-gradient artifact contract, including selected coordinates and represented stencils.
#pragma once
#include "common/longstaff_schwartz/price_gradients/exercise_replay_strategy.hpp"
#include "common/price_construction.cuh"
#include "common/price_gradients/stencil.hpp"
#include "common/price_gradients/mixed_sensitivity_stencil.cuh"
#include "common/price_gradients/sensitivity_request.hpp"
#include "common/price_gradients/sensitivity_graph_plan.hpp"
#include "common/price_gradients/sensitivity_stencil.cuh"
#include "common/price_gradients/time_configuration.hpp"
#include "common/price_gradients/sensitivity_strategy.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <optional>
#include <vector>

namespace ai_factory::workbench::datasets::price_gradients {
struct Recipe {
    std::filesystem::path model_input, product_input, dataset, catalog;
    std::string url, source_price_recipe;
    PriceConstruction construction;
    ::ai_factory::workbench::price_gradients::PriceGradientConfiguration configuration;
    ::ai_factory::workbench::price_gradients::TimeConfiguration time;
    bool exact_transition;
    ::ai_factory::workbench::price_gradients::SensitivityOrders orders =
        ::ai_factory::workbench::price_gradients::SensitivityOrders::first;
    std::filesystem::path curve_input;
    ::ai_factory::workbench::price_gradients::SensitivityStrategy
        sensitivity_strategy =
            ::ai_factory::workbench::price_gradients::
                SensitivityStrategy::mono;
    std::optional<
        ::ai_factory::workbench::longstaff_schwartz::price_gradients::
            ExerciseReplayStrategy
    > exercise_replay;
    std::optional<
        ::ai_factory::workbench::price_gradients::SensitivityRequest
    > sensitivity_request;
};
void append_exercise_replay_metadata(
    const Recipe&,
    nlohmann::ordered_json&
);
struct Results {
    std::vector<float> prices, price_errors, gradients, gradient_errors;
    std::vector<::ai_factory::workbench::price_gradients::Stencil> stencils;
    nlohmann::ordered_json execution;
    double wall_seconds = 0, kernel_seconds = 0;
    std::vector<float> diagonal_hessians, diagonal_hessian_errors;
    std::vector<::ai_factory::workbench::price_gradients::SensitivityStencil<4U>>
        diagonal_stencils;
    ::ai_factory::workbench::price_gradients::SensitivityGraphPlan
        sensitivity_graph;
    std::vector<float> mixed_hessians, mixed_hessian_errors;
    std::vector<
        ::ai_factory::workbench::price_gradients::MixedSensitivityStencil
    > mixed_stencils;
};
nlohmann::ordered_json sensitivity_metadata(const Recipe&, bool stochastic);
void write_dataset(const Recipe&, const Results&);
}  // namespace ai_factory::workbench::datasets::price_gradients

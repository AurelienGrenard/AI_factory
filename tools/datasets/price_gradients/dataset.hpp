// Price-gradient artifact contract, including selected coordinates and represented stencils.
#pragma once
#include "common/price_construction.cuh"
#include "common/price_gradients/stencil.hpp"
#include "common/equity/price_gradients/scenarios.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <vector>

namespace ai_factory::workbench::datasets::price_gradients {
struct Recipe {
    std::filesystem::path model_input, product_input, dataset, catalog;
    std::string url, source_price_recipe;
    PriceConstruction construction;
    ::ai_factory::workbench::price_gradients::PriceGradientConfiguration configuration;
    equity::price_gradients::TimeConfiguration time;
    bool exact_transition;
};
struct Results {
    std::vector<float> prices, price_errors, gradients, gradient_errors;
    std::vector<::ai_factory::workbench::price_gradients::Stencil> stencils;
    nlohmann::ordered_json execution;
    double wall_seconds = 0, kernel_seconds = 0;
};
nlohmann::ordered_json sensitivity_metadata(const Recipe&);
void write_dataset(const Recipe&, const Results&);
}  // namespace ai_factory::workbench::datasets::price_gradients

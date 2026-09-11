// Price-delta artifact contract; CUDA execution and certification are separate.
#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

struct PriceDeltaRecipe {
    std::filesystem::path model_input, product_input, dataset, catalog;
    std::string url, source_price_recipe, delta_method;
    double relative_bump_width = .01;
    unsigned simulation_steps_per_day = 0;
};

struct PriceDeltaResults {
    std::vector<float> prices, price_errors, deltas, delta_errors;
    std::vector<float> lower_spots, upper_spots, bump_widths;
    nlohmann::ordered_json execution;
    double wall_seconds = 0, kernel_seconds = 0;
};

void write_price_delta_dataset(const PriceDeltaRecipe&, const PriceDeltaResults&);

}  // namespace ai_factory::workbench::datasets

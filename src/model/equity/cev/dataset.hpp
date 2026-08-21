// CEV local-volatility model row.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::cev {

struct ModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float sigma;
    float beta;
};

static_assert(std::is_trivially_copyable_v<ModelParameters>);

std::vector<ModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::cev

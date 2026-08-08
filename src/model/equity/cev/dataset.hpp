// CEV local-volatility model row.
#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::cev {

struct CevModelParameters {
    float spot;
    float risk_free_rate;
    float dividend_yield;
    float sigma;
    float beta;
};

static_assert(std::is_trivially_copyable_v<CevModelParameters>);

std::vector<CevModelParameters> load_models(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::cev

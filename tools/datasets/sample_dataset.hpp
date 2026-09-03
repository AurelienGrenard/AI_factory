// Streaming assembly and publication of model-only sample artifacts.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

struct ModelSampleSeeds {
    std::uint64_t parameters;
    std::uint64_t schedule;
    std::uint64_t dynamics;
};

struct ModelSampleRecipe {
    std::string database_id;
    std::string model_family;
    std::filesystem::path dataset_path;
    std::filesystem::path catalog_path;
    std::string url;
    std::size_t production_parameter_count = 0U;
    std::size_t production_paths_per_parameter = 0U;
    std::uint32_t minimum_maturity_days = 0U;
    std::uint32_t maximum_maturity_days = 0U;
    ModelSampleSeeds seeds{};
    std::string numerical_method;
    nlohmann::ordered_json parameter_sampling;
    nlohmann::ordered_json output_descriptions;
    nlohmann::ordered_json time_grid;
};

struct ModelSampleExecution {
    std::size_t parameter_count = 0U;
    std::size_t paths_per_parameter = 0U;
    bool smoke_test = false;
    double wall_seconds = 0.0;
    double kernel_seconds = 0.0;
    nlohmann::ordered_json cuda_execution;
};

struct NamedSampleValues {
    std::string name;
    const std::vector<float>* values = nullptr;
};

using ParameterJsonFunction =
    std::function<nlohmann::ordered_json(std::size_t)>;

void write_model_sample_dataset(
    const ModelSampleRecipe& recipe,
    const ModelSampleExecution& execution,
    const ParameterJsonFunction& parameter_json,
    const std::vector<std::uint32_t>& maturity_days,
    const std::vector<NamedSampleValues>& outputs
);

void validate_model_sample_dataset_file(
    const std::filesystem::path& dataset_path,
    std::size_t expected_row_count,
    std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days
);

}  // namespace ai_factory::workbench::datasets

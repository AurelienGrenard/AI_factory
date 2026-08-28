// Streaming sample-result assembly; CUDA execution and model laws live elsewhere.
#include "tools/datasets/sample_dataset.hpp"

#include "common/sample/validation.cuh"
#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/sampling.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace ai_factory::workbench::datasets {
namespace {

std::size_t checked_sample_count(
    std::size_t parameter_count,
    std::size_t paths_per_parameter
) {
    return sample::sample_count(parameter_count, paths_per_parameter);
}

void validate_recipe(const ModelSampleRecipe& recipe) {
    if (recipe.database_id.empty() || recipe.model_family.empty()
        || recipe.numerical_method.empty()) {
        throw std::invalid_argument(
            "A sample recipe requires non-empty identifiers and method."
        );
    }
    validate_dataset_url(recipe.url);
    (void)checked_sample_count(
        recipe.production_parameter_count,
        recipe.production_paths_per_parameter
    );
    if (recipe.minimum_maturity_days == 0U
        || recipe.minimum_maturity_days > recipe.maximum_maturity_days) {
        throw std::invalid_argument(
            "Sample maturity bounds must be positive and ordered."
        );
    }
    if (!recipe.parameter_sampling.is_object()
        || !recipe.output_descriptions.is_object()
        || recipe.output_descriptions.empty()
        || !recipe.time_grid.is_object()) {
        throw std::invalid_argument(
            "Sample parameter, output and time-grid metadata are required."
        );
    }
}

void write_streamed_json(
    const ModelSampleRecipe& recipe,
    const ModelSampleExecution& execution,
    const ParameterJsonFunction& parameter_json,
    const std::vector<std::uint32_t>& maturity_days,
    const std::vector<NamedSampleValues>& outputs,
    std::size_t row_count
) {
    std::filesystem::create_directories(recipe.dataset_path.parent_path());
    std::ofstream output(recipe.dataset_path);
    if (!output) {
        throw std::runtime_error(
            "Cannot open sample JSON file: " + recipe.dataset_path.string()
        );
    }

    nlohmann::ordered_json envelope = {
        {"database_id", recipe.database_id},
        {"model_family", recipe.model_family},
        {"catalog", recipe.catalog_path.parent_path().generic_string()},
        {"url", recipe.url},
        {"row_count", row_count},
        {"time_convention", {
            {"unit", "business_day"},
            {"days_per_year", kBusinessDaysPerYear},
        }},
        {"construction", {
            {"parameter_count", execution.parameter_count},
            {"paths_per_parameter", execution.paths_per_parameter},
            {"row_order", "parameter-major, then path-major"},
            {"parameter_sampling", recipe.parameter_sampling},
            {"maturity_sampling", {
                {"distribution", "discrete uniform"},
                {"minimum_days", recipe.minimum_maturity_days},
                {"maximum_days", recipe.maximum_maturity_days},
            }},
        }},
        {"seeds", {
            {"parameters", recipe.seeds.parameters},
            {"schedule", recipe.seeds.schedule},
            {"dynamics", recipe.seeds.dynamics},
        }},
        {"timing", {
            {"wall_seconds", execution.wall_seconds},
            {"kernel_seconds", execution.kernel_seconds},
        }},
    };
    std::string prefix = envelope.dump(2);
    prefix.pop_back();
    output << prefix << ",\n  \"samples\": [\n";

    for (std::size_t sample_index = 0U;
         sample_index < row_count;
         ++sample_index) {
        const std::uint32_t days = maturity_days[sample_index];
        if (days < recipe.minimum_maturity_days
            || days > recipe.maximum_maturity_days) {
            throw std::runtime_error(
                "Generated sample maturity lies outside recipe bounds."
            );
        }
        const std::size_t parameter_index =
            sample_index / execution.paths_per_parameter;
        nlohmann::ordered_json values = nlohmann::ordered_json::object();
        for (const NamedSampleValues& named : outputs) {
            const float value = named.values->at(sample_index);
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Generated sample output '" + named.name
                    + "' is not finite at row "
                    + std::to_string(sample_index + 1U) + "."
                );
            }
            values[named.name] = value;
        }
        const nlohmann::ordered_json parameters =
            parameter_json(parameter_index);
        if (!parameters.is_object() || parameters.empty()) {
            throw std::runtime_error(
                "A serialized sample parameter row must be a non-empty object."
            );
        }
        const nlohmann::ordered_json row = {
            {"id", format_row_id(sample_index)},
            {"parameters", parameters},
            {"maturity_days", days},
            {
                "T",
                static_cast<float>(days)
                    / static_cast<float>(kBusinessDaysPerYear)
            },
            {"values", std::move(values)},
        };
        output << "    " << row.dump();
        if (sample_index + 1U != row_count) output << ',';
        output << '\n';
    }
    output << "  ]\n}\n";
    if (!output) {
        throw std::runtime_error(
            "Cannot write sample JSON file: " + recipe.dataset_path.string()
        );
    }
}

}  // namespace

void write_model_sample_dataset(
    const ModelSampleRecipe& recipe,
    const ModelSampleExecution& execution,
    const ParameterJsonFunction& parameter_json,
    const std::vector<std::uint32_t>& maturity_days,
    const std::vector<NamedSampleValues>& outputs
) {
    validate_recipe(recipe);
    const std::size_t row_count = checked_sample_count(
        execution.parameter_count,
        execution.paths_per_parameter
    );
    if (!parameter_json || maturity_days.size() != row_count
        || outputs.empty()) {
        throw std::invalid_argument(
            "Sample parameters, maturities and outputs must match the run."
        );
    }
    std::unordered_set<std::string> output_names;
    for (const NamedSampleValues& output : outputs) {
        if (output.name.empty() || output.values == nullptr
            || output.values->size() != row_count
            || !output_names.insert(output.name).second) {
            throw std::invalid_argument(
                "Sample outputs require unique names and complete vectors."
            );
        }
    }
    if (!execution.cuda_execution.is_object()
        || execution.cuda_execution.empty()
        || !std::isfinite(execution.wall_seconds)
        || execution.wall_seconds < 0.0
        || !std::isfinite(execution.kernel_seconds)
        || execution.kernel_seconds < 0.0) {
        throw std::invalid_argument(
            "Sample execution metadata and timings must be valid."
        );
    }

    write_streamed_json(
        recipe,
        execution,
        parameter_json,
        maturity_days,
        outputs,
        row_count
    );

    const std::size_t production_row_count = checked_sample_count(
        recipe.production_parameter_count,
        recipe.production_paths_per_parameter
    );
    nlohmann::ordered_json summary = {
        {"dataset_kind", "model samples"},
        {"model", recipe.model_family},
        {"numerical_method", recipe.numerical_method},
        {"implementation", "CUDA"},
        {"device", "gpu"},
        {"random_generator", "Philox-4x32-10"},
        {"parameter_count", recipe.production_parameter_count},
        {"paths_per_parameter", recipe.production_paths_per_parameter},
        {"row_order", "parameter-major, then path-major"},
    };
    for (const auto& [name, value] : execution.cuda_execution.items()) {
        summary[name] = value;
    }
    nlohmann::ordered_json catalog = {
        {"title", recipe.model_family + " model samples " + recipe.database_id},
        {"database_id", recipe.database_id},
        {"catalog", recipe.catalog_path.parent_path().generic_string()},
        {"url", recipe.url},
        {"row_count", production_row_count},
        {"time_convention", {
            {"unit", "business_day"},
            {"days_per_year", kBusinessDaysPerYear},
        }},
        {"summary", std::move(summary)},
        {"construction", {
            {"parameter_sampling", recipe.parameter_sampling},
            {"maturity_sampling", {
                {"distribution", "discrete uniform without modulo bias"},
                {"support", "integer business days"},
                {"minimum_days", recipe.minimum_maturity_days},
                {"maximum_days", recipe.maximum_maturity_days},
                {"year_fraction", "maturity_days / 252"},
            }},
        }},
        {"seeds", {
            {"parameters", recipe.seeds.parameters},
            {"schedule", recipe.seeds.schedule},
            {"dynamics", recipe.seeds.dynamics},
        }},
        {"outputs", recipe.output_descriptions},
        {"time_grid", recipe.time_grid},
        {"timing", {
            {"wall_seconds", format_duration(execution.wall_seconds)},
            {"kernel_seconds", format_duration(execution.kernel_seconds)},
        }},
    };
    if (execution.smoke_test) {
        catalog["smoke_test"] = {
            {"enabled", true},
            {"executed_row_count", row_count},
            {"production_row_count", production_row_count},
        };
    }
    write_catalog_yaml(recipe.catalog_path, catalog);
}

void validate_model_sample_dataset_file(
    const std::filesystem::path& dataset_path,
    std::size_t expected_row_count,
    std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days
) {
    const nlohmann::ordered_json document = read_json_file(dataset_path);
    if (document.at("row_count").get<std::size_t>() != expected_row_count
        || document.at("samples").size() != expected_row_count
        || document.at("time_convention").at("days_per_year")
            .get<std::uint32_t>() != kBusinessDaysPerYear) {
        throw std::runtime_error("Invalid model-sample dataset envelope.");
    }
    for (const auto& row : document.at("samples")) {
        const std::uint32_t days =
            row.at("maturity_days").get<std::uint32_t>();
        const float time = row.at("T").get<float>();
        if (!row.at("parameters").is_object()
            || row.at("parameters").empty()
            || !row.at("values").is_object()
            || row.at("values").empty()
            || days < minimum_maturity_days
            || days > maximum_maturity_days
            || time != static_cast<float>(days)
                / static_cast<float>(kBusinessDaysPerYear)) {
            throw std::runtime_error("Invalid model-sample dataset row.");
        }
        for (const auto& [name, value] : row.at("values").items()) {
            (void)name;
            if (!std::isfinite(value.get<float>())) {
                throw std::runtime_error("Non-finite model-sample value.");
            }
        }
    }
}

}  // namespace ai_factory::workbench::datasets

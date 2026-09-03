// Shared host pipeline for co-terminal Bermudan-swaption price datasets.
#pragma once

#include "common/longstaff_schwartz/launch.cuh"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/datasets/price_dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/tuning_profile.hpp"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace ai_factory::workbench::datasets {

struct BermudanSwaptionGenerationConfiguration {
    std::size_t paths_per_price;
    unsigned int threads_per_block;
    std::size_t blocks_per_price;
    std::uint64_t seed;
    std::string numerical_method;
    std::string regression_basis;
    std::string state_variables;
    std::string delta_t;
    nlohmann::ordered_json time_discretization;
    std::filesystem::path dataset_path;
    std::filesystem::path catalog_path;
    std::string url;
    std::string cuda_label;
};

inline BermudanSwaptionGenerationConfiguration
make_bermudan_swaption_generation_configuration(
    const std::string& model,
    const std::string& side,
    std::size_t paths_per_price,
    std::uint64_t seed,
    const std::string& numerical_method,
    const std::string& regression_basis,
    const std::string& state_variables,
    const std::string& delta_t,
    nlohmann::ordered_json time_discretization
) {
    const std::string product = "bermudan_" + side + "_swaptions";
    const std::string database_id =
        model + "_01__" + product + "_01__01";
    const std::string relative =
        "model/fixed_income/" + model + "/prices/" + product + "/"
        + database_id;
    return {
        paths_per_price,
        offline::cuda_tuning::kEarlyExerciseThreadsPerBlock,
        offline::cuda_tuning::kFixedIncomeLsmBlocksPerPrice,
        seed,
        numerical_method,
        regression_basis,
        state_variables,
        delta_t,
        std::move(time_discretization),
        "datasets/" + relative + ".json",
        "catalog/" + relative + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/" + relative + ".json",
        model + " Bermudan " + side + " swaption",
    };
}

inline BermudanSwaptionGenerationConfiguration
make_fitted_bermudan_swaption_generation_configuration(
    const std::string& model,
    const std::string& curve,
    const std::string& side,
    std::size_t paths_per_price,
    std::uint64_t seed,
    const std::string& numerical_method,
    const std::string& regression_basis,
    const std::string& state_variables
) {
    const std::string product = "bermudan_" + side + "_swaptions";
    const std::string database_id = model + "_01__" + curve + "_01__"
        + product + "_01__01";
    const std::string relative = "model/fixed_income/" + model + "/prices/"
        + curve + "/" + product + "/" + database_id;
    return {
        paths_per_price,
        offline::cuda_tuning::kEarlyExerciseThreadsPerBlock,
        offline::cuda_tuning::kFixedIncomeLsmBlocksPerPrice,
        seed,
        numerical_method,
        regression_basis,
        state_variables,
        "",
        {{"time_day_fraction", "1 / 252"}},
        "datasets/" + relative + ".json",
        "catalog/" + relative + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/" + relative + ".json",
        model + " " + curve + " Bermudan " + side + " swaption",
    };
}

inline nlohmann::ordered_json bermudan_swaption_catalog_sections(
    const BermudanSwaptionGenerationConfiguration& configuration
) {
    return {
        {"outputs", {
            {
                "price",
                {{"estimator", "Longstaff-Schwartz discounted policy cashflow mean"}}
            },
            {
                "standard_error",
                {{
                    "estimator",
                    "Conditional standard error of discounted policy cashflows"
                }}
            },
        }},
        {"exercise_policy", {
            {"method", "Longstaff-Schwartz"},
            {"exercise_dates", "co-terminal regular Bermudan schedule"},
            {"regression_basis", configuration.regression_basis},
            {"regression_state", configuration.state_variables},
            {"regression_target", "next policy cashflow discounted pathwise"},
            {"in_the_money_paths_only", true},
            {"solver", "FP64 normal equations and Cholesky on GPU"},
        }},
    };
}

inline nlohmann::ordered_json bermudan_swaption_cuda_execution(
    const BermudanSwaptionGenerationConfiguration& configuration,
    const longstaff_schwartz::LaunchResult& execution
) {
    nlohmann::ordered_json result = {
        {"threads_per_block", configuration.threads_per_block},
        {"blocks_per_price", execution.blocks_per_price},
        {"batch_count", execution.batch_count},
        {"kernel_launch_count", execution.kernel_launch_count},
        {"workspace_bytes", execution.workspace_bytes},
        {
            "tuning_profile",
            offline::cuda_tuning::metadata("fixed_income_early_exercise")
        },
    };
    for (const auto& [name, value] : configuration.time_discretization.items()) {
        result[name] = value;
    }
    return result;
}

template<typename Model, typename Launcher>
void generate_bermudan_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const std::vector<product::BermudanSwaptionParameters>& products,
    Launcher launcher,
    const BermudanSwaptionGenerationConfiguration& configuration
) {
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    longstaff_schwartz::LaunchResult execution{};
    const auto run = offline::cuda::run_monte_carlo(
        offline::cuda::inputs(models, products),
        result_count,
        [&](auto& resources) {
            launcher(
                resources.template input<0U>(),
                1U,
                products.data(),
                resources.template input<1U>(),
                1U,
                1U,
                std::min<std::size_t>(
                    configuration.paths_per_price, 4'096U
                ),
                resources.prices(),
                resources.standard_errors()
            );
        },
        [&](auto& resources) {
            execution = launcher(
                resources.template input<0U>(),
                models.size(),
                products.data(),
                resources.template input<1U>(),
                products.size(),
                result_count,
                configuration.paths_per_price,
                resources.prices(),
                resources.standard_errors()
            );
            longstaff_schwartz::validate_regression_diagnostics(
                execution, configuration.cuda_label.c_str()
            );
        }
    );

    write_monte_carlo_price_dataset(
        model_dataset_path,
        product_dataset_path,
        construction,
        run.prices,
        run.standard_errors,
        "Philox",
        configuration.dataset_path,
        configuration.catalog_path,
        configuration.url,
        configuration.numerical_method,
        configuration.paths_per_price,
        configuration.delta_t,
        bermudan_swaption_cuda_execution(configuration, execution),
        bermudan_swaption_catalog_sections(configuration),
        configuration.seed,
        run.wall_seconds,
        execution.kernel_seconds
    );
    validate_price_dataset_file(configuration.dataset_path);
}

template<typename Model, typename Curve, typename Launcher>
void generate_bermudan_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& curve_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const std::vector<Curve>& curves,
    const std::vector<product::BermudanSwaptionParameters>& products,
    Launcher launcher,
    const BermudanSwaptionGenerationConfiguration& configuration
) {
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const std::size_t result_count = price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    longstaff_schwartz::LaunchResult execution{};
    const auto run = offline::cuda::run_monte_carlo(
        offline::cuda::inputs(models, curves, products),
        result_count,
        [&](auto& resources) {
            launcher(
                resources.template input<0U>(),
                1U,
                resources.template input<1U>(),
                1U,
                products.data(),
                resources.template input<2U>(),
                1U,
                1U,
                std::min<std::size_t>(
                    configuration.paths_per_price, 4'096U
                ),
                resources.prices(),
                resources.standard_errors()
            );
        },
        [&](auto& resources) {
            execution = launcher(
                resources.template input<0U>(),
                models.size(),
                resources.template input<1U>(),
                curves.size(),
                products.data(),
                resources.template input<2U>(),
                products.size(),
                result_count,
                configuration.paths_per_price,
                resources.prices(),
                resources.standard_errors()
            );
            longstaff_schwartz::validate_regression_diagnostics(
                execution, configuration.cuda_label.c_str()
            );
        }
    );

    write_monte_carlo_price_dataset(
        model_dataset_path,
        curve_dataset_path,
        product_dataset_path,
        construction,
        run.prices,
        run.standard_errors,
        "Philox",
        configuration.dataset_path,
        configuration.catalog_path,
        configuration.url,
        configuration.numerical_method,
        configuration.paths_per_price,
        configuration.delta_t,
        bermudan_swaption_cuda_execution(configuration, execution),
        bermudan_swaption_catalog_sections(configuration),
        configuration.seed,
        run.wall_seconds,
        execution.kernel_seconds
    );
    validate_price_dataset_file(configuration.dataset_path);
}

template<typename Model, typename Launcher>
void generate_exact_bermudan_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const std::vector<product::BermudanSwaptionParameters>& products,
    Launcher launcher,
    const BermudanSwaptionGenerationConfiguration& configuration
) {
    generate_bermudan_swaption_prices(
        model_dataset_path,
        product_dataset_path,
        models,
        products,
        [launcher, &configuration](
            const Model* device_models,
            std::size_t model_count,
            const product::BermudanSwaptionParameters* host_products,
            const product::BermudanSwaptionParameters* device_products,
            std::size_t product_count,
            std::size_t result_count,
            std::size_t paths_per_price,
            float* device_prices,
            float* device_standard_errors
        ) {
            return launcher(
                device_models,
                model_count,
                host_products,
                device_products,
                product_count,
                PriceConstruction::Aligned,
                result_count,
                paths_per_price,
                1.0f / 252.0f,
                configuration.threads_per_block,
                configuration.blocks_per_price,
                configuration.seed,
                device_prices,
                device_standard_errors
            );
        },
        configuration
    );
}

template<typename Model, typename Launcher>
void generate_fixed_step_bermudan_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const std::vector<product::BermudanSwaptionParameters>& products,
    Launcher launcher,
    float dt,
    std::uint32_t simulation_steps_per_day,
    const BermudanSwaptionGenerationConfiguration& configuration
) {
    generate_bermudan_swaption_prices(
        model_dataset_path,
        product_dataset_path,
        models,
        products,
        [launcher, dt, simulation_steps_per_day, &configuration](
            const Model* device_models,
            std::size_t model_count,
            const product::BermudanSwaptionParameters* host_products,
            const product::BermudanSwaptionParameters* device_products,
            std::size_t product_count,
            std::size_t result_count,
            std::size_t paths_per_price,
            float* device_prices,
            float* device_standard_errors
        ) {
            return launcher(
                device_models,
                model_count,
                host_products,
                device_products,
                product_count,
                PriceConstruction::Aligned,
                result_count,
                paths_per_price,
                dt,
                simulation_steps_per_day,
                configuration.threads_per_block,
                configuration.blocks_per_price,
                configuration.seed,
                device_prices,
                device_standard_errors
            );
        },
        configuration
    );
}

template<typename Model, typename Curve, typename Launcher>
void generate_exact_fitted_bermudan_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& curve_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const std::vector<Curve>& curves,
    const std::vector<product::BermudanSwaptionParameters>& products,
    Launcher launcher,
    const BermudanSwaptionGenerationConfiguration& configuration
) {
    generate_bermudan_swaption_prices(
        model_dataset_path,
        curve_dataset_path,
        product_dataset_path,
        models,
        curves,
        products,
        [launcher, &configuration](
            const Model* device_models,
            std::size_t model_count,
            const Curve* device_curves,
            std::size_t curve_count,
            const product::BermudanSwaptionParameters* host_products,
            const product::BermudanSwaptionParameters* device_products,
            std::size_t product_count,
            std::size_t result_count,
            std::size_t paths_per_price,
            float* device_prices,
            float* device_standard_errors
        ) {
            return launcher(
                device_models,
                model_count,
                device_curves,
                curve_count,
                host_products,
                device_products,
                product_count,
                PriceConstruction::Aligned,
                result_count,
                paths_per_price,
                1.0f / 252.0f,
                configuration.threads_per_block,
                configuration.blocks_per_price,
                configuration.seed,
                device_prices,
                device_standard_errors
            );
        },
        configuration
    );
}

}  // namespace ai_factory::workbench::datasets

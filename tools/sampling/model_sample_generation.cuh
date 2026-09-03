// Generic host orchestration for model-only CUDA sample recipes.
#pragma once

#include "common/check_cuda.cuh"
#include "common/sample/validation.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/tuning_profile.hpp"
#include "tools/datasets/sample_dataset.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace ai_factory::workbench::offline::sampling {

struct ModelSampleProfile {
    unsigned int threads_per_block = 0U;
    std::size_t block_count_limit = 0U;
    const char* engine_family = "model_sample";
};

struct ModelSampleShape {
    std::size_t parameter_count = 0U;
    std::size_t paths_per_parameter = 0U;
    bool smoke_test = false;
    bool preflight = false;
};

enum class ModelSampleRunMode { production, smoke_test, preflight };

inline ModelSampleRunMode model_sample_run_mode(int argc, char** argv) {
    if (argc == 1) return ModelSampleRunMode::production;
    if (argc == 2 && std::string(argv[1]) == "--smoke-test") {
        return ModelSampleRunMode::smoke_test;
    }
    if (argc == 2 && std::string(argv[1]) == "--preflight") {
        return ModelSampleRunMode::preflight;
    }
    throw std::invalid_argument(
        "A model-sample generator accepts only --smoke-test or --preflight."
    );
}

inline ModelSampleShape resolve_model_sample_shape(
    const datasets::ModelSampleRecipe& recipe,
    ModelSampleRunMode run_mode
) {
    if (run_mode != ModelSampleRunMode::smoke_test) {
        return {
            recipe.production_parameter_count,
            recipe.production_paths_per_parameter,
            false,
            run_mode == ModelSampleRunMode::preflight,
        };
    }
    if (recipe.production_paths_per_parameter == 250U) {
        return {4U, 250U, true, false};
    }
    if (recipe.production_paths_per_parameter == 1U) {
        return {1'000U, 1U, true, false};
    }
    throw std::invalid_argument(
        "Smoke tests support the canonical 250-path and one-path layouts."
    );
}

inline datasets::ModelSampleRecipe smoke_sample_recipe(
    datasets::ModelSampleRecipe recipe
) {
    const std::string model = recipe.dataset_path
        .parent_path().parent_path().filename().string();
    const std::filesystem::path directory =
        std::filesystem::path("/tmp/ai_factory_sample_smoke")
        / model / recipe.database_id;
    recipe.dataset_path = directory / (recipe.database_id + ".json");
    recipe.catalog_path = directory / "dataset.yaml";
    return recipe;
}

inline std::size_t checked_bytes(std::size_t count, std::size_t width) {
    if (width != 0U
        && count > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("Sample memory estimate exceeds size_t.");
    }
    return count * width;
}

inline void validate_sample_memory_plan(
    std::size_t host_bytes,
    std::size_t device_bytes
) {
    std::size_t free_device_bytes = 0U;
    std::size_t total_device_bytes = 0U;
    check_cuda(
        cudaMemGetInfo(&free_device_bytes, &total_device_bytes),
        "model-sample CUDA memory query"
    );
    (void)total_device_bytes;
    if (static_cast<long double>(device_bytes)
        > 0.85L * static_cast<long double>(free_device_bytes)) {
        throw std::runtime_error(
            "The model-sample plan exceeds 85% of free device memory."
        );
    }
#if defined(__linux__)
    const long pages = ::sysconf(_SC_AVPHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages > 0L && page_size > 0L) {
        const long double available =
            static_cast<long double>(pages)
            * static_cast<long double>(page_size);
        if (static_cast<long double>(host_bytes) > 0.70L * available) {
            throw std::runtime_error(
                "The model-sample plan exceeds 70% of available host memory."
            );
        }
    }
#else
    (void)host_bytes;
#endif
}

inline std::size_t model_sample_block_count(
    std::size_t launch_sample_count,
    std::size_t paths_per_parameter,
    unsigned int threads_per_block,
    std::size_t block_count_limit
) {
    if (launch_sample_count == 0U || paths_per_parameter == 0U
        || threads_per_block == 0U || block_count_limit == 0U) {
        throw std::invalid_argument(
            "A model-sample launch requires positive geometry."
        );
    }
    const std::size_t work_items = paths_per_parameter == 1U
        ? (launch_sample_count + threads_per_block - 1U)
            / threads_per_block
        : (launch_sample_count + paths_per_parameter - 1U)
            / paths_per_parameter;
    return std::max<std::size_t>(
        1U,
        std::min(work_items, block_count_limit)
    );
}

template<typename Parameters, typename DeviceInput, typename ParameterFactory,
         typename InputFactory, typename ParameterJson, typename Launcher>
int generate_model_sample_dataset_impl(
    int argc,
    char** argv,
    datasets::ModelSampleRecipe recipe,
    const ModelSampleProfile& profile,
    const std::vector<std::string>& output_names,
    ParameterFactory&& parameter_factory,
    InputFactory&& input_factory,
    ParameterJson&& parameter_json,
    Launcher&& launcher
) {
    const ModelSampleRunMode run_mode = model_sample_run_mode(argc, argv);
    const ModelSampleShape shape = resolve_model_sample_shape(
        recipe, run_mode
    );
    if (shape.smoke_test) recipe = smoke_sample_recipe(std::move(recipe));
    const std::size_t sample_count = sample::sample_count(
        shape.parameter_count,
        shape.paths_per_parameter
    );
    if (output_names.empty()) {
        throw std::invalid_argument(
            "A model-sample generator requires at least one observable."
        );
    }

    const auto wall_start = std::chrono::steady_clock::now();
    std::vector<Parameters> parameters = std::invoke(
        std::forward<ParameterFactory>(parameter_factory),
        shape.parameter_count,
        recipe.seeds.parameters
    );
    if (parameters.size() != shape.parameter_count) {
        throw std::runtime_error(
            "The parameter factory returned the wrong row count."
        );
    }
    auto device_inputs = std::invoke(
        std::forward<InputFactory>(input_factory),
        parameters,
        recipe.maximum_maturity_days
    );
    if (device_inputs.size() != shape.parameter_count) {
        throw std::runtime_error(
            "The device-input factory returned the wrong row count."
        );
    }

    const std::size_t parameter_bytes = checked_bytes(
        parameters.size(), sizeof(Parameters)
    );
    const std::size_t input_bytes = checked_bytes(
        device_inputs.size(), sizeof(DeviceInput)
    );
    const std::size_t maturity_bytes = checked_bytes(
        sample_count, sizeof(std::uint32_t)
    );
    const std::size_t observable_bytes = checked_bytes(
        checked_bytes(sample_count, output_names.size()),
        sizeof(float)
    );
    validate_sample_memory_plan(
        parameter_bytes
            + (std::same_as<Parameters, DeviceInput> ? 0U : input_bytes)
            + maturity_bytes + observable_bytes,
        input_bytes + maturity_bytes + observable_bytes
    );

    cuda::DeviceBuffer<DeviceInput> device_parameters(device_inputs.size());
    cuda::DeviceBuffer<std::uint32_t> device_maturity_days(sample_count);
    device_parameters.copy_from(device_inputs.data());
    std::vector<cuda::DeviceBuffer<float>> device_outputs;
    device_outputs.reserve(output_names.size());
    for (std::size_t index = 0U; index < output_names.size(); ++index) {
        device_outputs.emplace_back(sample_count);
    }
    std::vector<float*> output_pointers;
    output_pointers.reserve(device_outputs.size());
    for (auto& output : device_outputs) {
        output_pointers.push_back(output.data());
    }

    const auto invoke_launch = [&](
        std::size_t launch_count,
        unsigned int threads_per_block
    ) {
        const std::size_t block_count = model_sample_block_count(
            launch_count,
            shape.paths_per_parameter,
            threads_per_block,
            profile.block_count_limit
        );
        std::invoke(
            launcher,
            device_parameters.data(),
            shape.parameter_count,
            shape.paths_per_parameter,
            recipe.minimum_maturity_days,
            recipe.maximum_maturity_days,
            0U,
            launch_count,
            threads_per_block,
            block_count,
            recipe.seeds.schedule,
            recipe.seeds.dynamics,
            device_maturity_days.data(),
            std::span<float*>(output_pointers)
        );
    };

    const std::size_t warmup_count = std::min<std::size_t>(
        sample_count,
        shape.paths_per_parameter == 1U
            ? 1'000U
            : 4U * shape.paths_per_parameter
    );
    invoke_launch(warmup_count, profile.threads_per_block);
    check_cuda(cudaDeviceSynchronize(), "model-sample CUDA warmup");

    cuda::Event start;
    cuda::Event stop;
    check_cuda(cudaEventRecord(start.get()), "model-sample timer start");
    invoke_launch(sample_count, profile.threads_per_block);
    check_cuda(cudaEventRecord(stop.get()), "model-sample timer stop");
    check_cuda(cudaEventSynchronize(stop.get()), "model-sample timer wait");
    float kernel_milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(
            &kernel_milliseconds, start.get(), stop.get()
        ),
        "model-sample elapsed time"
    );

    std::vector<std::uint32_t> maturity_days(sample_count);
    device_maturity_days.copy_to(maturity_days.data());
    std::vector<std::vector<float>> output_values(
        output_names.size(),
        std::vector<float>(sample_count)
    );
    for (std::size_t index = 0U; index < device_outputs.size(); ++index) {
        device_outputs[index].copy_to(output_values[index].data());
    }
    if (shape.preflight) {
        const auto expected_maturity_days = maturity_days;
        const auto expected_output_values = output_values;
        const unsigned int replay_threads_per_block =
            profile.threads_per_block == 128U ? 256U : 128U;
        invoke_launch(sample_count, replay_threads_per_block);
        check_cuda(
            cudaDeviceSynchronize(),
            "model-sample preflight replay"
        );
        device_maturity_days.copy_to(maturity_days.data());
        for (std::size_t index = 0U; index < device_outputs.size(); ++index) {
            device_outputs[index].copy_to(output_values[index].data());
        }
        if (maturity_days != expected_maturity_days
            || output_values != expected_output_values) {
            throw std::runtime_error(
                "Model-sample preflight changed with launch geometry."
            );
        }
        for (const std::uint32_t maturity : maturity_days) {
            if (maturity < recipe.minimum_maturity_days
                || maturity > recipe.maximum_maturity_days) {
                throw std::runtime_error(
                    "Model-sample preflight maturity lies outside bounds."
                );
            }
        }
        for (const auto& output : output_values) {
            for (const float value : output) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "Model-sample preflight produced a non-finite value."
                    );
                }
            }
        }
        const double preflight_wall_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall_start
            ).count();
        std::cout
            << "MODEL_SAMPLE_PREFLIGHT rows=" << sample_count
            << " finite=true deterministic_replay=true"
            << " primary_threads_per_block=" << profile.threads_per_block
            << " replay_threads_per_block=" << replay_threads_per_block
            << " kernel_seconds="
            << static_cast<double>(kernel_milliseconds) * 1.0e-3
            << " wall_seconds=" << preflight_wall_seconds << '\n';
        return 0;
    }
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

    std::vector<datasets::NamedSampleValues> named_outputs;
    named_outputs.reserve(output_names.size());
    for (std::size_t index = 0U; index < output_names.size(); ++index) {
        named_outputs.push_back({output_names[index], &output_values[index]});
    }
    const std::size_t production_row_count = sample::sample_count(
        recipe.production_parameter_count,
        recipe.production_paths_per_parameter
    );
    const std::size_t full_block_count = model_sample_block_count(
        production_row_count,
        recipe.production_paths_per_parameter,
        profile.threads_per_block,
        profile.block_count_limit
    );
    datasets::write_model_sample_dataset(
        recipe,
        {
            shape.parameter_count,
            shape.paths_per_parameter,
            shape.smoke_test,
            wall_seconds,
            static_cast<double>(kernel_milliseconds) * 1.0e-3,
            {
                {"threads_per_block", profile.threads_per_block},
                {"block_count", full_block_count},
                {"kernel_launch_count", 1U},
                {
                    "execution_strategy",
                    recipe.production_paths_per_parameter == 1U
                        ? "persistent thread grid-stride"
                        : "persistent parameter-block"
                },
                {
                    "tuning_profile",
                    cuda_tuning::metadata(profile.engine_family)
                },
            },
        },
        [&](std::size_t parameter_index) {
            return std::invoke(
                parameter_json,
                parameters.at(parameter_index)
            );
        },
        maturity_days,
        named_outputs
    );
    if (shape.smoke_test) {
        datasets::validate_model_sample_dataset_file(
            recipe.dataset_path,
            sample_count,
            recipe.minimum_maturity_days,
            recipe.maximum_maturity_days
        );
    }
    return 0;
}

template<typename Parameters, typename ParameterFactory,
         typename ParameterJson, typename Launcher>
int generate_model_sample_dataset(
    int argc,
    char** argv,
    datasets::ModelSampleRecipe recipe,
    const ModelSampleProfile& profile,
    const std::vector<std::string>& output_names,
    ParameterFactory&& parameter_factory,
    ParameterJson&& parameter_json,
    Launcher&& launcher
) {
    return generate_model_sample_dataset_impl<Parameters, Parameters>(
        argc,
        argv,
        std::move(recipe),
        profile,
        output_names,
        std::forward<ParameterFactory>(parameter_factory),
        [](const std::vector<Parameters>& parameters, std::uint32_t) {
            return std::span<const Parameters>(parameters);
        },
        std::forward<ParameterJson>(parameter_json),
        std::forward<Launcher>(launcher)
    );
}

template<typename Parameters, typename DeviceInput, typename ParameterFactory,
         typename InputFactory, typename ParameterJson, typename Launcher>
int generate_prepared_model_sample_dataset(
    int argc,
    char** argv,
    datasets::ModelSampleRecipe recipe,
    const ModelSampleProfile& profile,
    const std::vector<std::string>& output_names,
    ParameterFactory&& parameter_factory,
    InputFactory&& input_factory,
    ParameterJson&& parameter_json,
    Launcher&& launcher
) {
    return generate_model_sample_dataset_impl<Parameters, DeviceInput>(
        argc,
        argv,
        std::move(recipe),
        profile,
        output_names,
        std::forward<ParameterFactory>(parameter_factory),
        std::forward<InputFactory>(input_factory),
        std::forward<ParameterJson>(parameter_json),
        std::forward<Launcher>(launcher)
    );
}

}  // namespace ai_factory::workbench::offline::sampling

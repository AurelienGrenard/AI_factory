// Compose model and product adapters into a device-prepared terminal scenario.
#pragma once

#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/equity/price_gradients/terminal_scenario.cuh"
#include "common/price_gradients/sensitivity_parameter.cuh"
#include "common/simulation/calendar.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation = device_preparation;

template<typename ModelPreparation, typename ProductPreparation>
pg::SensitivityParameter resolve_terminal_parameter(std::string_view name) {
    constexpr auto model_count = ModelPreparation::parameter_names.size();
    constexpr auto product_count = ProductPreparation::parameter_names.size();
    static_assert(model_count + product_count + 1U <= 256U);
    bool found = false;
    pg::SensitivityParameter result{};
    for (std::size_t index = 0U;
         index < ModelPreparation::parameter_names.size();
         ++index) {
        if (ModelPreparation::parameter_names[index] != name) continue;
        if (found) {
            throw std::invalid_argument(
                "Ambiguous terminal sensitivity: " + std::string(name)
            );
        }
        result = {static_cast<std::uint8_t>(index)};
        found = true;
    }
    for (std::size_t index = 0U;
         index < ProductPreparation::parameter_names.size();
         ++index) {
        if (ProductPreparation::parameter_names[index] != name) continue;
        if (found) {
            throw std::invalid_argument(
                "Ambiguous terminal sensitivity: " + std::string(name)
            );
        }
        result = {
            static_cast<std::uint8_t>(model_count + index)
        };
        found = true;
    }
    if (name == "product.maturity_years") {
        if constexpr (!ModelPreparation::kSupportsMaturitySensitivity) {
            throw std::invalid_argument(
                "Maturity sensitivity is not supported by this model."
            );
        }
        if (found) {
            throw std::invalid_argument(
                "Ambiguous terminal sensitivity: " + std::string(name)
            );
        }
        return {
            static_cast<std::uint8_t>(model_count + product_count)
        };
    }
    if (!found) {
        throw std::invalid_argument(
            "Unsupported terminal sensitivity: " + std::string(name)
        );
    }
    return result;
}

template<
    typename ModelPreparation,
    typename ProductPreparation,
    bool EnableMaturitySensitivity =
        ModelPreparation::kSupportsMaturitySensitivity
>
struct ScenarioDevicePreparation {
    using ModelAdapter = ModelPreparation;
    using ProductAdapter = ProductPreparation;
    using Model = typename ModelPreparation::Model;
    using Product = typename ProductPreparation::Product;
    using Parameter = pg::SensitivityParameter;
    using Scenario = ::ai_factory::workbench::equity::price_gradients::Scenario<
        Model,
        Product
    >;
    static constexpr bool kSupportsMaturitySensitivity =
        EnableMaturitySensitivity;
    static constexpr bool kSupportsMaturityDiagonal =
        EnableMaturitySensitivity
        && ModelPreparation::kSupportsMaturityDiagonal;
    static constexpr std::size_t sensitivity_parameter_count =
        ModelPreparation::parameter_names.size()
        + ProductPreparation::parameter_names.size()
        + (EnableMaturitySensitivity ? 1U : 0U);
    static_assert(sensitivity_parameter_count <= 256U);
    static constexpr std::uint8_t model_parameter_count =
        static_cast<std::uint8_t>(ModelPreparation::parameter_names.size());
    static constexpr std::uint8_t product_parameter_count =
        static_cast<std::uint8_t>(ProductPreparation::parameter_names.size());

    static pg::SensitivityParameter resolve_parameter(std::string_view name) {
        bool found = false;
        pg::SensitivityParameter result{};
        for (std::size_t index = 0U;
             index < ModelPreparation::parameter_names.size();
             ++index) {
            if (ModelPreparation::parameter_names[index] != name) continue;
            if (found) {
                throw std::invalid_argument(
                    "Ambiguous sensitivity: " + std::string(name)
                );
            }
            result = {static_cast<std::uint8_t>(index)};
            found = true;
        }
        for (std::size_t index = 0U;
             index < ProductPreparation::parameter_names.size();
             ++index) {
            if (ProductPreparation::parameter_names[index] != name) continue;
            if (found) {
                throw std::invalid_argument(
                    "Ambiguous sensitivity: " + std::string(name)
                );
            }
            result = {
                static_cast<std::uint8_t>(model_parameter_count + index)
            };
            found = true;
        }
        if constexpr (EnableMaturitySensitivity) {
            if (name == "product.maturity_years") {
                if (found) {
                    throw std::invalid_argument(
                        "Ambiguous sensitivity: " + std::string(name)
                    );
                }
                return {
                    static_cast<std::uint8_t>(
                        model_parameter_count + product_parameter_count
                    )
                };
            }
        }
        if (!found) {
            throw std::invalid_argument(
                "Unsupported sensitivity: " + std::string(name)
            );
        }
        return result;
    }

    __host__ __device__ static bool finite(float value) {
#if defined(__CUDA_ARCH__)
        return ::isfinite(value);
#else
        return std::isfinite(value);
#endif
    }

    __host__ __device__ static bool make_central(
        const Model& model,
        const Product& product,
        pg::TimeConfiguration time,
        Scenario& central
    ) {
        if (!ModelPreparation::valid(model)
            || !ProductPreparation::valid(product)) {
            return false;
        }
        const std::uint64_t steps =
            static_cast<std::uint64_t>(product.maturity_days)
            * time.simulation_steps_per_day;
        if (steps == 0U || steps > static_cast<std::uint64_t>(0xffffffffU)) {
            return false;
        }
        const float maturity_years = static_cast<float>(steps) * time.dt;
        if (!finite(maturity_years) || !(maturity_years > 0.0f)) return false;
        central = {
            model,
            product,
            maturity_years,
            static_cast<std::uint32_t>(steps),
            maturity_years,
            static_cast<std::uint32_t>(steps),
            model.spot,
            1.0f,
            false,
            pg::CentralRequirement::payoff,
            {1.0f, 0.0f, 0.0f, 0.0f},
        };
        return true;
    }

    __host__ __device__ static bool is_maturity(Parameter parameter) {
        return EnableMaturitySensitivity
            && parameter.code
                == model_parameter_count + product_parameter_count;
    }

    __host__ __device__ static pg::SensitivityParameterOwner parameter_owner(
        Parameter parameter
    ) {
        if (parameter.code < model_parameter_count) {
            return pg::SensitivityParameterOwner::model;
        }
        if (parameter.code
            < model_parameter_count + product_parameter_count) {
            return pg::SensitivityParameterOwner::product;
        }
        return pg::SensitivityParameterOwner::maturity;
    }

    __host__ __device__ static float read_parameter(
        Parameter parameter,
        const Scenario& scenario
    ) {
        if (parameter.code < model_parameter_count) {
            return ModelPreparation::read(parameter.code, scenario.model);
        }
        if (parameter.code
            < model_parameter_count + product_parameter_count) {
            return ProductPreparation::read(
                parameter.code - model_parameter_count,
                scenario.product
            );
        }
        return is_maturity(parameter) ? scenario.maturity_years : ::nanf("");
    }

    __host__ __device__ static void write_parameter(
        Parameter parameter,
        Scenario& scenario,
        float value
    ) {
        if (parameter.code < model_parameter_count) {
            ModelPreparation::write(parameter.code, scenario.model, value);
        } else if (parameter.code
                   < model_parameter_count + product_parameter_count) {
            ProductPreparation::write(
                parameter.code - model_parameter_count,
                scenario.product,
                value
            );
        }
    }

    __host__ __device__ static bool apply_parameter(
        Parameter parameter,
        float endpoint,
        pg::TimeConfiguration time,
        Scenario& row
    ) {
        if (!is_maturity(parameter)) {
            write_parameter(parameter, row, endpoint);
            return true;
        }
        const double step_value = static_cast<double>(endpoint)
            / static_cast<double>(time.dt);
        if (!preparation::finite(step_value) || step_value < 1.0
            || step_value > static_cast<double>(0xffffffffU)) {
            return false;
        }
        row.step_count = static_cast<std::uint32_t>(::llround(step_value));
        if (static_cast<float>(row.step_count) * time.dt != endpoint) {
            return false;
        }
        if constexpr (requires {
            ProductPreparation::calendar(row.product);
        }) {
            const auto calendar = ProductPreparation::calendar(row.product);
            const auto prefix_steps =
                simulation::calendar_terminal_prefix_days(calendar)
                * static_cast<std::uint64_t>(
                    time.simulation_steps_per_day
                );
            if (static_cast<std::uint64_t>(row.step_count) <= prefix_steps) {
                return false;
            }
        }
        row.maturity_years = endpoint;
        return true;
    }

    __host__ __device__ static bool finalize_scenario(
        const Scenario& central,
        Scenario& row
    ) {
        if constexpr (ModelPreparation::kMultiplicativeSpot) {
            row.simulation_spot = central.model.spot;
            row.spot_scale = row.model.spot / central.model.spot;
        } else {
            row.simulation_spot = row.model.spot;
            row.spot_scale = 1.0f;
        }
        row.reuse_central = row.step_count == central.step_count
            && ModelPreparation::same_dynamics(row.model, central.model)
            && (ModelPreparation::kMultiplicativeSpot
                || row.model.spot == central.model.spot);
        return ModelPreparation::valid(row.model)
            && ProductPreparation::valid(row.product)
            && row.step_count > 0U;
    }

    __host__ __device__ static bool change_scenario(
        const Scenario& central,
        Parameter parameter,
        float endpoint,
        pg::TimeConfiguration time,
        Scenario& row
    ) {
        row = central;
        return apply_parameter(parameter, endpoint, time, row)
            && finalize_scenario(central, row);
    }

    __host__ __device__ static bool change_scenario_pair(
        const Scenario& central,
        Parameter first_parameter,
        float first_endpoint,
        Parameter second_parameter,
        float second_endpoint,
        pg::TimeConfiguration time,
        Scenario& row
    ) {
        row = central;
        return apply_parameter(first_parameter, first_endpoint, time, row)
            && apply_parameter(
                second_parameter, second_endpoint, time, row
            )
            && finalize_scenario(central, row);
    }

    __host__ __device__ static void finalize_mixed_scenario(
        Parameter first_parameter,
        const Scenario& first_row,
        Parameter second_parameter,
        const Scenario& second_row,
        Scenario& mixed_row
    ) {
        if constexpr (EnableMaturitySensitivity) {
            const Scenario* maturity_row = nullptr;
            if (is_maturity(first_parameter)) {
                maturity_row = &first_row;
            } else if (is_maturity(second_parameter)) {
                maturity_row = &second_row;
            }
            if (maturity_row != nullptr) {
                for (std::size_t normal = 0U; normal < 4U; ++normal) {
                    mixed_row.normal_weights[normal] =
                        maturity_row->normal_weights[normal];
                }
            }
        }
    }

    template<pg::SensitivityOrders Orders>
    __host__ __device__ static void finalize_task(
        pg::SensitivityTask<
            Scenario,
            pg::SensitivityTraits<Orders>::node_capacity
        >& task,
        Parameter parameter
    ) {
        if constexpr (pg::requests_second_v<Orders>) {
            task.central_requirement = pg::CentralRequirement::payoff;
        } else {
            task.central_requirement =
                task.stencil.kind != pg::StencilKind::centered
                ? pg::CentralRequirement::payoff
                : task.nodes[1U].reuse_central
                    || task.nodes[2U].reuse_central
                ? pg::CentralRequirement::state
                : pg::CentralRequirement::none;
        }
        const auto count = pg::active_node_count(task.stencil);
        for (std::size_t node = 1U; node < count; ++node) {
            task.nodes[node].central_requirement = task.central_requirement;
        }
        if (is_maturity(parameter)) {
            constexpr auto capacity =
                pg::SensitivityTraits<Orders>::node_capacity;
            const auto count = pg::active_node_count(task.stencil);
            float times[capacity]{};
            float weights[capacity][capacity]{};
            for (std::size_t node = 0U; node < count; ++node) {
                times[node] = task.nodes[node].maturity_years;
            }
            preparation::brownian_endpoint_weights(
                times, count, weights
            );
            for (std::size_t node = 0U; node < count; ++node) {
                for (std::size_t normal = 0U;
                     normal < capacity;
                     ++normal) {
                    task.nodes[node].normal_weights[normal] =
                        weights[node][normal];
                }
            }
        }
    }
};

template<typename ModelPreparation, typename ProductPreparation>
using TerminalDevicePreparation = ScenarioDevicePreparation<
    ModelPreparation,
    ProductPreparation,
    ModelPreparation::kSupportsMaturitySensitivity
>;

}  // namespace ai_factory::workbench::equity::price_gradients

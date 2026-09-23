// Compose fixed-income model and product adapters into compact device tasks.
#pragma once

#include "common/price_gradients/device_preparation.cuh"
#include "common/price_gradients/sensitivity_parameter.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ai_factory::workbench::fixed_income::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<typename Model, typename Product>
struct Scenario {
    Model model;
    Product product;
    float day_fraction;
};

template<typename ModelPreparation, typename ProductPreparation>
struct ScenarioDevicePreparation {
    using ModelAdapter = ModelPreparation;
    using ProductAdapter = ProductPreparation;
    using Model = typename ModelPreparation::Model;
    using Product = typename ProductPreparation::Product;
    using Parameter = pg::SensitivityParameter;
    using Scenario = fixed_income::price_gradients::Scenario<Model, Product>;
    static constexpr bool kSupportsMaturitySensitivity = false;

    static constexpr std::size_t sensitivity_parameter_count =
        ModelPreparation::parameter_names.size()
        + ProductPreparation::parameter_names.size();
    static_assert(sensitivity_parameter_count <= 256U);
    static constexpr std::uint8_t model_parameter_count =
        static_cast<std::uint8_t>(ModelPreparation::parameter_names.size());
    static constexpr std::uint8_t product_parameter_count =
        static_cast<std::uint8_t>(ProductPreparation::parameter_names.size());

    static Parameter resolve_parameter(std::string_view name) {
        bool found = false;
        Parameter result{};
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
        if (!found) {
            throw std::invalid_argument(
                "Unsupported sensitivity: " + std::string(name)
            );
        }
        return result;
    }

    __host__ __device__ static bool make_central(
        const Model& model,
        const Product& product,
        pg::TimeConfiguration time,
        Scenario& central
    ) {
        const float day_fraction = time.dt
            * static_cast<float>(time.simulation_steps_per_day);
        if (!ModelPreparation::valid(model)
            || !ProductPreparation::valid(product)
            || !pg::device_preparation::finite(day_fraction)
            || !(day_fraction > 0.0f)) {
            return false;
        }
        central = {model, product, day_fraction};
        return true;
    }

    __host__ __device__ static bool is_maturity(Parameter) {
        return false;
    }

    __host__ __device__ static pg::SensitivityParameterOwner parameter_owner(
        Parameter parameter
    ) {
        return parameter.code < model_parameter_count
            ? pg::SensitivityParameterOwner::model
            : pg::SensitivityParameterOwner::product;
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
        return ::nanf("");
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

    __host__ __device__ static bool change_scenario(
        const Scenario& central,
        Parameter parameter,
        float endpoint,
        pg::TimeConfiguration,
        Scenario& row
    ) {
        row = central;
        write_parameter(parameter, row, endpoint);
        return ModelPreparation::valid(row.model)
            && ProductPreparation::valid(row.product);
    }

    template<pg::SensitivityOrders Orders>
    __host__ __device__ static void finalize_task(
        pg::SensitivityTask<
            Scenario,
            pg::SensitivityTraits<Orders>::node_capacity
        >& task,
        Parameter
    ) {
        task.central_requirement = pg::CentralRequirement::payoff;
    }
};


template<typename Model, typename Curve, typename Product>
struct CurveScenario {
    Model model;
    Curve curve;
    Product product;
    float day_fraction;
};

template<
    typename ModelPreparation,
    typename CurvePreparation,
    typename ProductPreparation
>
struct CurveScenarioDevicePreparation {
    using ModelAdapter = ModelPreparation;
    using CurveAdapter = CurvePreparation;
    using ProductAdapter = ProductPreparation;
    using Model = typename ModelPreparation::Model;
    using Curve = typename CurvePreparation::Curve;
    using Product = typename ProductPreparation::Product;
    using Parameter = pg::SensitivityParameter;
    using Scenario = fixed_income::price_gradients::CurveScenario<
        Model, Curve, Product
    >;
    static constexpr bool kSupportsMaturitySensitivity = false;

    static constexpr std::size_t sensitivity_parameter_count =
        ModelPreparation::parameter_names.size()
        + CurvePreparation::parameter_names.size()
        + ProductPreparation::parameter_names.size();
    static_assert(sensitivity_parameter_count <= 256U);
    static constexpr std::uint8_t model_parameter_count =
        static_cast<std::uint8_t>(ModelPreparation::parameter_names.size());
    static constexpr std::uint8_t curve_parameter_count =
        static_cast<std::uint8_t>(CurvePreparation::parameter_names.size());
    static constexpr std::uint8_t product_parameter_count =
        static_cast<std::uint8_t>(ProductPreparation::parameter_names.size());

    static Parameter resolve_parameter(std::string_view name) {
        bool found = false;
        Parameter result{};
        const auto resolve = [&](const auto& names, std::uint8_t offset) {
            for (std::size_t index = 0U; index < names.size(); ++index) {
                if (names[index] != name) continue;
                if (found) {
                    throw std::invalid_argument(
                        "Ambiguous sensitivity: " + std::string(name)
                    );
                }
                result = {
                    static_cast<std::uint8_t>(offset + index)
                };
                found = true;
            }
        };
        resolve(ModelPreparation::parameter_names, 0U);
        resolve(CurvePreparation::parameter_names, model_parameter_count);
        resolve(
            ProductPreparation::parameter_names,
            static_cast<std::uint8_t>(
                model_parameter_count + curve_parameter_count
            )
        );
        if (!found) {
            throw std::invalid_argument(
                "Unsupported sensitivity: " + std::string(name)
            );
        }
        return result;
    }

    __host__ __device__ static bool make_central(
        const Model& model,
        const Curve& curve,
        const Product& product,
        pg::TimeConfiguration time,
        Scenario& central
    ) {
        const float day_fraction = time.dt
            * static_cast<float>(time.simulation_steps_per_day);
        if (!ModelPreparation::valid(model)
            || !CurvePreparation::valid(curve)
            || !ProductPreparation::valid(product)
            || !pg::device_preparation::finite(day_fraction)
            || !(day_fraction > 0.0f)) {
            return false;
        }
        central = {model, curve, product, day_fraction};
        return true;
    }

    __host__ __device__ static bool is_maturity(Parameter) {
        return false;
    }

    __host__ __device__ static pg::SensitivityParameterOwner parameter_owner(
        Parameter parameter
    ) {
        if (parameter.code < model_parameter_count) {
            return pg::SensitivityParameterOwner::model;
        }
        if (parameter.code
            < model_parameter_count + curve_parameter_count) {
            return pg::SensitivityParameterOwner::curve;
        }
        return pg::SensitivityParameterOwner::product;
    }

    __host__ __device__ static float read_parameter(
        Parameter parameter,
        const Scenario& scenario
    ) {
        if (parameter.code < model_parameter_count) {
            return ModelPreparation::read(parameter.code, scenario.model);
        }
        if (parameter.code
            < model_parameter_count + curve_parameter_count) {
            return CurvePreparation::read(
                parameter.code - model_parameter_count, scenario.curve
            );
        }
        if (parameter.code < sensitivity_parameter_count) {
            return ProductPreparation::read(
                parameter.code - model_parameter_count
                    - curve_parameter_count,
                scenario.product
            );
        }
        return ::nanf("");
    }

    __host__ __device__ static void write_parameter(
        Parameter parameter,
        Scenario& scenario,
        float value
    ) {
        if (parameter.code < model_parameter_count) {
            ModelPreparation::write(parameter.code, scenario.model, value);
        } else if (parameter.code
                   < model_parameter_count + curve_parameter_count) {
            CurvePreparation::write(
                parameter.code - model_parameter_count,
                scenario.curve,
                value
            );
        } else if (parameter.code < sensitivity_parameter_count) {
            ProductPreparation::write(
                parameter.code - model_parameter_count
                    - curve_parameter_count,
                scenario.product,
                value
            );
        }
    }

    __host__ __device__ static bool change_scenario(
        const Scenario& central,
        Parameter parameter,
        float endpoint,
        pg::TimeConfiguration,
        Scenario& row
    ) {
        row = central;
        write_parameter(parameter, row, endpoint);
        return ModelPreparation::valid(row.model)
            && CurvePreparation::valid(row.curve)
            && ProductPreparation::valid(row.product);
    }

    template<pg::SensitivityOrders Orders>
    __host__ __device__ static void finalize_task(
        pg::SensitivityTask<
            Scenario,
            pg::SensitivityTraits<Orders>::node_capacity
        >& task,
        Parameter
    ) {
        task.central_requirement = pg::CentralRequirement::payoff;
    }
};

}  // namespace ai_factory::workbench::fixed_income::price_gradients

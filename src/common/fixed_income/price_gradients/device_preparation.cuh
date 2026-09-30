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
namespace preparation = pg::device_preparation;

namespace detail {

template<typename ProductPreparation, typename Product>
__host__ __device__ bool prepare_terminal_time(
    const Product& product,
    pg::TimeConfiguration time,
    float& day_fraction,
    float& maturity_years,
    std::uint32_t& step_count
) {
    day_fraction = time.dt
        * static_cast<float>(time.simulation_steps_per_day);
    const std::uint64_t steps =
        static_cast<std::uint64_t>(
            ProductPreparation::terminal_maturity_days(product)
        ) * time.simulation_steps_per_day;
    if (!preparation::finite(day_fraction) || !(day_fraction > 0.0f)
        || steps == 0U
        || steps > static_cast<std::uint64_t>(0xffffffffU)) {
        return false;
    }
    step_count = static_cast<std::uint32_t>(steps);
    maturity_years = static_cast<float>(step_count) * time.dt;
    return preparation::finite(maturity_years)
        && maturity_years > 0.0f;
}

template<typename ProductPreparation, typename Scenario>
__host__ __device__ bool apply_terminal_time(
    float endpoint,
    pg::TimeConfiguration time,
    Scenario& row
) {
    const double step_value = static_cast<double>(endpoint)
        / static_cast<double>(time.dt);
    if (!preparation::finite(step_value) || step_value < 1.0
        || step_value > static_cast<double>(0xffffffffU)) {
        return false;
    }
    const auto steps = static_cast<std::uint32_t>(::llround(step_value));
    if (static_cast<float>(steps) * time.dt != endpoint) return false;
    const auto prefix_steps = static_cast<std::uint64_t>(
        ProductPreparation::terminal_prefix_days(row.product)
    ) * time.simulation_steps_per_day;
    if (static_cast<std::uint64_t>(steps) <= prefix_steps) return false;
    row.maturity_years = endpoint;
    row.step_count = steps;
    return true;
}

}  // namespace detail

template<typename Model, typename Product>
struct Scenario {
    Model model;
    Product product;
    float day_fraction;
    float maturity_years;
    std::uint32_t step_count = 1U;
    std::uint32_t central_step_count = 1U;
    const float* normal_weights = nullptr;
    bool reuse_central = false;
};

template<typename ModelPreparation, typename ProductPreparation>
struct ScenarioDevicePreparation {
    using ModelAdapter = ModelPreparation;
    using ProductAdapter = ProductPreparation;
    using Model = typename ModelPreparation::Model;
    using Product = typename ProductPreparation::Product;
    using Parameter = pg::SensitivityParameter;
    using Scenario = fixed_income::price_gradients::Scenario<Model, Product>;
    static constexpr bool kSupportsMaturitySensitivity =
        ProductPreparation::kSupportsMaturitySensitivity;
    static constexpr bool kSupportsMaturityDiagonal =
        kSupportsMaturitySensitivity;

    static constexpr std::size_t sensitivity_parameter_count =
        ModelPreparation::parameter_names.size()
        + ProductPreparation::parameter_names.size()
        + (kSupportsMaturitySensitivity ? 1U : 0U);
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
        if constexpr (kSupportsMaturitySensitivity) {
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

    __host__ __device__ static bool make_central(
        const Model& model,
        const Product& product,
        pg::TimeConfiguration time,
        Scenario& central
    ) {
        float day_fraction = 0.0f;
        float maturity_years = 0.0f;
        std::uint32_t step_count = 0U;
        if (!ModelPreparation::valid(model)
            || !ProductPreparation::valid(product)
            || !detail::prepare_terminal_time<ProductPreparation>(
                product,
                time,
                day_fraction,
                maturity_years,
                step_count
            )) {
            return false;
        }
        central = {
            model,
            product,
            day_fraction,
            maturity_years,
            step_count,
            step_count,
            nullptr,
            false,
        };
        return true;
    }

    __host__ __device__ static bool is_maturity(Parameter parameter) {
        return kSupportsMaturitySensitivity
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
        return is_maturity(parameter)
            ? scenario.maturity_years
            : ::nanf("");
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
        if (is_maturity(parameter)) {
            return detail::apply_terminal_time<ProductPreparation>(
                endpoint, time, row
            );
        }
        write_parameter(parameter, row, endpoint);
        return true;
    }

    __host__ __device__ static bool finalize_scenario(
        Scenario& row,
        bool model_changed
    ) {
        row.reuse_central = !model_changed;
        return ModelPreparation::valid(row.model)
            && ProductPreparation::valid(row.product);
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
            && finalize_scenario(
            row,
            parameter_owner(parameter)
                == pg::SensitivityParameterOwner::model
        );
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
        return apply_parameter(
                first_parameter, first_endpoint, time, row
            )
            && apply_parameter(
                second_parameter, second_endpoint, time, row
            )
            && finalize_scenario(
            row,
            parameter_owner(first_parameter)
                    == pg::SensitivityParameterOwner::model
                || parameter_owner(second_parameter)
                    == pg::SensitivityParameterOwner::model
        );
    }

    __host__ __device__ static void finalize_mixed_scenario(
        Parameter,
        const Scenario&,
        Parameter,
        const Scenario&,
        Scenario&
    ) {}

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
    float maturity_years;
    std::uint32_t step_count = 1U;
    std::uint32_t central_step_count = 1U;
    const float* normal_weights = nullptr;
    bool reuse_central = false;
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
    static constexpr bool kSupportsMaturitySensitivity =
        ProductPreparation::kSupportsMaturitySensitivity;
    static constexpr bool kSupportsMaturityDiagonal =
        kSupportsMaturitySensitivity;

    static constexpr std::size_t sensitivity_parameter_count =
        ModelPreparation::parameter_names.size()
        + CurvePreparation::parameter_names.size()
        + ProductPreparation::parameter_names.size()
        + (kSupportsMaturitySensitivity ? 1U : 0U);
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
        if constexpr (kSupportsMaturitySensitivity) {
            if (name == "product.maturity_years") {
                if (found) {
                    throw std::invalid_argument(
                        "Ambiguous sensitivity: " + std::string(name)
                    );
                }
                return {
                    static_cast<std::uint8_t>(
                        model_parameter_count
                        + curve_parameter_count
                        + product_parameter_count
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

    __host__ __device__ static bool make_central(
        const Model& model,
        const Curve& curve,
        const Product& product,
        pg::TimeConfiguration time,
        Scenario& central
    ) {
        float day_fraction = 0.0f;
        float maturity_years = 0.0f;
        std::uint32_t step_count = 0U;
        if (!ModelPreparation::valid(model)
            || !CurvePreparation::valid(curve)
            || !ProductPreparation::valid(product)
            || !detail::prepare_terminal_time<ProductPreparation>(
                product,
                time,
                day_fraction,
                maturity_years,
                step_count
            )) {
            return false;
        }
        central = {
            model,
            curve,
            product,
            day_fraction,
            maturity_years,
            step_count,
            step_count,
            nullptr,
            false,
        };
        return true;
    }

    __host__ __device__ static bool is_maturity(Parameter parameter) {
        return kSupportsMaturitySensitivity
            && parameter.code
                == model_parameter_count
                    + curve_parameter_count
                    + product_parameter_count;
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
        if (parameter.code
            < model_parameter_count
                + curve_parameter_count
                + product_parameter_count) {
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
            < model_parameter_count + curve_parameter_count) {
            return CurvePreparation::read(
                parameter.code - model_parameter_count, scenario.curve
            );
        }
        if (parameter.code
            < model_parameter_count
                + curve_parameter_count
                + product_parameter_count) {
            return ProductPreparation::read(
                parameter.code - model_parameter_count
                    - curve_parameter_count,
                scenario.product
            );
        }
        return is_maturity(parameter)
            ? scenario.maturity_years
            : ::nanf("");
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
        } else if (parameter.code
                   < model_parameter_count
                       + curve_parameter_count
                       + product_parameter_count) {
            ProductPreparation::write(
                parameter.code - model_parameter_count
                    - curve_parameter_count,
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
        if (is_maturity(parameter)) {
            return detail::apply_terminal_time<ProductPreparation>(
                endpoint, time, row
            );
        }
        write_parameter(parameter, row, endpoint);
        return true;
    }

    __host__ __device__ static bool finalize_scenario(
        Scenario& row,
        bool model_changed
    ) {
        row.reuse_central = !model_changed;
        return ModelPreparation::valid(row.model)
            && CurvePreparation::valid(row.curve)
            && ProductPreparation::valid(row.product);
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
            && finalize_scenario(
            row,
            parameter_owner(parameter)
                == pg::SensitivityParameterOwner::model
        );
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
        return apply_parameter(
                first_parameter, first_endpoint, time, row
            )
            && apply_parameter(
                second_parameter, second_endpoint, time, row
            )
            && finalize_scenario(
            row,
            parameter_owner(first_parameter)
                    == pg::SensitivityParameterOwner::model
                || parameter_owner(second_parameter)
                    == pg::SensitivityParameterOwner::model
        );
    }

    __host__ __device__ static void finalize_mixed_scenario(
        Parameter,
        const Scenario&,
        Parameter,
        const Scenario&,
        Scenario&
    ) {}

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

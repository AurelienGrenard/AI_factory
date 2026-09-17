// Host-resolved model/product coordinates and validated terminal scenario batches.
#pragma once

#include "common/price_construction.cuh"
#include "common/price_gradients/stencil.hpp"
#include "common/price_gradients/central_requirement.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct TimeConfiguration {
    float dt = 1.0f / 504.0f;
    std::uint32_t simulation_steps_per_day = 2U;
    void validate() const {
        if (!std::isfinite(dt) || !(dt > 0.0f) || simulation_steps_per_day == 0U)
            throw std::invalid_argument("Invalid gradient time grid.");
    }
};

template<typename Parameters>
struct ParameterField {
    std::string_view name;
    float Parameters::* member;
};

template<typename Model, typename Product>
struct Scenario {
    Model model;
    Product product;
    float maturity_years;
    std::uint32_t step_count;
    // Model used for simulation may retain central spot for homogeneous paths.
    float simulation_spot;
    float spot_scale;
    bool reuse_central;
    pg::CentralRequirement central_requirement;
    // Normalized Brownian endpoint in a three-normal, central-first coupling.
    float normal_weights[3];
};

template<typename Model, typename Product>
struct ScenarioPlan {
    using ScenarioType = Scenario<Model, Product>;
    pg::PriceGradientConfiguration configuration;
    TimeConfiguration time;
    std::size_t result_count;
    std::vector<ScenarioType> scenarios;  // row-major, then central/paired scenarios
    std::vector<pg::Stencil> stencils;   // row-major, then selected coordinate

    std::size_t sensitivity_count() const { return configuration.sensitivities.size(); }
    std::size_t scenario_count() const { return 1U + 2U * sensitivity_count(); }
};

// Preserve W(T) as the original first normal. Other dates are conditional
// Brownian bridges or extensions; using sqrt(t)*the same normal is not W(t).
inline std::array<std::array<float, 3>, 3> brownian_endpoint_weights(
    float central, float first, float second
) {
    std::array<std::array<double, 3>, 3> coefficients{};
    coefficients[0][0] = std::sqrt(static_cast<double>(central));
    const double times[3]{central, first, second};
    for (unsigned int index = 1; index < 3; ++index) {
        if (times[index] == times[0]) { coefficients[index] = coefficients[0]; continue; }
        if (index == 2U && times[2] == times[1]) { coefficients[2] = coefficients[1]; continue; }
        int left = -1, right = -1;
        for (unsigned int known = 0; known < index; ++known) {
            if (times[known] < times[index] && (left < 0 || times[known] > times[left])) left = known;
            if (times[known] > times[index] && (right < 0 || times[known] < times[right])) right = known;
        }
        const double left_time = left < 0 ? 0.0 : times[left];
        if (right < 0) {
            if (left >= 0) coefficients[index] = coefficients[left];
            coefficients[index][index] = std::sqrt(times[index] - left_time);
        } else {
            const double interval = times[right] - left_time;
            const double fraction = (times[index] - left_time) / interval;
            for (unsigned int normal = 0; normal < 3; ++normal)
                coefficients[index][normal] = (left < 0 ? 0.0 : (1.0 - fraction) * coefficients[left][normal])
                    + fraction * coefficients[right][normal];
            coefficients[index][index] = std::sqrt((times[index] - left_time) * (times[right] - times[index]) / interval);
        }
    }
    std::array<std::array<float, 3>, 3> result{};
    for (unsigned int index = 0; index < 3; ++index)
        for (unsigned int normal = 0; normal < 3; ++normal)
            result[index][normal] = static_cast<float>(coefficients[index][normal] / std::sqrt(times[index]));
    result[0] = {1.0f, 0.0f, 0.0f};
    return result;
}

template<typename ModelPolicy, typename ProductPolicy>
auto prepare_scenarios(
    std::span<const typename ModelPolicy::Parameters> models,
    std::span<const typename ProductPolicy::Parameters> products,
    PriceConstruction construction, TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration
) {
    using Model = typename ModelPolicy::Parameters;
    using Product = typename ProductPolicy::Parameters;
    using Row = Scenario<Model, Product>;
    constexpr auto maximum_count = ModelPolicy::fields.size() + ProductPolicy::fields.size() + 1U;
    configuration.validate(maximum_count);
    time.validate();
    if (construction != PriceConstruction::Aligned && construction != PriceConstruction::CartesianProduct)
        throw std::invalid_argument("Unknown price construction.");
    const std::size_t count = price_row_count(models.size(), products.size(), construction);
    const auto k = configuration.sensitivities.size();
    const auto n = 1U + 2U * k;
    if (count > std::numeric_limits<std::size_t>::max() / n / sizeof(Row))
        throw std::overflow_error("Gradient scenario batch size overflow.");
    struct Coordinate { float Model::* model = nullptr; float Product::* product = nullptr; bool maturity = false; };
    std::vector<Coordinate> coordinates;
    for (const auto& sensitivity : configuration.sensitivities) {
        Coordinate coordinate;
        for (const auto& field : ModelPolicy::fields)
            if (sensitivity.parameter == field.name) coordinate.model = field.member;
        for (const auto& field : ProductPolicy::fields)
            if (sensitivity.parameter == field.name) coordinate.product = field.member;
        coordinate.maturity = sensitivity.parameter == "product.maturity_years";
        if (coordinate.maturity) {
            if constexpr (requires { ModelPolicy::kSupportsMaturitySensitivity; }) {
                if (!ModelPolicy::kSupportsMaturitySensitivity)
                    throw std::invalid_argument("Unsupported sensitivity: " + sensitivity.parameter);
            }
            if constexpr (requires { ProductPolicy::kSupportsMaturitySensitivity; }) {
                if (!ProductPolicy::kSupportsMaturitySensitivity)
                    throw std::invalid_argument("Unsupported sensitivity: " + sensitivity.parameter);
            }
        }
        if (!coordinate.model && !coordinate.product && !coordinate.maturity)
            throw std::invalid_argument("Unsupported sensitivity: " + sensitivity.parameter);
        coordinates.push_back(coordinate);
    }
    ScenarioPlan<Model, Product> plan{configuration, time, count, {}, {}};
    plan.scenarios.reserve(count * n);
    plan.stencils.reserve(count * k);
    for (std::size_t index = 0; index < count; ++index) {
        const auto& model = models[construction == PriceConstruction::Aligned ? index : index / products.size()];
        const auto& product = products[construction == PriceConstruction::Aligned ? index : index % products.size()];
        if (!ModelPolicy::valid(model) || !ProductPolicy::valid(product))
            throw std::invalid_argument("Invalid central gradient row " + std::to_string(index));
        const std::uint64_t steps = std::uint64_t(product.maturity_days) * time.simulation_steps_per_day;
        if (steps == 0U || steps > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Gradient maturity step count outside uint32.");
        const float maturity_years = static_cast<float>(steps) * time.dt;
        if (!std::isfinite(maturity_years) || !(maturity_years > 0.0f))
            throw std::invalid_argument("Invalid represented gradient maturity.");
        const Row central{model, product, maturity_years, static_cast<std::uint32_t>(steps), model.spot, 1.0f,
            false, pg::CentralRequirement::payoff, {1.0f, 0.0f, 0.0f}};
        plan.scenarios.push_back(central);
        for (std::size_t sensitivity = 0; sensitivity < k; ++sensitivity) {
            const auto coordinate = coordinates[sensitivity];
            const auto& selection = configuration.sensitivities[sensitivity];
            const float value = coordinate.model ? model.*coordinate.model
                : coordinate.product ? product.*coordinate.product : maturity_years;
            auto changed = [&](float endpoint) {
                Row row = central;
                if (coordinate.model) row.model.*coordinate.model = endpoint;
                else if (coordinate.product) row.product.*coordinate.product = endpoint;
                else {
                    const double step_value = static_cast<double>(endpoint) / time.dt;
                    if (!std::isfinite(step_value) || step_value < 1.0
                        || step_value > std::numeric_limits<std::uint32_t>::max()) {
                        row.step_count = 0U;
                    } else {
                        row.step_count = static_cast<std::uint32_t>(std::llround(step_value));
                        // A represented date must be exactly the declared integer grid point.
                        if (static_cast<float>(row.step_count) * time.dt != endpoint) row.step_count = 0U;
                    }
                    row.maturity_years = endpoint;
                }
                if constexpr (ModelPolicy::kMultiplicativeSpot) {
                    row.simulation_spot = central.model.spot;
                    row.spot_scale = row.model.spot / central.model.spot;
                } else {
                    row.simulation_spot = row.model.spot;
                    row.spot_scale = 1.f;
                }
                row.reuse_central = row.step_count == central.step_count;
                for (const auto& field : ModelPolicy::fields) {
                    if constexpr (ModelPolicy::kMultiplicativeSpot)
                        if (field.member == &Model::spot) continue;
                    row.reuse_central = row.reuse_central
                        && row.model.*field.member == central.model.*field.member;
                }
                return row;
            };
            auto admissible = [&](float endpoint) {
                const auto row = changed(endpoint);
                return ModelPolicy::valid(row.model) && ProductPolicy::valid(row.product) && row.step_count > 0U;
            };
            try {
                auto stencil = [&] {
                    if (!coordinate.maturity) return pg::prepare_stencil(value, selection.bump, admissible);
                    const float h = selection.bump.scale == pg::BumpScale::relative
                        ? static_cast<float>(selection.bump.displacement) * value : static_cast<float>(selection.bump.displacement);
                    const double step_width = static_cast<double>(h) / time.dt;
                    if (!std::isfinite(step_width) || step_width < 1.0
                        || step_width > std::numeric_limits<std::uint32_t>::max())
                        throw std::invalid_argument("Maturity bump must contain a positive integer number of grid steps.");
                    const auto bump_steps = std::llround(step_width);
                    if (static_cast<float>(bump_steps) * time.dt != h)
                        throw std::invalid_argument("Maturity bump must be an integer multiple of dt.");
                    return pg::prepare_stencil(value, selection.bump, admissible, [&](int multiple) {
                        const auto endpoint_steps = static_cast<std::int64_t>(steps) + multiple * bump_steps;
                        return static_cast<float>(endpoint_steps) * time.dt;
                    });
                }();
                auto first = changed(stencil.first), second = changed(stencil.second);
                first.central_requirement = second.central_requirement = stencil.kind != pg::StencilKind::centered
                    ? pg::CentralRequirement::payoff : first.reuse_central || second.reuse_central
                    ? pg::CentralRequirement::state : pg::CentralRequirement::none;
                if (coordinate.maturity) {
                    const auto weights = brownian_endpoint_weights(maturity_years, first.maturity_years, second.maturity_years);
                    std::copy(weights[1].begin(), weights[1].end(), first.normal_weights);
                    std::copy(weights[2].begin(), weights[2].end(), second.normal_weights);
                }
                plan.stencils.push_back(stencil);
                plan.scenarios.push_back(first);
                plan.scenarios.push_back(second);
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument("Gradient row " + std::to_string(index) + ", "
                    + selection.parameter + ": " + error.what());
            }
        }
    }
    return plan;
}

}  // namespace ai_factory::workbench::equity::price_gradients

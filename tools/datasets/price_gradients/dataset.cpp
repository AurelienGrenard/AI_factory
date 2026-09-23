// Validate complete gradient rows before staging their native JSON/YAML artifacts.
#include "tools/datasets/price_gradients/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include "common/result_index.cuh"
#include <cmath>

namespace ai_factory::workbench::datasets::price_gradients {
namespace pg = ::ai_factory::workbench::price_gradients;

nlohmann::ordered_json sensitivity_metadata(
    const Recipe& recipe,
    bool stochastic
) {
    auto selections = nlohmann::ordered_json::array();
    for (const auto& selection : recipe.configuration.sensitivities)
        selections.push_back({{"parameter", selection.parameter}, {"displacement", selection.bump.displacement},
            {"scale", selection.bump.scale == pg::BumpScale::absolute ? "absolute" : "relative"},
            {"boundary", selection.bump.boundary == pg::BoundaryRule::central_only ? "central_only" : "central_then_one_sided_order2"}});
    nlohmann::ordered_json metadata{{"method", stochastic
            ? "finite_difference_shared_innovations"
            : "finite_difference"}, {"parameters", selections},
            {"source_price_recipe", recipe.source_price_recipe}};
    if (recipe.orders != pg::SensitivityOrders::first) {
        metadata["orders"] = recipe.orders == pg::SensitivityOrders::second
            ? nlohmann::ordered_json::array({"diagonal_second"})
            : nlohmann::ordered_json::array({"first", "diagonal_second"});
    }
    return metadata;
}

void write_dataset(const Recipe& recipe, const Results& result) {
    validate_dataset_url(recipe.url);
    recipe.configuration.validate(recipe.configuration.sensitivities.size());
    recipe.time.validate();
    const double reciprocal_dt = 1.0 / recipe.time.dt;
    if (reciprocal_dt < 1.0 || reciprocal_dt > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Gradient artifact requires an integer steps-per-year grid.");
    const auto steps_per_year = static_cast<std::uint32_t>(std::llround(reciprocal_dt));
    if (1.0f/static_cast<float>(steps_per_year) != recipe.time.dt)
        throw std::invalid_argument("Gradient artifact dt is not a reciprocal integer grid.");
    const auto models = read_json_file(recipe.model_input);
    const auto products = read_json_file(recipe.product_input);
    nlohmann::ordered_json curves;
    const nlohmann::ordered_json* curve_rows = nullptr;
    if (!recipe.curve_input.empty()) {
        curves = read_json_file(recipe.curve_input);
        curve_rows = &curves.at("curves");
    }
    const auto& model_rows = models.at("models");
    const auto& product_rows = products.at("products");
    const auto count = curve_rows == nullptr
        ? price_row_count(
            model_rows.size(), product_rows.size(), recipe.construction
        )
        : price_row_count(
            model_rows.size(),
            curve_rows->size(),
            product_rows.size(),
            recipe.construction
        );
    const auto k = recipe.configuration.sensitivities.size();
    const bool first_requested = recipe.orders == pg::SensitivityOrders::first
        || recipe.orders == pg::SensitivityOrders::first_and_second;
    const bool second_requested = recipe.orders == pg::SensitivityOrders::second
        || recipe.orders == pg::SensitivityOrders::first_and_second;
    if (!first_requested && !second_requested)
        throw std::invalid_argument("Gradient artifact has no requested sensitivity order.");
    if (k && count > std::numeric_limits<std::size_t>::max()/k) throw std::overflow_error("Gradient output cardinality overflow.");
    const bool stochastic = result.execution.at("paths_per_price").get<std::size_t>() != 0U;
    const auto& scenario_count = result.execution.at("scenario_count");
    const bool scenario_shape_valid = scenario_count.is_null()
        ? result.execution.value("materialized_scenario_count", 1U) == 0U
        : scenario_count.get<std::size_t>() == 1U+2U*k;
    if (result.prices.size() != count
        || result.gradients.size() != (first_requested ? count*k : 0U)
        || result.stencils.size() != count*k
        || result.diagonal_hessians.size() != (second_requested ? count*k : 0U)
        || result.diagonal_stencils.size() != (second_requested ? count*k : 0U)
        || result.price_errors.size() != (stochastic ? count : 0U)
        || result.gradient_errors.size() != (stochastic && first_requested ? count*k : 0U)
        || result.diagonal_hessian_errors.size() != (stochastic && second_requested ? count*k : 0U)
        || result.execution.at("sensitivity_count").get<std::size_t>() != k
        || !scenario_shape_valid)
        throw std::invalid_argument("Incomplete price-gradient outputs or inconsistent execution shape.");
    for (const auto* values : {&result.prices, &result.gradients, &result.price_errors,
            &result.gradient_errors, &result.diagonal_hessians,
            &result.diagonal_hessian_errors})
        for (float value : *values) if (!std::isfinite(value)) throw std::invalid_argument("Non-finite gradient artifact value.");
    for (const auto* errors : {&result.price_errors, &result.gradient_errors,
            &result.diagonal_hessian_errors})
        for (float error : *errors) if (error < 0) throw std::invalid_argument("Negative gradient sampling error.");
    if (!std::isfinite(result.wall_seconds) || !std::isfinite(result.kernel_seconds)
        || result.wall_seconds < 0 || result.kernel_seconds < 0) throw std::invalid_argument("Invalid gradient execution timing.");
    auto rows = nlohmann::ordered_json::array();
    for (std::size_t row = 0; row < count; ++row) {
        auto gradients = nlohmann::ordered_json::object();
        auto diagonal_hessians = nlohmann::ordered_json::object();
        auto stencils = nlohmann::ordered_json::object();
        auto errors = nlohmann::ordered_json::object();
        auto diagonal_errors = nlohmann::ordered_json::object();
        for (std::size_t i = 0; i < k; ++i) {
            const auto& name = recipe.configuration.sensitivities[i].parameter;
            const auto& s = result.stencils[row*k+i];
            if (!std::isfinite(s.central) || !std::isfinite(s.first) || !std::isfinite(s.second)
                || !std::isfinite(s.displacement) || !(s.displacement > 0)
                || !std::isfinite(s.first_weight) || !std::isfinite(s.second_weight)
                || !std::isfinite(s.represented_width) || s.first == s.second
                || s.represented_width != s.second-s.first)
                throw std::invalid_argument("Invalid represented gradient stencil.");
            const bool centered = s.kind == pg::StencilKind::centered;
            const bool forward = s.kind == pg::StencilKind::forward;
            const bool backward = s.kind == pg::StencilKind::backward;
            if (!(centered && s.first < s.central && s.central < s.second)
                && !(forward && s.central < s.first && s.first < s.second)
                && !(backward && s.second < s.first && s.first < s.central))
                throw std::invalid_argument("Stencil orientation disagrees with its endpoints.");
            if (first_requested) {
                gradients[name] = result.gradients[row*k+i];
                if (stochastic) errors[name] = result.gradient_errors[row*k+i];
            }
            if (second_requested) {
                const auto& diagonal = result.diagonal_stencils[row*k+i];
                if (diagonal.node_count < 3U || diagonal.node_count > 4U
                    || diagonal.parameter_values[0U] != s.central
                    || diagonal.parameter_values[1U] != s.first
                    || diagonal.parameter_values[2U] != s.second
                    || diagonal.kind != s.kind
                    || diagonal.displacement != s.displacement
                    || diagonal.represented_width != s.represented_width)
                    throw std::invalid_argument("Diagonal stencil disagrees with its first-order nodes.");
                for (unsigned int node = 0U; node < diagonal.node_count; ++node) {
                    if (!std::isfinite(diagonal.parameter_values[node])
                        || !std::isfinite(diagonal.second_weights[node]))
                        throw std::invalid_argument("Non-finite diagonal stencil node or weight.");
                }
                diagonal_hessians[name] = result.diagonal_hessians[row*k+i];
                if (stochastic) diagonal_errors[name] =
                    result.diagonal_hessian_errors[row*k+i];
            }
            stencils[name] = {{"kind", centered ? "centered" : forward ? "forward" : "backward"},
                {"central", s.central}, {"first", s.first}, {"second", s.second},
                {"displacement", s.displacement}, {"represented_width", s.represented_width},
                {"first_weight", s.first_weight}, {"second_weight", s.second_weight}};
            if (second_requested) {
                const auto& diagonal = result.diagonal_stencils[row*k+i];
                stencils[name]["node_count"] = diagonal.node_count;
                if (diagonal.node_count == 4U) {
                    stencils[name]["third"] = diagonal.parameter_values[3U];
                }
                auto weights = nlohmann::ordered_json::array();
                for (unsigned int node = 0U; node < diagonal.node_count; ++node)
                    weights.push_back(diagonal.second_weights[node]);
                stencils[name]["second_weights"] = std::move(weights);
            }
        }
        nlohmann::ordered_json outputs{{"price", result.prices[row]}};
        if (first_requested) outputs["gradients"] = gradients;
        if (second_requested) outputs["diagonal_hessians"] = diagonal_hessians;
        if (stochastic) {
            outputs["standard_error"] = result.price_errors[row];
            if (first_requested) outputs["gradient_standard_errors"] = errors;
            if (second_requested) outputs["diagonal_hessian_standard_errors"] = diagonal_errors;
        }
        nlohmann::ordered_json output_row{
            {"id", format_row_id(row)},
            {"stencils", stencils},
            {"outputs", outputs},
        };
        if (curve_rows == nullptr) {
            const auto indices = decode_model_product_result_index(
                row, product_rows.size(), recipe.construction
            );
            output_row["model_id"] =
                model_rows.at(indices.model_index).at("id");
            output_row["product_id"] =
                product_rows.at(indices.product_index).at("id");
        } else {
            const auto indices = decode_model_curve_product_result_index(
                row,
                curve_rows->size(),
                product_rows.size(),
                recipe.construction
            );
            output_row["model_id"] =
                model_rows.at(indices.model_index).at("id");
            output_row["curve_id"] =
                curve_rows->at(indices.curve_index).at("id");
            output_row["product_id"] =
                product_rows.at(indices.product_index).at("id");
        }
        rows.push_back(std::move(output_row));
    }
    const auto reference = [](const auto& input) {
        return nlohmann::ordered_json{{"id", input.at("database_id")}, {"catalog", input.at("catalog")}, {"url", input.at("url")}};
    };
    const auto construction = recipe.construction == PriceConstruction::Aligned
        ? nlohmann::ordered_json{{"method", "Aligned"}}
        : nlohmann::ordered_json{
            {"method", "Cartesian product"},
            {"order", curve_rows == nullptr
                ? "model, product" : "model, curve, product"},
        };
    nlohmann::ordered_json metadata{{"title", recipe.dataset.stem().string()}, {"database_id", recipe.dataset.stem().string()},
        {"catalog", recipe.catalog.parent_path().generic_string()}, {"url", recipe.url}, {"row_count", count},
        {"time_convention", products.at("time_convention")}, {"model_dataset", reference(models)}, {"product_dataset", reference(products)},
        {"price_construction", construction}, {"sensitivity", sensitivity_metadata(recipe, stochastic)},
        {"summary", result.execution}, {"validation", {{"status","pending"},{"verified",false}}},
        {"qualification", "bounded implementation checks; bump, discretization and performance not globally qualified"},
        {"standard_error_scope", stochastic ? "paired sampling error; excludes finite-difference and discretization bias" : "not applicable"},
        {"timing", {{"wall_seconds",result.wall_seconds},{"kernel_seconds",result.kernel_seconds}}}};
    if (curve_rows != nullptr) {
        metadata["curve_dataset"] = reference(curves);
    }
    if (recipe.exact_transition)
        metadata["time_representation"] = {{"kind","exact_terminal_transition"},
            {"contractual_days_per_year",steps_per_year/recipe.time.simulation_steps_per_day},
            {"maturity_bump_steps_per_year",steps_per_year}};
    else metadata["time_grid"] = {{"steps_per_year",steps_per_year},
        {"simulation_steps_per_day",recipe.time.simulation_steps_per_day},
        {"delta_t","1 / " + std::to_string(steps_per_year)}};
    auto document = metadata;
    document.erase("title");
    document["results"] = std::move(rows);
    write_json_file(recipe.dataset, document);
    nlohmann::ordered_json receipt_execution = result.execution;
    receipt_execution["paths_per_price"] = result.execution.value(
        "paths_per_price",
        result.execution.value("monte_carlo_paths_per_price", 0U)
    );
    write_generation_receipt(
        recipe.catalog,
        count,
        receipt_execution,
        result.wall_seconds,
        result.kernel_seconds
    );
}
}  // namespace ai_factory::workbench::datasets::price_gradients

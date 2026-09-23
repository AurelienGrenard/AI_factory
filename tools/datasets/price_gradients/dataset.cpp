// Validate complete gradient rows before staging their native JSON/YAML artifacts.
#include "tools/datasets/price_gradients/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include "common/result_index.cuh"
#include <cmath>

namespace ai_factory::workbench::datasets::price_gradients {
namespace pg = ::ai_factory::workbench::price_gradients;

nlohmann::ordered_json sensitivity_metadata(const Recipe& recipe) {
    auto selections = nlohmann::ordered_json::array();
    for (const auto& selection : recipe.configuration.sensitivities)
        selections.push_back({{"parameter", selection.parameter}, {"displacement", selection.bump.displacement},
            {"scale", selection.bump.scale == pg::BumpScale::absolute ? "absolute" : "relative"},
            {"boundary", selection.bump.boundary == pg::BoundaryRule::central_only ? "central_only" : "central_then_one_sided_order2"}});
    return {{"method", "finite_difference_shared_innovations"}, {"parameters", selections},
            {"source_price_recipe", recipe.source_price_recipe}};
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
    const auto models = read_json_file(recipe.model_input), products = read_json_file(recipe.product_input);
    const auto& model_rows = models.at("models");
    const auto& product_rows = products.at("products");
    const auto count = price_row_count(model_rows.size(), product_rows.size(), recipe.construction);
    const auto k = recipe.configuration.sensitivities.size();
    if (k && count > std::numeric_limits<std::size_t>::max()/k) throw std::overflow_error("Gradient output cardinality overflow.");
    const bool stochastic = result.execution.at("paths_per_price").get<std::size_t>() != 0U;
    if (result.prices.size() != count || result.gradients.size() != count*k || result.stencils.size() != count*k
        || result.price_errors.size() != (stochastic ? count : 0U) || result.gradient_errors.size() != (stochastic ? count*k : 0U)
        || result.execution.at("sensitivity_count").get<std::size_t>() != k
        || result.execution.at("scenario_count").get<std::size_t>() != 1U+2U*k)
        throw std::invalid_argument("Incomplete price-gradient outputs or inconsistent execution shape.");
    for (const auto* values : {&result.prices, &result.gradients, &result.price_errors, &result.gradient_errors})
        for (float value : *values) if (!std::isfinite(value)) throw std::invalid_argument("Non-finite gradient artifact value.");
    for (const auto* errors : {&result.price_errors, &result.gradient_errors})
        for (float error : *errors) if (error < 0) throw std::invalid_argument("Negative gradient sampling error.");
    if (!std::isfinite(result.wall_seconds) || !std::isfinite(result.kernel_seconds)
        || result.wall_seconds < 0 || result.kernel_seconds < 0) throw std::invalid_argument("Invalid gradient execution timing.");
    auto rows = nlohmann::ordered_json::array();
    for (std::size_t row = 0; row < count; ++row) {
        auto gradients = nlohmann::ordered_json::object();
        auto stencils = nlohmann::ordered_json::object();
        auto errors = nlohmann::ordered_json::object();
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
            gradients[name] = result.gradients[row*k+i];
            if (stochastic) errors[name] = result.gradient_errors[row*k+i];
            stencils[name] = {{"kind", centered ? "centered" : forward ? "forward" : "backward"},
                {"central", s.central}, {"first", s.first}, {"second", s.second},
                {"displacement", s.displacement}, {"represented_width", s.represented_width},
                {"first_weight", s.first_weight}, {"second_weight", s.second_weight}};
        }
        const auto indices = decode_model_product_result_index(row, product_rows.size(), recipe.construction);
        nlohmann::ordered_json outputs{{"price", result.prices[row]}, {"gradients", gradients}};
        if (stochastic) { outputs["standard_error"] = result.price_errors[row]; outputs["gradient_standard_errors"] = errors; }
        rows.push_back({{"id", format_row_id(row)}, {"model_id", model_rows.at(indices.model_index).at("id")},
            {"product_id", product_rows.at(indices.product_index).at("id")}, {"stencils", stencils}, {"outputs", outputs}});
    }
    const auto reference = [](const auto& input) {
        return nlohmann::ordered_json{{"id", input.at("database_id")}, {"catalog", input.at("catalog")}, {"url", input.at("url")}};
    };
    const auto construction = recipe.construction == PriceConstruction::Aligned
        ? nlohmann::ordered_json{{"method", "Aligned"}}
        : nlohmann::ordered_json{{"method", "Cartesian product"}, {"order", "model, product"}};
    nlohmann::ordered_json metadata{{"title", recipe.dataset.stem().string()}, {"database_id", recipe.dataset.stem().string()},
        {"catalog", recipe.catalog.parent_path().generic_string()}, {"url", recipe.url}, {"row_count", count},
        {"time_convention", products.at("time_convention")}, {"model_dataset", reference(models)}, {"product_dataset", reference(products)},
        {"price_construction", construction}, {"sensitivity", sensitivity_metadata(recipe)},
        {"summary", result.execution}, {"validation", {{"status","pending"},{"verified",false}}},
        {"qualification", "bounded implementation checks; bump, discretization and performance not globally qualified"},
        {"standard_error_scope", stochastic ? "paired sampling error; excludes finite-difference and discretization bias" : "not applicable"},
        {"timing", {{"wall_seconds",result.wall_seconds},{"kernel_seconds",result.kernel_seconds}}}};
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

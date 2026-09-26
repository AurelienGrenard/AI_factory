// Validate complete sensitivity rows before staging native JSON/YAML artifacts.
#include "tools/datasets/price_gradients/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include "common/result_index.cuh"

#include <cmath>
#include <limits>
#include <string>

namespace ai_factory::workbench::datasets::price_gradients {
namespace pg = ::ai_factory::workbench::price_gradients;

namespace {

pg::SensitivityRequest effective_request(const Recipe& recipe) {
    return recipe.sensitivity_request.value_or(
        pg::SensitivityRequest{recipe.orders}
    );
}

pg::SensitivityGraphPlan effective_graph(const Recipe& recipe) {
    return pg::make_sensitivity_graph_plan(
        effective_request(recipe),
        recipe.configuration.sensitivities.size()
    );
}

bool same_graph(
    const pg::SensitivityGraphPlan& first,
    const pg::SensitivityGraphPlan& second
) {
    return first.first == second.first
        && first.diagonal_second == second.diagonal_second
        && first.mixed_second == second.mixed_second
        && first.coordinate_uses == second.coordinate_uses
        && first.node_capacity == second.node_capacity;
}

std::string mixed_name(
    const Recipe& recipe,
    pg::SensitivityPair pair
) {
    return recipe.configuration.sensitivities[pair.first].parameter + "|"
        + recipe.configuration.sensitivities[pair.second].parameter;
}

std::size_t checked_product(
    std::size_t first,
    std::size_t second,
    const char* label
) {
    if (first != 0U
        && second > std::numeric_limits<std::size_t>::max() / first) {
        throw std::overflow_error(label);
    }
    return first * second;
}

}  // namespace

nlohmann::ordered_json sensitivity_metadata(
    const Recipe& recipe,
    bool stochastic
) {
    const auto graph = effective_graph(recipe);
    auto selections = nlohmann::ordered_json::array();
    for (const auto& selection : recipe.configuration.sensitivities) {
        selections.push_back({
            {"parameter", selection.parameter},
            {"displacement", selection.bump.displacement},
            {"scale", selection.bump.scale == pg::BumpScale::absolute
                ? "absolute" : "relative"},
            {"boundary", selection.bump.boundary
                == pg::BoundaryRule::central_only
                    ? "central_only"
                    : "central_then_one_sided_order2"},
        });
    }
    nlohmann::ordered_json metadata{
        {"method", stochastic
            ? "finite_difference_shared_innovations"
            : "finite_difference"},
        {"parameters", selections},
        {"source_price_recipe", recipe.source_price_recipe},
    };
    auto orders = nlohmann::ordered_json::array();
    if (!graph.first.empty()) orders.push_back("first");
    if (!graph.diagonal_second.empty()) orders.push_back("diagonal_second");
    if (!graph.mixed_second.empty()) orders.push_back("mixed_second");
    if (orders != nlohmann::ordered_json::array({"first"})
        || recipe.sensitivity_request.has_value()) {
        metadata["orders"] = std::move(orders);
    }
    const auto request = effective_request(recipe);
    if (request.has_selected_outputs()) {
        const auto selected_names = [&](const auto& coordinates) {
            auto result = nlohmann::ordered_json::array();
            for (const auto coordinate : coordinates) {
                result.push_back(
                    recipe.configuration.sensitivities[
                        coordinate
                    ].parameter
                );
            }
            return result;
        };
        if (!graph.first.empty()) {
            metadata["first"] = selected_names(graph.first);
        }
        if (!graph.diagonal_second.empty()) {
            metadata["diagonal_second"] =
                selected_names(graph.diagonal_second);
        }
    }
    if (!graph.mixed_second.empty()) {
        if (request.all_mixed_second && !request.has_selected_outputs()) {
            metadata["mixed_second"] = "all";
        } else {
            auto pairs = nlohmann::ordered_json::array();
            for (const auto pair : graph.mixed_second) {
                pairs.push_back({
                    {"first", recipe.configuration.sensitivities[
                        pair.first
                    ].parameter},
                    {"second", recipe.configuration.sensitivities[
                        pair.second
                    ].parameter},
                });
            }
            metadata["mixed_second"] = std::move(pairs);
        }
    }
    return metadata;
}

void write_dataset(const Recipe& recipe, const Results& result) {
    validate_dataset_url(recipe.url);
    recipe.configuration.validate(recipe.configuration.sensitivities.size());
    recipe.time.validate();

    const double reciprocal_dt = 1.0 / recipe.time.dt;
    if (reciprocal_dt < 1.0
        || reciprocal_dt > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Gradient artifact requires an integer steps-per-year grid."
        );
    }
    const auto steps_per_year =
        static_cast<std::uint32_t>(std::llround(reciprocal_dt));
    if (1.0f / static_cast<float>(steps_per_year) != recipe.time.dt) {
        throw std::invalid_argument(
            "Gradient artifact dt is not a reciprocal integer grid."
        );
    }

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

    const auto sensitivity_count =
        recipe.configuration.sensitivities.size();
    const auto graph = effective_graph(recipe);
    if (!result.sensitivity_graph.coordinate_uses.empty()
        && !same_graph(graph, result.sensitivity_graph)) {
        throw std::invalid_argument(
            "Gradient artifact graph differs from its recipe."
        );
    }
    const auto first_count = graph.first.size();
    const auto diagonal_count = graph.diagonal_second.size();
    const auto mixed_count = graph.mixed_second.size();
    if (graph.output_count() == 1U) {
        throw std::invalid_argument(
            "Gradient artifact has no requested sensitivity output."
        );
    }

    const auto axis_stencil_count = checked_product(
        count, sensitivity_count, "Gradient stencil cardinality overflow."
    );
    const auto first_output_count = checked_product(
        count, first_count, "Gradient output cardinality overflow."
    );
    const auto diagonal_output_count = checked_product(
        count, diagonal_count, "Diagonal output cardinality overflow."
    );
    const auto mixed_output_count = checked_product(
        count, mixed_count, "Mixed output cardinality overflow."
    );
    const bool stochastic =
        result.execution.at("paths_per_price").get<std::size_t>() != 0U;
    const auto& scenario_count = result.execution.at("scenario_count");
    const bool scenario_shape_valid = scenario_count.is_null()
        ? result.execution.value("materialized_scenario_count", 1U) == 0U
        : scenario_count.get<std::size_t>()
            == 1U + 2U * sensitivity_count;

    if (result.prices.size() != count
        || result.gradients.size() != first_output_count
        || result.stencils.size() != axis_stencil_count
        || result.diagonal_hessians.size() != diagonal_output_count
        || result.diagonal_stencils.size()
            != (diagonal_count == 0U ? 0U : axis_stencil_count)
        || result.mixed_hessians.size() != mixed_output_count
        || result.mixed_stencils.size() != mixed_output_count
        || result.price_errors.size() != (stochastic ? count : 0U)
        || result.gradient_errors.size()
            != (stochastic ? first_output_count : 0U)
        || result.diagonal_hessian_errors.size()
            != (stochastic ? diagonal_output_count : 0U)
        || result.mixed_hessian_errors.size()
            != (stochastic ? mixed_output_count : 0U)
        || result.execution.at("sensitivity_count").get<std::size_t>()
            != sensitivity_count
        || !scenario_shape_valid) {
        throw std::invalid_argument(
            "Incomplete price-gradient outputs or inconsistent execution shape."
        );
    }

    for (const auto* values : {
             &result.prices,
             &result.gradients,
             &result.price_errors,
             &result.gradient_errors,
             &result.diagonal_hessians,
             &result.diagonal_hessian_errors,
             &result.mixed_hessians,
             &result.mixed_hessian_errors,
         }) {
        for (const float value : *values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Non-finite gradient artifact value."
                );
            }
        }
    }
    for (const auto* errors : {
             &result.price_errors,
             &result.gradient_errors,
             &result.diagonal_hessian_errors,
             &result.mixed_hessian_errors,
         }) {
        for (const float error : *errors) {
            if (error < 0.0f) {
                throw std::invalid_argument(
                    "Negative gradient sampling error."
                );
            }
        }
    }
    if (!std::isfinite(result.wall_seconds)
        || !std::isfinite(result.kernel_seconds)
        || result.wall_seconds < 0.0
        || result.kernel_seconds < 0.0) {
        throw std::invalid_argument("Invalid gradient execution timing.");
    }

    auto rows = nlohmann::ordered_json::array();
    for (std::size_t row = 0U; row < count; ++row) {
        auto gradients = nlohmann::ordered_json::object();
        auto diagonal_hessians = nlohmann::ordered_json::object();
        auto mixed_hessians = nlohmann::ordered_json::object();
        auto stencils = nlohmann::ordered_json::object();
        auto mixed_stencils = nlohmann::ordered_json::object();
        auto gradient_errors = nlohmann::ordered_json::object();
        auto diagonal_errors = nlohmann::ordered_json::object();
        auto mixed_errors = nlohmann::ordered_json::object();

        for (std::size_t coordinate = 0U;
             coordinate < sensitivity_count;
             ++coordinate) {
            if (graph.coordinate_uses[coordinate]
                == pg::SensitivityCoordinateUse::none) {
                continue;
            }
            const auto& name =
                recipe.configuration.sensitivities[coordinate].parameter;
            const auto& stencil =
                result.stencils[row * sensitivity_count + coordinate];
            if (!std::isfinite(stencil.central)
                || !std::isfinite(stencil.first)
                || !std::isfinite(stencil.second)
                || !std::isfinite(stencil.displacement)
                || !(stencil.displacement > 0.0f)
                || !std::isfinite(stencil.first_weight)
                || !std::isfinite(stencil.second_weight)
                || !std::isfinite(stencil.represented_width)
                || stencil.first == stencil.second
                || stencil.represented_width
                    != stencil.second - stencil.first) {
                throw std::invalid_argument(
                    "Invalid represented gradient stencil."
                );
            }
            const bool centered =
                stencil.kind == pg::StencilKind::centered;
            const bool forward =
                stencil.kind == pg::StencilKind::forward;
            const bool backward =
                stencil.kind == pg::StencilKind::backward;
            if (!(centered
                    && stencil.first < stencil.central
                    && stencil.central < stencil.second)
                && !(forward
                    && stencil.central < stencil.first
                    && stencil.first < stencil.second)
                && !(backward
                    && stencil.second < stencil.first
                    && stencil.first < stencil.central)) {
                throw std::invalid_argument(
                    "Stencil orientation disagrees with its endpoints."
                );
            }

            stencils[name] = {
                {"kind", centered
                    ? "centered" : forward ? "forward" : "backward"},
                {"central", stencil.central},
                {"first", stencil.first},
                {"second", stencil.second},
                {"displacement", stencil.displacement},
                {"represented_width", stencil.represented_width},
                {"first_weight", stencil.first_weight},
                {"second_weight", stencil.second_weight},
            };

            if (pg::has_coordinate_use(
                    graph.coordinate_uses[coordinate],
                    pg::SensitivityCoordinateUse::diagonal_second
                )) {
                const auto& diagonal =
                    result.diagonal_stencils[
                        row * sensitivity_count + coordinate
                    ];
                if (diagonal.node_count < 3U
                    || diagonal.node_count > 4U
                    || diagonal.parameter_values[0U] != stencil.central
                    || diagonal.parameter_values[1U] != stencil.first
                    || diagonal.parameter_values[2U] != stencil.second
                    || diagonal.kind != stencil.kind
                    || diagonal.displacement != stencil.displacement
                    || diagonal.represented_width
                        != stencil.represented_width) {
                    throw std::invalid_argument(
                        "Diagonal stencil disagrees with its first-order nodes."
                    );
                }
                for (unsigned int node = 0U;
                     node < diagonal.node_count;
                     ++node) {
                    if (!std::isfinite(diagonal.parameter_values[node])
                        || !std::isfinite(diagonal.second_weights[node])) {
                        throw std::invalid_argument(
                            "Non-finite diagonal stencil node or weight."
                        );
                    }
                }
                stencils[name]["node_count"] = diagonal.node_count;
                if (diagonal.node_count == 4U) {
                    stencils[name]["third"] =
                        diagonal.parameter_values[3U];
                }
                auto weights = nlohmann::ordered_json::array();
                for (unsigned int node = 0U;
                     node < diagonal.node_count;
                     ++node) {
                    weights.push_back(diagonal.second_weights[node]);
                }
                stencils[name]["second_weights"] = std::move(weights);
            }
        }

        for (std::size_t selected = 0U;
             selected < first_count;
             ++selected) {
            const auto& name = recipe.configuration.sensitivities[
                graph.first[selected]
            ].parameter;
            gradients[name] = result.gradients[
                row * first_count + selected
            ];
            if (stochastic) {
                gradient_errors[name] = result.gradient_errors[
                    row * first_count + selected
                ];
            }
        }
        for (std::size_t selected = 0U;
             selected < diagonal_count;
             ++selected) {
            const auto& name = recipe.configuration.sensitivities[
                graph.diagonal_second[selected]
            ].parameter;
            diagonal_hessians[name] = result.diagonal_hessians[
                row * diagonal_count + selected
            ];
            if (stochastic) {
                diagonal_errors[name] =
                    result.diagonal_hessian_errors[
                        row * diagonal_count + selected
                    ];
            }
        }
        for (std::size_t selected = 0U;
             selected < mixed_count;
             ++selected) {
            const auto label = mixed_name(
                recipe, graph.mixed_second[selected]
            );
            mixed_hessians[label] = result.mixed_hessians[
                row * mixed_count + selected
            ];
            if (stochastic) {
                mixed_errors[label] = result.mixed_hessian_errors[
                    row * mixed_count + selected
                ];
            }
            const auto& stencil =
                result.mixed_stencils[row * mixed_count + selected];
            if (stencil.node_count == 0U
                || stencil.node_count > pg::kMixedSensitivityNodeCapacity) {
                throw std::invalid_argument(
                    "Invalid mixed sensitivity stencil cardinality."
                );
            }
            auto weights = nlohmann::ordered_json::array();
            auto first_nodes = nlohmann::ordered_json::array();
            auto second_nodes = nlohmann::ordered_json::array();
            for (unsigned int node = 0U;
                 node < stencil.node_count;
                 ++node) {
                if (!std::isfinite(stencil.weights[node])
                    || stencil.first_local_nodes[node] > 2U
                    || stencil.second_local_nodes[node] > 2U) {
                    throw std::invalid_argument(
                        "Invalid mixed sensitivity stencil support."
                    );
                }
                weights.push_back(stencil.weights[node]);
                first_nodes.push_back(stencil.first_local_nodes[node]);
                second_nodes.push_back(stencil.second_local_nodes[node]);
            }
            mixed_stencils[label] = {
                {"node_count", stencil.node_count},
                {"weights", std::move(weights)},
                {"first_local_nodes", std::move(first_nodes)},
                {"second_local_nodes", std::move(second_nodes)},
            };
        }

        nlohmann::ordered_json outputs{{"price", result.prices[row]}};
        if (first_count != 0U) outputs["gradients"] = gradients;
        if (diagonal_count != 0U) {
            outputs["diagonal_hessians"] = diagonal_hessians;
        }
        if (mixed_count != 0U) outputs["mixed_hessians"] = mixed_hessians;
        if (stochastic) {
            outputs["standard_error"] = result.price_errors[row];
            if (first_count != 0U) {
                outputs["gradient_standard_errors"] = gradient_errors;
            }
            if (diagonal_count != 0U) {
                outputs["diagonal_hessian_standard_errors"] =
                    diagonal_errors;
            }
            if (mixed_count != 0U) {
                outputs["mixed_hessian_standard_errors"] = mixed_errors;
            }
        }

        nlohmann::ordered_json output_row{
            {"id", format_row_id(row)},
            {"stencils", stencils},
            {"outputs", outputs},
        };
        if (mixed_count != 0U) {
            output_row["mixed_stencils"] = mixed_stencils;
        }
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
        return nlohmann::ordered_json{
            {"id", input.at("database_id")},
            {"catalog", input.at("catalog")},
            {"url", input.at("url")},
        };
    };
    const auto construction =
        recipe.construction == PriceConstruction::Aligned
            ? nlohmann::ordered_json{{"method", "Aligned"}}
            : nlohmann::ordered_json{
                {"method", "Cartesian product"},
                {"order", curve_rows == nullptr
                    ? "model, product" : "model, curve, product"},
            };
    nlohmann::ordered_json metadata{
        {"title", recipe.dataset.stem().string()},
        {"database_id", recipe.dataset.stem().string()},
        {"catalog", recipe.catalog.parent_path().generic_string()},
        {"url", recipe.url},
        {"row_count", count},
        {"time_convention", products.at("time_convention")},
        {"model_dataset", reference(models)},
        {"product_dataset", reference(products)},
        {"price_construction", construction},
        {"sensitivity", sensitivity_metadata(recipe, stochastic)},
        {"summary", result.execution},
        {"validation", {{"status", "pending"}, {"verified", false}}},
        {"qualification", "bounded implementation checks; bump, discretization and performance not globally qualified"},
        {"standard_error_scope", stochastic
            ? "paired sampling error; excludes finite-difference and discretization bias"
            : "not applicable"},
        {"timing", {
            {"wall_seconds", result.wall_seconds},
            {"kernel_seconds", result.kernel_seconds},
        }},
    };
    if (curve_rows != nullptr) {
        metadata["curve_dataset"] = reference(curves);
    }
    if (recipe.exact_transition) {
        metadata["time_representation"] = {
            {"kind", "exact_terminal_transition"},
            {"contractual_days_per_year",
                steps_per_year / recipe.time.simulation_steps_per_day},
            {"maturity_bump_steps_per_year", steps_per_year},
        };
    } else {
        metadata["time_grid"] = {
            {"steps_per_year", steps_per_year},
            {"simulation_steps_per_day",
                recipe.time.simulation_steps_per_day},
            {"delta_t", "1 / " + std::to_string(steps_per_year)},
        };
    }

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

// Parameter-dataset document assembly and publication.
#include "tools/datasets/parameter_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include "datasets/validation.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ai_factory::workbench::datasets {
namespace {
nlohmann::ordered_json database_rows(
    const std::vector<ParameterRow>& parameters
) {
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        rows.push_back({
            {"id", format_row_id(index)},
            {"parameters", parameters[index]},
        });
    }
    return rows;
}

void write_parameter_dataset(
    const std::string& database_id,
    const std::string& family,
    const std::string& family_key,
    const std::string& row_key,
    const std::string& definition_key,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& definition,
    const GeneratedRows& generated,
    const nlohmann::ordered_json& metadata
) {
    if (generated.rows.empty()) {
        throw std::invalid_argument("A parameter dataset cannot be empty.");
    }
    validate_dataset_url(url);

    nlohmann::ordered_json dataset = {
        {"database_id", database_id},
        {family_key, family},
        {"row_count", generated.rows.size()},
    };
    for (const auto& [key, value] : metadata.items()) dataset[key] = value;
    dataset[row_key] = database_rows(generated.rows);
    write_json_file(dataset_path, dataset);

    nlohmann::ordered_json recipe = {
        {"schema_version", 1},
        {"kind", row_key == "models" ? "model_parameters"
                : row_key == "curves" ? "curve_parameters"
                : "product_parameters"},
        {"dataset_id", database_id},
        {"generator", (catalog_path.parent_path() / "generator.cpp").generic_string()},
        {"output", {
            {"path", dataset_path.generic_string()},
            {"format", "json"},
        }},
        {family_key, family},
        {"row_count", generated.rows.size()},
    };
    if (metadata.contains("parameterization")) {
        recipe["parameterization"] = metadata.at("parameterization");
    }
    if (metadata.contains("source_model_dataset")) {
        recipe["inputs"] = {
            {"source_model", metadata.at("source_model_dataset")}
        };
    }
    if (metadata.contains("time_convention")) {
        recipe["time_convention"] = metadata.at("time_convention");
    }
    if (metadata.contains("quote_convention")) {
        recipe["quote_convention"] = metadata.at("quote_convention");
    }
    recipe["parameters"] = parameter_descriptions;
    if (row_key == "products") {
        auto compact_payoff = definition;
        compact_payoff.erase("payoff_homogeneity");
        compact_payoff.erase("spot_level_quote");
        recipe[definition_key] = compact_payoff;
    } else if (row_key != "curves") {
        recipe[definition_key] = definition;
    }
    recipe["construction"] = generated.construction;
    write_yaml_document(catalog_path.parent_path() / "recipe.yaml", recipe);

    write_generation_receipt(
        catalog_path,
        generated.rows.size(),
        {{"mode", "native_parameter_generator"}},
        0.0,
        0.0
    );
}

}  // namespace

void write_model_dataset(
    const std::string& database_id,
    const std::string& model_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& dynamics,
    const GeneratedRows& generated
) {
    write_parameter_dataset(
        database_id, model_family, "model_family", "models", "dynamics",
        dataset_path, catalog_path, url, parameter_descriptions, dynamics,
        generated,
        {}
    );
}

void write_factorized_model_dataset(
    const std::filesystem::path& source_dataset_path,
    const std::string& database_id,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url
) {
    const auto source = read_json_file(source_dataset_path);
    validate_model_dataset(source);
    GeneratedRows generated;
    nlohmann::ordered_json descriptions =
        nlohmann::ordered_json::object();
    for (const auto& row : source.at("models")) {
        auto parameters = row.at("parameters");
        parameters.erase("risk_free_rate");
        parameters.erase("rho");
        parameters.erase("correlation");
        if (parameters.empty())
            throw std::invalid_argument("Empty factorized model row");
        for (const auto& [name, value] : parameters.items())
            descriptions[name] = "Model parameter.";
        generated.rows.push_back(std::move(parameters));
    }
    generated.construction = {
        {"method", "projection of complete model parameters"},
        {"source_model_dataset", source_dataset_path.generic_string()},
        {"excluded", {"risk_free_rate", "rho", "correlation"}}
    };
    write_parameter_dataset(
        database_id, source.at("model_family").get<std::string>(),
        "model_family", "models", "dynamics", dataset_path, catalog_path,
        url, descriptions,
        {{"source_model_dataset", source_dataset_path.generic_string()}},
        generated,
        {{"parameterization", "factorized"},
         {"source_model_dataset", source_dataset_path.generic_string()}}
    );
    validate_model_dataset_file(dataset_path);
}

void write_core_factorized_model_dataset(
    const std::filesystem::path& source_dataset_path,
    const std::string& database_id,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    std::size_t core_prefix_count,
    std::size_t row_count,
    std::uint64_t seed
) {
    const auto source = read_json_file(source_dataset_path);
    validate_model_dataset(source);
    const auto& source_rows = source.at("models");
    if (row_count == 0U || core_prefix_count < row_count
        || core_prefix_count > source_rows.size())
        throw std::invalid_argument("Invalid core factor sample size");
    std::vector<std::size_t> indices(core_prefix_count);
    std::iota(indices.begin(), indices.end(), 0U);
    std::mt19937_64 random(seed);
    std::shuffle(indices.begin(), indices.end(), random);
    indices.resize(row_count);
    GeneratedRows generated;
    nlohmann::ordered_json descriptions =
        nlohmann::ordered_json::object();
    for (const auto index : indices) {
        auto parameters = source_rows.at(index).at("parameters");
        parameters.erase("risk_free_rate");
        parameters.erase("rho");
        parameters.erase("correlation");
        if (parameters.empty())
            throw std::invalid_argument("Empty factorized core row");
        for (const auto& [name, value] : parameters.items())
            descriptions[name] = "Model factor parameter.";
        generated.rows.push_back(std::move(parameters));
    }
    generated.construction = {
        {"method", "seeded sample without replacement from core prefix"},
        {"source_model_dataset", source_dataset_path.generic_string()},
        {"source_core_prefix_rows", core_prefix_count},
        {"sample_seed", seed},
        {"source_row_indices", indices},
        {"excluded", {"risk_free_rate", "rho", "correlation"}}
    };
    write_parameter_dataset(
        database_id, source.at("model_family").get<std::string>(),
        "model_family", "models", "dynamics", dataset_path, catalog_path,
        url, descriptions,
        {{"source_model_dataset", source_dataset_path.generic_string()}},
        generated,
        {{"parameterization", "factorized"},
         {"source_model_dataset", source_dataset_path.generic_string()}}
    );
    validate_model_dataset_file(dataset_path);
}

void write_curve_dataset(
    const std::string& database_id,
    const std::string& curve_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& curve_definition,
    const GeneratedRows& generated
) {
    write_parameter_dataset(
        database_id, curve_family, "curve_family", "curves", "curve",
        dataset_path, catalog_path, url, parameter_descriptions,
        curve_definition, generated,
        {}
    );
}

void write_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated,
    const nlohmann::ordered_json& quote_convention
) {
    nlohmann::ordered_json metadata = {{"time_convention", {
        {"unit", "business_day"},
        {"days_per_year", kBusinessDaysPerYear},
    }}};
    if (!quote_convention.empty()) {
        metadata["quote_convention"] = quote_convention;
    }
    write_parameter_dataset(
        database_id, product_family, "product_family", "products", "payoff",
        dataset_path, catalog_path, url, parameter_descriptions, payoff,
        generated, metadata
    );
}


void write_fixed_income_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated,
    const nlohmann::ordered_json& quote_convention
) {
    write_parameter_dataset(
        database_id, product_family, "product_family", "products", "payoff",
        dataset_path, catalog_path, url, parameter_descriptions, payoff,
        generated,
        {{"time_convention", {
            {"unit", "business_day"},
            {"days_per_year", kBusinessDaysPerYear},
        }}, {"quote_convention", quote_convention}}
    );
}

void write_spot_relative_equity_product_dataset(
    const std::string& database_id,
    const std::string& product_family,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const nlohmann::ordered_json& parameter_descriptions,
    const nlohmann::ordered_json& payoff,
    const GeneratedRows& generated
) {
    auto descriptions = parameter_descriptions;
    for (const char* name : {
            "strike", "trigger_strike", "payoff_strike", "barrier",
            "lower_barrier", "upper_barrier", "autocall_barrier",
            "coupon_barrier", "protection_barrier"
        }) {
        if (descriptions.contains(name)) {
            descriptions[name] =
                "Dimensionless ratio of the quoted spot level to the "
                "central model spot S0; 1.0 means 100% of S0.";
        }
    }
    auto resolved_payoff = payoff;
    resolved_payoff["spot_level_quote"] = {
        {"unit", "ratio_to_central_spot"},
        {"resolution", "Each quoted strike/barrier is multiplied by the "
            "unbumped model S0 once for each model-product pair."},
        {"sensitivity", "S0 reference and resolved contract levels remain "
            "fixed when model.spot is bumped; product-level derivatives "
            "are with respect to the quoted ratio."},
    };
    nlohmann::ordered_json quote_convention = nlohmann::ordered_json::object();
    for (const char* name : {
            "strike", "trigger_strike", "payoff_strike", "barrier",
            "lower_barrier", "upper_barrier", "autocall_barrier",
            "coupon_barrier", "protection_barrier"
        }) {
        if (descriptions.contains(name)) {
            quote_convention[name] = {
                {"mode", "ratio_to_central_spot"},
                {"unit", "dimensionless"},
                {"effective_level", std::string(name) + " * S0"},
            };
        }
    }
    quote_convention["reference_policy"] =
        "Resolved at the unbumped model spot S0 for each model-product pair; "
        "absolute contract levels remain fixed for model bumps.";
    write_product_dataset(
        database_id, product_family, dataset_path, catalog_path, url,
        descriptions, resolved_payoff, generated, quote_convention
    );
}

}  // namespace ai_factory::workbench::datasets

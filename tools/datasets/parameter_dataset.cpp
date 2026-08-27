// Parameter-dataset document assembly and publication.
#include "tools/datasets/parameter_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

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
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", generated.rows.size()},
    };
    for (const auto& [key, value] : metadata.items()) dataset[key] = value;
    dataset[row_key] = database_rows(generated.rows);
    write_json_file(dataset_path, dataset);

    nlohmann::ordered_json catalog = {
        {"title", family + " parameter dataset " + database_id},
        {"database_id", database_id},
        {family_key, family},
        {"catalog", catalog_path.parent_path().generic_string()},
        {"url", url},
        {"row_count", generated.rows.size()},
    };
    for (const auto& [key, value] : metadata.items()) catalog[key] = value;
    catalog["parameters"] = parameter_descriptions;
    catalog[definition_key] = definition;
    catalog["construction"] = generated.construction;
    write_catalog_yaml(catalog_path, catalog);
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
    const GeneratedRows& generated
) {
    write_parameter_dataset(
        database_id, product_family, "product_family", "products", "payoff",
        dataset_path, catalog_path, url, parameter_descriptions, payoff,
        generated,
        {{"time_convention", {
            {"unit", "business_day"},
            {"days_per_year", kBusinessDaysPerYear},
        }}}
    );
}

}  // namespace ai_factory::workbench::datasets

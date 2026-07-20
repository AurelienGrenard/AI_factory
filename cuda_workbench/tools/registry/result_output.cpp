// Implementation of the canonical Monte Carlo result JSON/YAML writer.
// It reconstructs result identities and row mappings from source databases.
#include "tools/registry/result_output.hpp"

#include <nlohmann/json.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ai_factory::workbench::result_output {

void write_monte_carlo_result(
    const std::filesystem::path& model_json_path,
    const std::filesystem::path& product_json_path,
    ConstructionMethod construction,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    const std::string& engine,
    const std::string& result_version,
    const std::filesystem::path& output_root,
    const std::filesystem::path& generation_script,
    const std::filesystem::path& source_file,
    const std::string& numerical_method,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::uint64_t first_seed,
    double wall_seconds,
    double kernel_seconds
) {
    const nlohmann::json model_document = registry::read_json(model_json_path);
    const nlohmann::json product_document = registry::read_json(product_json_path);
    const auto& model_rows = model_document.at("models");
    const auto& product_rows = product_document.at("products");
    const std::size_t result_count = registry::constructed_row_count(
        model_rows.size(), product_rows.size(), construction
    );
    if (prices.size() != result_count
        || standard_errors.size() != result_count) {
        throw std::invalid_argument(
            "Result vectors must match the constructed input row count."
        );
    }

    const std::string model_database_id =
        model_document.at("database_id").get<std::string>();
    const std::string product_database_id =
        product_document.at("database_id").get<std::string>();
    const std::string database_id =
        model_database_id + "__" + product_database_id
        + "__" + engine + "_" + result_version;
    const std::filesystem::path json_path =
        output_root / "data" / (database_id + ".json");
    const std::filesystem::path yaml_path =
        output_root / "specifications" / (database_id + ".yaml");
    const std::string construction_rule =
        construction == ConstructionMethod::Aligned
        ? "aligned row pairing"
        : "Cartesian product in model-major order";

    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t index = 0; index < result_count; ++index) {
        const std::size_t model_index =
            construction == ConstructionMethod::Aligned
            ? index
            : index / product_rows.size();
        const std::size_t product_index =
            construction == ConstructionMethod::Aligned
            ? index
            : index % product_rows.size();
        rows.push_back({
            {"id", registry::row_id(index)},
            {"model_id", model_rows.at(model_index).at("id").get<std::string>()},
            {"product_id", product_rows.at(product_index).at("id").get<std::string>()},
            {"seed", first_seed + index},
            {"outputs", {
                {"price", prices[index]},
                {"standard_error", standard_errors[index]},
            }},
        });
    }

    const nlohmann::ordered_json model_database = {
        {"id", model_database_id},
        {"json_path", model_json_path.generic_string()},
    };
    const nlohmann::ordered_json product_database = {
        {"id", product_database_id},
        {"json_path", product_json_path.generic_string()},
    };
    const nlohmann::ordered_json timing = {
        {"wall_seconds", wall_seconds},
        {"kernel_seconds", kernel_seconds},
    };

    const nlohmann::ordered_json document = {
        {"database_id", database_id},
        {"specification", yaml_path.generic_string()},
        {"generation_script", generation_script.generic_string()},
        {"row_count", result_count},
        {"model_database", model_database},
        {"product_database", product_database},
        {"timing", timing},
        {"results", rows},
    };
    registry::write_text_file(json_path, document.dump(2) + "\n");

    const std::string device =
        engine.find("gpu") != std::string::npos ? "gpu" : "cpu";
    const std::string implementation =
        device == "gpu" ? "CUDA" : "C++";
    std::ostringstream yaml;
    yaml << std::setprecision(10)
         << "title: " << nlohmann::json(
                model_database_id + " x " + product_database_id + " " + engine
            ).dump() << '\n'
         << "database_id: " << database_id << '\n'
         << "json_path: " << nlohmann::json(json_path.generic_string()).dump() << '\n'
         << "generation_script: "
         << nlohmann::json(generation_script.generic_string()).dump() << '\n'
         << "summary:\n"
         << "  row_count: " << result_count << '\n'
         << "  monte_carlo_paths_per_price: "
         << monte_carlo_paths_per_price << '\n'
         << "  model: "
         << nlohmann::json(model_document.at("model_family").get<std::string>()).dump()
         << '\n'
         << "  numerical_method: " << nlohmann::json(numerical_method).dump() << '\n'
         << "  payoff: "
         << nlohmann::json(product_document.at("product_family").get<std::string>()).dump()
         << '\n'
         << "  implementation: " << implementation << '\n'
         << "  device: " << device << '\n'
         << "  threads_per_block: " << threads_per_block << '\n'
         << "  source_files:\n"
         << "  - " << nlohmann::json(source_file.generic_string()).dump() << '\n'
         << "time_grid:\n"
         << "  rule: nearest integer step count to target dt\n"
         << "  target_dt: " << target_dt << '\n'
         << "  step_count: round(maturity / target_dt)\n"
         << "  effective_dt: maturity / step_count\n"
         << "outputs:\n"
         << "  price:\n"
         << "    estimator: Monte Carlo discounted payoff mean\n"
         << "  standard_error:\n"
         << "    estimator: Monte Carlo standard error of discounted payoff\n"
         << "model_database:\n"
         << "  id: " << model_database_id << '\n'
         << "  json_path: " << nlohmann::json(model_json_path.generic_string()).dump()
         << '\n'
         << "product_database:\n"
         << "  id: " << product_database_id << '\n'
         << "  json_path: " << nlohmann::json(product_json_path.generic_string()).dump()
         << '\n'
         << "result_construction:\n"
         << "  rule: " << nlohmann::json(construction_rule).dump() << '\n'
         << "timing:\n"
         << "  wall_seconds: " << wall_seconds << '\n'
         << "  kernel_seconds: " << kernel_seconds << '\n';
    registry::write_text_file(yaml_path, yaml.str());
}

}  // namespace ai_factory::workbench::result_output

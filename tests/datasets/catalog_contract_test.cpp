// Validate split catalogue metadata and construction invariants.
#include "common/result_index.cuh"
#include "common/time_configuration.cuh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open metadata: " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

bool has_key(const std::string& document, const std::string& key) {
    return document.find("\n" + key + ":") != std::string::npos
        || document.rfind(key + ":", 0U) == 0U
        || document.find("\"" + key + "\":") != std::string::npos;
}

void validate_recipe(const std::filesystem::path& path) {
    const std::string recipe = read_text(path);
    for (const std::string& key : {
             "schema_version", "kind", "dataset_id", "generator", "output"
         }) {
        require(has_key(recipe, key), "recipe lacks " + key + ": " + path.string());
    }
    require(
        recipe.find("generator.cpp") != std::string::npos
            && recipe.find("datasets/") != std::string::npos
            && recipe.find("json") != std::string::npos,
        "recipe does not name its generator and formatted output: " + path.string()
    );
    require(
        !has_key(recipe, "validation") && !has_key(recipe, "timing"),
        "recipe contains validation or execution timing: " + path.string()
    );
}

void validate_generation(const std::filesystem::path& path) {
    const std::string generation = read_text(path);
    for (const std::string& key : {
             "schema_version", "status", "recipe", "artifact", "execution",
             "timing", "record_sha256"
         }) {
        require(
            has_key(generation, key),
            "generation receipt lacks " + key + ": " + path.string()
        );
    }
    for (const std::string& forbidden : {
             "dataset_id", "database_id", "validation", "outputs",
             "sensitivity", "time_grid", "price_construction"
         }) {
        require(
            !has_key(generation, forbidden),
            "generation receipt repeats recipe/validation field " + forbidden
                + ": " + path.string()
        );
    }
    require(
        std::filesystem::is_regular_file(path.parent_path() / "recipe.yaml"),
        "generation receipt has no adjacent recipe"
    );
}


}  // namespace

int main() {
    using namespace ai_factory::workbench;

    for (const std::uint32_t days_per_year : {252U, 360U, 365U}) {
        const time::DayFractionTimeConfiguration configuration{
            1.0f / static_cast<float>(days_per_year)
        };
        require(
            std::fabs(time::year_fraction(days_per_year, configuration) - 1.0f)
                <= 2.0f * std::numeric_limits<float>::epsilon(),
            "contract days and model years disagree"
        );
    }

    require(price_row_count(100U, 100U, PriceConstruction::Aligned) == 100U,
            "aligned construction count changed");
    require(price_row_count(100U, 100U, PriceConstruction::CartesianProduct)
                == 10'000U,
            "two-input Cartesian count changed");
    require(price_row_count(100U, 100U, 100U, PriceConstruction::Aligned)
                == 100U,
            "three-input aligned count changed");
    require(price_row_count(
                100U, 100U, 100U, PriceConstruction::CartesianProduct
            ) == 1'000'000U,
            "three-input Cartesian count changed");

    const auto product = decode_model_product_result_index(
        7U, 3U, PriceConstruction::CartesianProduct
    );
    require(product.model_index == 2U && product.product_index == 1U,
            "model/product order changed");
    const auto curve = decode_model_curve_product_result_index(
        17U, 2U, 4U, PriceConstruction::CartesianProduct
    );
    require(curve.model_index == 2U && curve.curve_index == 0U
                && curve.product_index == 1U,
            "model/curve/product order changed");

    bool rejected_overflow = false;
    try {
        static_cast<void>(price_row_count(
            std::numeric_limits<std::size_t>::max(), 2U,
            PriceConstruction::CartesianProduct
        ));
    } catch (const std::overflow_error&) {
        rejected_overflow = true;
    }
    require(rejected_overflow, "Cartesian cardinality accepted overflow");

    std::size_t generators = 0U;
    std::size_t recipes = 0U;
    std::size_t generations = 0U;
    for (const auto& entry : std::filesystem::recursive_directory_iterator("catalog")) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename();
        require(name != "dataset.yaml", "obsolete dataset.yaml remains in catalog");
        if (name == "generator.cpp") {
            ++generators;
            require(
                std::filesystem::is_regular_file(entry.path().parent_path() / "recipe.yaml"),
                "generator has no adjacent recipe: " + entry.path().string()
            );
        } else if (name == "recipe.yaml") {
            ++recipes;
            validate_recipe(entry.path());
        } else if (name == "generation.yaml") {
            ++generations;
            validate_generation(entry.path());
        } else if (name == "validation.yaml") {
            throw std::runtime_error(
                "validation.yaml is deferred and must not live in catalog: "
                + entry.path().string()
            );
        }
    }
    require(generators > 0U && recipes == generators,
            "catalogue must contain exactly one recipe per generator");
    require(generations > 0U,
            "materialized generation metadata is missing");

    const std::string heston = read_text(
        "catalog/model/equity/markovian/heston/parameters/heston_01/recipe.yaml"
    );
    require(heston.find("core_share: 0.9") != std::string::npos
                && heston.find("stress_share: 0.1") != std::string::npos,
            "parameter recipe lost its ordered core/stress construction");
    const std::string bermudan_gradient = read_text(
        "catalog/model/fixed_income/vasicek/price_gradients/"
        "bermudan_payer_swaptions/"
        "vasicek_01__bermudan_payer_swaptions_01__01_"
        "price_gradient_diagonal_hessian_frozen_policy/recipe.yaml"
    );
    require(bermudan_gradient.find(
                "\"exercise_replay\": \"frozen_regression_policy\""
            ) != std::string::npos,
            "public Bermudan gradient lost its frozen regression policy");
}

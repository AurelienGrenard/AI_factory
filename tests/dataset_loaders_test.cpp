// Exercise JSON loader invariants and generated-artifact validation.
#include "curve/nelson_siegel/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "model/g2/dataset.hpp"
#include "model/g2_plus_plus/dataset.hpp"
#include "model/heston/dataset.hpp"
#include "model/hull_white/dataset.hpp"
#include "model/ornstein_uhlenbeck/dataset.hpp"
#include "model/vasicek/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "product/asian_option/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "product/rate_option/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "product/lookback_option/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "product/down_and_in_option/dataset.hpp"
#include "product/down_and_out_option/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "product/up_and_in_option/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

const std::filesystem::path test_path =
    std::filesystem::temp_directory_path()
    / "ai_factory_dataset_loader_test.json";

// Stop the test at the first broken loader contract.
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Replace the temporary one-row dataset before each loader call.
void write_document(const nlohmann::json& document) {
    std::ofstream output(test_path);
    if (!output) throw std::runtime_error("cannot write loader test dataset");
    output << document;
}

// Check one accepted row and one rejected value with a useful row message.
template <typename Loader>
void check_loader(
    const std::string& name,
    const std::string& collection,
    const std::string& invalid_field,
    const nlohmann::json& invalid_value,
    const std::string& expected_reason,
    nlohmann::json document,
    Loader loader
) {
    write_document(document);
    require(loader(test_path).size() == 1U, name + " rejected a valid row");

    document.at(collection).at(0).at("parameters").at(invalid_field) =
        invalid_value;
    write_document(document);
    try {
        static_cast<void>(loader(test_path));
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        require(
            message.find("invalid_001") != std::string::npos,
            name + " error does not identify the invalid row"
        );
        require(
            message.find(expected_reason) != std::string::npos,
            name + " error does not identify the invalid field"
        );
        return;
    }
    throw std::runtime_error(name + " accepted an invalid row");
}

// Require one structural error to name the database and missing field.
template <typename Validator>
void check_missing_field(
    const std::string& name,
    const std::string& field,
    nlohmann::json document,
    Validator validator
) {
    document.erase(field);
    try {
        validator(document);
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        require(
            message.find("test_01") != std::string::npos,
            name + " structure error does not identify the database"
        );
        require(
            message.find(field) != std::string::npos,
            name + " structure error does not identify the missing field"
        );
        return;
    }
    throw std::runtime_error(name + " accepted a missing structural field");
}

// Build the common one-row JSON envelope used by every loader.
nlohmann::json one_row(
    const std::string& collection,
    const nlohmann::json& parameters
) {
    const std::string family_field = collection == "models"
        ? "model_family" : collection == "curves"
        ? "curve_family" : "product_family";
    return {
        {"database_id", "test_01"},
        {family_field, "Test family"},
        {"catalog", "catalog/test/test_01"},
        {"url", "https://datasets.ai-factory.example/test_01.json"},
        {"row_count", 1U},
        {collection, {{
            {"id", "invalid_001"},
            {"parameters", parameters},
        }}},
    };
}

}  // namespace

// Validate every model, curve, and product loader without generating datasets.
int main() {
    namespace curve = ai_factory::workbench::curve::nelson_siegel;
    namespace svensson = ai_factory::workbench::curve::svensson;
    namespace g2 = ai_factory::workbench::model::g2;
    namespace g2_plus_plus = ai_factory::workbench::model::g2_plus_plus;
    namespace heston = ai_factory::workbench::heston;
    namespace hull_white = ai_factory::workbench::model::hull_white;
    namespace ou = ai_factory::workbench::model::ornstein_uhlenbeck;
    namespace vasicek = ai_factory::workbench::model::vasicek;
    namespace product = ai_factory::workbench::product;
    namespace datasets = ai_factory::workbench::datasets;
    using ai_factory::workbench::OptionSide;

    check_missing_field(
        "Model", "model_family",
        one_row("models", {{"value", 1.0f}}),
        datasets::validate_model_dataset
    );
    check_missing_field(
        "Curve", "curve_family",
        one_row("curves", {{"value", 1.0f}}),
        datasets::validate_curve_dataset
    );
    check_missing_field(
        "Product", "product_family",
        one_row("products", {{"value", 1.0f}}),
        datasets::validate_product_dataset
    );

    // Exercise the post-generation file validators for parameter datasets.
    write_document(one_row("models", {{"value", 1.0f}}));
    datasets::validate_model_dataset_file(test_path);
    write_document(one_row("curves", {{"value", 1.0f}}));
    datasets::validate_curve_dataset_file(test_path);
    write_document(one_row("products", {{"value", 1.0f}}));
    datasets::validate_product_dataset_file(test_path);

    check_loader(
        "G2", "models", "correlation", 1.1f, "correlation",
        one_row("models", {
            {"mean_reversion_x", 0.15f},
            {"volatility_x", 0.01f},
            {"mean_reversion_y", 0.70f},
            {"volatility_y", 0.008f},
            {"correlation", -0.40f},
            {"initial_state_x", 0.02f},
            {"initial_state_y", 0.01f},
        }),
        g2::load_models
    );
    check_loader(
        "G2++", "models", "mean_reversion_x", 0.0f, "mean reversions",
        one_row("models", {
            {"mean_reversion_x", 0.15f},
            {"volatility_x", 0.01f},
            {"mean_reversion_y", 0.70f},
            {"volatility_y", 0.008f},
            {"correlation", -0.40f},
        }),
        g2_plus_plus::load_models
    );
    check_loader(
        "Heston", "models", "spot", 0.0f, "spot",
        one_row("models", {
            {"spot", 1.0f},
            {"risk_free_rate", 0.03f},
            {"dividend_yield", 0.01f},
            {"initial_variance", 0.04f},
            {"kappa", 1.5f},
            {"theta", 0.04f},
            {"gamma", 0.3f},
            {"rho", -0.5f},
        }),
        heston::load_models
    );
    check_loader(
        "Ornstein-Uhlenbeck", "models", "volatility", -0.01f,
        "volatility",
        one_row("models", {
            {"mean_reversion", 0.2f},
            {"volatility", 0.01f},
            {"initial_state", 0.03f},
        }),
        ou::load_models
    );
    check_loader(
        "Vasicek", "models", "mean_reversion", 0.0f, "mean_reversion",
        one_row("models", {
            {"mean_reversion", 0.2f},
            {"long_term_mean", 0.04f},
            {"volatility", 0.01f},
            {"initial_state", 0.03f},
        }),
        vasicek::load_models
    );
    check_loader(
        "Hull-White", "models", "volatility", -0.01f, "volatility",
        one_row("models", {
            {"mean_reversion", 0.2f},
            {"volatility", 0.01f},
        }),
        hull_white::load_models
    );
    check_loader(
        "Nelson-Siegel", "curves", "tau", 0.0f, "tau",
        one_row("curves", {
            {"beta0", 0.03f},
            {"beta1", -0.01f},
            {"beta2", 0.02f},
            {"tau", 2.0f},
        }),
        curve::load_curves
    );
    check_loader(
        "Svensson", "curves", "tau2", 1.0f, "tau2",
        one_row("curves", {
            {"beta0", 0.03f},
            {"beta1", -0.01f},
            {"beta2", 0.02f},
            {"beta3", -0.01f},
            {"tau1", 2.0f},
            {"tau2", 7.0f},
        }),
        svensson::load_curves
    );
    check_loader(
        "European call", "products", "strike", 0.0f, "strike",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_european_options
    );
    check_loader(
        "European put", "products", "strike", 0.0f, "strike",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_european_options
    );
    check_loader(
        "Asian call", "products", "maturity", 0.0f, "maturity",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_asian_options
    );
    check_loader(
        "Asian put", "products", "maturity", 0.0f, "maturity",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_asian_options
    );
    check_loader(
        "Geometric Asian call", "products", "strike", 0.0f, "strike",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_geometric_asian_options
    );
    check_loader(
        "Geometric Asian put", "products", "maturity", 0.0f, "maturity",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_geometric_asian_options
    );
    check_loader(
        "Forward-start call", "products", "reset_time", 1.0f,
        "must precede",
        one_row("products", {
            {"moneyness", 1.0f}, {"reset_time", 0.5f}, {"maturity", 1.0f},
        }),
        product::load_forward_start_options
    );
    check_loader(
        "Forward-start put", "products", "moneyness", 0.0f, "moneyness",
        one_row("products", {
            {"moneyness", 1.0f}, {"reset_time", 0.5f}, {"maturity", 1.0f},
        }),
        product::load_forward_start_options
    );
    check_loader(
        "Up-and-out call", "products", "barrier", 0.9f, "exceed strike",
        one_row("products", {
            {"strike", 1.0f}, {"barrier", 1.2f}, {"maturity", 1.0f},
        }),
        product::load_up_and_out_options
    );
    check_loader(
        "Down-and-out put", "products", "barrier", 1.1f, "below strike",
        one_row("products", {
            {"strike", 1.0f}, {"barrier", 0.8f}, {"maturity", 1.0f},
        }),
        product::load_down_and_out_options
    );
    check_loader(
        "Up-and-in call", "products", "barrier", 0.9f, "exceed strike",
        one_row("products", {
            {"strike", 1.0f}, {"barrier", 1.2f}, {"maturity", 1.0f},
        }),
        product::load_up_and_in_options
    );
    check_loader(
        "Down-and-in put", "products", "barrier", 1.1f, "below strike",
        one_row("products", {
            {"strike", 1.0f}, {"barrier", 0.8f}, {"maturity", 1.0f},
        }),
        product::load_down_and_in_options
    );
    check_loader(
        "Up one-touch", "products", "cash_payoff", 0.0f, "cash_payoff",
        one_row("products", {
            {"barrier", 1.2f}, {"cash_payoff", 1.0f}, {"maturity", 1.0f},
        }),
        product::load_up_one_touches
    );
    check_loader(
        "Up no-touch", "products", "barrier", 0.0f, "barrier",
        one_row("products", {
            {"barrier", 1.2f}, {"cash_payoff", 1.0f}, {"maturity", 1.0f},
        }),
        product::load_up_no_touches
    );
    check_loader(
        "Double-knock-out call", "products", "lower_barrier", 1.0f,
        "lie between",
        one_row("products", {
            {"strike", 1.0f}, {"lower_barrier", 0.8f},
            {"upper_barrier", 1.2f}, {"maturity", 1.0f},
        }),
        product::load_double_knock_out_options
    );
    check_loader(
        "Double-knock-out put", "products", "upper_barrier", 1.0f,
        "lie between",
        one_row("products", {
            {"strike", 1.0f}, {"lower_barrier", 0.8f},
            {"upper_barrier", 1.2f}, {"maturity", 1.0f},
        }),
        product::load_double_knock_out_options
    );
    check_loader(
        "Digital call", "products", "cash_payoff", 0.0f, "cash_payoff",
        one_row("products", {
            {"strike", 1.0f},
            {"maturity", 1.0f},
            {"cash_payoff", 1.0f},
        }),
        product::load_digital_options
    );
    check_loader(
        "Digital put", "products", "cash_payoff", 0.0f, "cash_payoff",
        one_row("products", {
            {"strike", 1.0f},
            {"maturity", 1.0f},
            {"cash_payoff", 1.0f},
        }),
        product::load_digital_options
    );
    check_loader(
        "Asset-or-nothing call", "products", "strike", 0.0f, "strike",
        one_row("products", {
            {"strike", 1.0f},
            {"maturity", 1.0f},
        }),
        product::load_asset_or_nothing_options
    );
    check_loader(
        "Asset-or-nothing put", "products", "strike", 0.0f, "strike",
        one_row("products", {
            {"strike", 1.0f},
            {"maturity", 1.0f},
        }),
        product::load_asset_or_nothing_options
    );
    check_loader(
        "Gap call", "products", "payoff_strike", 1.1f,
        "must not exceed",
        one_row("products", {
            {"trigger_strike", 1.0f},
            {"payoff_strike", 0.95f},
            {"maturity", 1.0f},
        }),
        [](const std::filesystem::path& path) {
            return product::load_gap_options(path, OptionSide::call);
        }
    );
    check_loader(
        "Gap put", "products", "payoff_strike", 0.9f,
        "must not be below",
        one_row("products", {
            {"trigger_strike", 1.0f},
            {"payoff_strike", 1.05f},
            {"maturity", 1.0f},
        }),
        [](const std::filesystem::path& path) {
            return product::load_gap_options(path, OptionSide::put);
        }
    );
    check_loader(
        "Straddle", "products", "strike", 0.0f, "strike",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_straddles
    );
    check_loader(
        "Lookback option", "products", "strike", -1.0f, "strike",
        one_row("products", {{"strike", 1.0f}, {"maturity", 1.0f}}),
        product::load_lookback_options
    );
    check_loader(
        "American put", "products", "exercise_interval", 1.0f,
        "exercise_interval",
        one_row("products", {
            {"strike", 1.0f},
            {"maturity", 1.0f},
            {"exercise_interval", 1.0f / 12.0f},
        }),
        product::load_american_options
    );
    check_loader(
        "American call", "products", "exercise_interval", 1.0f,
        "exercise_interval",
        one_row("products", {
            {"strike", 1.0f},
            {"maturity", 1.0f},
            {"exercise_interval", 1.0f / 12.0f},
        }),
        product::load_american_options
    );
    check_loader(
        "Phoenix autocall", "products", "coupon_barrier", 0.50f,
        "barriers",
        one_row("products", {
            {"maturity", 2.0f},
            {"observation_interval", 0.25f},
            {"autocall_barrier", 1.0f},
            {"coupon_barrier", 0.70f},
            {"protection_barrier", 0.60f},
            {"annual_coupon_rate", 0.08f},
        }),
        product::load_phoenix_autocalls
    );
    check_loader(
        "Phoenix Memory autocall", "products", "coupon_barrier", 0.50f,
        "barriers",
        one_row("products", {
            {"maturity", 2.0f},
            {"observation_interval", 0.25f},
            {"autocall_barrier", 1.0f},
            {"coupon_barrier", 0.70f},
            {"protection_barrier", 0.60f},
            {"annual_coupon_rate", 0.08f},
        }),
        product::load_phoenix_memory_autocalls
    );
    check_loader(
        "Athena autocall", "products", "protection_barrier", 1.10f,
        "barriers",
        one_row("products", {
            {"maturity", 2.0f},
            {"observation_interval", 0.25f},
            {"autocall_barrier", 1.0f},
            {"protection_barrier", 0.60f},
            {"annual_coupon_rate", 0.08f},
        }),
        product::load_athena_autocalls
    );
    check_loader(
        "Cliquet", "products", "global_floor", 0.40f,
        "global bounds",
        one_row("products", {
            {"maturity", 2.0f},
            {"observation_interval", 0.25f},
            {"participation_rate", 1.0f},
            {"local_floor", -0.05f},
            {"local_cap", 0.05f},
            {"global_floor", 0.0f},
            {"global_cap", 0.30f},
        }),
        product::load_cliquets
    );
    check_loader(
        "Range Accrual", "products", "lower_barrier", 1.10f,
        "barriers",
        one_row("products", {
            {"maturity", 2.0f},
            {"observation_interval", 0.25f},
            {"lower_barrier", 0.80f},
            {"upper_barrier", 1.20f},
            {"coupon_rate", 0.08f},
        }),
        product::load_range_accruals
    );
    check_loader(
        "Caplet", "products", "fixing_time", 0.0f, "fixing_time",
        one_row("products", {
            {"notional", 1.0f},
            {"strike", 0.04f},
            {"fixing_time", 1.0f},
            {"payment_time", 1.5f},
            {"accrual_period", 0.5f},
        }),
        product::load_rate_options
    );
    check_loader(
        "Floorlet", "products", "payment_time", 1.0f, "payment_time",
        one_row("products", {
            {"notional", 1.0f},
            {"strike", 0.04f},
            {"fixing_time", 1.0f},
            {"payment_time", 1.5f},
            {"accrual_period", 0.5f},
        }),
        product::load_rate_options
    );
    check_loader(
        "Zero-coupon bond call", "products", "strike", 0.0f, "strike",
        one_row("products", {
            {"notional", 1.0f},
            {"strike", 0.97f},
            {"option_expiry", 1.0f},
            {"bond_maturity", 1.5f},
        }),
        product::load_zero_coupon_bond_options
    );
    check_loader(
        "Zero-coupon bond put", "products", "bond_maturity", 1.0f,
        "bond_maturity",
        one_row("products", {
            {"notional", 1.0f},
            {"strike", 0.97f},
            {"option_expiry", 1.0f},
            {"bond_maturity", 1.5f},
        }),
        product::load_zero_coupon_bond_options
    );
    // Price datasets may omit the curve or require it consistently per row.
    nlohmann::json price_document = {
        {"database_id", "test_price_01"},
        {"catalog", "catalog/price/test_price_01"},
        {"url", "https://datasets.ai-factory.example/test_price_01.json"},
        {"row_count", 1U},
        {"model_dataset", {
            {"id", "test_model_01"},
            {"catalog", "catalog/model/test_model_01"},
            {"url", "https://datasets.ai-factory.example/test_model_01.json"},
        }},
        {"product_dataset", {
            {"id", "test_product_01"},
            {"catalog", "catalog/product/test_product_01"},
            {"url", "https://datasets.ai-factory.example/test_product_01.json"},
        }},
        {"timing", {{"wall_seconds", 1.0}, {"kernel_seconds", 0.5}}},
        {"results", {{
            {"id", "000001"},
            {"model_id", "000001"},
            {"product_id", "000001"},
            {"outputs", {{"price", 0.1f}}},
        }}},
    };
    datasets::validate_price_dataset(price_document);
    price_document["curve_dataset"] = {
        {"id", "test_curve_01"},
        {"catalog", "catalog/curve/test_curve_01"},
        {"url", "https://datasets.ai-factory.example/test_curve_01.json"},
    };
    bool rejected_missing_curve_id = false;
    try {
        datasets::validate_price_dataset(price_document);
    } catch (const std::invalid_argument& error) {
        rejected_missing_curve_id =
            std::string(error.what()).find("curve_id") != std::string::npos;
    }
    require(
        rejected_missing_curve_id,
        "price structure accepted a missing curve_id"
    );
    price_document["results"][0]["curve_id"] = "000001";
    datasets::validate_price_dataset(price_document);
    write_document(price_document);
    datasets::validate_price_dataset_file(test_path);

    // Duplicate row identifiers and non-finite outputs are rejected early.
    nlohmann::json duplicate_models = one_row(
        "models", {{"value", 1.0f}}
    );
    duplicate_models["row_count"] = 2U;
    duplicate_models["models"].push_back(duplicate_models["models"][0]);
    bool rejected_duplicate_id = false;
    try {
        datasets::validate_model_dataset(duplicate_models);
    } catch (const std::invalid_argument& error) {
        rejected_duplicate_id =
            std::string(error.what()).find("must be unique")
            != std::string::npos;
    }
    require(rejected_duplicate_id, "model structure accepted duplicate ids");

    price_document["results"][0]["outputs"]["price"] =
        std::numeric_limits<double>::infinity();
    bool rejected_non_finite_price = false;
    try {
        datasets::validate_price_dataset(price_document);
    } catch (const std::invalid_argument& error) {
        rejected_non_finite_price =
            std::string(error.what()).find("finite") != std::string::npos;
    }
    require(
        rejected_non_finite_price,
        "price structure accepted a non-finite output"
    );
    price_document["results"][0]["outputs"]["price"] = 0.1f;

    price_document["row_count"] = 2U;
    bool rejected_structure = false;
    try {
        datasets::validate_price_dataset(price_document);
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        rejected_structure =
            message.find("test_price_01") != std::string::npos
            && message.find("row_count") != std::string::npos;
    }
    require(rejected_structure, "price structure error is not informative");

    std::filesystem::remove(test_path);
}

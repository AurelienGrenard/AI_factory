// Every factor-only model base loads with neutral legacy correlation fields.
#include "model/equity/markovian/bates/dataset.hpp"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "model/equity/markovian/cev/dataset.hpp"
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "model/equity/markovian/kou/dataset.hpp"
#include "model/equity/markovian/merton/dataset.hpp"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "model/fixed_income/g2/dataset.hpp"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "model/fixed_income/constant_rate/dataset.hpp"
#include "common/dataset_validation.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

int main() {
    using namespace ai_factory::workbench;
    const auto check = [](auto load, const std::filesystem::path& path) {
        std::ifstream stream(path);
        if (!stream) throw std::runtime_error("Missing factor dataset: " + path.string());
        nlohmann::json document;
        stream >> document;
        if (document.at("parameterization") != "factorized"
            || document.at("models").size() != 1000U)
            throw std::runtime_error("Invalid factor dataset: " + path.string());
        for (const auto& row : document.at("models")) {
            const auto& p = row.at("parameters");
            if (p.contains("risk_free_rate") || p.contains("rho")
                || p.contains("correlation"))
                throw std::runtime_error("Exogenous field in " + path.string());
        }
        const auto models = load(path);
        if (models.size() != 1000U)
            throw std::runtime_error("Wrong factor count: " + path.string());
        for (const auto& model : models) {
            if constexpr (requires { model.risk_free_rate; })
                if (model.risk_free_rate != 0.0f)
                    throw std::runtime_error("Equity rate was not neutral");
            if constexpr (requires { model.rho; })
                if (model.rho != 0.0f)
                    throw std::runtime_error("Internal rho was not neutral");
            if constexpr (requires { model.correlation; })
                if (model.correlation != 0.0f)
                    throw std::runtime_error("Internal correlation was not neutral");
            if constexpr (requires { model.process.correlation; })
                if (model.process.correlation != 0.0f)
                    throw std::runtime_error("Rate correlation was not neutral");
        }
    };
    check(model::equity::bates::load_models, "datasets/model/equity/markovian/bates/factor_parameters/bates_factors_01.json");
    check(model::equity::black_scholes::load_models, "datasets/model/equity/markovian/black_scholes/factor_parameters/black_scholes_factors_01.json");
    check(model::equity::cev::load_models, "datasets/model/equity/markovian/cev/factor_parameters/cev_factors_01.json");
    check(model::equity::heston::load_models, "datasets/model/equity/markovian/heston/factor_parameters/heston_factors_01.json");
    check(model::equity::heston_3_2::load_models, "datasets/model/equity/markovian/heston_3_2/factor_parameters/heston_3_2_factors_01.json");
    check(model::equity::kou::load_models, "datasets/model/equity/markovian/kou/factor_parameters/kou_factors_01.json");
    check(model::equity::merton::load_models, "datasets/model/equity/markovian/merton/factor_parameters/merton_factors_01.json");
    check(model::equity::normal_inverse_gaussian::load_models, "datasets/model/equity/markovian/normal_inverse_gaussian/factor_parameters/normal_inverse_gaussian_factors_01.json");
    check(model::equity::sabr::load_models, "datasets/model/equity/markovian/sabr/factor_parameters/sabr_factors_01.json");
    check(model::equity::schobel_zhu::load_models, "datasets/model/equity/markovian/schobel_zhu/factor_parameters/schobel_zhu_factors_01.json");
    check(model::equity::stein_stein::load_models, "datasets/model/equity/markovian/stein_stein/factor_parameters/stein_stein_factors_01.json");
    check(model::equity::variance_gamma::load_models, "datasets/model/equity/markovian/variance_gamma/factor_parameters/variance_gamma_factors_01.json");
    check(model::equity::log_modulated_rough_bergomi::load_models, "datasets/model/equity/rough/log_modulated_rough_bergomi/factor_parameters/log_modulated_rough_bergomi_factors_01.json");
    check(model::equity::quadratic_rough_heston::load_models, "datasets/model/equity/rough/quadratic_rough_heston/factor_parameters/quadratic_rough_heston_factors_01.json");
    check(model::equity::rough_bergomi::load_models, "datasets/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_factors_01.json");
    check(model::equity::rough_heston::load_models, "datasets/model/equity/rough/rough_heston/factor_parameters/rough_heston_factors_01.json");
    check(model::equity::rough_sabr::load_models, "datasets/model/equity/rough/rough_sabr/factor_parameters/rough_sabr_factors_01.json");
    check(model::equity::rough_stein_stein::load_models, "datasets/model/equity/rough/rough_stein_stein/factor_parameters/rough_stein_stein_factors_01.json");
    check(model::fixed_income::g2::load_models, "datasets/model/fixed_income/g2/factor_parameters/g2_factors_01.json");
    check(model::fixed_income::g2_plus_plus::load_models, "datasets/model/fixed_income/g2_plus_plus/factor_parameters/g2_plus_plus_factors_01.json");
    std::ifstream invalid_source(
        "datasets/model/equity/markovian/heston/factor_parameters/heston_factors_01.json");
    nlohmann::json invalid;
    invalid_source >> invalid;
    invalid.at("models").at(0).at("parameters")["rho"] = 0.3;
    bool rejected = false;
    try { datasets::validate_model_dataset(invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected)
        throw std::runtime_error("Factorized row accepted embedded rho");
    const auto rates = model::fixed_income::constant_rate::load_models(
        "datasets/model/fixed_income/constant_rate/parameters/constant_rates_01.json");
    if (rates.size() != 7U || rates[2].rate != 0.0f)
        throw std::runtime_error("Constant rate dataset failed to load");
}

// Verify loaders and gradient policies share each model/product parameter domain.
#include "model/equity/markovian/bates/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/black_scholes/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/cev/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/heston/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/heston_3_2/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/kou/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/merton/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/sabr/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/schobel_zhu/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/stein_stein/price_gradients/device_preparation.cuh"
#include "model/equity/markovian/variance_gamma/price_gradients/device_preparation.cuh"
#include "model/fixed_income/cir/price_gradients/device_preparation.cuh"
#include "model/fixed_income/g2/price_gradients/device_preparation.cuh"
#include "model/fixed_income/g2_plus_plus/price_gradients/device_preparation.cuh"
#include "model/fixed_income/hull_white/price_gradients/device_preparation.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/price_gradients/device_preparation.cuh"
#include "model/fixed_income/vasicek/price_gradients/device_preparation.cuh"
#include "curve/nelson_siegel/price_gradients/device_preparation.cuh"
#include "curve/svensson/price_gradients/device_preparation.cuh"
#include "product/american_option/price_gradients/device_preparation.cuh"
#include "product/asset_or_nothing_option/price_gradients/device_preparation.cuh"
#include "product/bermudan_swaption/price_gradients/device_preparation.cuh"
#include "product/digital_option/price_gradients/device_preparation.cuh"
#include "product/european_option/price_gradients/device_preparation.cuh"
#include "product/european_swaption/price_gradients/device_preparation.cuh"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace model = ai_factory::workbench::model;
namespace curve = ai_factory::workbench::curve;
namespace product = ai_factory::workbench::product;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename Parameters, typename Policy, typename Domain, typename Validate>
void require_shared_domain(
    const Parameters& valid,
    const Parameters& invalid,
    Domain domain,
    Validate validate,
    const char* message
) {
    require(domain(valid) && Policy::valid(valid), message);
    require(!domain(invalid) && !Policy::valid(invalid), message);
    validate(valid, "test: ");
    bool rejected = false;
    try {
        validate(invalid, "test: ");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

int main() {
    try {
        namespace bates = model::equity::bates;
        auto bates_valid = bates::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f,
            0.1f, -0.05f, 0.2f
        };
        auto bates_invalid = bates_valid;
        bates_invalid.jump_intensity = -0.01f;
        require_shared_domain<
            bates::ModelParameters,
            bates::price_gradients::DevicePreparation
        >(
            bates_valid, bates_invalid, bates::valid_parameters,
            bates::validate_parameters, "Bates domains diverged."
        );

        namespace bs = model::equity::black_scholes;
        auto bs_valid = bs::ModelParameters{1.0f, 0.0f, 0.0f, 0.2f};
        auto bs_invalid = bs_valid;
        bs_invalid.volatility = 0.0f;
        require_shared_domain<bs::ModelParameters, bs::price_gradients::DevicePreparation>(
            bs_valid, bs_invalid, bs::valid_parameters, bs::validate_parameters,
            "Black-Scholes domains diverged."
        );

        namespace heston = model::equity::heston;
        auto heston_valid = heston::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f
        };
        auto heston_invalid = heston_valid;
        heston_invalid.rho = 1.01f;
        require_shared_domain<
            heston::ModelParameters,
            heston::price_gradients::DevicePreparation
        >(
            heston_valid, heston_invalid, heston::valid_parameters,
            heston::validate_parameters, "Heston domains diverged."
        );

        namespace heston_3_2 = model::equity::heston_3_2;
        auto heston_3_2_valid = heston_3_2::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f
        };
        auto heston_3_2_invalid = heston_3_2_valid;
        heston_3_2_invalid.initial_variance = 0.0f;
        require_shared_domain<
            heston_3_2::ModelParameters,
            heston_3_2::price_gradients::DevicePreparation
        >(
            heston_3_2_valid, heston_3_2_invalid,
            heston_3_2::valid_parameters,
            heston_3_2::validate_parameters,
            "Heston 3/2 domains diverged."
        );

        namespace cev = model::equity::cev;
        auto cev_valid = cev::ModelParameters{1.0f, 0.0f, 0.0f, 0.2f, 0.5f};
        auto cev_invalid = cev_valid;
        cev_invalid.beta = 1.0f;
        require_shared_domain<cev::ModelParameters, cev::price_gradients::DevicePreparation>(
            cev_valid, cev_invalid, cev::valid_parameters,
            cev::validate_parameters, "CEV domains diverged."
        );

        namespace merton = model::equity::merton;
        auto merton_valid = merton::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.2f, 0.1f, -0.05f, 0.2f
        };
        auto merton_invalid = merton_valid;
        merton_invalid.jump_intensity = -0.01f;
        require_shared_domain<
            merton::ModelParameters,
            merton::price_gradients::DevicePreparation
        >(
            merton_valid, merton_invalid, merton::valid_parameters,
            merton::validate_parameters, "Merton domains diverged."
        );

        namespace kou = model::equity::kou;
        auto kou_valid = kou::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.2f, 0.1f, 0.4f, 4.0f, 5.0f
        };
        auto kou_invalid = kou_valid;
        kou_invalid.positive_jump_rate = 2.0f;
        require_shared_domain<
            kou::ModelParameters,
            kou::price_gradients::DevicePreparation
        >(
            kou_valid, kou_invalid, kou::valid_parameters,
            kou::validate_parameters, "Kou domains diverged."
        );

        namespace nig = model::equity::normal_inverse_gaussian;
        auto nig_valid = nig::ModelParameters{
            1.0f, 0.0f, 0.0f, 8.0f, -2.0f, 0.5f
        };
        auto nig_invalid = nig_valid;
        nig_invalid.alpha = 1.5f;
        require_shared_domain<
            nig::ModelParameters,
            nig::price_gradients::DevicePreparation
        >(
            nig_valid, nig_invalid, nig::valid_parameters,
            nig::validate_parameters, "NIG domains diverged."
        );

        namespace sabr = model::equity::sabr;
        auto sabr_valid = sabr::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.2f, 0.4f, -0.5f, 0.7f
        };
        auto sabr_invalid = sabr_valid;
        sabr_invalid.beta = 1.01f;
        require_shared_domain<
            sabr::ModelParameters,
            sabr::price_gradients::DevicePreparation
        >(
            sabr_valid, sabr_invalid, sabr::valid_parameters,
            sabr::validate_parameters, "SABR domains diverged."
        );

        namespace schobel_zhu = model::equity::schobel_zhu;
        auto schobel_zhu_valid = schobel_zhu::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.2f, 1.5f, 0.2f, 0.3f, -0.7f
        };
        auto schobel_zhu_invalid = schobel_zhu_valid;
        schobel_zhu_invalid.correlation = 1.0f;
        require_shared_domain<
            schobel_zhu::ModelParameters,
            schobel_zhu::price_gradients::DevicePreparation
        >(
            schobel_zhu_valid, schobel_zhu_invalid,
            schobel_zhu::valid_parameters,
            schobel_zhu::validate_parameters,
            "Schobel-Zhu domains diverged."
        );

        namespace stein_stein = model::equity::stein_stein;
        auto stein_stein_valid = stein_stein::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.2f, 1.5f, 0.3f, -0.7f
        };
        auto stein_stein_invalid = stein_stein_valid;
        stein_stein_invalid.volatility_of_volatility = 0.0f;
        require_shared_domain<
            stein_stein::ModelParameters,
            stein_stein::price_gradients::DevicePreparation
        >(
            stein_stein_valid, stein_stein_invalid,
            stein_stein::valid_parameters,
            stein_stein::validate_parameters,
            "Stein-Stein domains diverged."
        );

        namespace vg = model::equity::variance_gamma;
        auto vg_valid = vg::ModelParameters{
            1.0f, 0.0f, 0.0f, 0.2f, 0.2f, -0.1f
        };
        auto vg_invalid = vg_valid;
        vg_invalid.theta = 3.0f;
        require_shared_domain<
            vg::ModelParameters,
            vg::price_gradients::DevicePreparation
        >(
            vg_valid, vg_invalid, vg::valid_parameters,
            vg::validate_parameters, "Variance-Gamma domains diverged."
        );

        namespace cir = model::fixed_income::cir;
        auto cir_valid = cir::ModelParameters{{0.3f, 0.04f, 0.1f}, 0.03f};
        auto cir_invalid = cir_valid;
        cir_invalid.process.mean_reversion = 0.0f;
        require_shared_domain<
            cir::ModelParameters,
            cir::price_gradients::DevicePreparation
        >(
            cir_valid, cir_invalid, cir::valid_parameters,
            cir::validate_parameters, "CIR domains diverged."
        );


        namespace g2 = model::fixed_income::g2;
        auto g2_valid = g2::ModelParameters{
            {0.1f, 0.01f, 0.2f, 0.02f, -0.5f}, {0.0f, 0.0f}
        };
        auto g2_invalid = g2_valid;
        g2_invalid.process.correlation = 1.01f;
        require_shared_domain<
            g2::ModelParameters,
            g2::price_gradients::DevicePreparation
        >(
            g2_valid, g2_invalid, g2::valid_parameters,
            g2::validate_parameters, "G2 domains diverged."
        );

        namespace g2pp = model::fixed_income::g2_plus_plus;
        auto g2pp_valid = g2pp::ModelParameters{{
            0.1f, 0.01f, 0.2f, 0.02f, -0.5f
        }};
        auto g2pp_invalid = g2pp_valid;
        g2pp_invalid.process.mean_reversion_y = 0.0f;
        require_shared_domain<
            g2pp::ModelParameters,
            g2pp::price_gradients::DevicePreparation
        >(
            g2pp_valid, g2pp_invalid, g2pp::valid_parameters,
            g2pp::validate_parameters, "G2++ domains diverged."
        );

        namespace hull_white = model::fixed_income::hull_white;
        auto hull_white_valid = hull_white::ModelParameters{0.1f, 0.01f};
        auto hull_white_invalid = hull_white_valid;
        hull_white_invalid.volatility = -0.01f;
        require_shared_domain<
            hull_white::ModelParameters,
            hull_white::price_gradients::DevicePreparation
        >(
            hull_white_valid, hull_white_invalid,
            hull_white::valid_parameters, hull_white::validate_parameters,
            "Hull-White domains diverged."
        );

        namespace ou = model::fixed_income::ornstein_uhlenbeck;
        auto ou_valid = ou::ModelParameters{{0.1f, 0.01f}, 0.03f};
        auto ou_invalid = ou_valid;
        ou_invalid.initial_state =
            std::numeric_limits<float>::quiet_NaN();
        require_shared_domain<
            ou::ModelParameters,
            ou::price_gradients::DevicePreparation
        >(
            ou_valid, ou_invalid, ou::valid_parameters,
            ou::validate_parameters, "OU domains diverged."
        );

        namespace vasicek = model::fixed_income::vasicek;
        auto vasicek_valid =
            vasicek::ModelParameters{{0.1f, 0.03f, 0.01f}, 0.03f};
        auto vasicek_invalid = vasicek_valid;
        vasicek_invalid.process.mean_reversion = 0.0f;
        require_shared_domain<
            vasicek::ModelParameters,
            vasicek::price_gradients::DevicePreparation
        >(
            vasicek_valid, vasicek_invalid, vasicek::valid_parameters,
            vasicek::validate_parameters, "Vasicek domains diverged."
        );

        namespace ns = curve::nelson_siegel;
        auto ns_valid = ns::NelsonSiegelParameters{0.03f, -0.01f, 0.01f, 1.5f};
        auto ns_invalid = ns_valid;
        ns_invalid.tau = 0.0f;
        require_shared_domain<
            ns::NelsonSiegelParameters,
            ns::price_gradients::DevicePreparation
        >(
            ns_valid, ns_invalid, ns::valid_parameters,
            ns::validate_parameters, "Nelson-Siegel domains diverged."
        );

        namespace svensson = curve::svensson;
        auto svensson_valid = svensson::SvenssonParameters{
            0.03f, -0.01f, 0.01f, 0.005f, 1.0f, 3.0f
        };
        auto svensson_invalid = svensson_valid;
        svensson_invalid.tau2 = 0.5f;
        require_shared_domain<
            svensson::SvenssonParameters,
            svensson::price_gradients::DevicePreparation
        >(
            svensson_valid, svensson_invalid, svensson::valid_parameters,
            svensson::validate_parameters, "Svensson domains diverged."
        );

        auto european_valid = product::EuropeanOptionParameters{1.0f, 252U};
        auto european_invalid = european_valid;
        european_invalid.strike = 0.0f;
        require_shared_domain<
            product::EuropeanOptionParameters,
            product::european_option::price_gradients::DevicePreparation
        >(
            european_valid, european_invalid,
            product::european_option::valid_parameters,
            product::european_option::validate_parameters,
            "European-option domains diverged."
        );

        auto asset_or_nothing_valid =
            product::AssetOrNothingOptionParameters{1.0f, 252U};
        auto asset_or_nothing_invalid = asset_or_nothing_valid;
        asset_or_nothing_invalid.strike = 0.0f;
        require_shared_domain<
            product::AssetOrNothingOptionParameters,
            product::asset_or_nothing_option::price_gradients::DevicePreparation
        >(
            asset_or_nothing_valid, asset_or_nothing_invalid,
            product::asset_or_nothing_option::valid_parameters,
            product::asset_or_nothing_option::validate_parameters,
            "Asset-or-nothing-option domains diverged."
        );

        auto digital_valid =
            product::DigitalOptionParameters{1.0f, 252U, 1.0f};
        auto digital_invalid = digital_valid;
        digital_invalid.cash_payoff = 0.0f;
        require_shared_domain<
            product::DigitalOptionParameters,
            product::digital_option::price_gradients::DevicePreparation
        >(
            digital_valid, digital_invalid,
            product::digital_option::valid_parameters,
            product::digital_option::validate_parameters,
            "Digital-option domains diverged."
        );

        auto american_valid = product::AmericanOptionParameters{1.0f, 252U, 21U};
        auto american_invalid = american_valid;
        american_invalid.exercise_interval_days = 252U;
        require_shared_domain<
            product::AmericanOptionParameters,
            product::american_option::price_gradients::DevicePreparation
        >(
            american_valid, american_invalid,
            product::american_option::valid_parameters,
            product::american_option::validate_parameters,
            "American-option domains diverged."
        );

        auto swaption_valid = product::RegularEuropeanSwaptionParameters{
            1.0f, 0.03f, 0.5f, 252U, 126U, 4U
        };
        auto swaption_invalid = swaption_valid;
        swaption_invalid.accrual_fraction =
            std::numeric_limits<float>::quiet_NaN();
        require_shared_domain<
            product::RegularEuropeanSwaptionParameters,
            product::european_swaption::price_gradients::DevicePreparation
        >(
            swaption_valid, swaption_invalid,
            product::european_swaption::valid_parameters,
            product::european_swaption::validate_parameters,
            "European-swaption domains diverged."
        );


        auto bermudan_valid = product::BermudanSwaptionParameters{
            1.0f, 0.03f, 0.5f, 252U, 126U, 6U, 3U
        };
        auto bermudan_invalid = bermudan_valid;
        bermudan_invalid.exercise_count = 1U;
        require_shared_domain<
            product::BermudanSwaptionParameters,
            product::bermudan_swaption::price_gradients::DevicePreparation
        >(
            bermudan_valid, bermudan_invalid,
            product::bermudan_swaption::valid_parameters,
            product::bermudan_swaption::validate_parameters,
            "Bermudan-swaption domains diverged."
        );

        std::cout << "Shared gradient parameter domains passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

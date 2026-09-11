// Bounded public catalogue coverage, with production-path LSM checks on three models.
#include "model/equity/markovian/bates/product/european_option.cuh"
#include "model/equity/markovian/bates/product/european_option_price_delta.cuh"
#include "model/equity/markovian/cev/product/european_option.cuh"
#include "model/equity/markovian/cev/product/european_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/european_option.cuh"
#include "model/equity/markovian/heston/product/european_option_price_delta.cuh"
#include "model/equity/markovian/heston_3_2/product/european_option.cuh"
#include "model/equity/markovian/heston_3_2/product/european_option_price_delta.cuh"
#include "model/equity/markovian/kou/product/european_option.cuh"
#include "model/equity/markovian/kou/product/european_option_price_delta.cuh"
#include "model/equity/markovian/merton/product/european_option.cuh"
#include "model/equity/markovian/merton/product/european_option_price_delta.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/european_option.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/european_option_price_delta.cuh"
#include "model/equity/markovian/sabr/product/european_option.cuh"
#include "model/equity/markovian/sabr/product/european_option_price_delta.cuh"
#include "model/equity/markovian/schobel_zhu/product/european_option.cuh"
#include "model/equity/markovian/schobel_zhu/product/european_option_price_delta.cuh"
#include "model/equity/markovian/stein_stein/product/european_option.cuh"
#include "model/equity/markovian/stein_stein/product/european_option_price_delta.cuh"
#include "model/equity/markovian/variance_gamma/product/european_option.cuh"
#include "model/equity/markovian/variance_gamma/product/european_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/european_option.cuh"
#include "model/equity/markovian/black_scholes/product/european_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/digital_option.cuh"
#include "model/equity/markovian/black_scholes/product/digital_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option.cuh"
#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/gap_option.cuh"
#include "model/equity/markovian/black_scholes/product/gap_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/forward_start_option.cuh"
#include "model/equity/markovian/black_scholes/product/forward_start_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/geometric_asian_option.cuh"
#include "model/equity/markovian/black_scholes/product/geometric_asian_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/straddle.cuh"
#include "model/equity/markovian/black_scholes/product/straddle_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/range_accrual.cuh"
#include "model/equity/markovian/black_scholes/product/range_accrual_price_delta.cuh"
#include "model/equity/markovian/bates/product/athena_autocall.cuh"
#include "model/equity/markovian/bates/product/athena_autocall_price_delta.cuh"
#include "model/equity/markovian/bates/product/phoenix_memory_autocall.cuh"
#include "model/equity/markovian/bates/product/phoenix_memory_autocall_price_delta.cuh"
#include "model/equity/markovian/bates/product/cliquet.cuh"
#include "model/equity/markovian/bates/product/cliquet_price_delta.cuh"
#include "model/equity/markovian/bates/product/forward_start_option.cuh"
#include "model/equity/markovian/bates/product/forward_start_option_price_delta.cuh"
#include "model/equity/markovian/bates/product/range_accrual.cuh"
#include "model/equity/markovian/bates/product/range_accrual_price_delta.cuh"
#include "model/equity/markovian/bates/product/up_and_out_option.cuh"
#include "model/equity/markovian/bates/product/up_and_out_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/american_option.cuh"
#include "model/equity/markovian/black_scholes/product/american_option_price_delta.cuh"
#include "model/equity/markovian/bates/product/american_option.cuh"
#include "model/equity/markovian/bates/product/american_option_price_delta.cuh"
#include "model/equity/markovian/cev/product/american_option.cuh"
#include "model/equity/markovian/cev/product/american_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/american_option.cuh"
#include "model/equity/markovian/heston/product/american_option_price_delta.cuh"
#include "model/equity/markovian/kou/product/american_option.cuh"
#include "model/equity/markovian/kou/product/american_option_price_delta.cuh"
#include "model/equity/markovian/merton/product/american_option.cuh"
#include "model/equity/markovian/merton/product/american_option_price_delta.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/american_option.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/american_option_price_delta.cuh"
#include "model/equity/markovian/schobel_zhu/product/american_option.cuh"
#include "model/equity/markovian/schobel_zhu/product/american_option_price_delta.cuh"
#include "model/equity/markovian/variance_gamma/product/american_option.cuh"
#include "model/equity/markovian/variance_gamma/product/american_option_price_delta.cuh"
#include "tests/price_delta/catalogue/fixtures/model_parameters.hpp"
#include "tests/price_delta/catalogue/fixtures/public_launcher_parity.cuh"

int main() {
    using namespace ai_factory::workbench;
    using price_delta_test::compare_public;
    namespace fixtures = price_delta_test::fixtures;
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        compare_public<true, false>("bates/european_option", fixtures::bates,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::bates::launch_bates_european_option_cuda<OptionSide::call>,
            model::equity::bates::launch_bates_european_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("cev/european_option", fixtures::cev,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::cev::launch_cev_european_option_cuda<OptionSide::call>,
            model::equity::cev::launch_cev_european_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("heston/european_option", fixtures::heston,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::heston::launch_heston_european_option_cuda<OptionSide::call>,
            model::equity::heston::launch_heston_european_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("heston_3_2/european_option", fixtures::heston_3_2,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::heston_3_2::launch_heston_3_2_european_option_cuda<OptionSide::call>,
            model::equity::heston_3_2::launch_heston_3_2_european_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("kou/european_option", fixtures::kou,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::kou::launch_kou_european_option_cuda<OptionSide::call>,
            model::equity::kou::launch_kou_european_option_price_delta_cuda<OptionSide::call>, 1.f/252.f);
        compare_public<true, false>("merton/european_option", fixtures::merton,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::merton::launch_merton_european_option_cuda<OptionSide::call>,
            model::equity::merton::launch_merton_european_option_price_delta_cuda<OptionSide::call>, 1.f/252.f);
        compare_public<true, false>("normal_inverse_gaussian/european_option", fixtures::normal_inverse_gaussian,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_european_option_cuda<OptionSide::call>,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_european_option_price_delta_cuda<OptionSide::call>, 1.f/252.f);
        compare_public<true, false>("sabr/european_option", fixtures::sabr,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::sabr::launch_sabr_european_option_cuda<OptionSide::call>,
            model::equity::sabr::launch_sabr_european_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("schobel_zhu/european_option", fixtures::schobel_zhu,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::schobel_zhu::launch_schobel_zhu_european_option_cuda<OptionSide::call>,
            model::equity::schobel_zhu::launch_schobel_zhu_european_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("stein_stein/european_option", fixtures::stein_stein,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::stein_stein::launch_stein_stein_european_option_cuda<OptionSide::call>,
            model::equity::stein_stein::launch_stein_stein_european_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("variance_gamma/european_option", fixtures::variance_gamma,
            product::EuropeanOptionParameters{1.1f,21U}, 4096U,
            model::equity::variance_gamma::launch_variance_gamma_european_option_cuda<OptionSide::call>,
            model::equity::variance_gamma::launch_variance_gamma_european_option_price_delta_cuda<OptionSide::call>, 1.f/252.f);
        compare_public<false, false>("black_scholes/european_option", fixtures::black_scholes,
            product::EuropeanOptionParameters{1.1f,21U}, 0U,
            model::equity::black_scholes::launch_black_scholes_european_option_cuda<OptionSide::call>,
            model::equity::black_scholes::launch_black_scholes_european_option_price_delta_cuda<OptionSide::call>, 1.f/252.f);
        compare_public<false, false>("black_scholes/digital_option", fixtures::black_scholes,
            product::DigitalOptionParameters{1.1f,21U,1.f}, 0U,
            model::equity::black_scholes::launch_black_scholes_digital_option_cuda<OptionSide::put>,
            model::equity::black_scholes::launch_black_scholes_digital_option_price_delta_cuda<OptionSide::put>, 1.f/252.f);
        compare_public<false, false>("black_scholes/asset_or_nothing_option", fixtures::black_scholes,
            product::AssetOrNothingOptionParameters{1.1f,21U}, 0U,
            model::equity::black_scholes::launch_black_scholes_asset_or_nothing_option_cuda<OptionSide::call>,
            model::equity::black_scholes::launch_black_scholes_asset_or_nothing_option_price_delta_cuda<OptionSide::call>, 1.f/252.f);
        compare_public<false, false>("black_scholes/gap_option", fixtures::black_scholes,
            product::GapOptionParameters{1.1f,1.2f,21U}, 0U,
            model::equity::black_scholes::launch_black_scholes_gap_option_cuda<OptionSide::put>,
            model::equity::black_scholes::launch_black_scholes_gap_option_price_delta_cuda<OptionSide::put>, 1.f/252.f);
        compare_public<false, false>("black_scholes/forward_start_option", fixtures::black_scholes,
            product::ForwardStartOptionParameters{1.f,7U,21U}, 0U,
            model::equity::black_scholes::launch_black_scholes_forward_start_option_cuda<OptionSide::call>,
            model::equity::black_scholes::launch_black_scholes_forward_start_option_price_delta_cuda<OptionSide::call>, 1.f/252.f);
        compare_public<false, false>("black_scholes/geometric_asian_option", fixtures::black_scholes,
            product::GeometricAsianOptionParameters{1.1f,21U}, 0U,
            model::equity::black_scholes::launch_black_scholes_geometric_asian_option_cuda<OptionSide::put>,
            model::equity::black_scholes::launch_black_scholes_geometric_asian_option_price_delta_cuda<OptionSide::put>, 1.f/504.f,2U);
        compare_public<false, false>("black_scholes/straddle", fixtures::black_scholes,
            product::StraddleParameters{1.1f,21U}, 0U,
            model::equity::black_scholes::launch_black_scholes_straddle_cuda,
            model::equity::black_scholes::launch_black_scholes_straddle_price_delta_cuda, 1.f/252.f);
        compare_public<false, false>("black_scholes/range_accrual", fixtures::black_scholes,
            product::RangeAccrualParameters{21U,7U,.9f,1.3f,.05f}, 0U,
            model::equity::black_scholes::launch_black_scholes_range_accrual_cuda,
            model::equity::black_scholes::launch_black_scholes_range_accrual_price_delta_cuda, 1.f/252.f);
        compare_public<true, false>("bates/athena_autocall", fixtures::bates,
            product::AthenaAutocallParameters{21U,7U,1.f,.7f,.05f}, 4096U,
            model::equity::bates::launch_bates_athena_autocall_cuda,
            model::equity::bates::launch_bates_athena_autocall_price_delta_cuda, 1.f/504.f,2U);
        compare_public<true, false>("bates/phoenix_memory_autocall", fixtures::bates,
            product::PhoenixMemoryAutocallParameters{21U,7U,1.f,.8f,.7f,.05f}, 4096U,
            model::equity::bates::launch_bates_phoenix_memory_autocall_cuda,
            model::equity::bates::launch_bates_phoenix_memory_autocall_price_delta_cuda, 1.f/504.f,2U);
        compare_public<true, false>("bates/cliquet", fixtures::bates,
            product::CliquetParameters{21U,7U,1.f,-.1f,.1f,-.2f,.2f}, 4096U,
            model::equity::bates::launch_bates_cliquet_cuda,
            model::equity::bates::launch_bates_cliquet_price_delta_cuda, 1.f/504.f,2U);
        compare_public<true, false>("bates/forward_start_option", fixtures::bates,
            product::ForwardStartOptionParameters{1.f,7U,21U}, 4096U,
            model::equity::bates::launch_bates_forward_start_option_cuda<OptionSide::call>,
            model::equity::bates::launch_bates_forward_start_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, false>("bates/range_accrual", fixtures::bates,
            product::RangeAccrualParameters{21U,7U,.9f,1.3f,.05f}, 4096U,
            model::equity::bates::launch_bates_range_accrual_cuda,
            model::equity::bates::launch_bates_range_accrual_price_delta_cuda, 1.f/504.f,2U);
        compare_public<true, false>("bates/up_and_out_option", fixtures::bates,
            product::UpAndOutOptionParameters{1.1f,1.21f,21U}, 4096U,
            model::equity::bates::launch_bates_up_and_out_option_cuda<OptionSide::call>,
            model::equity::bates::launch_bates_up_and_out_option_price_delta_cuda<OptionSide::call>, 1.f/504.f,2U);
        compare_public<true, true>("black_scholes/american_option", fixtures::black_scholes,
            product::AmericanOptionParameters{1.1f,8U,7U}, 1048576U,
            model::equity::black_scholes::launch_black_scholes_american_option_cuda<OptionSide::put>,
            model::equity::black_scholes::launch_black_scholes_american_option_price_delta_cuda<OptionSide::put>, 1.f/252.f);
        compare_public<true, true>("bates/american_option", fixtures::bates,
            product::AmericanOptionParameters{1.1f,8U,7U}, 4096U,
            model::equity::bates::launch_bates_american_option_cuda<OptionSide::put>,
            model::equity::bates::launch_bates_american_option_price_delta_cuda<OptionSide::put>, 1.f/504.f,2U);
        compare_public<true, true>("cev/american_option", fixtures::cev,
            product::AmericanOptionParameters{1.1f,8U,7U}, 1048576U,
            model::equity::cev::launch_cev_american_option_cuda<OptionSide::put>,
            model::equity::cev::launch_cev_american_option_price_delta_cuda<OptionSide::put>, 1.f/504.f,2U);
        compare_public<true, true>("heston/american_option", fixtures::heston,
            product::AmericanOptionParameters{1.1f,8U,7U}, 1048576U,
            model::equity::heston::launch_heston_american_option_cuda<OptionSide::put>,
            model::equity::heston::launch_heston_american_option_price_delta_cuda<OptionSide::put>, 1.f/504.f,2U);
        compare_public<true, true>("kou/american_option", fixtures::kou,
            product::AmericanOptionParameters{1.1f,8U,7U}, 4096U,
            model::equity::kou::launch_kou_american_option_cuda<OptionSide::put>,
            model::equity::kou::launch_kou_american_option_price_delta_cuda<OptionSide::put>, 1.f/252.f);
        compare_public<true, true>("merton/american_option", fixtures::merton,
            product::AmericanOptionParameters{1.1f,8U,7U}, 4096U,
            model::equity::merton::launch_merton_american_option_cuda<OptionSide::put>,
            model::equity::merton::launch_merton_american_option_price_delta_cuda<OptionSide::put>, 1.f/252.f);
        compare_public<true, true>("normal_inverse_gaussian/american_option", fixtures::normal_inverse_gaussian,
            product::AmericanOptionParameters{1.1f,8U,7U}, 4096U,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_american_option_cuda<OptionSide::put>,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_american_option_price_delta_cuda<OptionSide::put>, 1.f/252.f);
        compare_public<true, true>("schobel_zhu/american_option", fixtures::schobel_zhu,
            product::AmericanOptionParameters{1.1f,8U,7U}, 4096U,
            model::equity::schobel_zhu::launch_schobel_zhu_american_option_cuda<OptionSide::put>,
            model::equity::schobel_zhu::launch_schobel_zhu_american_option_price_delta_cuda<OptionSide::put>, 1.f/504.f,2U);
        compare_public<true, true>("variance_gamma/american_option", fixtures::variance_gamma,
            product::AmericanOptionParameters{1.1f,8U,7U}, 4096U,
            model::equity::variance_gamma::launch_variance_gamma_american_option_cuda<OptionSide::put>,
            model::equity::variance_gamma::launch_variance_gamma_american_option_price_delta_cuda<OptionSide::put>, 1.f/252.f);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

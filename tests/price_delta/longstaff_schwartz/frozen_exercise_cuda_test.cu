// BS/Heston/CEV public LSM deltas: central bits, time zero, maturity and bounded refits.
#include "tests/price_delta/cuda_test_support.cuh"
#include "common/equity/price_delta/spot_bump.cuh"
#include "model/equity/markovian/black_scholes/product/american_option.cuh"
#include "model/equity/markovian/black_scholes/product/american_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/american_option.cuh"
#include "model/equity/markovian/heston/product/american_option_price_delta.cuh"
#include "model/equity/markovian/cev/product/american_option.cuh"
#include "model/equity/markovian/cev/product/american_option_price_delta.cuh"

#include <array>
#include <bit>
#include <cmath>
#include <iostream>

using namespace ai_factory::workbench;
using price_delta_test::DeviceArray;
using price_delta_test::require;
namespace bs = model::equity::black_scholes;
namespace heston = model::equity::heston;
namespace cev = model::equity::cev;

template<OptionSide Side, typename Model, typename Price, typename Delta, typename... Time>
void compare(const char* name, Model model, Price price, Delta delta, Time... time) {
    constexpr std::size_t rows = 4, paths = 4096, blocks = 8;
    constexpr std::uint64_t seed = 91531;
    std::array<Model, rows> models{model, model, model, model};
    models[1].spot = .15f;
    models[1].risk_free_rate = .2f;
    const std::array<product::AmericanOptionParameters, rows> products{{
        {1.0f, 63, 7}, {1.0f, 21, 7}, {1.0f, 8, 7}, {1.05f, 20, 7}
    }};
    DeviceArray<Model> device_models(rows);
    DeviceArray<product::AmericanOptionParameters> device_products(rows);
    DeviceArray<float> base_prices(rows), base_errors(rows), prices(rows), errors(rows);
    DeviceArray<float> deltas(rows), delta_errors(rows);
    auto upload = [&](const auto& data, auto* destination) {
        check_cuda(cudaMemcpy(destination, data.data(), sizeof(data), cudaMemcpyHostToDevice), "upload LSM fixture");
    };
    auto download = [&](const float* source) {
        std::array<float, rows> values{};
        check_cuda(cudaMemcpy(values.data(), source, sizeof(values), cudaMemcpyDeviceToHost), "download LSM fixture");
        return values;
    };
    upload(products, device_products.data);
    for (unsigned threads : {128U, 256U}) {
        upload(models, device_models.data);
        const auto base = price(device_models.data, rows, products.data(), device_products.data,
            rows, PriceConstruction::Aligned, rows, paths, time..., threads, blocks, seed,
            base_prices.data, base_errors.data);
        longstaff_schwartz::validate_regression_diagnostics(base, name);
        const auto expected_prices = download(base_prices.data);
        const auto expected_errors = download(base_errors.data);
        for (float width : {.01f, .005f}) {
            const auto result = delta(models.data(), device_models.data, rows,
                products.data(), device_products.data, rows, PriceConstruction::Aligned,
                rows, paths, time..., threads, blocks, seed,
                equity::price_delta::SpotBumpConfiguration{width},
                prices.data, errors.data, deltas.data, delta_errors.data);
            longstaff_schwartz::validate_regression_diagnostics(result, name);
            require(result.kernel_launch_count == base.kernel_launch_count + 2 * base.batch_count,
                    "Delta must add two passes, not extra regressions");
            require(result.workspace_bytes >= base.workspace_bytes + rows * paths * 8,
                    "Stopping trace missing from workspace budget");
            const auto actual_prices = download(prices.data), actual_errors = download(errors.data);
            const auto actual_deltas = download(deltas.data), actual_delta_errors = download(delta_errors.data);
            for (std::size_t row = 0; row < rows; ++row) {
                require(std::bit_cast<std::uint32_t>(actual_prices[row]) == std::bit_cast<std::uint32_t>(expected_prices[row]),
                        "Central LSM price bits differ");
                require(std::bit_cast<std::uint32_t>(actual_errors[row]) == std::bit_cast<std::uint32_t>(expected_errors[row]),
                        "Central LSM error bits differ");
                require(std::isfinite(actual_deltas[row]) && std::isfinite(actual_delta_errors[row])
                        && actual_delta_errors[row] >= 0, "Invalid paired delta statistics");
            }
            if constexpr (Side == OptionSide::put) {
                require(actual_prices[1] == 1.0f - models[1].spot && actual_errors[1] == 0,
                        "Fixture must exercise at time zero");
                const auto bump = equity::price_delta::prepare_spot_bump(models[1].spot, {width});
                const float expected = ((1.0f - bump.upper) - (1.0f - bump.lower)) / bump.width;
                require(actual_deltas[1] == expected && actual_delta_errors[1] == 0,
                        "Initial exercise was not frozen");
            }
            // Refit comparisons are bounded diagnostics, not a policy-bias certificate.
            if (threads == 128U && width == .01f) {
                std::array<std::array<float, rows>, 2> bumped_prices{};
                for (unsigned side = 0; side < 2; ++side) {
                    auto bumped_models = models;
                    for (std::size_t row = 0; row < rows; ++row) {
                        const auto bump = equity::price_delta::prepare_spot_bump(models[row].spot, {width});
                        bumped_models[row].spot = side == 0 ? bump.lower : bump.upper;
                    }
                    upload(bumped_models, device_models.data);
                    const auto refit = price(device_models.data, rows, products.data(), device_products.data,
                        rows, PriceConstruction::Aligned, rows, paths, time..., threads, blocks, seed,
                        base_prices.data, base_errors.data);
                    longstaff_schwartz::validate_regression_diagnostics(refit, name);
                    bumped_prices[side] = download(base_prices.data);
                }
                upload(models, device_models.data);
                for (std::size_t row = 0; row < rows; ++row) {
                    const auto bump = equity::price_delta::prepare_spot_bump(models[row].spot, {width});
                    const float refit = (bumped_prices[1][row] - bumped_prices[0][row]) / bump.width;
                    std::cout << name << " " << option_side_name(Side) << " row=" << row
                              << " frozen=" << actual_deltas[row] << " se=" << actual_delta_errors[row]
                              << " refit=" << refit << " gap=" << actual_deltas[row] - refit << '\n';
                }
            }
            std::cout << name << " " << option_side_name(Side) << " threads=" << threads
                      << " bump=" << width << " price_ms=" << base.kernel_seconds * 1000
                      << " price_delta_ms=" << result.kernel_seconds * 1000 << '\n';
        }
    }
}

template<OptionSide Side>
void run() {
    compare<Side>("black_scholes", bs::ModelParameters{1.0f, .03f, .01f, .2f},
        bs::launch_black_scholes_american_option_cuda<Side>,
        bs::launch_black_scholes_american_option_price_delta_cuda<Side>, 1.0f / 252);
    compare<Side>("heston", heston::ModelParameters{1.0f, .03f, .01f, .04f, 1.5f, .04f, .4f, -.7f},
        heston::launch_heston_american_option_cuda<Side>,
        heston::launch_heston_american_option_price_delta_cuda<Side>, 1.0f / 504, 2U);
    compare<Side>("cev", cev::ModelParameters{1.0f, .03f, .01f, .3f, .6f},
        cev::launch_cev_american_option_cuda<Side>,
        cev::launch_cev_american_option_price_delta_cuda<Side>, 1.0f / 504, 2U);
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        run<OptionSide::call>();
        run<OptionSide::put>();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

// Bounded BS/Heston/CEV CRN checks: central bits, paired delta, stopping and launches.
#include "common/equity/price_delta/multiplicative_spot_path.cuh"
#include "common/equity/price_delta/coupled_spot_paths.cuh"
#include "common/equity/price_delta/path_product_policy.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/monte_carlo/monte_carlo_price_delta_kernel.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/cev/price_delta_dynamics_impl.cuh"
#include "model/equity/markovian/sabr/price_delta_dynamics_impl.cuh"
#include "model/equity/markovian/heston/product/european_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/up_and_out_option_price_delta.cuh"
#include "model/equity/markovian/cev/product/european_option_price_delta.cuh"
#include "model/equity/markovian/cev/product/up_and_out_option_price_delta.cuh"
#include "product/european_option/pricing_policy.cuh"
#include "product/digital_option/pricing_policy.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"
#include "tests/price_delta/cuda_test_support.cuh"

#include <bit>
#include <iostream>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace delta = equity::price_delta;
namespace bs = model::equity::black_scholes;
namespace heston = model::equity::heston;
namespace cev = model::equity::cev;
namespace sabr = model::equity::sabr;

using price_delta_test::require;
using price_delta_test::DeviceArray;

template<typename BasePolicy, typename DeltaPolicy>
__global__ void compare_paths(
    typename BasePolicy::ModelParameters model,
    typename BasePolicy::ProductParameters product,
    typename BasePolicy::TimeConfiguration time,
    float* errors
) {
    const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
    const auto key = philox::make_key(719U);
    const delta::SpotBumpConfiguration config{};
    const auto paired = DeltaPolicy::evaluate_path(
        DeltaPolicy::prepare_row(model, product, config, time), key, path);
    const float central = BasePolicy::evaluate_path(
        BasePolicy::prepare_row(model, product, time), key, path);
    const auto bump = delta::prepare_spot_bump(model.spot, config);
    auto lower = model;
    auto upper = model;
    lower.spot = bump.lower;
    upper.spot = bump.upper;
    const float minus = BasePolicy::evaluate_path(
        BasePolicy::prepare_row(lower, product, time), key, path);
    const float plus = BasePolicy::evaluate_path(
        BasePolicy::prepare_row(upper, product, time), key, path);
    const float reference_delta = (plus - minus) / bump.width;
    errors[2U * path] = __float_as_uint(central) == __float_as_uint(paired.price)
        ? 0.0f : 1.0f;
    errors[2U * path + 1U] = fabsf(paired.delta - reference_delta);
}

template<typename BaseSchedule, typename DeltaSchedule, typename Path, typename Product,
         typename PublicLauncher = std::nullptr_t>
void run_case(
    const char* name, typename Path::ModelParameters model,
    typename Product::ProductParameters product,
    typename BaseSchedule::TimeConfiguration time,
    float delta_tolerance = 0.003f, PublicLauncher public_launcher = nullptr
) {
    using Base = equity::PathProductMonteCarloPricingPolicy<BaseSchedule, Product>;
    using Paired = delta::PathProductPriceDeltaPolicy<DeltaSchedule, Product, Path>;
    constexpr std::size_t rows = 3U;
    constexpr std::size_t paths = 4096U;
    std::vector<typename Path::ModelParameters> models(rows, model);
    std::vector<typename Product::ProductParameters> products(rows, product);
    DeviceArray<typename Path::ModelParameters> device_models(rows);
    DeviceArray<typename Product::ProductParameters> device_products(rows);
    DeviceArray<float> outputs(6U * rows);
    check_cuda(cudaMemcpy(device_models.data, models.data(), sizeof(model) * rows,
                          cudaMemcpyHostToDevice), name);
    check_cuda(cudaMemcpy(device_products.data, products.data(), sizeof(product) * rows,
                          cudaMemcpyHostToDevice), name);
    const auto inputs = make_model_product_device_inputs(device_models.data, rows,
        device_products.data, rows, PriceConstruction::Aligned);
    const delta::SpotBumpConfiguration bump{};
    for (unsigned int threads : {128U, 256U}) {
        monte_carlo::launch_monte_carlo_cuda<Base>(inputs,
            {products.data(), rows, PriceConstruction::Aligned}, rows, 0U, rows,
            paths, time, threads, 2U, 719U, outputs.data, outputs.data + rows,
            name, "price", name);
        // Two batches, nonzero offset, and a persistent block in the first batch.
        for (std::size_t offset : {0U, 2U}) {
            if constexpr (std::is_same_v<PublicLauncher, std::nullptr_t>) {
                monte_carlo::launch_monte_carlo_price_delta_cuda<Paired>(
                {inputs, bump}, {models.data(), rows, products.data(), rows,
                    PriceConstruction::Aligned, bump}, rows, offset,
                offset == 0U ? 2U : 1U, paths, time, threads, 1U, 719U,
                outputs.data + 2U * rows, outputs.data + 3U * rows,
                outputs.data + 4U * rows, outputs.data + 5U * rows, name, "delta");
            } else {
                public_launcher(models.data(), device_models.data, rows,
                    products.data(), device_products.data, rows, PriceConstruction::Aligned,
                    rows, offset, offset == 0U ? 2U : 1U, paths, time.dt,
                    time.simulation_steps_per_day, threads, 1U, 719U, bump,
                    outputs.data + 2U * rows, outputs.data + 3U * rows,
                    outputs.data + 4U * rows, outputs.data + 5U * rows);
            }
        }
        float results[6U * rows];
        check_cuda(cudaMemcpy(results, outputs.data, sizeof(results), cudaMemcpyDeviceToHost), name);
        for (std::size_t row = 0; row < rows; ++row) {
            require(std::bit_cast<unsigned int>(results[row])
                == std::bit_cast<unsigned int>(results[2U * rows + row]), "central price bits differ");
            require(std::bit_cast<unsigned int>(results[rows + row])
                == std::bit_cast<unsigned int>(results[3U * rows + row]), "central SE bits differ");
            for (std::size_t column = 0; column < 6U; ++column)
                require(std::isfinite(results[column * rows + row]), "non-finite result");
            require(results[5U * rows + row] >= 0.0f, "negative delta SE");
        }
    }
    DeviceArray<float> errors(512U);
    compare_paths<Base, Paired><<<2U, 128U>>>(model, product, time, errors.data);
    check_cuda(cudaGetLastError(), "path comparison");
    float host_errors[512U];
    check_cuda(cudaMemcpy(host_errors, errors.data, sizeof(host_errors), cudaMemcpyDeviceToHost), name);
    float maximum_error = 0.0f;
    for (std::size_t path = 0; path < 256U; ++path) {
        require(host_errors[2U * path] == 0.0f, "central path bits differ");
        const float error = host_errors[2U * path + 1U];
        require(std::isfinite(error) && error <= delta_tolerance, "paired vs resimulated delta differs");
        maximum_error = std::max(maximum_error, error);
    }
    std::cout << name << ": price/SE/path bitwise; max bumped delta difference "
              << maximum_error << '\n';
}
}  // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        using BsPath = delta::MultiplicativeSpotPath<bs::DynamicsPolicy>;
        using HestonPath = delta::MultiplicativeSpotPath<heston::DynamicsPolicy>;
        using CevPath = delta::CoupledSpotPaths<cev::PriceDeltaDynamics>;
        using BsTerminal = simulation::ExactTransitionTerminalSchedule<bs::DynamicsPolicy>;
        using HestonTerminal = simulation::FixedStepTerminalSchedule<heston::DynamicsPolicy>;
        using CevTerminal = simulation::FixedStepTerminalSchedule<cev::DynamicsPolicy>;
        using CevDeltaTerminal = simulation::FixedStepTerminalSchedule<cev::PriceDeltaDynamics>;
        using Call = product::EuropeanOptionPathPolicy<OptionSide::call>;
        using Put = product::EuropeanOptionPathPolicy<OptionSide::put>;
        using Digital = product::DigitalOptionPathPolicy<OptionSide::call>;
        using Barrier = product::UpAndOutOptionPathPolicy<OptionSide::call>;
        const bs::ModelParameters bs_model{1.2f, .03f, .01f, .2f};
        const heston::ModelParameters heston_model{1.2f,.03f,.01f,.04f,1.5f,.04f,.4f,-.7f};
        const cev::ModelParameters cev_model{1.2f,.03f,.01f,.3f,.6f};
        const simulation::FixedStepTimeConfiguration fixed{1.0f / 504.0f, 2U};
        using SabrPath = delta::CoupledSpotPaths<sabr::PriceDeltaDynamics>;
        using SabrTerminal = simulation::FixedStepTerminalSchedule<sabr::DynamicsPolicy>;
        using SabrDeltaTerminal = simulation::FixedStepTerminalSchedule<sabr::PriceDeltaDynamics>;
        run_case<SabrTerminal, SabrDeltaTerminal, SabrPath, Put>("SABR normalized volatility",
            {1.2f,.03f,.01f,.2f,.4f,-.5f,.6f}, {1.1f,21U}, fixed, 0.0f);
        run_case<BsTerminal, BsTerminal, BsPath, Call>("BS exact call", bs_model, {1.1f,63U}, {1.f/252.f});
        run_case<BsTerminal, BsTerminal, BsPath, Digital>("BS digital", bs_model, {1.2f,63U,1.f}, {1.f/252.f});
        run_case<HestonTerminal, HestonTerminal, HestonPath, Call>("Heston call", heston_model, {1.1f,63U}, fixed,
            .003f, heston::launch_heston_european_option_price_delta_cuda<OptionSide::call>);
        run_case<CevTerminal, CevDeltaTerminal, CevPath, Put>("CEV put", cev_model, {1.1f,63U}, fixed,
            0.0f, cev::launch_cev_european_option_price_delta_cuda<OptionSide::put>);
        using HestonDense = simulation::FixedStepDenseSchedule<heston::DynamicsPolicy>;
        run_case<HestonDense, HestonDense, HestonPath, Barrier>("Heston near barrier", heston_model, {1.0f,1.202f,20U}, fixed,
            .003f, heston::launch_heston_up_and_out_option_price_delta_cuda<OptionSide::call>);
        using CevDense = simulation::FixedStepDenseSchedule<cev::DynamicsPolicy>;
        using CevDeltaDense = simulation::FixedStepDenseSchedule<cev::PriceDeltaDynamics>;
        run_case<CevDense, CevDeltaDense, CevPath, Barrier>("CEV near barrier", cev_model, {1.0f,1.202f,20U}, fixed,
            0.0f, cev::launch_cev_up_and_out_option_price_delta_cuda<OptionSide::call>);
        for (float width : {0.0f, -1.0f, 2.0f, 1.e-12f}) {
            bool rejected = false;
            try { delta::validate_spot_bump(1.f, {width}); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid bump accepted");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

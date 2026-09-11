// Rough Heston/QRH: public price parity, independently prepared bumps and factor sharing.
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/rough_heston/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/product/european_option.cuh"
#include "model/equity/rough/rough_heston/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_heston/product/up_and_out_option.cuh"
#include "model/equity/rough/rough_heston/product/up_and_out_option_price_delta.cuh"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/quadratic_rough_heston/dynamics_impl.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/european_option.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/european_option_price_delta.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/up_and_out_option.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/up_and_out_option_price_delta.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/price_delta/path_product_policy.cuh"
#include "common/equity/price_delta/multiplicative_spot_path.cuh"
#include "product/european_option/pricing_policy.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"
#include "tests/price_delta/cuda_test_support.cuh"
#include <bit>
#include <cstring>
#include <iostream>

namespace {
using namespace ai_factory::workbench;
using price_delta_test::require;
using price_delta_test::DeviceArray;
namespace delta = equity::price_delta;
namespace rh = model::equity::rough_heston;
namespace qrh = model::equity::quadratic_rough_heston;
constexpr simulation::FixedStepTimeConfiguration time_grid{1.f/504.f, 2U};

template<typename Schedule, typename Product>
__global__ void compare_paths(typename Schedule::Dynamics::Parameters model,
    typename Product::ProductParameters product,
    const typename Schedule::Dynamics::PreparedDynamics* prepared,
    delta::SpotBumpConfiguration configuration, float* errors) {
    const simulation::FixedStepTimeConfiguration time_grid{1.f/504.f, 2U};
    using Dynamics = typename Schedule::Dynamics;
    using Scalar = equity::PathProductMonteCarloPricingPolicy<Schedule, Product>;
    using Paired = delta::PathProductPriceDeltaPolicy<Schedule, Product,
        delta::MultiplicativeSpotPath<Dynamics>>;
    const auto path = threadIdx.x;
    const auto key = philox::make_key(771U);
    const auto endpoints = delta::prepare_spot_bump(model.spot, configuration);
    const auto paired = Paired::evaluate_path(
        Paired::prepare_row(model, product, prepared[0], configuration, time_grid), key, path);
    const float central = Scalar::evaluate_path(Scalar::prepare_row(model, product, prepared[0], time_grid), key, path);
    auto lower = model, upper = model;
    lower.spot = endpoints.lower;
    upper.spot = endpoints.upper;
    const float lo = Scalar::evaluate_path(Scalar::prepare_row(lower, product, prepared[1], time_grid), key, path);
    const float hi = Scalar::evaluate_path(Scalar::prepare_row(upper, product, prepared[2], time_grid), key, path);
    errors[2*path] = __float_as_uint(central) == __float_as_uint(paired.price) ? 0.f : 1.f;
    errors[2*path+1] = fabsf(paired.delta - (hi-lo)/endpoints.width);
}

template<typename Schedule, typename Product, typename Prepare, typename Price, typename Paired>
void run_case(const char* name, typename Schedule::Dynamics::Parameters model,
    typename Product::ProductParameters product, Prepare prepare, Price price, Paired paired,
    std::size_t paths) {
    using Model = typename Schedule::Dynamics::Parameters;
    using Prepared = typename Schedule::Dynamics::PreparedDynamics;
    using Parameters = typename Product::ProductParameters;
    Model models[3]{model, model, model};
    Parameters products[3]{product, product, product};
    DeviceArray<Model> dm(3);
    DeviceArray<Parameters> dp(3);
    DeviceArray<Prepared> dd(3);
    DeviceArray<float> out(18), errors(512);
    check_cuda(cudaMemcpy(dm.data, models, sizeof(models), cudaMemcpyHostToDevice), "models");
    check_cuda(cudaMemcpy(dp.data, products, sizeof(products), cudaMemcpyHostToDevice), "products");
    float maximum_oracle_error = 0.f;
    for (float width : {.01f, .005f}) {
        const delta::SpotBumpConfiguration bump{width};
        Prepared central[3]{prepare(model), prepare(model), prepare(model)};
        check_cuda(cudaMemcpy(dd.data, central, sizeof(central), cudaMemcpyHostToDevice), "prepared");
        for (unsigned threads : {128U, 256U}) {
            price(dm.data, 3U, dd.data, 3U, products, dp.data, 3U, PriceConstruction::Aligned,
                3U, 0U, 3U, paths, time_grid.dt, 2U, threads, 2U, 771U, out.data, out.data+3);
            for (std::size_t offset : {0U, 2U}) {
                paired(models, dm.data, 3U, dd.data, 3U, products, dp.data, 3U, PriceConstruction::Aligned,
                    3U, offset, offset == 0 ? 2U : 1U, paths, time_grid.dt, 2U, threads, 1U, 771U,
                    bump, out.data+6, out.data+9, out.data+12, out.data+15);
            }
            float result[18];
            check_cuda(cudaMemcpy(result, out.data, sizeof(result), cudaMemcpyDeviceToHost), "results");
            for (unsigned i=0; i<18; ++i) require(std::isfinite(result[i]), "non-finite rough output");
            for (unsigned i=0; i<6; ++i)
                require(std::bit_cast<unsigned>(result[i]) == std::bit_cast<unsigned>(result[i+6]),
                        "rough central price/error bits differ");
            for (unsigned i=15; i<18; ++i) require(result[i] >= 0, "negative paired error");
        }
        const auto endpoints = delta::prepare_spot_bump(model.spot, bump);
        auto lower = model, upper = model;
        lower.spot = endpoints.lower;
        upper.spot = endpoints.upper;
        Prepared scenarios[3]{prepare(model), prepare(lower), prepare(upper)};
        for (unsigned i=1; i<3; ++i) {
            auto factors = scenarios[i];
            factors.initial_log_spot = scenarios[0].initial_log_spot;
            require(std::memcmp(&factors, &scenarios[0], sizeof(Prepared)) == 0,
                    "rough factor preparation depends on S0");
        }
        check_cuda(cudaMemcpy(dd.data, scenarios, sizeof(scenarios), cudaMemcpyHostToDevice), "bumped preparations");
        compare_paths<Schedule, Product><<<1,256>>>(model, product, dd.data, bump, errors.data);
        check_cuda(cudaGetLastError(), "path oracle");
        float result[512];
        check_cuda(cudaMemcpy(result, errors.data, sizeof(result), cudaMemcpyDeviceToHost), "oracle results");
        for (unsigned i=0; i<256; ++i) {
            maximum_oracle_error = std::max(maximum_oracle_error, result[2*i+1]);
            require(result[2*i] == 0, "rough pathwise central bits differ");
            require(std::isfinite(result[2*i+1]) && result[2*i+1] < .005f,
                    "rough paired delta differs from original bumped transitions");
        }
    }
    std::cout << name << ": central price/error/path bits; prepared-bump oracle; paths=" << paths << "; max_delta_gap=" << maximum_oracle_error << '\n';
}

template<std::size_t N>
void run_models() {
    std::cout << "factor_count=" << N << '\n';
    using Call = product::EuropeanOptionPathPolicy<OptionSide::call>;
    using Put = product::EuropeanOptionPathPolicy<OptionSide::put>;
    using Barrier = product::UpAndOutOptionPathPolicy<OptionSide::call>;
    const rh::ModelParameters h{1.2f,.03f,.01f,.04f,1.5f,.06f,.3f,.15f,-.7f};
    const qrh::ModelParameters q{1.2f,.03f,.01f,.1f,.1f,.1f,.04f,1.f,.2f,.15f};
    const auto ph = [](const auto& m) { return rh::prepare_dynamics<N>(m, 21.f/252.f, time_grid.dt); };
    const auto pq = [](const auto& m) { return qrh::prepare_dynamics<N>(m, 21.f/252.f, time_grid.dt); };
    const auto paths = N == 7U ? (1U<<20U) : 4096U;
    run_case<simulation::FixedStepTerminalSchedule<rh::DynamicsPolicy<N>>, Call>("rough Heston call", h, {1.1f,4U}, ph,
        rh::launch_rough_heston_european_option_cuda<OptionSide::call,N>,
        rh::launch_rough_heston_european_option_price_delta_cuda<OptionSide::call,N>, paths);
    run_case<simulation::FixedStepTerminalSchedule<qrh::DynamicsPolicy<N>>, Put>("QRH put", q, {1.3f,4U}, pq,
        qrh::launch_quadratic_rough_heston_european_option_cuda<OptionSide::put,N>,
        qrh::launch_quadratic_rough_heston_european_option_price_delta_cuda<OptionSide::put,N>, paths);
    run_case<simulation::FixedStepDenseSchedule<rh::DynamicsPolicy<N>>, Barrier>("rough Heston barrier", h, {1.f,1.202f,21U}, ph,
        rh::launch_rough_heston_up_and_out_option_cuda<OptionSide::call,N>,
        rh::launch_rough_heston_up_and_out_option_price_delta_cuda<OptionSide::call,N>, 4096U);
    run_case<simulation::FixedStepDenseSchedule<qrh::DynamicsPolicy<N>>, Barrier>("QRH barrier", q, {1.f,1.202f,21U}, pq,
        qrh::launch_quadratic_rough_heston_up_and_out_option_cuda<OptionSide::call,N>,
        qrh::launch_quadratic_rough_heston_up_and_out_option_price_delta_cuda<OptionSide::call,N>, 4096U);
}
}
int main() {
    try {
        int devices=0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        // Inspect the production factor count first when diagnostics deduplicate names.
        run_models<7>(); run_models<3>(); run_models<2>();
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

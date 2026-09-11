// Public FFT pricing: central bits, shared-random-number bumps and chunk boundaries.
#include "model/equity/rough/rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/rough_bergomi/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/product/up_and_out_option.cuh"
#include "model/equity/rough/rough_bergomi/product/up_and_out_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/european_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/up_and_out_option.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/up_and_out_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/product/european_option.cuh"
#include "model/equity/rough/rough_stein_stein/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/product/up_and_out_option.cuh"
#include "model/equity/rough/rough_stein_stein/product/up_and_out_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/product/european_option.cuh"
#include "model/equity/rough/rough_sabr/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/product/up_and_out_option.cuh"
#include "model/equity/rough/rough_sabr/product/up_and_out_option_price_delta.cuh"
#include "tests/price_delta/cuda_test_support.cuh"
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>

namespace {
using namespace ai_factory::workbench;
using price_delta_test::require;
using price_delta_test::DeviceArray;

template<typename Model, typename Product, typename Price, typename Paired>
void run_case(const char* name, Model model, Product product, Price price, Paired paired,
              std::size_t paths) {
    constexpr float day = 1.f/252.f, dt = 1.f/504.f;
    constexpr std::size_t rows = 3U, steps = 8U;
    Model models[rows]{model, model, model};
    Product products[rows]{product, product, product};
    DeviceArray<Model> dm(rows);
    DeviceArray<Product> dp(rows);
    DeviceArray<float> out(24);
    const auto maximum = volterra::plan_hybrid_fft_price_delta_workspace(steps, paths, 4096U);
    DeviceArray<unsigned char> workspace(maximum.workspace_bytes);
    check_cuda(cudaMemcpy(dp.data, products, sizeof(products), cudaMemcpyHostToDevice), "products");
    double scalar_seconds=0, paired_seconds=0;
    for (float width : {.01f, .005f}) {
        const equity::price_delta::SpotBumpConfiguration bump{width};
        if (paths == 1048576U && width != .01f) continue;
        for (std::size_t chunk : {256U, 4096U}) {
            if (paths == 1048576U && chunk != 4096U) continue;
            // All three rows, including a nonzero result index; identical independent streams.
            check_cuda(cudaMemcpy(dm.data, models, sizeof(models), cudaMemcpyHostToDevice), "models");
            auto start=std::chrono::steady_clock::now();
            for (std::size_t row=0; row<rows; ++row)
                price(dm.data, rows, dp.data, rows, PriceConstruction::Aligned, rows, row, paths,
                      day, dt, steps, chunk, workspace.data, maximum.workspace_bytes, 971U,
                      out.data, out.data+3);
            check_cuda(cudaDeviceSynchronize(), "scalar");
            scalar_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            start=std::chrono::steady_clock::now();
            for (std::size_t row=0; row<rows; ++row)
                paired(models, dm.data, rows, products, dp.data, rows, PriceConstruction::Aligned,
                       rows, row, paths, day, dt, steps, chunk, workspace.data, maximum.workspace_bytes,
                       971U, bump, out.data+6, out.data+9, out.data+12, out.data+15);
            check_cuda(cudaDeviceSynchronize(), "paired");
            paired_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            float result[18];
            check_cuda(cudaMemcpy(result, out.data, sizeof(result), cudaMemcpyDeviceToHost), "results");
            for (float v : result) require(std::isfinite(v), "non-finite FFT output");
            for (unsigned i=0; i<6; ++i)
                require(std::bit_cast<unsigned>(result[i]) == std::bit_cast<unsigned>(result[i+6]),
                        "FFT central price/error bits differ");
            for (unsigned i=15; i<18; ++i) require(result[i]>=0, "negative paired error");
            Model lower[rows], upper[rows];
            for (unsigned i=0;i<rows;++i) {
                auto endpoints=equity::price_delta::prepare_spot_bump(models[i].spot,bump);
                lower[i]=models[i]; upper[i]=models[i];
                lower[i].spot=endpoints.lower; upper[i].spot=endpoints.upper;
            }
            for (unsigned scenario=0;scenario<2;++scenario) {
                check_cuda(cudaMemcpy(dm.data, scenario==0?lower:upper, sizeof(models), cudaMemcpyHostToDevice), "bumps");
                for (std::size_t row=0;row<rows;++row)
                    price(dm.data, rows, dp.data, rows, PriceConstruction::Aligned, rows, row, paths,
                          day, dt, steps, chunk, workspace.data, maximum.workspace_bytes, 971U,
                          out.data+18+scenario*3,out.data);
            }
            float oracle[6];
            check_cuda(cudaMemcpy(oracle,out.data+18,sizeof(oracle),cudaMemcpyDeviceToHost),"oracle");
            for(unsigned i=0;i<rows;++i) {
                const auto endpoints=equity::price_delta::prepare_spot_bump(models[i].spot,bump);
                const float reference=(oracle[i+3]-oracle[i])/endpoints.width;
                require(std::abs(result[12+i]-reference)<.002f, "FFT independently prepared bump mismatch");
            }
        }
    }
    if (paths == 4097U) {
        check_cuda(cudaMemcpy(dm.data, models, sizeof(models), cudaMemcpyHostToDevice), "restore");
        auto submit = [&](const Model* mirror, std::size_t selected_steps, std::size_t bytes,
                          equity::price_delta::SpotBumpConfiguration bump, float* deltas) {
            paired(mirror, dm.data, 1U, products, dp.data, rows, PriceConstruction::CartesianProduct,
                   rows, 2U, paths, day, dt, selected_steps, 4096U, workspace.data, bytes,
                   971U, bump, out.data, out.data+3, deltas, out.data+9);
        };
        submit(models, steps, maximum.workspace_bytes, {.01f}, out.data+6);
        price(dm.data, 1U, dp.data, rows, PriceConstruction::CartesianProduct, rows, 2U, paths,
              day, dt, steps, 4096U, workspace.data, maximum.workspace_bytes, 971U,
              out.data+12, out.data+15);
        float result[18];
        check_cuda(cudaMemcpy(result, out.data, sizeof(result), cudaMemcpyDeviceToHost), "Cartesian");
        require(std::bit_cast<unsigned>(result[2]) == std::bit_cast<unsigned>(result[14]), "Cartesian price bits");
        require(std::bit_cast<unsigned>(result[5]) == std::bit_cast<unsigned>(result[17]), "Cartesian error bits");
        auto rejects = [](auto invoke) {
            bool caught=false;
            try { invoke(); } catch(const std::invalid_argument&) { caught=true; }
            require(caught, "invalid FFT delta launch accepted");
        };
        rejects([&] { submit(nullptr,steps,maximum.workspace_bytes,{.01f},out.data+6); });
        rejects([&] { submit(models,steps-1,maximum.workspace_bytes,{.01f},out.data+6); });
        rejects([&] { submit(models,steps,maximum.workspace_bytes-1,{.01f},out.data+6); });
        rejects([&] { submit(models,steps,maximum.workspace_bytes,{1e-12f},out.data+6); });
        rejects([&] { submit(models,steps,maximum.workspace_bytes,{.01f},nullptr); });
    }
    std::cout<<name<<": paths="<<paths<<" central bits; independent bump prices; scalar_s="
             <<scalar_seconds<<" paired_s="<<paired_seconds<<'\n';
}
}
int main() {
    using namespace ai_factory::workbench;
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
    try {
        namespace rough_bergomi = model::equity::rough_bergomi;
        run_case("rough_bergomi european_option", rough_bergomi::ModelParameters{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f},
            product::EuropeanOptionParameters{1.2f,4U}, rough_bergomi::launch_rough_bergomi_european_option_cuda<OptionSide::call>,
            rough_bergomi::launch_rough_bergomi_european_option_price_delta_cuda<OptionSide::call>,
            1048576U);
        run_case("rough_bergomi up_and_out_option", rough_bergomi::ModelParameters{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f},
            product::UpAndOutOptionParameters{1.1f,1.202f,4U}, rough_bergomi::launch_rough_bergomi_up_and_out_option_cuda<OptionSide::call>,
            rough_bergomi::launch_rough_bergomi_up_and_out_option_price_delta_cuda<OptionSide::call>,
            4097U);
        namespace log_modulated_rough_bergomi = model::equity::log_modulated_rough_bergomi;
        run_case("log_modulated_rough_bergomi european_option", log_modulated_rough_bergomi::ModelParameters{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f,1.f,1.5f},
            product::EuropeanOptionParameters{1.2f,4U}, log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_european_option_cuda<OptionSide::call>,
            log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_european_option_price_delta_cuda<OptionSide::call>,
            1048576U);
        run_case("log_modulated_rough_bergomi up_and_out_option", log_modulated_rough_bergomi::ModelParameters{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f,1.f,1.5f},
            product::UpAndOutOptionParameters{1.1f,1.202f,4U}, log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_up_and_out_option_cuda<OptionSide::call>,
            log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_up_and_out_option_price_delta_cuda<OptionSide::call>,
            4097U);
        namespace rough_stein_stein = model::equity::rough_stein_stein;
        run_case("rough_stein_stein european_option", rough_stein_stein::ModelParameters{1.2f,.03f,.01f,.2f,1.5f,.3f,.1f,-.7f},
            product::EuropeanOptionParameters{1.2f,4U}, rough_stein_stein::launch_rough_stein_stein_european_option_cuda<OptionSide::call>,
            rough_stein_stein::launch_rough_stein_stein_european_option_price_delta_cuda<OptionSide::call>,
            1048576U);
        run_case("rough_stein_stein up_and_out_option", rough_stein_stein::ModelParameters{1.2f,.03f,.01f,.2f,1.5f,.3f,.1f,-.7f},
            product::UpAndOutOptionParameters{1.1f,1.202f,4U}, rough_stein_stein::launch_rough_stein_stein_up_and_out_option_cuda<OptionSide::call>,
            rough_stein_stein::launch_rough_stein_stein_up_and_out_option_price_delta_cuda<OptionSide::call>,
            4097U);
        namespace rough_sabr = model::equity::rough_sabr;
        run_case("rough_sabr european_option", rough_sabr::ModelParameters{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f,.6f},
            product::EuropeanOptionParameters{1.2f,4U}, rough_sabr::launch_rough_sabr_european_option_cuda<OptionSide::call>,
            rough_sabr::launch_rough_sabr_european_option_price_delta_cuda<OptionSide::call>,
            1048576U);
        run_case("rough_sabr up_and_out_option", rough_sabr::ModelParameters{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f,.6f},
            product::UpAndOutOptionParameters{1.1f,1.202f,4U}, rough_sabr::launch_rough_sabr_up_and_out_option_cuda<OptionSide::call>,
            rough_sabr::launch_rough_sabr_up_and_out_option_price_delta_cuda<OptionSide::call>,
            4097U);
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}

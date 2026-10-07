// Causal FFT versus the same model cells evaluated by direct convolution.
#include "common/check_cuda.cuh"
#include "common/volterra/causal_fft_pricer.cuh"
#include "model/equity/rough/rough_heston/causal_fft_pricing.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_pricing.cuh"
#include "product/asian_option/pricing_policy.cuh"
#include "product/european_option/pricing_policy.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace causal = ai_factory::workbench::volterra::causal_fft;

constexpr std::size_t paths = 513U;
constexpr std::size_t rows = 2U;
constexpr std::uint64_t seed = 912000001ULL;

template<unsigned Steps, class Path, class Product, class Schedule>
__global__ void direct_reference(
    const causal::PreparedRow<Path, Product, Schedule>* prepared_rows,
    const float* weights,
    float* payoffs
) {
    const std::size_t row_index = blockIdx.y;
    const std::size_t path = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_index >= rows || path >= paths) return;
    const auto row = prepared_rows[row_index];
    auto state = Path::initial_state(row.model);
    auto handler = Product::make_handler(row.product);
    auto cursor = Schedule::make_cursor(row.schedule);
    equity::PathProductObservationAdapter<
        Path, typename Product::Handler, Product::kObservationCoordinate
    > observer{handler};
    bool active = Schedule::on_initial_state(
        row.schedule, cursor, state, observer
    );
    float source[Steps];
    for (unsigned step = 0U; step < Steps; ++step) {
        float far = 0.0f;
        for (unsigned prior = 0U; prior < step; ++prior) {
            far = fmaf(
                weights[row_index * Steps + step - prior - 1U],
                source[prior], far
            );
        }
        source[step] = active ? Path::advance_cell(
            row.model, row.kernel, far, row.key,
            path, step, state
        ) : 0.0f;
        if (active) active = Schedule::on_step(
            row.schedule, cursor, step, state, observer
        );
    }
    payoffs[row_index * paths + path] =
        Product::template finalize<Path>(row.product, state, handler);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<unsigned Steps, class Path, class Product, class Schedule>
void exercise(
    const std::vector<typename Path::Parameters>& models,
    const std::vector<typename Product::ProductParameters>& products,
    const char* label
) {
    constexpr unsigned steps = Steps;
    using Model = typename Path::Parameters;
    using ProductParameters = typename Product::ProductParameters;
    const auto plan = causal::plan_workspace<Path, Product, Schedule>(
        steps, paths, 256U, rows
    );
    Model* device_models = nullptr;
    ProductParameters* device_products = nullptr;
    float *device_prices = nullptr, *device_errors = nullptr;
    float* device_payoffs = nullptr;
    void* workspace = nullptr;
    check_cuda(cudaMalloc(&device_models, rows * sizeof(Model)), "causal model allocation");
    check_cuda(cudaMalloc(&device_products, rows * sizeof(ProductParameters)),
               "causal product allocation");
    check_cuda(cudaMalloc(&device_prices, rows * sizeof(float)), "causal price allocation");
    check_cuda(cudaMalloc(&device_errors, rows * sizeof(float)), "causal error allocation");
    check_cuda(cudaMalloc(&device_payoffs, rows * paths * sizeof(float)),
               "direct payoff allocation");
    check_cuda(cudaMalloc(&workspace, plan.workspace_bytes), "causal workspace allocation");
    check_cuda(cudaMemcpy(device_models, models.data(), rows * sizeof(Model),
                          cudaMemcpyHostToDevice), "causal model upload");
    check_cuda(cudaMemcpy(device_products, products.data(),
                          rows * sizeof(ProductParameters),
                          cudaMemcpyHostToDevice), "causal product upload");
    auto launch = [&] {
        if constexpr (std::is_same_v<
            Path, model::equity::rough_heston::CausalFftPathPolicy
        >) {
            model::equity::rough_heston::launch_rough_heston_causal_fft_cuda<
                Product, Schedule
            >(
                device_models, rows, device_products, rows,
                PriceConstruction::Aligned, rows, 0U, rows, paths,
                {1.0f / 252.0f, 1.0f / 252.0f}, steps,
                32U, 256U, rows, workspace, plan.workspace_bytes,
                seed, device_prices, device_errors
            );
        } else {
            model::equity::quadratic_rough_heston::
                launch_quadratic_rough_heston_causal_fft_cuda<
                    Product, Schedule
                >(
                    device_models, rows, device_products, rows,
                    PriceConstruction::Aligned, rows, 0U, rows, paths,
                    {1.0f / 252.0f, 1.0f / 252.0f}, steps,
                    32U, 256U, rows, workspace, plan.workspace_bytes,
                    seed, device_prices, device_errors
                );
        }
    };
    launch();
    check_cuda(cudaDeviceSynchronize(), "causal pricing synchronize");
    std::vector<float> prices(rows), errors(rows), payoffs(rows * paths);
    check_cuda(cudaMemcpy(prices.data(), device_prices, rows * sizeof(float),
                          cudaMemcpyDeviceToHost), "causal price download");
    check_cuda(cudaMemcpy(errors.data(), device_errors, rows * sizeof(float),
                          cudaMemcpyDeviceToHost), "causal error download");
    const auto* bytes = static_cast<const unsigned char*>(workspace);
    const auto* prepared_rows = reinterpret_cast<const causal::PreparedRow<
        Path, Product, Schedule>*>(bytes + plan.row_offset);
    const auto* weights = reinterpret_cast<const float*>(
        bytes + plan.weight_offset
    );
    direct_reference<Steps, Path, Product, Schedule><<<
        dim3(static_cast<unsigned>((paths + 255U) / 256U), rows), 256U
    >>>(
        prepared_rows, weights, device_payoffs
    );
    check_cuda(cudaGetLastError(), "direct reference launch");
    check_cuda(cudaMemcpy(payoffs.data(), device_payoffs,
                          rows * paths * sizeof(float), cudaMemcpyDeviceToHost),
               "direct payoff download");
    for (std::size_t row = 0U; row < rows; ++row) {
        double sum = 0.0, sq = 0.0;
        for (std::size_t path = 0U; path < paths; ++path) {
            const double payoff = payoffs[row * paths + path];
            sum += payoff;
            sq += payoff * payoff;
        }
        const double reference = sum / paths;
        const double reference_error = std::sqrt(
            std::max(0.0, sq - paths * reference * reference)
                / (paths * (paths - 1U))
        );
        std::printf("CAUSAL_FFT,%s,row=%zu,fft=%.9f,direct=%.9f,se=%.9f\n",
                    label, row, prices[row], reference, errors[row]);
        require(std::isfinite(prices[row]) && std::isfinite(errors[row])
                && errors[row] > 0.0f, "invalid causal FFT moments");
        require(std::abs(prices[row] - reference) < 1.0e-3,
                "causal FFT disagrees with direct convolution");
        require(std::abs(errors[row] - reference_error) < 1.0e-3,
                "causal FFT standard error disagrees with direct convolution");
    }
    launch();
    check_cuda(cudaDeviceSynchronize(), "causal replay synchronize");
    std::vector<float> replay(rows);
    check_cuda(cudaMemcpy(replay.data(), device_prices, rows * sizeof(float),
                          cudaMemcpyDeviceToHost), "causal replay download");
    require(replay == prices, "causal FFT replay is not deterministic");
    cudaFree(device_models);
    cudaFree(device_products);
    cudaFree(device_prices);
    cudaFree(device_errors);
    cudaFree(device_payoffs);
    cudaFree(workspace);
}
}  // namespace

int main() {
    using Call = product::EuropeanOptionPathPolicy<OptionSide::call>;
    using Asian = product::AsianOptionPathPolicy<OptionSide::call>;
    using Barrier = product::UpAndOutOptionPathPolicy<OptionSide::call>;
    namespace rh = model::equity::rough_heston;
    namespace qrh = model::equity::quadratic_rough_heston;
    const std::vector<rh::ModelParameters> heston{
        {1.0f, .02f, .01f, .04f, .30f, .02f, .20f, .20f, -.70f},
        {1.1f, .01f, .00f, .05f, .40f, .02f, .15f, .28f, -.50f},
    };
    const std::vector<qrh::ModelParameters> quadratic{
        {1.0f, .02f, .01f, .10f, .384f, .095f, .0025f, 1.2f, .8f, .20f},
        {1.1f, .01f, .00f, .12f, .35f, .09f, .003f, 1.0f, .7f, .28f},
    };
    const std::vector<product::EuropeanOptionParameters> calls{
        {1.0f, 128U}, {1.05f, 128U}
    };
    const std::vector<product::AsianOptionParameters> asians{
        {1.0f, 128U}, {1.05f, 128U}
    };
    const std::vector<product::UpAndOutOptionParameters> barriers{
        {1.0f, 1.08f, 128U}, {1.05f, 1.16f, 128U}
    };
    exercise<128U, rh::CausalFftPathPolicy, Call, volterra::TerminalHybridSchedule>(
        heston, calls, "rough_heston_call"
    );
    exercise<128U, rh::CausalFftPathPolicy, Asian, volterra::DenseHybridSchedule>(
        heston, asians, "rough_heston_asian"
    );
    exercise<128U, rh::CausalFftPathPolicy, Barrier,
             volterra::DenseHybridSchedule>(
        heston, barriers, "rough_heston_up_and_out"
    );
    exercise<128U, qrh::CausalFftPathPolicy, Call, volterra::TerminalHybridSchedule>(
        quadratic, calls, "quadratic_rough_heston_call"
    );
    exercise<128U, qrh::CausalFftPathPolicy, Asian, volterra::DenseHybridSchedule>(
        quadratic, asians, "quadratic_rough_heston_asian"
    );
    const std::vector<product::EuropeanOptionParameters> short_calls{
        {1.0f, 95U}, {1.05f, 95U}
    };
    exercise<95U, rh::CausalFftPathPolicy, Call,
             volterra::TerminalHybridSchedule>(
        heston, short_calls, "rough_heston_call_95_steps"
    );
    const std::vector<product::AsianOptionParameters> long_asians{
        {1.0f, 130U}, {1.05f, 130U}
    };
    exercise<130U, qrh::CausalFftPathPolicy, Asian,
             volterra::DenseHybridSchedule>(
        quadratic, long_asians, "quadratic_rough_heston_asian_130_steps"
    );
    const std::vector<product::EuropeanOptionParameters> annual_calls{
        {1.0f, 252U}, {1.05f, 252U}
    };
    exercise<504U, rh::CausalFftPathPolicy, Call,
             volterra::TerminalHybridSchedule>(
        heston, annual_calls, "rough_heston_call_dt_1_504"
    );
    exercise<504U, qrh::CausalFftPathPolicy, Call,
             volterra::TerminalHybridSchedule>(
        quadratic, annual_calls, "quadratic_rough_heston_call_dt_1_504"
    );
}

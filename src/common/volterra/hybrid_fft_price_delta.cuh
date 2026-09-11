// Dedicated paired payoff kernels consume the original FFT preparation and convolution.
#pragma once
#include "common/equity/price_delta/path_product_observers.cuh"
#include "common/equity/price_delta/volterra_spot_paths.cuh"
#include "common/volterra/hybrid_fft_price_delta_workspace.cuh"
#include "common/volterra/hybrid_fft_pricer.cuh"

namespace ai_factory::workbench::volterra::hybrid_fft {
namespace spot_delta = ::ai_factory::workbench::equity::price_delta;

template<typename Row, typename Paths, typename Convolution>
__global__ void evaluate_price_delta_paths_kernel(
    std::size_t offset, std::size_t count, const Row* prepared,
    const typename Row::Path::Parameters* model,
    const typename Row::Product::ProductParameters* product,
    spot_delta::SpotBumpConfiguration configuration, float day_fraction,
    const float* variances, const float2* convolutions,
    PartialMoments* price_partials, PartialMoments* delta_partials) {
    using Product = typename Row::Product;
    using Observers = spot_delta::PairedProductObservers<Product, Paths>;
    using Scenario = typename Observers::ScenarioObserver;
    static_assert(sizeof(Scenario) <= simulation::kMaximumObservationHandlerBytes);
    struct BumpedRow {
        typename Paths::Dynamics::PreparedModel dynamics;
        typename Paths::Prepared paths;
        typename Product::PreparedProduct products[3];
        float width;
    };
    static_assert(sizeof(Row) + sizeof(BumpedRow) <= 2048U,
                  "FFT paired prepared rows exceed the shared budget; use a compact view.");
    __shared__ Row row;
    __shared__ BumpedRow bumped;
    if (threadIdx.x == 0U) {
        row = *prepared;
        const auto endpoints = spot_delta::prepare_spot_bump(model->spot, configuration);
        auto lower = *model, upper = *model;
        lower.spot = endpoints.lower;
        upper.spot = endpoints.upper;
        const equity::ProductPreparationContext context{day_fraction, row.schedule.maturity_years};
        bumped = {Paths::prepare_dynamics(*model, row.model, row.schedule.time_step, endpoints),
            Paths::prepare(endpoints),
            {row.product, Product::prepare_product(lower, *product, context),
                          Product::prepare_product(upper, *product, context)}, endpoints.width};
    }
    __syncthreads();
    const std::size_t local = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    double price = 0, price_square = 0, delta = 0, delta_square = 0;
    if (local < count) {
        Scenario scenarios[3]{{Product::make_handler(bumped.products[0])},
                              {Product::make_handler(bumped.products[1])},
                              {Product::make_handler(bumped.products[2])}};
        typename Observers::Observer observer{bumped.paths, scenarios};
        const auto convolution = [&] {
            if constexpr (std::is_same_v<Convolution, FftPathConvolution>)
                return FftPathConvolution{convolutions, local / 2U, (local & 1U) != 0U};
            else return DirectPathConvolution{};
        }();
        simulate_observed_path<typename Paths::Dynamics>(row, bumped.dynamics,
            offset + local, variances, convolution, observer);
        float payoffs[3];
        #pragma unroll
        for (unsigned i=0; i<3U; ++i)
            payoffs[i] = Product::template finalize<spot_delta::SpotObservationPolicy>(
                bumped.products[i], scenarios[i].terminal, scenarios[i].handler);
        price = static_cast<double>(payoffs[0]);
        price_square = price * price;
        // The paired subtraction and division remain FP32.
        delta = static_cast<double>((payoffs[2] - payoffs[1]) / bumped.width);
        delta_square = delta * delta;
    }
    const auto central = reductions::reduce_block(price, price_square);
    const auto index = offset / tuning::kPricingPathThreads + blockIdx.x;
    if (threadIdx.x == 0U) price_partials[index] = {central.sum, central.sumsq};
    __syncthreads();
    const auto paired = reductions::reduce_block(delta, delta_square);
    if (threadIdx.x == 0U) delta_partials[index] = {paired.sum, paired.sumsq};
}

template<typename ModelPath, typename Product, typename Paths>
struct PriceDeltaPathConsumer {
    const typename ModelPath::Parameters* model;
    const typename Product::ProductParameters* product;
    spot_delta::SpotBumpConfiguration bump;
    float day_fraction;
    PartialMoments* delta_partials;
    float* deltas;
    float* errors;

    template<typename Convolution, typename Row>
    static void validate_resources() {
        constexpr std::size_t shared = 2U * (tuning::kPricingPathThreads / 32U) * sizeof(double);
        int active = 0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
            evaluate_price_delta_paths_kernel<Row, Paths, Convolution>,
            tuning::kPricingPathThreads, shared), "FFT price-delta occupancy");
        if (active == 0) throw std::invalid_argument("FFT paired path kernel has no resident block.");
    }

    template<typename Convolution, typename Row>
    void submit(std::size_t offset, std::size_t count, const Row* row,
        const float* variances, const float2* convolutions, PartialMoments* partials,
        const char* name, const char* variant) const {
        constexpr auto kernel = evaluate_price_delta_paths_kernel<Row, Paths, Convolution>;
        const dim3 grid(static_cast<unsigned>(hybrid_fft_partial_moment_count(count)));
        constexpr std::size_t shared = 2U * (tuning::kPricingPathThreads / 32U) * sizeof(double);
        report_cuda_kernel_phase_launch_if_enabled(name, variant, "paired_path_evaluation", kernel,
            grid, dim3(tuning::kPricingPathThreads), shared);
        evaluate_price_delta_paths_kernel<Row, Paths, Convolution><<<grid, tuning::kPricingPathThreads, shared>>>(
            offset, count, row, model, product, bump, day_fraction, variances, convolutions,
            partials, delta_partials);
        check_cuda(cudaGetLastError(), "Volterra paired path evaluation");
    }
    template<typename Row>
    void evaluate(std::size_t offset, std::size_t count, const Row* row,
        const float* variances, const float2* convolutions, PartialMoments* partials,
        const char* name, const char* variant) const {
        submit<FftPathConvolution>(offset, count, row, variances, convolutions, partials, name, variant);
    }
#if AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT > 0
    template<typename Row>
    void evaluate_direct(std::size_t offset, std::size_t count, const Row* row,
        const float* variances, PartialMoments* partials,
        const char* name, const char* variant) const {
        submit<DirectPathConvolution>(offset, count, row, variances, nullptr, partials, name, variant);
    }
#endif
    void finish(const PartialMoments* partials, std::size_t paths, std::size_t result,
        float* prices, float* price_errors, const char* name, const char* variant) const {
        PricePathConsumer{}.finish(partials, paths, result, prices, price_errors, name, variant);
        // Reuse precisely the scalar finalization order for the paired moments.
        PricePathConsumer{}.finish(delta_partials, paths, result, deltas, errors, name, "delta");
    }
};

template<typename Kernel, typename ModelPath, typename Product, typename Schedule, typename Paths>
void launch_price_delta_cuda(
    const typename ModelPath::Parameters* host_models,
    const typename ModelPath::Parameters* device_models, std::size_t model_count,
    const typename Product::ProductParameters* host_products,
    const typename Product::ProductParameters* device_products, std::size_t product_count,
    PriceConstruction construction, std::size_t result_count, std::size_t result_index,
    std::size_t paths, HybridTimeConfiguration time, std::size_t steps, std::size_t chunk,
    void* workspace, std::size_t workspace_bytes, std::uint64_t seed,
    spot_delta::SpotBumpConfiguration bump,
    float* prices, float* price_errors, float* deltas, float* delta_errors,
    const char* name, const char* variant) {
    if (!host_models || !host_products) throw std::invalid_argument("FFT delta requires host input mirrors.");
    validate_model_product_construction(model_count, product_count, construction, result_count);
    if (result_index >= result_count) throw std::invalid_argument("FFT delta result is outside the batch.");
    const auto indices = decode_model_product_result_index(result_index, product_count, construction);
    spot_delta::validate_spot_bump(host_models[indices.model_index].spot, bump);
    const auto calendar = Product::calendar(host_products[indices.product_index]);
    simulation::validate_calendar(calendar);
    validate_time_configuration(time);
    if (Schedule::execution_step_count(calendar, time) != steps)
        throw std::invalid_argument("FFT delta step count contradicts its calendar/time grid.");
    validate_device_pointer(deltas, "device_deltas");
    validate_device_pointer(delta_errors, "device_delta_errors");
    const auto required = required_hybrid_fft_price_delta_workspace_bytes(steps, paths, chunk);
    if (workspace_bytes < required) throw std::invalid_argument("FFT delta workspace is too small.");
    validate_device_pointer(workspace, "device_workspace");
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    const auto base_bytes = required_hybrid_fft_workspace_bytes(steps, paths, chunk);
    auto* delta_partials = reinterpret_cast<PartialMoments*>(static_cast<unsigned char*>(workspace) + base_bytes);
    const PriceDeltaPathConsumer<ModelPath, Product, Paths> consumer{
        device_models + indices.model_index, device_products + indices.product_index,
        bump, time.day_fraction, delta_partials, deltas, delta_errors};
    using Row = PreparedRow<Kernel, ModelPath, Product, Schedule>;
#if AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT > 0
    if (steps <= AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT)
        consumer.template validate_resources<DirectPathConvolution, Row>();
    else
#endif
        consumer.template validate_resources<FftPathConvolution, Row>();
    launch_path_pipeline_cuda<Kernel, ModelPath, Product, Schedule>(
        device_models, model_count, device_products, product_count, construction, result_count,
        result_index, paths, time, steps, chunk, workspace, workspace_bytes, seed,
        prices, price_errors, name, variant, consumer);
}

}  // namespace ai_factory::workbench::volterra::hybrid_fft

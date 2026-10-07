// Path-product evaluation on a cached Gaussian-Volterra convolution.
#pragma once

#include "common/volterra/hybrid_path_simulation.cuh"
#include "common/volterra/pricing_row.cuh"
#include "common/volterra/price_gradients/path_node_evaluation.cuh"

namespace ai_factory::workbench::volterra::price_gradients {

template<typename KernelPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename HybridSchedule>
struct GaussianPathEvaluation {
    using NodePolicy = RoughPathPayoffNodePolicy<
        ModelPathPolicy, ProductPolicy, HybridSchedule
    >;
    static constexpr bool kReuseCentralValue = false;
    static constexpr bool kBatchSharedPath = true;
    using Metadata = typename NodePolicy::Metadata;

    template<typename TerminalRow, typename Scenario>
    __device__ static void evaluate_batch(
        const TerminalRow& terminal_row,
        const typename ModelPathPolicy::PreparedModel& model,
        const Scenario* scenarios,
        const Metadata* metadata,
        const std::uint16_t* indices,
        std::size_t count,
        ::ai_factory::workbench::price_gradients::TimeConfiguration time,
        std::size_t path,
        const float* variances,
        hybrid_fft::FftPathConvolution convolution,
        float* values
    ) {
        RoughPathPayoffBatch<
            ModelPathPolicy, ProductPolicy, HybridSchedule
        > batch{};
        for (std::size_t i = 0U; i < count; ++i)
            batch.add(metadata[indices[i]]);
        const auto& schedule = metadata[indices[0U]].schedule;
        PricingRow<
            KernelPolicy, ModelPathPolicy, ProductPolicy,
            HybridSchedule
        > row{
            terminal_row.kernel, model, batch.slots[0U].product,
            schedule, terminal_row.key, terminal_row.sqrt_time_step
        };
        const auto state = hybrid_fft::simulate_observed_path<
            ModelPathPolicy
        >(row, model, path, variances, convolution, batch);
        batch.finalize(state, values);
    }

    template<typename TerminalRow, typename Scenario>
    __device__ static float evaluate(
        const TerminalRow& terminal_row,
        const typename ModelPathPolicy::PreparedModel& model,
        const Scenario& scenario,
        const Metadata& metadata,
        ::ai_factory::workbench::price_gradients::TimeConfiguration time,
        std::size_t path,
        const float* variances,
        hybrid_fft::FftPathConvolution convolution
    ) {
        static_assert(ModelPathPolicy::kNativeLogSpot);
        const auto& schedule = metadata.schedule;
        PricingRow<
            KernelPolicy, ModelPathPolicy, ProductPolicy,
            HybridSchedule
        > row{
            terminal_row.kernel, model, metadata.product, schedule,
            terminal_row.key, terminal_row.sqrt_time_step
        };
        auto handler = ProductPolicy::make_handler(metadata.product);
        ScaledRoughPathObserver<
            ModelPathPolicy, decltype(handler),
            ProductPolicy::kObservationCoordinate
        > observer{
            handler, metadata.spot_scale, metadata.log_scale
        };
        auto state = hybrid_fft::simulate_observed_path<ModelPathPolicy>(
            row, model, path, variances, convolution, observer
        );
        state.log_spot += metadata.log_scale;
        return ProductPolicy::template finalize<ModelPathPolicy>(
            metadata.product, state, handler
        );
    }
};

}  // namespace ai_factory::workbench::volterra::price_gradients

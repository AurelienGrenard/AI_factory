// CUDA adapter for the two-channel price/standard-error checkpoint contract.
#pragma once

#include "tools/cuda/generation_checkpoint.hpp"
#include "tools/cuda/pricing_runner.cuh"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ai_factory::workbench::offline::cuda {

class MonteCarloGenerationCheckpoint {
public:
    explicit MonteCarloGenerationCheckpoint(std::size_t result_count)
        : storage_(result_count, {"price", "standard_error"}) {
        if (storage_.enabled()) {
            kernel_timer_ = std::make_unique<KernelDurationAccumulator>();
        }
    }

    bool enabled() const noexcept { return storage_.enabled(); }
    std::size_t completed_prices() const noexcept {
        return storage_.completed_prices();
    }
    std::size_t resumed_prices() const noexcept {
        return storage_.resumed_prices();
    }

    void start_batch() {
        if (kernel_timer_) kernel_timer_->start_batch();
    }

    template<class Execution>
    void commit(Execution& execution, std::size_t offset, std::size_t count) {
        if (!enabled()) return;
        kernel_timer_->finish_batch();
        prices_.resize(count);
        standard_errors_.resize(count);
        execution.copy_prices_range_to(prices_.data(), offset, count);
        execution.copy_standard_errors_range_to(
            standard_errors_.data(), offset, count
        );
        storage_.commit(
            offset,
            count,
            {
                std::span<const float>(prices_),
                std::span<const float>(standard_errors_),
            }
        );
    }

    void restore(MonteCarloRun& run) const {
        storage_.restore_prefix({
            std::span<float>(run.prices),
            std::span<float>(run.standard_errors),
        });
    }

    void restore_kernel_seconds(MonteCarloRun& run) const noexcept {
        if (kernel_timer_) run.kernel_seconds = kernel_timer_->seconds();
    }

private:
    GenerationCheckpoint storage_;
    std::vector<float> prices_;
    std::vector<float> standard_errors_;
    std::unique_ptr<KernelDurationAccumulator> kernel_timer_;
};

}  // namespace ai_factory::workbench::offline::cuda

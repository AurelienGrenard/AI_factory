// CUDA resource ownership and execution metrics shared by LSM launchers.
#pragma once

#include "common/longstaff_schwartz/regression_status.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <string>

namespace ai_factory::workbench::longstaff_schwartz {

struct RegressionDiagnosticSummary {
    std::size_t successful_regression_count = 0U;
    std::size_t no_candidate_count = 0U;
    std::size_t insufficient_candidate_count = 0U;
    std::size_t non_finite_statistics_count = 0U;
    std::size_t factorization_failure_count = 0U;
    std::size_t non_finite_coefficient_count = 0U;
    std::size_t affected_result_count = 0U;
    std::size_t first_fatal_result_index =
        std::numeric_limits<std::size_t>::max();
    RegressionStatus first_fatal_status = RegressionStatus::success;

    bool has_fatal_failure() const noexcept {
        return affected_result_count != 0U;
    }
};

struct LaunchResult {
    double kernel_seconds = 0.0;
    std::size_t batch_count = 0U;
    std::size_t kernel_launch_count = 0U;
    std::size_t maximum_prices_per_batch = 0U;
    std::size_t blocks_per_price = 0U;
    std::size_t workspace_bytes = 0U;
    RegressionDiagnosticSummary regression_diagnostics{};
};

std::string regression_diagnostic_message(
    const RegressionDiagnosticSummary& diagnostics,
    const char* product_name
);

void validate_regression_diagnostics(
    const LaunchResult& result,
    const char* product_name
);

struct WorkspaceBudget {
    std::size_t free_bytes;
    std::size_t total_bytes;
    std::size_t safety_margin;
    std::size_t available_bytes;
};

WorkspaceBudget query_workspace_budget(const char* product_name);

class LaunchResources {
public:
    LaunchResources(std::size_t workspace_bytes, const char* product_name);
    ~LaunchResources();

    LaunchResources(const LaunchResources&) = delete;
    LaunchResources& operator=(const LaunchResources&) = delete;

    unsigned char* workspace() const noexcept;
    void start_batch();
    double finish_batch();

private:
    unsigned char* workspace_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
};

}  // namespace ai_factory::workbench::longstaff_schwartz

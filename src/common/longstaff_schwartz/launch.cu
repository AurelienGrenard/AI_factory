// CUDA resource ownership and execution timing for LSM launchers.
#include "common/longstaff_schwartz/launch.cuh"

#include "common/check_cuda.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::longstaff_schwartz {
namespace {

constexpr std::size_t kOneGiB = 1ULL << 30U;

std::string operation(const char* action, const char* product_name) {
    return std::string(action) + ' ' + product_name;
}

}  // namespace

std::string regression_diagnostic_message(
    const RegressionDiagnosticSummary& diagnostics,
    const char* product_name
) {
    if (!diagnostics.has_fatal_failure()) return {};
    return std::string(product_name) + " result row "
        + std::to_string(diagnostics.first_fatal_result_index)
        + " has a fatal Longstaff-Schwartz regression status: "
        + regression_status_name(diagnostics.first_fatal_status)
        + " (affected rows="
        + std::to_string(diagnostics.affected_result_count)
        + ", insufficient="
        + std::to_string(diagnostics.insufficient_candidate_count)
        + ", non-finite statistics="
        + std::to_string(diagnostics.non_finite_statistics_count)
        + ", factorization failures="
        + std::to_string(diagnostics.factorization_failure_count)
        + ", non-finite coefficients="
        + std::to_string(diagnostics.non_finite_coefficient_count)
        + ").";
}

void validate_regression_diagnostics(
    const LaunchResult& result,
    const char* product_name
) {
    const std::string message = regression_diagnostic_message(
        result.regression_diagnostics, product_name
    );
    if (!message.empty()) throw std::runtime_error(message);
}

WorkspaceBudget query_workspace_budget(const char* product_name) {
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    const std::string query = operation("cudaMemGetInfo", product_name);
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), query.c_str());
    const std::size_t safety_margin = std::max(kOneGiB, free_bytes / 10U);
    if (free_bytes <= safety_margin) {
        throw std::runtime_error(
            "Insufficient free GPU memory after the safety margin."
        );
    }
    return {
        free_bytes,
        total_bytes,
        safety_margin,
        free_bytes - safety_margin,
    };
}

LaunchResources::LaunchResources(
    std::size_t workspace_bytes,
    const char* product_name
) {
    try {
        const std::string allocate = operation("cudaMalloc", product_name);
        check_cuda(
            cudaMalloc(&workspace_, workspace_bytes), allocate.c_str()
        );
        check_cuda(cudaEventCreate(&start_event_), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event_), "cudaEventCreate stop");
    } catch (...) {
        if (start_event_ != nullptr) cudaEventDestroy(start_event_);
        if (stop_event_ != nullptr) cudaEventDestroy(stop_event_);
        if (workspace_ != nullptr) cudaFree(workspace_);
        throw;
    }
}

LaunchResources::~LaunchResources() {
    if (start_event_ != nullptr) cudaEventDestroy(start_event_);
    if (stop_event_ != nullptr) cudaEventDestroy(stop_event_);
    if (workspace_ != nullptr) cudaFree(workspace_);
}

unsigned char* LaunchResources::workspace() const noexcept {
    return workspace_;
}

void LaunchResources::start_batch() {
    check_cuda(cudaEventRecord(start_event_), "cudaEventRecord start");
}

double LaunchResources::finish_batch() {
    check_cuda(cudaEventRecord(stop_event_), "cudaEventRecord stop");
    check_cuda(
        cudaEventSynchronize(stop_event_), "cudaEventSynchronize stop"
    );
    float milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start_event_, stop_event_),
        "cudaEventElapsedTime batch"
    );
    return static_cast<double>(milliseconds) * 1.0e-3;
}

}  // namespace ai_factory::workbench::longstaff_schwartz

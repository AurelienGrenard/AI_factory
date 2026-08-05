// CUDA resource ownership and execution metrics shared by LSM launchers.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz {

struct LaunchResult {
    double kernel_seconds;
    std::size_t batch_count;
    std::size_t kernel_launch_count;
    std::size_t maximum_prices_per_batch;
    std::size_t blocks_per_price;
    std::size_t workspace_bytes;
};

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

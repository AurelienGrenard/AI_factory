#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ai_factory::cuda::detail {

constexpr int kThreadsPerBlock = 256;

inline void check_cuda(cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(error)
        );
    }
}

class ReusableDeviceBuffer {
public:
    ReusableDeviceBuffer() = default;
    ReusableDeviceBuffer(const ReusableDeviceBuffer&) = delete;
    ReusableDeviceBuffer& operator=(const ReusableDeviceBuffer&) = delete;

    ~ReusableDeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void* ensure(std::size_t byte_capacity, const char* context) {
        if (byte_capacity <= byte_capacity_) {
            return data_;
        }
        void* replacement = nullptr;
        check_cuda(cudaMalloc(&replacement, byte_capacity), context);
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = replacement;
        byte_capacity_ = byte_capacity;
        return data_;
    }

private:
    void* data_ = nullptr;
    std::size_t byte_capacity_ = 0U;
};

template <std::size_t BufferCount>
class ReusableCudaWorkspace {
public:
    ReusableCudaWorkspace() {
        check_cuda(cudaEventCreate(&start_), "cudaEventCreate reusable start");
        try {
            check_cuda(cudaEventCreate(&stop_), "cudaEventCreate reusable stop");
        } catch (...) {
            cudaEventDestroy(start_);
            start_ = nullptr;
            throw;
        }
    }

    ReusableCudaWorkspace(const ReusableCudaWorkspace&) = delete;
    ReusableCudaWorkspace& operator=(const ReusableCudaWorkspace&) = delete;

    ~ReusableCudaWorkspace() {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    template <typename T>
    T* buffer(std::size_t index, std::size_t count, const char* context) {
        if (index >= BufferCount) {
            throw std::out_of_range("Reusable CUDA buffer index out of range.");
        }
        return static_cast<T*>(buffers_[index].ensure(count * sizeof(T), context));
    }

    cudaEvent_t start_event() const { return start_; }
    cudaEvent_t stop_event() const { return stop_; }

private:
    std::array<ReusableDeviceBuffer, BufferCount> buffers_{};
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

template <typename Tag, std::size_t BufferCount>
ReusableCudaWorkspace<BufferCount>& reusable_cuda_workspace() {
    static thread_local ReusableCudaWorkspace<BufferCount> workspace;
    return workspace;
}

template <typename Row>
struct DeviceWorkspace {
    Row* rows = nullptr;
    MonteCarloOutput* outputs = nullptr;
    std::size_t row_capacity = 0;
    cudaEvent_t start{};
    cudaEvent_t stop{};
};

template <typename Row>
void allocate_workspace(DeviceWorkspace<Row>& workspace, std::size_t row_capacity) {
    const auto row_bytes = row_capacity * sizeof(Row);
    const auto output_bytes = row_capacity * sizeof(MonteCarloOutput);
    check_cuda(cudaMalloc(&workspace.rows, row_bytes), "cudaMalloc workspace rows");
    check_cuda(
        cudaMalloc(&workspace.outputs, output_bytes),
        "cudaMalloc workspace outputs"
    );
    check_cuda(cudaEventCreate(&workspace.start), "cudaEventCreate workspace start");
    check_cuda(cudaEventCreate(&workspace.stop), "cudaEventCreate workspace stop");
    workspace.row_capacity = row_capacity;
}

template <typename Row>
void release_workspace(DeviceWorkspace<Row>& workspace) {
    cudaFree(workspace.rows);
    cudaFree(workspace.outputs);
    if (workspace.start != nullptr) {
        cudaEventDestroy(workspace.start);
    }
    if (workspace.stop != nullptr) {
        cudaEventDestroy(workspace.stop);
    }
    workspace.rows = nullptr;
    workspace.outputs = nullptr;
    workspace.row_capacity = 0;
    workspace.start = nullptr;
    workspace.stop = nullptr;
}

template <typename Row>
void run_kernel_with_workspace(
    DeviceWorkspace<Row>& workspace,
    const Row* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing,
    void (*kernel)(
        const Row*,
        std::size_t,
        std::size_t,
        std::size_t,
        MonteCarloOutput*
    )
) {
    if (row_count > workspace.row_capacity) {
        throw std::runtime_error("CUDA workspace row capacity exceeded.");
    }
    const auto row_bytes = row_count * sizeof(Row);
    const auto output_bytes = row_count * sizeof(MonteCarloOutput);

    check_cuda(
        cudaMemcpy(workspace.rows, host_rows, row_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy rows"
    );
    check_cuda(cudaEventRecord(workspace.start), "cudaEventRecord start");

    kernel<<<
        static_cast<unsigned int>(row_count),
        kThreadsPerBlock,
        2U * kThreadsPerBlock * sizeof(double)
    >>>(workspace.rows, row_count, num_paths, num_steps, workspace.outputs);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventRecord(workspace.stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(workspace.stop), "cudaEventSynchronize stop");

    float elapsed_ms = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&elapsed_ms, workspace.start, workspace.stop),
        "kernel timing"
    );
    check_cuda(
        cudaMemcpy(
            host_outputs,
            workspace.outputs,
            output_bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy outputs"
    );

    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }

}

template <typename Row>
void run_kernel_common(
    const Row* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing,
    void (*kernel)(
        const Row*,
        std::size_t,
        std::size_t,
        std::size_t,
        MonteCarloOutput*
    )
) {
    DeviceWorkspace<Row> workspace{};
    allocate_workspace(workspace, row_count);
    try {
        run_kernel_with_workspace(
            workspace,
            host_rows,
            row_count,
            num_paths,
            num_steps,
            host_outputs,
            timing,
            kernel
        );
    } catch (...) {
        release_workspace(workspace);
        throw;
    }
    release_workspace(workspace);
}

}  // namespace ai_factory::cuda::detail

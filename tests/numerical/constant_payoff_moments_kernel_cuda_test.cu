// Production-length raw-moment sums must not reject deterministic FP32 payoffs.
#include "common/reductions.cuh"
#include "common/check_cuda.cuh"
#include <array>
#include <cmath>
#include <iostream>

using namespace ai_factory::workbench;

struct Result { double price, error, centered, scale; };

__global__ void constant_moments(std::size_t paths, Result* results) {
    const float value = expf(-.028979286551475525f * (21.f / 252.f))
        * (1.f + .09619792550802231f * (21.f / 252.f));
    double sum = 0, sumsq = 0;
    for (std::size_t path = threadIdx.x; path < paths; path += blockDim.x) {
        sum += static_cast<double>(value);
        sumsq += static_cast<double>(value) * static_cast<double>(value);
    }
    const auto total = reductions::reduce_block(sum, sumsq);
    if (!threadIdx.x) {
        auto& result = results[0];
        reductions::compute_statistics(total, paths, result.price, result.error,
            paths / blockDim.x + (paths % blockDim.x != 0U));
        const double mean = total.sum / static_cast<double>(paths);
        result.centered = total.sumsq - static_cast<double>(paths) * mean * mean;
        result.scale = total.sumsq;
        // Materially inconsistent moments must still fail closed.
        double invalid_price, invalid_error;
        reductions::compute_statistics({1024.0, 1.0}, 1024, invalid_price, invalid_error, 4096);
        if (isfinite(invalid_price) || isfinite(invalid_error)) result.price = nan("");
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    Result* device = nullptr;
    try {
        check_cuda(cudaMalloc(&device, sizeof(Result)), "allocate constant moments");
        bool valid = true;
        for (unsigned threads : {128U, 256U, 512U}) {
            constant_moments<<<1, threads, 2 * (threads / 32) * sizeof(double)>>>(1048576, device);
            Result value{};
            check_cuda(cudaMemcpy(&value, device, sizeof(value), cudaMemcpyDeviceToHost), "constant moments");
            std::cout << threads << " threads: price=" << value.price << " error=" << value.error
                      << " centered/scale=" << value.centered / value.scale << '\n';
            valid = valid && std::isfinite(value.price) && std::isfinite(value.error) && value.error <= 1.e-8;
        }
        cudaFree(device);
        return valid ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        cudaFree(device);
        return 1;
    }
}

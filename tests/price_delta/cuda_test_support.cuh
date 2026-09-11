// Small RAII device fixtures and fail-fast assertions shared by price-delta tests.
#pragma once

#include "common/check_cuda.cuh"
#include <cstddef>
#include <stdexcept>

namespace price_delta_test {

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename T>
struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(std::size_t count) {
        ai_factory::workbench::check_cuda(cudaMalloc(&data, count * sizeof(T)), "allocate fixture");
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

}  // namespace price_delta_test

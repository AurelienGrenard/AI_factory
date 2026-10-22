// Block-cooperative padded linear convolution built on a caller-selected FFT.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench::volterra {

// Loader(index) supplies one packed complex input and Store(index, value)
// consumes the inverse result. Length must already include the zero padding
// required to turn the circular FFT result into the requested linear one.
template<unsigned int Length, class Forward, class Inverse,
         class Loader, class Store>
__device__ __forceinline__ void execute_padded_linear_convolution(
    const float2* kernel_spectrum,
    Loader&& load,
    Store&& store,
    typename Forward::value_type* shared_values
) {
    using Complex = typename Forward::value_type;
    Complex thread_data[Forward::storage_size];

    if (Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::input_ept; ++item) {
            const unsigned int index =
                item * Forward::stride + threadIdx.x;
            reinterpret_cast<float2*>(thread_data)[item] = load(index);
        }
    }

    Forward().execute(thread_data, shared_values);
    constexpr float inverse_length = 1.0f / static_cast<float>(Length);
    if (Forward::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Forward::output_ept; ++item) {
            const unsigned int frequency =
                item * Forward::stride + threadIdx.x;
            if (frequency < Length) {
                const float2 value =
                    reinterpret_cast<float2*>(thread_data)[item];
                const float2 kernel = kernel_spectrum[frequency];
                reinterpret_cast<float2*>(thread_data)[item] = {
                    (value.x * kernel.x - value.y * kernel.y)
                        * inverse_length,
                    (value.x * kernel.y + value.y * kernel.x)
                        * inverse_length,
                };
            }
        }
    }
    Inverse().execute(thread_data, shared_values);

    // The store is allowed to reuse the FFT scratch itself. Ensure every
    // working group has stopped reading that storage before the first output
    // element overwrites it.
    __syncthreads();

    if (Inverse::working_group::is_thread_active()) {
        #pragma unroll
        for (unsigned int item = 0U; item < Inverse::output_ept; ++item) {
            const unsigned int index =
                item * Inverse::stride + threadIdx.x;
            store(index, reinterpret_cast<float2*>(thread_data)[item]);
        }
    }
}

}  // namespace ai_factory::workbench::volterra

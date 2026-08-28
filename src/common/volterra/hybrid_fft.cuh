// cuFFTDx types and deterministic Gaussian access shared by Volterra engines.
#pragma once

#include "common/philox.cuh"

#include <cufftdx.hpp>
#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::volterra::hybrid_fft {

#ifndef AI_FACTORY_CUFFTDX_ARCHITECTURE
#error "AI_FACTORY_CUFFTDX_ARCHITECTURE must name the reviewed cuFFTDx SM profile"
#endif

// Random access to the exact normal produced by UniformSequence and
// NormalPairCache at one scalar index. Cooperative FFT lanes can therefore
// generate Brownian cells independently while preserving the canonical
// (key, path, local_group) Philox mapping.
__device__ __forceinline__ float normal_at(
    philox::PhiloxKey key,
    std::uint64_t path,
    std::uint64_t normal_index
) {
    const std::uint64_t pair_index = normal_index >> 1U;
    const philox::RandomQuad uniforms = philox::uniform_quad(
        key,
        path,
        pair_index >> 1U
    );
    const bool upper_pair = (pair_index & 1U) != 0U;
    const philox::NormalPair normals = philox::box_muller(
        upper_pair ? uniforms.third : uniforms.first,
        upper_pair ? uniforms.fourth : uniforms.second
    );
    return (normal_index & 1U) == 0U ? normals.first : normals.second;
}

template<unsigned int Length, unsigned int ElementsPerThread,
         unsigned int FftsPerBlock>
struct FftTypes {
    using Base = decltype(
        cufftdx::Block()
        + cufftdx::Size<Length>()
        + cufftdx::Type<cufftdx::fft_type::c2c>()
        + cufftdx::Precision<float>()
        + cufftdx::SM<AI_FACTORY_CUFFTDX_ARCHITECTURE>()
        + cufftdx::ElementsPerThread<ElementsPerThread>()
        + cufftdx::FFTsPerBlock<FftsPerBlock>()
    );
    using Forward = decltype(
        Base() + cufftdx::Direction<cufftdx::fft_direction::forward>()
    );
    using Inverse = decltype(
        Base() + cufftdx::Direction<cufftdx::fft_direction::inverse>()
    );
};

}  // namespace ai_factory::workbench::volterra::hybrid_fft

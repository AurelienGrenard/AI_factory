// Central compile-time geometry profile for Volterra FFT pricing and sampling.
#pragma once

#include <cstdint>
#include <stdexcept>

#ifndef AI_FACTORY_CUDA_VOLTERRA_PRICING_PATH_THREADS
#define AI_FACTORY_CUDA_VOLTERRA_PRICING_PATH_THREADS 256
#endif
#ifndef AI_FACTORY_CUDA_VOLTERRA_PRICING_FINALIZATION_THREADS
#define AI_FACTORY_CUDA_VOLTERRA_PRICING_FINALIZATION_THREADS 256
#endif

namespace ai_factory::workbench::volterra::tuning {

inline constexpr unsigned int kPricingPathThreads =
    AI_FACTORY_CUDA_VOLTERRA_PRICING_PATH_THREADS;
inline constexpr unsigned int kPricingFinalizationThreads =
    AI_FACTORY_CUDA_VOLTERRA_PRICING_FINALIZATION_THREADS;

template<
    unsigned int Length,
    unsigned int PricingElementsPerThread,
    unsigned int PricingFftsPerBlock,
    unsigned int SamplingElementsPerThread,
    unsigned int SamplingFftsPerBlock
>
struct HybridFftSpecialization {
    static constexpr unsigned int kLength = Length;
    static constexpr unsigned int kPricingElementsPerThread =
        PricingElementsPerThread;
    static constexpr unsigned int kPricingFftsPerBlock = PricingFftsPerBlock;
    static constexpr unsigned int kSamplingElementsPerThread =
        SamplingElementsPerThread;
    static constexpr unsigned int kSamplingFftsPerBlock = SamplingFftsPerBlock;
};

// These are the measured SM89 reference choices. Pricing and sampling fields
// remain independent even where their current values coincide, so a future
// architecture profile can tune either consumer without changing its engine.
template<unsigned int MaximumFftLength, typename Callback>
void dispatch_hybrid_fft_specialization(
    std::uint32_t step_count,
    Callback&& callback
) {
    if (step_count <= 8U) {
        callback.template operator()<
            HybridFftSpecialization<16U, 8U, 16U, 8U, 16U>
        >();
    } else if (step_count <= 32U) {
        callback.template operator()<
            HybridFftSpecialization<64U, 8U, 8U, 8U, 8U>
        >();
    } else if (step_count <= 64U) {
        callback.template operator()<
            HybridFftSpecialization<128U, 8U, 8U, 8U, 8U>
        >();
    } else if (step_count <= 128U) {
        callback.template operator()<
            HybridFftSpecialization<256U, 16U, 8U, 16U, 8U>
        >();
    } else if (step_count <= 256U) {
        callback.template operator()<
            HybridFftSpecialization<512U, 8U, 2U, 8U, 2U>
        >();
    } else if (step_count <= 512U) {
        callback.template operator()<
            HybridFftSpecialization<1024U, 16U, 1U, 16U, 1U>
        >();
    } else if (step_count <= 1024U) {
        callback.template operator()<
            HybridFftSpecialization<2048U, 16U, 1U, 16U, 1U>
        >();
    } else if constexpr (MaximumFftLength >= 4096U) {
        if (step_count <= 2048U) {
            callback.template operator()<
                HybridFftSpecialization<4096U, 16U, 1U, 16U, 1U>
            >();
        } else if constexpr (MaximumFftLength >= 8192U) {
            if (step_count <= 4096U) {
                callback.template operator()<
                    HybridFftSpecialization<8192U, 16U, 1U, 16U, 1U>
                >();
            } else {
                throw std::invalid_argument(
                    "Volterra step count exceeds the tuning profile."
                );
            }
        } else {
            throw std::invalid_argument(
                "Volterra step count exceeds the tuning profile."
            );
        }
    } else {
        throw std::invalid_argument(
            "Volterra step count exceeds the tuning profile."
        );
    }
}

static_assert(kPricingPathThreads >= 32U);
static_assert(kPricingPathThreads <= 1024U);
static_assert(kPricingPathThreads % 32U == 0U);
static_assert(kPricingFinalizationThreads >= 32U);
static_assert(kPricingFinalizationThreads <= 1024U);
static_assert(kPricingFinalizationThreads % 32U == 0U);

}  // namespace ai_factory::workbench::volterra::tuning

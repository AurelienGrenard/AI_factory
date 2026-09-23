// Nelson-Siegel curve coordinates for compact device preparation.
#pragma once

#include "curve/nelson_siegel/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::curve::nelson_siegel::price_gradients {

struct DevicePreparation {
    using Curve = NelsonSiegelParameters;
    static constexpr std::array parameter_names{
        std::string_view{"curve.beta0"},
        std::string_view{"curve.beta1"},
        std::string_view{"curve.beta2"},
        std::string_view{"curve.tau"},
    };

    __host__ __device__ static bool valid(const Curve& curve) {
        return valid_parameters(curve);
    }

    __host__ __device__ static float read(std::uint8_t index, const Curve& curve) {
        switch (index) {
        case 0U: return curve.beta0;
        case 1U: return curve.beta1;
        case 2U: return curve.beta2;
        case 3U: return curve.tau;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index, Curve& curve, float value
    ) {
        switch (index) {
        case 0U: curve.beta0 = value; break;
        case 1U: curve.beta1 = value; break;
        case 2U: curve.beta2 = value; break;
        case 3U: curve.tau = value; break;
        }
    }
};

}  // namespace ai_factory::workbench::curve::nelson_siegel::price_gradients

// Flat-curve coordinate for compact device sensitivity preparation.
#pragma once

#include "curve/flat/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::curve::flat::price_gradients {

struct DevicePreparation {
    using Curve = FlatCurveParameters;
    static constexpr std::array parameter_names{
        std::string_view{"curve.rate"},
    };

    __host__ __device__ static bool valid(const Curve& curve) {
        return valid_parameters(curve);
    }

    __host__ __device__ static float read(std::uint8_t index, const Curve& curve) {
        return index == 0U ? curve.rate : ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index, Curve& curve, float value
    ) {
        if (index == 0U) curve.rate = value;
    }
};

}  // namespace ai_factory::workbench::curve::flat::price_gradients

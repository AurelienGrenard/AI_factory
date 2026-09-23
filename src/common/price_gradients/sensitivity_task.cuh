// Row-local nodes and central-path requirement for one selected coordinate.
#pragma once

#include "common/price_gradients/central_requirement.hpp"
#include "common/price_gradients/configuration.hpp"
#include "common/price_gradients/sensitivity_stencil.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::price_gradients {

template<typename Parameter>
struct SensitivitySpec {
    Parameter parameter;
    BumpConfiguration bump;
};

template<typename Node, std::size_t NodeCapacity>
struct SensitivityNodes {
    Node values[NodeCapacity]{};

    __host__ __device__ Node& operator[](std::size_t index) {
        return values[index];
    }

    __host__ __device__ const Node& operator[](std::size_t index) const {
        return values[index];
    }
};

template<std::size_t NodeCapacity>
struct SensitivityValues {
    float values[NodeCapacity]{};

    __host__ __device__ float& operator[](std::size_t index) {
        return values[index];
    }

    __host__ __device__ float operator[](std::size_t index) const {
        return values[index];
    }
};

struct SensitivityResult {
    float first = 0.0f;
    float second = 0.0f;
};

template<typename Node, std::size_t NodeCapacity>
struct SensitivityTask {
    SensitivityNodes<Node, NodeCapacity> nodes{};
    SensitivityStencil<NodeCapacity> stencil{};
    CentralRequirement central_requirement = CentralRequirement::none;
};

}  // namespace ai_factory::workbench::price_gradients

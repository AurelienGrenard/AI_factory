// Replay the eight Kou rounding regressions across launch geometries at 2^20 paths.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/kou/product/american_option.cuh"
#include "tests/longstaff_schwartz/fixtures/kou_ill_conditioned_rows.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {
using namespace ai_factory::workbench;
namespace kou = model::equity::kou;

struct DeviceRow {
    kou::ModelParameters* model = nullptr;
    product::AmericanOptionParameters* product = nullptr;
    float* outputs = nullptr;
    ~DeviceRow() {
        cudaFree(model);
        cudaFree(product);
        cudaFree(outputs);
    }
};

template<OptionSide Side>
void check_case(const ai_factory::tests::longstaff_schwartz::KouRegressionCase& row) {
    DeviceRow device;
    check_cuda(cudaMalloc(&device.model, sizeof(*device.model)), "allocate Kou model");
    check_cuda(cudaMalloc(&device.product, sizeof(*device.product)), "allocate American product");
    check_cuda(cudaMalloc(&device.outputs, 2U * sizeof(float)), "allocate pricing outputs");
    check_cuda(cudaMemcpy(device.model, &row.model, sizeof(row.model), cudaMemcpyHostToDevice), "copy model");
    check_cuda(cudaMemcpy(device.product, &row.product, sizeof(row.product), cudaMemcpyHostToDevice), "copy product");
    std::array<float, 2> first{};
    bool reference_set = false;
    for (auto [threads, blocks] : {std::pair{128U, 128U}, std::pair{256U, 64U}, std::pair{512U, 32U}}) {
        const auto execution = kou::launch_kou_american_option_cuda<Side>(
            device.model, 1U, &row.product, device.product, 1U,
            PriceConstruction::Aligned, 1U, 1U << 20U, 1.0f / 252.0f,
            threads, blocks, row.seed, device.outputs, device.outputs + 1U
        );
        longstaff_schwartz::validate_regression_diagnostics(execution, "Kou geometry regression");
        std::array<float, 2> values{};
        check_cuda(cudaMemcpy(values.data(), device.outputs, sizeof(values), cudaMemcpyDeviceToHost), "copy prices");
        if (!std::isfinite(values[0]) || !std::isfinite(values[1]) || values[1] < 0.0f) {
            throw std::runtime_error("Invalid Kou result");
        }
        if (reference_set && values != first) {
            throw std::runtime_error("Kou price/error changes with geometry");
        }
        first = values;
        reference_set = true;
        if constexpr (Side == OptionSide::put) {
            if (std::abs(values[0] - row.put_price) > 1e-6f + 1e-6f * std::abs(row.put_price)
                || std::abs(values[1] - row.put_standard_error) > 1e-8f + 1e-5f * std::abs(row.put_standard_error)) {
                throw std::runtime_error("Kou differs from the high-precision regression reference");
            }
        }
    }
}
}  // namespace

int main() {
    int devices = 0;
    const auto availability = cudaGetDeviceCount(&devices);
    if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver
        || devices == 0) return 77;
    check_cuda(availability, "query test GPU");
    for (const auto& row : ai_factory::tests::longstaff_schwartz::kKouRegressionCases) {
        check_case<OptionSide::put>(row);
        check_case<OptionSide::call>(row);
    }
}

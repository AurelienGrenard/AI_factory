// Check the bounded LSM residual correction, its precision and status accounting.
#include "common/check_cuda.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
namespace lsm = ai_factory::workbench::longstaff_schwartz;
using ai_factory::workbench::check_cuda;
using Regressor = lsm::NormalEquationRegressor<
    lsm::basis::LaguerrePolynomialTwoFactorBasis,
    lsm::RegressionRefinement::normal_residual
>;
constexpr std::size_t kSize = Regressor::kBasisSize;

struct Observation {
    Regressor::Features features;
    double target;
};

template<typename T>
struct ManagedArray {
    T* data = nullptr;
    explicit ManagedArray(std::size_t count) {
        check_cuda(cudaMallocManaged(&data, count * sizeof(T)), "allocate test data");
    }
    ~ManagedArray() { cudaFree(data); }
    ManagedArray(const ManagedArray&) = delete;
    ManagedArray& operator=(const ManagedArray&) = delete;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Independent CPU accumulation and pivoted elimination, not native Cholesky.
std::array<double, kSize> reference(const std::vector<Observation>& observations) {
    static_assert(std::numeric_limits<long double>::digits >= 64);
    constexpr std::size_t count = kSize * kSize + kSize;
    std::array<long double, count> sums{}, errors{};
    const auto add = [&](std::size_t index, long double value) {
        const long double corrected = value - errors[index];
        const long double updated = sums[index] + corrected;
        errors[index] = (updated - sums[index]) - corrected;
        sums[index] = updated;
    };
    for (const auto& observation : observations) {
        for (std::size_t row = 0U; row < kSize; ++row) {
            const long double feature = observation.features.values[row];
            for (std::size_t column = 0U; column < kSize; ++column) {
                add(row * kSize + column,
                    feature * observation.features.values[column]);
            }
            add(kSize * kSize + row, feature * observation.target);
        }
    }
    double trace = 0.0;
    for (std::size_t row = 0U; row < kSize; ++row) {
        trace += static_cast<double>(sums[row * kSize + row]);
    }
    const double ridge = lsm::kRidgeRelative * trace / static_cast<double>(kSize);
    for (std::size_t row = 0U; row < kSize; ++row) sums[row * kSize + row] += ridge;
    long double* const rhs = sums.data() + kSize * kSize;
    for (std::size_t column = 0U; column < kSize; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1U; row < kSize; ++row) {
            if (std::abs(sums[row * kSize + column])
                > std::abs(sums[pivot * kSize + column])) pivot = row;
        }
        require(sums[pivot * kSize + column] != 0.0L, "Singular CPU reference");
        for (std::size_t index = column; index < kSize; ++index) {
            std::swap(sums[column * kSize + index], sums[pivot * kSize + index]);
        }
        std::swap(rhs[column], rhs[pivot]);
        for (std::size_t row = column + 1U; row < kSize; ++row) {
            const long double factor = sums[row * kSize + column]
                / sums[column * kSize + column];
            for (std::size_t index = column + 1U; index < kSize; ++index) {
                sums[row * kSize + index] -= factor * sums[column * kSize + index];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    std::array<double, kSize> coefficients{};
    for (std::size_t row = kSize; row-- > 0U;) {
        for (std::size_t column = row + 1U; column < kSize; ++column) {
            rhs[row] -= sums[row * kSize + column] * rhs[column];
        }
        rhs[row] /= sums[row * kSize + row];
        coefficients[row] = static_cast<double>(rhs[row]);
    }
    return coefficients;
}

template<bool Residual>
__global__ void partials_kernel(
    const Observation* observations, std::size_t count, double* partials,
    const double* coefficients, const lsm::RegressionStatus* status
) {
    if constexpr (Residual) {
        if (*status != lsm::RegressionStatus::success) return;
    }
    double values[Residual ? kSize : Regressor::kRegressionValueCount] = {};
    for (std::size_t path = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         path < count; path += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto& observation = observations[path];
        if constexpr (Residual) {
            Regressor::accumulate_residual(
                observation.features, observation.target, coefficients, values);
        } else {
            Regressor::accumulate(observation.features, observation.target, values);
        }
    }
    if constexpr (Residual) {
        Regressor::reduce_and_store_residual_partials(values, 0U, blockIdx.x, gridDim.x, partials);
    } else {
        Regressor::reduce_and_store_partials(values, 0U, blockIdx.x, gridDim.x, partials);
    }
}

template<bool Correction>
__global__ void solve_kernel(
    std::size_t blocks, const double* partials, double* coefficients,
    lsm::RegressionStatus* status, lsm::RegressionDiagnostics* diagnostics
) {
    Regressor::solve_for_row<Correction>(
        1U, 0U, 0U, blocks, partials, coefficients, status, diagnostics);
}

void run_case(float width, unsigned int threads, unsigned int blocks) {
    constexpr std::size_t count = 16384U;
    std::vector<Observation> observations(count);
    for (std::size_t index = 0U; index < count; ++index) {
        const float primary = 0.9f + width
            * (static_cast<float>(index % 997U) / 997.0f - 0.5f);
        const float secondary = std::log(primary);
        const float laguerre = 1.0f - primary;
        const float cashflow = 0.12f + 0.03f * std::sin(static_cast<float>(index) * 0.019f);
        observations[index] = {{
            {1.0f, laguerre, std::fma(0.5f * primary, primary, 1.0f - 2.0f * primary),
             secondary, secondary * secondary, laguerre * secondary}
        }, static_cast<double>(0.97f) * static_cast<double>(cashflow)};
    }
    const auto expected = reference(observations);
    ManagedArray<Observation> inputs(count);
    ManagedArray<double> partials(Regressor::kRegressionValueCount * blocks);
    ManagedArray<double> coefficients(kSize);
    ManagedArray<lsm::RegressionStatus> status(1U);
    ManagedArray<lsm::RegressionDiagnostics> diagnostics(1U);
    std::copy(observations.begin(), observations.end(), inputs.data);
    *diagnostics.data = {};
    const auto shared = Regressor::shared_bytes(threads);
    partials_kernel<false><<<blocks, threads, shared>>>(
        inputs.data, count, partials.data, coefficients.data, status.data);
    solve_kernel<false><<<1, threads, shared>>>(
        blocks, partials.data, coefficients.data, status.data, diagnostics.data);
    check_cuda(cudaDeviceSynchronize(), "initial regression");
    require(*status.data == lsm::RegressionStatus::success, "Initial solve failed");
    require(diagnostics.data->successful_regression_count == 0U,
        "Success recorded before the correction");
    partials_kernel<true><<<blocks, threads, shared>>>(
        inputs.data, count, partials.data, coefficients.data, status.data);
    solve_kernel<true><<<1, threads, shared>>>(
        blocks, partials.data, coefficients.data, status.data, diagnostics.data);
    check_cuda(cudaDeviceSynchronize(), "residual correction");
    require(*status.data == lsm::RegressionStatus::success, "Correction failed");
    require(diagnostics.data->successful_regression_count == 1U
        && diagnostics.data->fatal_failure_count == 0U, "Correction diagnostics counted twice");
    double error = 0.0;
    for (std::size_t index = 0U; index < kSize; ++index) {
        error = std::max(error, std::abs(coefficients.data[index] - expected[index])
            / (1.0 + std::abs(expected[index])));
    }
    require(error <= 5e-7, "Residual coefficients disagree with the CPU reference");
    std::cout << "width=" << width << " threads=" << threads << " blocks=" << blocks
              << " scaled_coefficient_error=" << error << '\n';

    // Failed corrections are fatal, not a second success or silent fallback.
    *diagnostics.data = {};
    partials.data[Regressor::kGramValueCount * blocks] = std::numeric_limits<double>::quiet_NaN();
    solve_kernel<true><<<1, threads, shared>>>(
        blocks, partials.data, coefficients.data, status.data, diagnostics.data);
    check_cuda(cudaDeviceSynchronize(), "non-finite correction");
    require(*status.data == lsm::RegressionStatus::non_finite_statistics
        && diagnostics.data->fatal_failure_count == 1U
        && diagnostics.data->successful_regression_count == 0U,
        "Non-finite residual was not recorded as one fatal failure");
    for (std::size_t candidates : std::array<std::size_t, 2>{0U, kSize}) {
        std::fill_n(partials.data, Regressor::kRegressionValueCount * blocks, 0.0);
        partials.data[(Regressor::kRegressionValueCount - 1U) * blocks] = candidates;
        *diagnostics.data = {};
        solve_kernel<false><<<1, threads, shared>>>(
            blocks, partials.data, coefficients.data, status.data, diagnostics.data);
        partials_kernel<true><<<blocks, threads, shared>>>(
            inputs.data, count, partials.data, coefficients.data, status.data);
        solve_kernel<true><<<1, threads, shared>>>(
            blocks, partials.data, coefficients.data, status.data, diagnostics.data);
        check_cuda(cudaDeviceSynchronize(), "skipped correction");
        require(diagnostics.data->successful_regression_count == 0U
            && diagnostics.data->fatal_failure_count == 0U
            && diagnostics.data->no_candidate_count + diagnostics.data->insufficient_candidate_count == 1U,
            "Skipped correction changed the initial status");
    }
}
}  // namespace

int main() {
    int devices = 0;
    const auto availability = cudaGetDeviceCount(&devices);
    if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver
        || devices == 0) return 77;
    check_cuda(availability, "query test GPU");
    for (float width : {0.6f, 0.01f}) {
        for (auto [threads, blocks] : {std::pair{128U, 32U}, std::pair{256U, 64U}, std::pair{512U, 128U}}) {
            run_case(width, threads, blocks);
        }
    }
}

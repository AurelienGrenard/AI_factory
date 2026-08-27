// Exercise all small-basis families and the shared FP64 normal equations.
#include "common/check_cuda.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/basis/hinge.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/concepts.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/longstaff_schwartz/workspace.cuh"
#include "common/simulation/early_exercise_schedule.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

namespace lsm = ai_factory::workbench::longstaff_schwartz;
namespace basis = ai_factory::workbench::longstaff_schwartz::basis;

using TwoFactorRegressor =
    lsm::NormalEquationRegressor<basis::LaguerrePolynomialTwoFactorBasis>;
using LaguerreRegressor =
    lsm::NormalEquationRegressor<basis::OneFactorLaguerreBasis<3U>>;
using HermiteRegressor =
    lsm::NormalEquationRegressor<basis::OneFactorHermiteBasis<2U>>;
using HingeRegressor =
    lsm::NormalEquationRegressor<basis::StandardizedFiveKnotHingeBasis>;
using StatusRegressor =
    lsm::NormalEquationRegressor<basis::OneFactorHermiteBasis<0U>>;

inline constexpr std::size_t kStatusCaseCount = 5U;

static_assert(lsm::SmallLinearRegressor<TwoFactorRegressor>);
static_assert(lsm::SmallLinearRegressor<LaguerreRegressor>);
static_assert(lsm::SmallLinearRegressor<HermiteRegressor>);
static_assert(lsm::SmallLinearRegressor<HingeRegressor>);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(float actual, float expected, float tolerance) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        throw std::runtime_error("Unexpected device basis value.");
    }
}

void validate_schedule_and_workspace_planning() {
    using ai_factory::workbench::simulation::MaturityAlignedExerciseCalendar;
    using ai_factory::workbench::simulation::
        maturity_aligned_exercise_count;
    using ai_factory::workbench::simulation::
        maturity_aligned_first_exercise_days;

    constexpr MaturityAlignedExerciseCalendar stubbed{365U, 30U};
    static_assert(maturity_aligned_exercise_count(stubbed) == 13U);
    static_assert(maturity_aligned_first_exercise_days(stubbed) == 5U);
    bool invalid_calendar_rejected = false;
    try {
        ai_factory::workbench::simulation::validate_exercise_calendar(
            MaturityAlignedExerciseCalendar{30U, 30U}
        );
    } catch (const std::invalid_argument&) {
        invalid_calendar_rejected = true;
    }
    require(
        invalid_calendar_rejected,
        "An exercise interval at maturity was accepted."
    );

    const lsm::WorkspaceDescriptor descriptor{
        64U,
        alignof(double),
        {
            {sizeof(float), alignof(float)},
            {sizeof(float), alignof(float)},
        },
        TwoFactorRegressor::kBasisSize,
        TwoFactorRegressor::kRegressionValueCount,
    };
    const std::vector<lsm::EarlyExerciseRowPlan> rows{
        {2U, 128U},
        {5U, 320U},
        {1U, 64U},
    };
    constexpr std::size_t paths_per_price = 64U;
    constexpr std::size_t blocks_per_price = 4U;
    const lsm::WorkspaceLayout first_two = lsm::make_workspace_layout(
        descriptor,
        2U,
        rows[0].state_value_count + rows[1].state_value_count,
        paths_per_price,
        blocks_per_price,
        "planning-test"
    );
    const lsm::ExecutionPlan plan = lsm::plan_batches(
        rows,
        descriptor,
        paths_per_price,
        blocks_per_price,
        first_two.total_bytes,
        "planning-test"
    );
    require(plan.batches.size() == 2U, "Unexpected workspace batch count.");
    require(
        plan.batches[0].result_offset == 0U
            && plan.batches[0].result_count == 2U
            && plan.batches[0].maximum_regression_count == 5U,
        "The first workspace batch is inconsistent."
    );
    require(
        plan.batches[1].result_offset == 2U
            && plan.batches[1].result_count == 1U
            && plan.batches[1].maximum_regression_count == 1U,
        "The second workspace batch is inconsistent."
    );
    require(
        plan.maximum_workspace_bytes == first_two.total_bytes,
        "The workspace maximum does not match the limiting batch."
    );

    bool insufficient_workspace_rejected = false;
    try {
        static_cast<void>(lsm::plan_batches(
            rows,
            descriptor,
            paths_per_price,
            blocks_per_price,
            1U,
            "planning-test"
        ));
    } catch (const std::runtime_error&) {
        insufficient_workspace_rejected = true;
    }
    require(
        insufficient_workspace_rejected,
        "A row larger than the workspace budget was accepted."
    );
}

__global__ void evaluate_bases_kernel(float* output) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) return;

    const TwoFactorRegressor::Features two_factor = TwoFactorRegressor::evaluate(
        {0.5f, 0.25f}
    );
    for (std::size_t index = 0U;
         index < TwoFactorRegressor::kBasisSize;
         ++index) {
        output[index] = two_factor.values[index];
    }

    const LaguerreRegressor::Features laguerre =
        LaguerreRegressor::evaluate(0.5f);
    for (std::size_t index = 0U;
         index < LaguerreRegressor::kBasisSize;
         ++index) {
        output[6U + index] = laguerre.values[index];
    }

    const HermiteRegressor::Features hermite =
        HermiteRegressor::evaluate(2.0f);
    for (std::size_t index = 0U;
         index < HermiteRegressor::kBasisSize;
         ++index) {
        output[10U + index] = hermite.values[index];
    }

    const HingeRegressor::Features hinge = HingeRegressor::evaluate(0.5f);
    for (std::size_t index = 0U;
         index < HingeRegressor::kBasisSize;
         ++index) {
        output[13U + index] = hinge.values[index];
    }
}

__global__ void fit_hermite_kernel(
    double* partials,
    double* coefficients,
    lsm::RegressionStatus* status,
    lsm::RegressionDiagnostics* diagnostics
) {
    double statistics[HermiteRegressor::kRegressionValueCount] = {};
    for (std::size_t sample = threadIdx.x;
         sample < 1024U;
         sample += blockDim.x) {
        const float x = -2.0f
            + 4.0f * (
                static_cast<float>(sample) + 0.5f
            ) / 1024.0f;
        const HermiteRegressor::Features features =
            HermiteRegressor::evaluate(x);
        const double target =
            0.25 * static_cast<double>(features.values[0])
            - 0.5 * static_cast<double>(features.values[1])
            + 0.75 * static_cast<double>(features.values[2]);
        HermiteRegressor::accumulate(features, target, statistics);
    }
    HermiteRegressor::reduce_and_store_partials(
        statistics,
        0U,
        0U,
        1U,
        partials
    );
    __syncthreads();
    HermiteRegressor::solve_for_row(
        1U,
        0U,
        0U,
        1U,
        partials,
        coefficients,
        status,
        diagnostics
    );
}

__global__ void classify_regression_statuses_kernel(
    const double* partials,
    double* coefficients,
    lsm::RegressionStatus* statuses,
    lsm::RegressionDiagnostics* diagnostics
) {
    const std::size_t row = blockIdx.x;
    StatusRegressor::solve_for_row(
        1U,
        0U,
        row,
        1U,
        partials,
        coefficients,
        statuses,
        diagnostics
    );
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    validate_schedule_and_workspace_planning();

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "test cudaGetDeviceCount");

    float* device_basis_values = nullptr;
    double* device_partials = nullptr;
    double* device_coefficients = nullptr;
    lsm::RegressionStatus* device_status = nullptr;
    lsm::RegressionDiagnostics* device_diagnostics = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_basis_values, 20U * sizeof(float)),
            "cudaMalloc basis values"
        );
        check_cuda(
            cudaMalloc(
                &device_partials,
                std::max(
                    HermiteRegressor::kRegressionValueCount,
                    kStatusCaseCount
                        * StatusRegressor::kRegressionValueCount
                ) * sizeof(double)
            ),
            "cudaMalloc regression partials"
        );
        check_cuda(
            cudaMalloc(
                &device_coefficients,
                std::max(
                    HermiteRegressor::kBasisSize,
                    kStatusCaseCount * StatusRegressor::kBasisSize
                ) * sizeof(double)
            ),
            "cudaMalloc regression coefficients"
        );
        check_cuda(
            cudaMalloc(
                &device_status,
                kStatusCaseCount * sizeof(lsm::RegressionStatus)
            ),
            "cudaMalloc regression status"
        );
        check_cuda(
            cudaMalloc(
                &device_diagnostics,
                kStatusCaseCount * sizeof(lsm::RegressionDiagnostics)
            ),
            "cudaMalloc regression diagnostics"
        );
        check_cuda(
            cudaMemset(
                device_diagnostics,
                0,
                kStatusCaseCount * sizeof(lsm::RegressionDiagnostics)
            ),
            "cudaMemset regression diagnostics"
        );

        evaluate_bases_kernel<<<1U, 1U>>>(device_basis_values);
        check_cuda(cudaGetLastError(), "evaluate basis families");

        constexpr unsigned int threads_per_block = 256U;
        fit_hermite_kernel<<<
            1U,
            threads_per_block,
            HermiteRegressor::shared_bytes(threads_per_block)
        >>>(
            device_partials,
            device_coefficients,
            device_status,
            device_diagnostics
        );
        check_cuda(cudaGetLastError(), "fit Hermite regression");

        float basis_values[20]{};
        double coefficients[HermiteRegressor::kBasisSize]{};
        lsm::RegressionStatus status = lsm::RegressionStatus::no_candidates;
        lsm::RegressionDiagnostics diagnostics{};
        check_cuda(
            cudaMemcpy(
                basis_values,
                device_basis_values,
                sizeof(basis_values),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy basis values"
        );
        check_cuda(
            cudaMemcpy(
                coefficients,
                device_coefficients,
                sizeof(coefficients),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy regression coefficients"
        );
        check_cuda(
            cudaMemcpy(
                &status,
                device_status,
                sizeof(status),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy regression status"
        );
        check_cuda(
            cudaMemcpy(
                &diagnostics,
                device_diagnostics,
                sizeof(diagnostics),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy regression diagnostics"
        );

        const float expected_two_factor[6] = {
            1.0f, 0.5f, 0.125f, 0.25f, 0.0625f, 0.125f,
        };
        for (std::size_t index = 0U; index < 6U; ++index) {
            require_close(
                basis_values[index], expected_two_factor[index], 1.0e-7f
            );
        }
        require_close(basis_values[10U], 1.0f, 1.0e-7f);
        require_close(basis_values[11U], 2.0f, 1.0e-7f);
        require_close(basis_values[12U], 3.0f, 1.0e-7f);
        const float expected_hinge[7] = {
            1.0f, 0.5f, 2.5f, 1.5f, 0.5f, 0.0f, 0.0f,
        };
        for (std::size_t index = 0U; index < 7U; ++index) {
            require_close(
                basis_values[13U + index], expected_hinge[index], 1.0e-7f
            );
        }

        require(
            status == lsm::RegressionStatus::success,
            "The synthetic Hermite regression was not solved."
        );
        require(
            diagnostics.successful_regression_count == 1U
                && diagnostics.fatal_failure_count == 0U,
            "The successful regression diagnostics are inconsistent."
        );
        require(
            std::fabs(coefficients[0] - 0.25) < 1.0e-8
                && std::fabs(coefficients[1] + 0.5) < 1.0e-8
                && std::fabs(coefficients[2] - 0.75) < 1.0e-8,
            "The synthetic Hermite coefficients were not recovered."
        );

        double status_partials[
            kStatusCaseCount * StatusRegressor::kRegressionValueCount
        ]{};
        const auto set_status_case = [&] (
            std::size_t row,
            double gram,
            double right_hand_side,
            double candidate_count
        ) {
            const std::size_t offset =
                row * StatusRegressor::kRegressionValueCount;
            status_partials[offset] = gram;
            status_partials[offset + 1U] = right_hand_side;
            status_partials[offset + 2U] = candidate_count;
        };
        set_status_case(0U, 0.0, 0.0, 0.0);
        set_status_case(1U, 1.0, 1.0, 1.0);
        set_status_case(2U, 0.0, 0.0, 2.0);
        set_status_case(3U, NAN, 0.0, 2.0);
        set_status_case(4U, 4.0e-14, DBL_MAX, 2.0);
        check_cuda(
            cudaMemcpy(
                device_partials,
                status_partials,
                sizeof(status_partials),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy typed regression cases"
        );
        check_cuda(
            cudaMemset(
                device_diagnostics,
                0,
                kStatusCaseCount * sizeof(lsm::RegressionDiagnostics)
            ),
            "cudaMemset typed regression diagnostics"
        );
        constexpr unsigned int status_threads_per_block = 32U;
        classify_regression_statuses_kernel<<<
            kStatusCaseCount,
            status_threads_per_block,
            StatusRegressor::shared_bytes(status_threads_per_block)
        >>>(
            device_partials,
            device_coefficients,
            device_status,
            device_diagnostics
        );
        check_cuda(cudaGetLastError(), "classify regression statuses");

        lsm::RegressionStatus statuses[kStatusCaseCount]{};
        lsm::RegressionDiagnostics typed_diagnostics[kStatusCaseCount]{};
        check_cuda(
            cudaMemcpy(
                statuses,
                device_status,
                sizeof(statuses),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy typed regression statuses"
        );
        check_cuda(
            cudaMemcpy(
                typed_diagnostics,
                device_diagnostics,
                sizeof(typed_diagnostics),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy typed regression diagnostics"
        );
        const lsm::RegressionStatus expected_statuses[kStatusCaseCount] = {
            lsm::RegressionStatus::no_candidates,
            lsm::RegressionStatus::insufficient_candidates,
            lsm::RegressionStatus::factorization_failure,
            lsm::RegressionStatus::non_finite_statistics,
            lsm::RegressionStatus::non_finite_coefficients,
        };
        for (std::size_t row = 0U; row < kStatusCaseCount; ++row) {
            require(
                statuses[row] == expected_statuses[row],
                "A regression failure cause was misclassified."
            );
            const bool expected_fatal = row >= 2U;
            require(
                (typed_diagnostics[row].fatal_failure_count != 0U)
                    == expected_fatal,
                "A regression status has the wrong publication policy."
            );
        }
    } catch (...) {
        if (device_basis_values != nullptr) cudaFree(device_basis_values);
        if (device_partials != nullptr) cudaFree(device_partials);
        if (device_coefficients != nullptr) cudaFree(device_coefficients);
        if (device_status != nullptr) cudaFree(device_status);
        if (device_diagnostics != nullptr) cudaFree(device_diagnostics);
        throw;
    }

    check_cuda(cudaFree(device_basis_values), "cudaFree basis values");
    check_cuda(cudaFree(device_partials), "cudaFree regression partials");
    check_cuda(cudaFree(device_coefficients), "cudaFree regression coefficients");
    check_cuda(cudaFree(device_status), "cudaFree regression status");
    check_cuda(
        cudaFree(device_diagnostics),
        "cudaFree regression diagnostics"
    );
}

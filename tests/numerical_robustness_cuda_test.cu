// Exercise failure signaling and cancellation-sensitive numerical primitives.
#include "common/check_cuda.cuh"
#include "common/fixed_income/jamshidian.cuh"
#include "common/fixed_income/jamshidian_cooperative.cuh"
#include "common/fixed_income/mean_reverting_gaussian.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/exercise_decision.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/reductions.cuh"
#include "curve/nelson_siegel/instantaneous_forward.cuh"
#include "curve/svensson/instantaneous_forward.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

namespace fixed_income = ai_factory::workbench::fixed_income;
namespace mean_reverting_gaussian =
    ai_factory::workbench::fixed_income::mean_reverting_gaussian;
namespace lsm = ai_factory::workbench::longstaff_schwartz;
namespace nelson_siegel = ai_factory::workbench::curve::nelson_siegel;
namespace reductions = ai_factory::workbench::reductions;
namespace schobel_zhu =
    ai_factory::workbench::model::equity::schobel_zhu;
namespace svensson = ai_factory::workbench::curve::svensson;

struct EmptyParameters {};

struct UnitAffineProvider {
    __device__ fixed_income::OneFactorAffineBondCoefficients
    affine_bond_coefficients(
        const EmptyParameters&,
        float,
        float
    ) const {
        return {0.0f, 1.0f};
    }
};

struct CollapsedBracketAffineProvider {
    __device__ fixed_income::OneFactorAffineBondCoefficients
    affine_bond_coefficients(
        const EmptyParameters&,
        float,
        float
    ) const {
        return {1.0e8f, 1.0f};
    }
};

struct TwoPaymentSchedule {
    __device__ bool valid() const { return true; }
    __device__ std::uint32_t payment_count() const { return 2U; }
    __device__ float accrual_fraction(std::uint32_t) const { return 1.0f; }
    __device__ float payment_time(std::uint32_t payment) const {
        return 2.0f + static_cast<float>(payment);
    }
};

using ConstantRegressor = lsm::NormalEquationRegressor<
    lsm::basis::OneFactorHermiteBasis<0U>
>;

struct SchobelZhuGridPoint {
    float mean_reversion;
    float delta_t;
};

inline constexpr std::size_t kSchobelZhuGridSize = 10U;

__host__ __device__ constexpr SchobelZhuGridPoint schobel_zhu_grid_point(
    std::size_t index
) {
    switch (index) {
        case 0U: return {0.030000f, 1.0f / 504.0f};
        case 1U: return {0.068745f, 1.0f / 504.0f};
        case 2U: return {0.157523f, 1.0f / 504.0f};
        case 3U: return {0.360823f, 1.0f / 504.0f};
        case 4U: return {0.826337f, 1.0f / 504.0f};
        case 5U: return {1.892929f, 1.0f / 504.0f};
        case 6U: return {4.335608f, 1.0f / 504.0f};
        case 7U: return {10.00000f, 1.0f / 504.0f};
        case 8U: return {0.030000f, 1.0e-6f};
        default: return {10.00000f, 1.0e-6f};
    }
}

struct NumericalResults {
    float unconverged_jamshidian_boundary;
    float collapsed_scalar_jamshidian_boundary;
    float collapsed_cooperative_jamshidian_boundary;
    double regular_price;
    double regular_standard_error;
    double invalid_price;
    double invalid_standard_error;
    double roundoff_price;
    double roundoff_standard_error;
    double undersized_price;
    double undersized_standard_error;
    double empty_price;
    double empty_standard_error;
    double nonfinite_price;
    double nonfinite_standard_error;
    double infinite_price;
    double infinite_standard_error;
    double fp64_prediction;
    float fp64_exercise_cashflow;
    float narrowed_exercise_cashflow;
    double nelson_siegel_forward;
    double svensson_forward;
    float stationary_volatility;
    schobel_zhu::PreparedModel small_mean_reversion_model;
    schobel_zhu::PreparedModel schobel_zhu_grid[kSchobelZhuGridSize];
    std::uint32_t schobel_zhu_philox_mapping_preserved;
    float insufficient_candidate_price;
    float insufficient_candidate_standard_error;
    float fatal_regression_price;
    float fatal_regression_standard_error;
    std::uint32_t insufficient_candidate_invalidated;
    std::uint32_t fatal_regression_invalidated;
};

__global__ void exercise_numerical_robustness_kernel(
    NumericalResults* output
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;

    const float unconverged_boundary = fixed_income::jamshidian_state_boundary<
        UnitAffineProvider,
        EmptyParameters,
        TwoPaymentSchedule,
        0U
    >(
        UnitAffineProvider{},
        EmptyParameters{},
        1.0f,
        0.1f,
        TwoPaymentSchedule{}
    );
    const float collapsed_scalar_boundary =
        fixed_income::jamshidian_state_boundary(
            CollapsedBracketAffineProvider{},
            EmptyParameters{},
            1.0f,
            0.1f,
            TwoPaymentSchedule{}
        );
    float collapsed_log_A[2] = {1.0e8f, 1.0e8f};
    float collapsed_B[2] = {1.0f, 1.0f};
    float unused_option_values[2] = {};
    const fixed_income::CooperativeJamshidianWorkspace collapsed_workspace{
        collapsed_log_A,
        collapsed_B,
        unused_option_values,
    };
    const float collapsed_cooperative_boundary =
        fixed_income::jamshidian_state_boundary_from_coefficients(
            0.1f,
            TwoPaymentSchedule{},
            collapsed_workspace
        );

    double regular_price = 0.0;
    double regular_standard_error = 0.0;
    reductions::compute_statistics(
        {6.0, 14.0},
        3U,
        regular_price,
        regular_standard_error
    );

    double invalid_price = 0.0;
    double invalid_standard_error = 0.0;
    reductions::compute_statistics(
        {2.0, 1.0},
        2U,
        invalid_price,
        invalid_standard_error
    );

    double roundoff_price = 0.0;
    double roundoff_standard_error = 0.0;
    reductions::compute_statistics(
        {2.0, 1.9999999999999998},
        2U,
        roundoff_price,
        roundoff_standard_error
    );

    double undersized_price = 0.0;
    double undersized_standard_error = 0.0;
    reductions::compute_statistics(
        {1.0, 1.0},
        1U,
        undersized_price,
        undersized_standard_error
    );

    double empty_price = 0.0;
    double empty_standard_error = 0.0;
    reductions::compute_statistics(
        {0.0, 0.0},
        0U,
        empty_price,
        empty_standard_error
    );

    double nonfinite_price = 0.0;
    double nonfinite_standard_error = 0.0;
    reductions::compute_statistics(
        {nan(""), 1.0},
        2U,
        nonfinite_price,
        nonfinite_standard_error
    );

    double infinite_price = 0.0;
    double infinite_standard_error = 0.0;
    reductions::compute_statistics(
        {CUDART_INF, CUDART_INF},
        2U,
        infinite_price,
        infinite_standard_error
    );

    const ConstantRegressor::Features constant_features =
        ConstantRegressor::evaluate(0.0f);
    const double coefficients[1] = {1.0 + 0x1p-30};
    const double fp64_prediction = ConstantRegressor::predict(
        constant_features, coefficients
    );
    const double boundary_coefficients[1] = {1.0 - 0x1p-30};
    const double boundary_prediction = ConstantRegressor::predict(
        constant_features, boundary_coefficients
    );
    const float fp64_exercise_cashflow = lsm::select_exercise_cashflow(
        1.0f, boundary_prediction, 0.25f
    );
    const float narrowed_exercise_cashflow = lsm::select_exercise_cashflow(
        1.0f,
        static_cast<double>(static_cast<float>(boundary_prediction)),
        0.25f
    );
    const double nelson_siegel_forward =
        nelson_siegel::instantaneous_forward_formula(
            0.02, -0.01, 0.03, 1.25
        );
    const double svensson_forward = svensson::instantaneous_forward_formula(
        0.02, -0.01, 0.03, 0.015, 1.5, 4.0, 2.25
    );
    const float stationary_volatility =
        mean_reverting_gaussian::volatility_from_stationary_deviation(
            0.025f, 0.7f
        );

    constexpr schobel_zhu::ModelParameters small_mean_reversion_parameters{
        1.0f,
        0.03f,
        0.01f,
        0.22f,
        1.0e-8f,
        0.20f,
        0.35f,
        -0.6f,
    };
    const schobel_zhu::PreparedModel small_mean_reversion_model =
        schobel_zhu::prepare_model(
            small_mean_reversion_parameters,
            1.0e-4f
        );

    schobel_zhu::PreparedModel schobel_zhu_grid[kSchobelZhuGridSize];
    for (std::size_t index = 0U;
         index < kSchobelZhuGridSize;
         ++index) {
        const SchobelZhuGridPoint point = schobel_zhu_grid_point(index);
        schobel_zhu::ModelParameters parameters =
            small_mean_reversion_parameters;
        parameters.mean_reversion = point.mean_reversion;
        schobel_zhu_grid[index] = schobel_zhu::prepare_model(
            parameters, point.delta_t
        );
    }

    const ai_factory::workbench::philox::PhiloxKey mapping_key =
        ai_factory::workbench::philox::make_key(0x5a17c9e3ULL);
    schobel_zhu::DynamicsPolicy::RandomContext policy_random(
        mapping_key, 41U
    );
    schobel_zhu::DynamicsPolicy::RandomContext explicit_random(
        mapping_key, 41U
    );
    schobel_zhu::State policy_state =
        schobel_zhu::initial_state(small_mean_reversion_model);
    schobel_zhu::State explicit_state = policy_state;
    schobel_zhu::DynamicsPolicy::simulate_one_step(
        small_mean_reversion_model, policy_random, policy_state
    );
    const float explicit_ou_normal =
        ai_factory::workbench::philox::next_normal(
            explicit_random.uniforms, explicit_random.normals
        );
    const float explicit_increment_residual =
        ai_factory::workbench::philox::next_normal(
            explicit_random.uniforms, explicit_random.normals
        );
    const float explicit_asset_residual =
        ai_factory::workbench::philox::next_normal(
            explicit_random.uniforms, explicit_random.normals
        );
    schobel_zhu::one_step_transition(
        small_mean_reversion_model,
        explicit_ou_normal,
        explicit_increment_residual,
        explicit_asset_residual,
        explicit_state
    );
    const float policy_next_normal =
        ai_factory::workbench::philox::next_normal(
            policy_random.uniforms, policy_random.normals
        );
    const float explicit_next_normal =
        ai_factory::workbench::philox::next_normal(
            explicit_random.uniforms, explicit_random.normals
        );
    const std::uint32_t schobel_zhu_philox_mapping_preserved =
        static_cast<std::uint32_t>(
            policy_state.log_spot == explicit_state.log_spot
            && policy_state.volatility == explicit_state.volatility
            && policy_next_normal == explicit_next_normal
        );

    lsm::RegressionDiagnostics insufficient_diagnostics{};
    lsm::record_regression_status(
        lsm::RegressionStatus::insufficient_candidates,
        insufficient_diagnostics
    );
    float insufficient_candidate_price = 2.0f;
    float insufficient_candidate_standard_error = 0.1f;
    const std::uint32_t insufficient_candidate_invalidated =
        static_cast<std::uint32_t>(
            lsm::invalidate_regression_result_if_fatal(
                insufficient_diagnostics,
                0U,
                &insufficient_candidate_price,
                &insufficient_candidate_standard_error
            )
        );
    lsm::RegressionDiagnostics fatal_diagnostics{};
    lsm::record_regression_status(
        lsm::RegressionStatus::factorization_failure,
        fatal_diagnostics
    );
    float fatal_regression_price = 2.0f;
    float fatal_regression_standard_error = 0.1f;
    const std::uint32_t fatal_regression_invalidated =
        static_cast<std::uint32_t>(
            lsm::invalidate_regression_result_if_fatal(
                fatal_diagnostics,
                0U,
                &fatal_regression_price,
                &fatal_regression_standard_error
            )
        );

    *output = {
        unconverged_boundary,
        collapsed_scalar_boundary,
        collapsed_cooperative_boundary,
        regular_price,
        regular_standard_error,
        invalid_price,
        invalid_standard_error,
        roundoff_price,
        roundoff_standard_error,
        undersized_price,
        undersized_standard_error,
        empty_price,
        empty_standard_error,
        nonfinite_price,
        nonfinite_standard_error,
        infinite_price,
        infinite_standard_error,
        fp64_prediction,
        fp64_exercise_cashflow,
        narrowed_exercise_cashflow,
        nelson_siegel_forward,
        svensson_forward,
        stationary_volatility,
        small_mean_reversion_model,
        {},
        schobel_zhu_philox_mapping_preserved,
        insufficient_candidate_price,
        insufficient_candidate_standard_error,
        fatal_regression_price,
        fatal_regression_standard_error,
        insufficient_candidate_invalidated,
        fatal_regression_invalidated,
    };
    for (std::size_t index = 0U;
         index < kSchobelZhuGridSize;
         ++index) {
        output->schobel_zhu_grid[index] = schobel_zhu_grid[index];
    }
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "test cudaGetDeviceCount");

    NumericalResults* device_results = nullptr;
    check_cuda(
        cudaMalloc(&device_results, sizeof(NumericalResults)),
        "cudaMalloc numerical results"
    );
    exercise_numerical_robustness_kernel<<<1U, 1U>>>(device_results);
    check_cuda(cudaGetLastError(), "exercise numerical robustness");

    NumericalResults results{};
    check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy numerical results"
    );
    check_cuda(cudaFree(device_results), "cudaFree numerical results");

    require(
        std::isnan(results.unconverged_jamshidian_boundary),
        "An unconverged Jamshidian boundary was published."
    );
    require(
        std::isnan(results.collapsed_scalar_jamshidian_boundary)
            && std::isnan(
                results.collapsed_cooperative_jamshidian_boundary
            ),
        "A stagnated FP32 Jamshidian bracket published a finite boundary."
    );
    require(
        std::fabs(results.regular_price - 2.0) < 1.0e-15
            && std::fabs(
                results.regular_standard_error - std::sqrt(1.0 / 3.0)
            ) < 1.0e-15,
        "Valid Monte Carlo statistics changed."
    );
    require(
        std::isnan(results.invalid_price)
            && std::isnan(results.invalid_standard_error),
        "An invalid negative variance was clamped to zero."
    );
    require(
        results.roundoff_price == 1.0
            && results.roundoff_standard_error == 0.0,
        "A roundoff-sized negative variance was not tolerated."
    );
    require(
        std::isnan(results.undersized_price)
            && std::isnan(results.undersized_standard_error),
        "Statistics were published for fewer than two samples."
    );
    require(
        std::isnan(results.empty_price)
            && std::isnan(results.empty_standard_error),
        "Statistics were published for an empty sample."
    );
    require(
        std::isnan(results.nonfinite_price)
            && std::isnan(results.nonfinite_standard_error),
        "Non-finite moment sums were published as statistics."
    );
    require(
        std::isnan(results.infinite_price)
            && std::isnan(results.infinite_standard_error),
        "Infinite moment sums were published as statistics."
    );
    require(
        results.fp64_prediction > 1.0,
        "LSM prediction coefficients were narrowed to FP32."
    );
    require(
        results.fp64_exercise_cashflow == 1.0f
            && results.narrowed_exercise_cashflow == 0.25f
            && results.fp64_exercise_cashflow
                - results.narrowed_exercise_cashflow == 0.75f,
        "The FP64 LSM exercise boundary did not preserve its price decision."
    );
    require(
        results.insufficient_candidate_invalidated == 0U
            && results.insufficient_candidate_price == 2.0f
            && results.insufficient_candidate_standard_error == 0.1f,
        "An explicitly skipped underdetermined regression invalidated a price."
    );
    require(
        results.fatal_regression_invalidated == 1U
            && std::isnan(results.fatal_regression_price)
            && std::isnan(results.fatal_regression_standard_error),
        "A fatal regression failure remained publishable."
    );
    require(
        std::fabs(
            results.nelson_siegel_forward
            - nelson_siegel::instantaneous_forward_formula(
                0.02, -0.01, 0.03, 1.25
            )
        ) < 1.0e-14
            && std::fabs(
                results.svensson_forward
                - svensson::instantaneous_forward_formula(
                    0.02, -0.01, 0.03, 0.015, 1.5, 4.0, 2.25
                )
            ) < 1.0e-14
            && std::fabs(
                results.stationary_volatility
                - mean_reverting_gaussian::
                    volatility_from_stationary_deviation(0.025f, 0.7f)
            ) < 1.0e-7f,
        "Host and device canonical fixed-income formulas disagree."
    );
    require(
        std::isfinite(
            results.small_mean_reversion_model
                .volatility_standard_deviation
        )
            && std::isfinite(
                results.small_mean_reversion_model
                    .endpoint_increment_correlation
            )
            && std::fabs(
                results.small_mean_reversion_model
                    .endpoint_increment_correlation
                - 1.0f
            ) < 1.0e-5f,
        "The small-kappa Schobel-Zhu transition is unstable."
    );
    for (std::size_t index = 0U;
         index < kSchobelZhuGridSize;
         ++index) {
        const SchobelZhuGridPoint point = schobel_zhu_grid_point(index);
        const double mean_reversion =
            static_cast<double>(point.mean_reversion);
        const double delta_t = static_cast<double>(point.delta_t);
        const double scaled_time = mean_reversion * delta_t;
        const double endpoint_variance =
            -std::expm1(-2.0 * scaled_time) / (2.0 * mean_reversion);
        const double expected_decay = std::exp(-scaled_time);
        const double expected_standard_deviation =
            0.35 * std::sqrt(endpoint_variance);
        const double expected_endpoint_correlation =
            -std::expm1(-scaled_time)
            / (mean_reversion * std::sqrt(delta_t * endpoint_variance));
        const schobel_zhu::PreparedModel& actual =
            results.schobel_zhu_grid[index];
        require(
            std::fabs(
                static_cast<double>(actual.volatility_decay)
                - expected_decay
            ) <= 3.0e-6 * std::max(1.0, std::fabs(expected_decay))
                && std::fabs(
                    static_cast<double>(
                        actual.volatility_standard_deviation
                    ) - expected_standard_deviation
                ) <= 3.0e-6 * std::max(
                    1.0, std::fabs(expected_standard_deviation)
                )
                && std::fabs(
                    static_cast<double>(
                        actual.endpoint_increment_correlation
                    ) - expected_endpoint_correlation
                ) <= 3.0e-6,
            "Schobel-Zhu preparation differs from the FP64 reference grid."
        );
    }
    require(
        results.schobel_zhu_philox_mapping_preserved == 1U,
        "Schobel-Zhu changed its three-normal Philox mapping."
    );
}

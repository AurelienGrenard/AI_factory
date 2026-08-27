// Generic sampling contract: policies, sources, calendars and execution modes.
#include "common/check_cuda.cuh"
#include "common/sample.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/markovian/black_scholes/sample.cuh"

// Generic generated-source tests instantiate the dynamics in this CUDA TU.
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace black_scholes = model::equity::black_scholes;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename Exception, typename Function>
void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

template<typename T>
T* allocate_device(std::size_t count, const char* operation) {
    T* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, count * sizeof(T)), operation);
    return pointer;
}

template<typename T>
std::vector<T> copy_to_host(
    const T* device_values,
    std::size_t count,
    const char* operation
) {
    std::vector<T> values(count);
    check_cuda(
        cudaMemcpy(
            values.data(),
            device_values,
            count * sizeof(T),
            cudaMemcpyDeviceToHost
        ),
        operation
    );
    return values;
}

template<typename T>
void copy_to_device(
    T* device_values,
    const std::vector<T>& values,
    const char* operation
) {
    check_cuda(
        cudaMemcpy(
            device_values,
            values.data(),
            values.size() * sizeof(T),
            cudaMemcpyHostToDevice
        ),
        operation
    );
}

bool differs(const std::vector<float>& left, const std::vector<float>& right) {
    return std::memcmp(
        left.data(),
        right.data(),
        left.size() * sizeof(float)
    ) != 0;
}

struct BlackScholesGenerationConfiguration {
    sample::UniformBounds risk_free_rate;
    sample::UniformBounds dividend_yield;
    sample::UniformBounds volatility;
};

struct BlackScholesParameterSamplingPolicy {
    using Configuration = BlackScholesGenerationConfiguration;
    using Parameters = black_scholes::ModelParameters;

    static void validate(const Configuration& configuration) {
        sample::validate_uniform_bounds(
            configuration.risk_free_rate,
            "generated Black-Scholes risk-free rate"
        );
        sample::validate_uniform_bounds(
            configuration.dividend_yield,
            "generated Black-Scholes dividend yield"
        );
        sample::validate_uniform_bounds(
            configuration.volatility,
            "generated Black-Scholes volatility"
        );
        if (!(configuration.volatility.minimum > 0.0f)) {
            throw std::invalid_argument(
                "Generated Black-Scholes volatility must be positive."
            );
        }
    }

    __device__ __forceinline__ static float uniform(
        sample::UniformBounds bounds,
        philox::UniformSequence& uniforms
    ) {
        return fmaf(
            bounds.maximum - bounds.minimum,
            uniforms.next(),
            bounds.minimum
        );
    }

    __device__ __forceinline__ static Parameters sample(
        const Configuration& configuration,
        philox::PhiloxKey key,
        std::uint64_t parameter_index
    ) {
        philox::UniformSequence uniforms(key, parameter_index);
        return {
            1.0f,
            uniform(configuration.risk_free_rate, uniforms),
            uniform(configuration.dividend_yield, uniforms),
            uniform(configuration.volatility, uniforms),
        };
    }
};

static_assert(sample::ParameterSamplingPolicy<
    BlackScholesParameterSamplingPolicy
>);

using TerminalSchedule = simulation::ExactTransitionTerminalSchedule<
    black_scholes::DynamicsPolicy
>;
using TerminalPolicy = sample::ModelSamplingPolicy<
    TerminalSchedule,
    sample::SpotSampleObservation<black_scholes::DynamicsPolicy>
>;
using ThreeDateSchedule = simulation::ExactTransitionCalendarSchedule<
    black_scholes::DynamicsPolicy,
    3U
>;
using ThreeDatePolicy = sample::ModelSamplingPolicy<
    ThreeDateSchedule,
    sample::SpotSampleObservation<black_scholes::DynamicsPolicy>
>;

static_assert(sample::SamplingPolicy<TerminalPolicy>);
static_assert(sample::SamplingPolicy<ThreeDatePolicy>);

void test_host_contract_validation() {
    require(
        sample::kSampleDayFraction == 1.0f / 252.0f,
        "The sample day convention is not 1/252."
    );
    require(
        sample::kSampleFixedStepDt == 1.0f / 504.0f
            && sample::kSampleSimulationStepsPerDay == 2U,
        "The discretized sample grid is not two steps per day."
    );
    require_throws<std::invalid_argument>(
        [] { (void)sample::sample_count(0U, 1U); },
        "A zero parameter count was accepted."
    );
    require_throws<std::overflow_error>(
        [] {
            (void)sample::sample_count(
                std::numeric_limits<std::size_t>::max(),
                2U
            );
        },
        "A sample-count overflow was accepted."
    );
    require_throws<std::invalid_argument>(
        [] { simulation::validate_calendar(simulation::MaturityCalendar{0U}); },
        "A zero-day maturity was accepted."
    );
    require_throws<std::invalid_argument>(
        [] {
            sample::UniformMaturityCalendarSource{{504U, 63U}, nullptr}
                .validate(1U);
        },
        "Reversed maturity-day bounds were accepted."
    );
    require_throws<std::invalid_argument>(
        [] {
            using Calendar = ThreeDateSchedule::Calendar;
            sample::RandomIncreasingCalendarSource<Calendar, 3U>{
                {63U, 80U, 10U}, nullptr, 0U
            }.validate(1U);
        },
        "An infeasible increasing calendar was accepted."
    );
}

void validate_finite_positive(const std::vector<float>& values) {
    for (float value : values) {
        require(
            std::isfinite(value) && value > 0.0f,
            "A sampled Black-Scholes spot is not finite and positive."
        );
    }
}

void test_model_bindings() {
    constexpr std::size_t parameter_count = 16U;
    const std::vector<black_scholes::ModelParameters> parameters(
        parameter_count,
        {1.0f, 0.03f, 0.01f, 0.25f}
    );
    auto* device_parameters = allocate_device<black_scholes::ModelParameters>(
        parameter_count,
        "sample parameter allocation"
    );
    copy_to_device(
        device_parameters,
        parameters,
        "sample parameter copy"
    );

    float* first = allocate_device<float>(
        parameter_count,
        "first terminal sample allocation"
    );
    float* replay = allocate_device<float>(
        parameter_count,
        "replay terminal sample allocation"
    );
    float* batched = allocate_device<float>(
        parameter_count,
        "batched terminal sample allocation"
    );

    black_scholes::launch_black_scholes_terminal_samples_cuda(
        device_parameters, parameter_count, 1U, 252U, 0U, parameter_count,
        128U, 4U, 881000001ULL, first
    );
    black_scholes::launch_black_scholes_terminal_samples_cuda(
        device_parameters, parameter_count, 1U, 252U, 0U, parameter_count,
        256U, 2U, 881000001ULL, replay
    );
    black_scholes::launch_black_scholes_terminal_samples_cuda(
        device_parameters, parameter_count, 1U, 252U, 0U, 8U,
        128U, 2U, 881000001ULL, batched
    );
    black_scholes::launch_black_scholes_terminal_samples_cuda(
        device_parameters, parameter_count, 1U, 252U, 8U, 8U,
        512U, 1U, 881000001ULL, batched
    );
    check_cuda(cudaDeviceSynchronize(), "model-binding sample synchronize");

    const auto first_host = copy_to_host(
        first, parameter_count, "first terminal sample copy"
    );
    const auto replay_host = copy_to_host(
        replay, parameter_count, "replay terminal sample copy"
    );
    const auto batched_host = copy_to_host(
        batched, parameter_count, "batched terminal sample copy"
    );
    require(
        std::memcmp(
            first_host.data(),
            replay_host.data(),
            parameter_count * sizeof(float)
        ) == 0,
        "Samples changed with the CUDA launch geometry."
    );
    require(
        std::memcmp(
            first_host.data(),
            batched_host.data(),
            parameter_count * sizeof(float)
        ) == 0,
        "Samples changed with batch boundaries."
    );
    validate_finite_positive(first_host);

    constexpr std::size_t packaged_parameter_count = 4U;
    constexpr std::size_t paths_per_parameter = 250U;
    constexpr std::size_t packaged_sample_count =
        packaged_parameter_count * paths_per_parameter;
    float* packaged = allocate_device<float>(
        packaged_sample_count,
        "packaged sample allocation"
    );
    black_scholes::launch_black_scholes_terminal_samples_cuda(
        device_parameters, packaged_parameter_count, paths_per_parameter,
        252U, 0U, packaged_sample_count, 128U, 4U, 881000002ULL, packaged
    );
    check_cuda(cudaDeviceSynchronize(), "packaged sample synchronize");
    const auto packaged_host = copy_to_host(
        packaged,
        packaged_sample_count,
        "packaged sample copy"
    );
    validate_finite_positive(packaged_host);
    for (std::size_t parameter = 0U;
         parameter < packaged_parameter_count;
         ++parameter) {
        const auto first_path = packaged_host.begin()
            + parameter * paths_per_parameter;
        require(
            std::adjacent_find(
                first_path,
                first_path + paths_per_parameter,
                std::not_equal_to<float>{}
            ) != first_path + paths_per_parameter,
            "Conditional paths inside one parameter package are identical."
        );
    }

    constexpr std::uint32_t observation_count = 3U;
    float* calendar = allocate_device<float>(
        observation_count * parameter_count,
        "calendar sample allocation"
    );
    black_scholes::launch_black_scholes_calendar_samples_cuda(
        device_parameters, parameter_count, 1U, 21U, 21U,
        observation_count, 0U, parameter_count, 128U, 4U, 881000003ULL,
        calendar
    );
    check_cuda(cudaDeviceSynchronize(), "calendar sample synchronize");
    validate_finite_positive(copy_to_host(
        calendar,
        observation_count * parameter_count,
        "calendar sample copy"
    ));

    check_cuda(cudaFree(calendar), "calendar sample free");
    check_cuda(cudaFree(packaged), "packaged sample free");
    check_cuda(cudaFree(batched), "batched sample free");
    check_cuda(cudaFree(replay), "replay sample free");
    check_cuda(cudaFree(first), "first sample free");
    check_cuda(cudaFree(device_parameters), "sample parameter free");
}

struct GeneratedRun {
    std::vector<float> spots;
    std::vector<std::uint32_t> maturity_days;
};

GeneratedRun generate_terminal_rows(
    sample::SamplingSeeds seeds,
    sample::SampleExecutionStrategy strategy,
    unsigned int threads_per_block,
    std::size_t block_count
) {
    constexpr std::size_t sample_count = 1024U;
    float* device_spots = allocate_device<float>(
        sample_count,
        "generated spot allocation"
    );
    std::uint32_t* device_days = allocate_device<std::uint32_t>(
        sample_count,
        "generated maturity-day allocation"
    );
    const sample::GeneratedParameterSource<
        BlackScholesParameterSamplingPolicy
    > parameters{
        {{0.001f, 0.08f}, {0.0f, 0.06f}, {0.08f, 0.45f}}
    };
    const sample::UniformMaturityCalendarSource maturities{
        {63U, 504U},
        device_days,
    };
    sample::launch_samples_cuda<TerminalPolicy>(
        parameters,
        maturities,
        {sample_count, 1U, 0U, sample_count},
        sample::kExactSampleTimeConfiguration,
        {threads_per_block, block_count},
        seeds,
        {device_spots},
        "black_scholes.generated_samples",
        "terminal",
        "generated terminal sample kernel",
        strategy
    );
    check_cuda(cudaDeviceSynchronize(), "generated terminal synchronize");
    GeneratedRun result{
        copy_to_host(
            device_spots,
            sample_count,
            "generated spot copy"
        ),
        copy_to_host(
            device_days,
            sample_count,
            "generated maturity-day copy"
        ),
    };
    check_cuda(cudaFree(device_days), "generated maturity-day free");
    check_cuda(cudaFree(device_spots), "generated spot free");
    return result;
}

void test_generated_sources_and_domains() {
    const sample::SamplingSeeds seeds{
        710000101ULL,
        810000101ULL,
        910000101ULL,
    };
    const GeneratedRun grid_stride = generate_terminal_rows(
        seeds,
        sample::SampleExecutionStrategy::thread_grid_stride,
        128U,
        4U
    );
    const GeneratedRun parameter_blocks = generate_terminal_rows(
        seeds,
        sample::SampleExecutionStrategy::parameter_block,
        256U,
        8U
    );
    require(
        grid_stride.maturity_days == parameter_blocks.maturity_days,
        "Generated maturities changed with execution strategy."
    );
    require(
        std::memcmp(
            grid_stride.spots.data(),
            parameter_blocks.spots.data(),
            grid_stride.spots.size() * sizeof(float)
        ) == 0,
        "Generated samples changed with execution strategy."
    );
    for (std::uint32_t day : grid_stride.maturity_days) {
        require(
            day >= 63U && day <= 504U,
            "A generated maturity lies outside its day bounds."
        );
        const float years = static_cast<float>(day)
            * sample::kSampleDayFraction;
        require(
            std::fabs(
                years - static_cast<float>(day) / 252.0f
            ) <= 1.0e-6f,
            "Day-to-year conversion is not the canonical 1/252 mapping."
        );
    }
    validate_finite_positive(grid_stride.spots);

    const GeneratedRun changed_parameters = generate_terminal_rows(
        {seeds.parameters + 1U, seeds.schedule, seeds.dynamics},
        sample::SampleExecutionStrategy::thread_grid_stride,
        128U,
        4U
    );
    require(
        grid_stride.maturity_days == changed_parameters.maturity_days,
        "The parameter RNG domain changed generated maturities."
    );
    require(
        differs(grid_stride.spots, changed_parameters.spots),
        "Changing the parameter seed did not change samples."
    );

    const GeneratedRun changed_schedule = generate_terminal_rows(
        {seeds.parameters, seeds.schedule + 1U, seeds.dynamics},
        sample::SampleExecutionStrategy::thread_grid_stride,
        128U,
        4U
    );
    require(
        grid_stride.maturity_days != changed_schedule.maturity_days,
        "Changing the schedule seed did not change maturities."
    );

    const GeneratedRun changed_dynamics = generate_terminal_rows(
        {seeds.parameters, seeds.schedule, seeds.dynamics + 1U},
        sample::SampleExecutionStrategy::thread_grid_stride,
        128U,
        4U
    );
    require(
        grid_stride.maturity_days == changed_dynamics.maturity_days,
        "The dynamics RNG domain changed generated maturities."
    );
    require(
        differs(grid_stride.spots, changed_dynamics.spots),
        "Changing the dynamics seed did not change samples."
    );
}

void test_random_increasing_calendar() {
    constexpr std::size_t sample_count = 256U;
    constexpr std::size_t observation_count = 3U;
    float* device_spots = allocate_device<float>(
        observation_count * sample_count,
        "random-calendar spot allocation"
    );
    std::uint32_t* device_days = allocate_device<std::uint32_t>(
        observation_count * sample_count,
        "random-calendar day allocation"
    );
    const sample::GeneratedParameterSource<
        BlackScholesParameterSamplingPolicy
    > parameters{
        {{0.001f, 0.08f}, {0.0f, 0.06f}, {0.08f, 0.45f}}
    };
    using Calendar = typename ThreeDateSchedule::Calendar;
    const sample::RandomIncreasingCalendarSource<Calendar, 3U> calendars{
        {63U, 504U, 21U},
        device_days,
        sample_count,
    };
    sample::launch_samples_cuda<ThreeDatePolicy>(
        parameters,
        calendars,
        {sample_count, 1U, 0U, sample_count},
        sample::kExactSampleTimeConfiguration,
        {128U, 4U},
        {710000201ULL, 810000201ULL, 910000201ULL},
        {device_spots},
        "black_scholes.generated_samples",
        "random_calendar",
        "generated random-calendar sample kernel"
    );
    check_cuda(cudaDeviceSynchronize(), "random-calendar synchronize");
    const auto days = copy_to_host(
        device_days,
        observation_count * sample_count,
        "random-calendar day copy"
    );
    validate_finite_positive(copy_to_host(
        device_spots,
        observation_count * sample_count,
        "random-calendar spot copy"
    ));
    for (std::size_t sample_index = 0U;
         sample_index < sample_count;
         ++sample_index) {
        std::uint32_t previous_day = 0U;
        for (std::size_t observation = 0U;
             observation < observation_count;
             ++observation) {
            const std::uint32_t day =
                days[observation * sample_count + sample_index];
            require(
                day >= 63U && day <= 504U,
                "A random observation day lies outside its bounds."
            );
            if (observation != 0U) {
                require(
                    day - previous_day >= 21U,
                    "A random calendar violates its minimum day interval."
                );
            }
            previous_day = day;
        }
    }
    check_cuda(cudaFree(device_days), "random-calendar day free");
    check_cuda(cudaFree(device_spots), "random-calendar spot free");
}

}  // namespace

int main() {
    test_host_contract_validation();

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "sample test cudaGetDeviceCount");

    test_model_bindings();
    test_generated_sources_and_domains();
    test_random_increasing_calendar();
}

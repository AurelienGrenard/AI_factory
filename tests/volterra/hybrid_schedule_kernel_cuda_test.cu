// Verify every rounded Volterra observation reaches its handler and final payoff.
#include "common/check_cuda.cuh"
#include "common/volterra/hybrid_schedule.cuh"
#include "product/athena_autocall/pricing_policy.cuh"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
namespace wb = ai_factory::workbench;
namespace volterra = wb::volterra;
struct Result { unsigned int observations; unsigned int final_step; float payoff; bool ordered; };
struct Handler {
    wb::product::AthenaAutocallPathPolicy::Handler payoff;
    unsigned int count = 0U;
    bool ordered = true;
    __device__ bool on_initial_state(float spot) { return payoff.on_initial_value(spot); }
    __device__ bool on_observation(unsigned int observation, float spot) {
        ordered = ordered && observation == count;
        ++count;
        return payoff.on_observation(observation, spot);
    }
};

template<typename Schedule>
__global__ void exercise(typename Schedule::Calendar calendar,
                         volterra::HybridTimeConfiguration time, unsigned int steps,
                         unsigned int observations, Result* output) {
    const auto schedule = Schedule::prepare(calendar, time, steps);
    auto cursor = Schedule::make_cursor(schedule);
    Handler handler{{observations, 10.0f, 0.8f, 0.01f, 1.0f}};
    bool keep_running = Schedule::on_initial_state(schedule, cursor, 1.0f, handler);
    unsigned int final_step = 0;
    for (unsigned int step = 0; step < steps && keep_running; ++step) {
        keep_running = Schedule::on_step(schedule, cursor, step, 1.0f, handler);
        if (!keep_running) final_step = step + 1;
    }
    *output = {handler.count, final_step, handler.payoff.discounted_payoff, handler.ordered};
}

template<typename Schedule>
void check(typename Schedule::Calendar calendar, unsigned int observations, float dt) {
    using wb::check_cuda;
    const volterra::HybridTimeConfiguration time{1.0f/252.0f, dt};
    const unsigned int steps = Schedule::execution_step_count(calendar, time);
    Result* device = nullptr;
    check_cuda(cudaMalloc(&device, sizeof(Result)), "Volterra schedule allocation");
    check_cuda(cudaMemset(device, 0, sizeof(Result)), "Volterra schedule storage initialization");
    exercise<Schedule><<<1, 1>>>(calendar, time, steps, observations, device);
    check_cuda(cudaGetLastError(), "Volterra schedule launch");
    Result actual{};
    check_cuda(cudaMemcpy(&actual, device, sizeof(Result), cudaMemcpyDeviceToHost), "Volterra schedule copy");
    check_cuda(cudaFree(device), "Volterra schedule release");
    if (actual.observations != observations || actual.final_step != steps
        || actual.payoff != 1.0f || !actual.ordered) {
        throw std::runtime_error("Volterra lost a contractual event or its final payment");
    }
}
}  // namespace

int main() {
    int count = 0;
    const auto availability = cudaGetDeviceCount(&count);
    if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver || count == 0) return 77;
    wb::check_cuda(availability, "Volterra schedule device discovery");
    for (float dt : {1.0f/504.0f, 1.0f/378.0f, 1.0f/63.0f}) {
        check<volterra::RegularHybridSchedule>({1U, 5U}, 5U, dt);
        check<volterra::StubbedRegularHybridSchedule>({1U, 1U, 5U}, 5U, dt);
        check<volterra::StubbedRegularHybridSchedule>({1U, 2U, 5U}, 5U, dt);
        check<volterra::CalendarHybridSchedule<2U>>({{1U, 4U}}, 2U, dt);
    }
    // Date projection is cumulative, positive and exact on the canonical grid.
    const std::array<unsigned int, 5> projected{2U, 3U, 5U, 6U, 8U};
    for (unsigned int day = 1; day <= 5; ++day) {
        if (volterra::rounded_observation_step(day, 5U, 8U) != projected[day-1]
            || volterra::rounded_observation_step(day, 5U, 10U) != 2U*day
            || volterra::rounded_observation_step(day, 5U, 1U) != 1U) {
            throw std::runtime_error("Volterra cumulative grid projection is incorrect");
        }
    }
    std::cout << "12 schedules: all events ordered; final Athena payment delivered\n";
}

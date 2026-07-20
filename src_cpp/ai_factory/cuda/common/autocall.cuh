#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cmath>

namespace ai_factory::cuda::autocall_detail {

#ifdef __CUDACC__
#define AI_FACTORY_HOST_DEVICE __host__ __device__
#else
#define AI_FACTORY_HOST_DEVICE
#endif

struct PathMetrics {
    double discounted_payoff = 0.0;
    double autocall = 0.0;
    double autocall_time = 0.0;
    double coupon_payment_frequency = 0.0;
    double total_coupon = 0.0;
    double capital_loss = 0.0;
    double loss_redemption = 0.0;
};

struct PathState {
    std::size_t unpaid_coupon_count = 0U;
    std::size_t coupon_payment_count = 0U;
    double discounted_payoff = 0.0;
    double total_coupon = 0.0;
    bool called = false;
};

AI_FACTORY_HOST_DEVICE inline bool observe(
    const AutocallTerms& terms,
    double performance,
    std::size_t observation,
    double observation_time,
    double rate,
    PathState& state
) {
    ++state.unpaid_coupon_count;
    double cash = 0.0;
    if (performance >= terms.coupon_barrier) {
        cash = terms.coupon_rate * static_cast<double>(state.unpaid_coupon_count);
        state.total_coupon += cash;
        state.unpaid_coupon_count = 0U;
        ++state.coupon_payment_count;
    }
    if (observation >= terms.first_autocall_observation
        && performance >= terms.autocall_barrier) {
        cash += 1.0;
        state.called = true;
    }
    state.discounted_payoff += exp(-rate * observation_time) * cash;
    return state.called;
}

AI_FACTORY_HOST_DEVICE inline PathMetrics finish(
    const AutocallTerms& terms,
    double terminal_performance,
    double maturity,
    double rate,
    std::size_t call_observation,
    PathState state
) {
    PathMetrics metrics{};
    if (!state.called) {
        const bool loss = terminal_performance < terms.protection_barrier;
        const double redemption = loss ? terminal_performance : 1.0;
        state.discounted_payoff += exp(-rate * maturity) * redemption;
        metrics.capital_loss = loss ? 1.0 : 0.0;
        metrics.loss_redemption = loss ? redemption : 0.0;
    } else {
        metrics.autocall = 1.0;
        metrics.autocall_time =
            maturity * static_cast<double>(call_observation)
            / static_cast<double>(terms.observation_count);
    }
    metrics.discounted_payoff = state.discounted_payoff;
    metrics.coupon_payment_frequency =
        static_cast<double>(state.coupon_payment_count)
        / static_cast<double>(terms.observation_count);
    metrics.total_coupon = state.total_coupon;
    return metrics;
}

#undef AI_FACTORY_HOST_DEVICE

}  // namespace ai_factory::cuda::autocall_detail

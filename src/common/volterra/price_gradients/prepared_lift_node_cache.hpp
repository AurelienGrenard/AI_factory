// Host cache for model-node dynamics of a fixed-factor Volterra lift.
#pragma once

#include "common/volterra/fractional_kernel_approximation.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace ai_factory::workbench::volterra::price_gradients {

// Policy supplies Model, Prepared and prepare(model, fitted_kernel, horizon, dt).
// The model key is byte-exact: a represented float bump must never silently
// alias a different prepared transition. The fitted kernel itself depends only
// on H while horizon and dt remain fixed for the entire sensitivity request.
template<std::size_t FactorCount, typename Policy>
class PreparedLiftNodeCache {
public:
    using Model = typename Policy::Model;
    using Prepared = typename Policy::Prepared;
    using Kernel = ExponentialKernel<FactorCount>;

    static_assert(std::is_trivially_copyable_v<Model>);
    static_assert(std::is_trivially_copyable_v<Prepared>);

    PreparedLiftNodeCache(float approximation_horizon, float dt)
        : horizon_(approximation_horizon), dt_(dt) {
        if (!std::isfinite(horizon_) || !std::isfinite(dt_)
            || !(horizon_ > dt_) || !(dt_ > 0.0f)) {
            throw std::invalid_argument(
                "Lift-node approximation horizon must exceed its time step."
            );
        }
    }

    // Fit distinct H endpoints independently before preparing model rows.
    // Keeping all map updates on this thread makes index() deterministic.
    void prefit(std::span<const float> hurst_values) {
        struct Pending { std::uint32_t bits; float value; };
        std::vector<Pending> pending;
        std::unordered_set<std::uint32_t> seen;
        for (const float value : hurst_values) {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            if (kernels_by_hurst_.contains(bits) || !seen.insert(bits).second)
                continue;
            pending.push_back({bits, value});
        }
        if (pending.empty()) return;
        std::vector<Kernel> fitted(pending.size());
        const auto hardware = std::max(1U, std::thread::hardware_concurrency());
        const auto workers = std::min<std::size_t>(
            pending.size(), std::min<std::size_t>(hardware, 8U)
        );
        std::atomic<std::size_t> cursor{0U};
        std::vector<std::exception_ptr> failures(workers);
        auto fit = [&](std::size_t worker) {
            try {
                while (true) {
                    const auto item = cursor.fetch_add(1U);
                    if (item >= pending.size()) break;
                    fitted[item] = fit_positive_fractional_kernel_l2<
                        FactorCount
                    >(pending[item].value, horizon_, dt_);
                }
            } catch (...) {
                failures[worker] = std::current_exception();
            }
        };
        if (workers == 1U) {
            fit(0U);
        } else {
            std::vector<std::jthread> threads;
            threads.reserve(workers);
            for (std::size_t worker = 0U; worker < workers; ++worker)
                threads.emplace_back(fit, worker);
            for (auto& thread : threads) thread.join();
        }
        for (const auto& failure : failures)
            if (failure) std::rethrow_exception(failure);
        for (std::size_t item = 0U; item < pending.size(); ++item)
            kernels_by_hurst_.emplace(pending[item].bits, fitted[item]);
    }

    std::uint32_t index(const Model& model) {
        const std::string key(
            reinterpret_cast<const char*>(&model), sizeof(Model)
        );
        if (const auto found = prepared_indices_.find(key);
            found != prepared_indices_.end()) {
            return found->second;
        }
        if (prepared_.size() >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Lift-node prepared index overflow.");
        }
        const auto hurst = std::bit_cast<std::uint32_t>(
            model.hurst_exponent
        );
        auto kernel = kernels_by_hurst_.find(hurst);
        if (kernel == kernels_by_hurst_.end()) {
            kernel = kernels_by_hurst_.emplace(
                hurst,
                fit_positive_fractional_kernel_l2<FactorCount>(
                    model.hurst_exponent, horizon_, dt_
                )
            ).first;
        }
        const auto index = static_cast<std::uint32_t>(prepared_.size());
        prepared_.push_back(Policy::prepare(
            model, kernel->second, horizon_, dt_
        ));
        prepared_indices_.emplace(key, index);
        return index;
    }

    bool contains_prepared(const Model& model) const {
        const std::string key(
            reinterpret_cast<const char*>(&model), sizeof(Model)
        );
        return prepared_indices_.contains(key);
    }

    // The compatibility price-delta launchers receive already fitted dynamics.
    // Register them without refitting H or changing the caller's lift horizon.
    std::uint32_t insert_prepared(const Model& model, const Prepared& value) {
        const std::string key(
            reinterpret_cast<const char*>(&model), sizeof(Model)
        );
        if (const auto found = prepared_indices_.find(key);
            found != prepared_indices_.end()) {
            return found->second;
        }
        if (prepared_.size() >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Lift-node prepared index overflow.");
        }
        const auto index = static_cast<std::uint32_t>(prepared_.size());
        prepared_.push_back(value);
        prepared_indices_.emplace(key, index);
        return index;
    }

    const std::vector<Prepared>& values() const noexcept { return prepared_; }
    std::size_t unique_kernel_count() const noexcept {
        return kernels_by_hurst_.size();
    }

private:
    float horizon_;
    float dt_;
    std::unordered_map<std::string, std::uint32_t> prepared_indices_;
    std::unordered_map<std::uint32_t, Kernel> kernels_by_hurst_;
    std::vector<Prepared> prepared_;
};

}  // namespace ai_factory::workbench::volterra::price_gradients

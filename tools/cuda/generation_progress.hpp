// Non-blocking sidecar progress for long-running offline CUDA generators.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace ai_factory::workbench::offline::cuda {

class GenerationProgress {
public:
    explicit GenerationProgress(std::size_t total_prices) noexcept
        : total_prices_(total_prices),
          started_(std::chrono::steady_clock::now()) {
        try {
            const char* configured = std::getenv(
                "AI_FACTORY_GENERATION_PROGRESS"
            );
            if (configured == nullptr || *configured == '\0'
                || total_prices_ == 0U) {
                return;
            }
            if (cudaGetDevice(&device_) != cudaSuccess) return;
            path_ = configured;
            if (const char* journal = std::getenv(
                    "AI_FACTORY_GENERATION_PROGRESS_LOG"
                ); journal != nullptr && *journal != '\0') {
                journal_path_ = journal;
            }
            enabled_ = true;
            write_sidecar("running", 0U);
            worker_ = std::thread([this] { monitor(); });
        } catch (...) {
            enabled_ = false;
        }
    }

    ~GenerationProgress() noexcept {
        stop_monitor();
        clear_events();
        if (enabled_ && !finished_) {
            write_sidecar("stopped", completed_prices_.load());
        }
    }

    GenerationProgress(const GenerationProgress&) = delete;
    GenerationProgress& operator=(const GenerationProgress&) = delete;

    bool enabled() const noexcept { return enabled_; }

    // Record a marker on the default stream. The monitoring thread observes
    // completion with cudaEventQuery; this call never waits for the GPU.
    void record_cuda_progress(std::size_t completed_prices) noexcept {
        if (!enabled_ || !should_record(completed_prices)) return;
        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming)
            != cudaSuccess) {
            return;
        }
        if (cudaEventRecord(event) != cudaSuccess) {
            cudaEventDestroy(event);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            events_.push_back({event, std::min(completed_prices, total_prices_)});
        }
        condition_.notify_one();
    }

    // Use this only at an existing host-side completion boundary.
    void record_host_progress(std::size_t completed_prices) noexcept {
        if (!enabled_) return;
        publish_completed(std::min(completed_prices, total_prices_));
    }

    // Call after the existing runner synchronization has completed.
    void complete() noexcept {
        if (!enabled_) return;
        stop_monitor();
        clear_events();
        completed_prices_.store(total_prices_);
        finished_ = true;
        write_sidecar("complete", total_prices_);
    }

private:
    struct Marker {
        cudaEvent_t event;
        std::size_t completed_prices;
    };

    bool should_record(std::size_t completed_prices) noexcept {
        completed_prices = std::min(completed_prices, total_prices_);
        if (completed_prices < next_marker_) return false;
        const std::size_t regular_step = std::max<std::size_t>(
            1U, (total_prices_ + 999U) / 1000U
        );
        next_marker_ = completed_prices + (
            completed_prices < 16U ? 1U : regular_step
        );
        return true;
    }

    void monitor() noexcept {
        if (cudaSetDevice(device_) != cudaSuccess) return;
        while (!stopping_.load()) {
            Marker marker{};
            bool has_marker = false;
            {
                std::unique_lock lock(mutex_);
                condition_.wait_for(lock, std::chrono::milliseconds(250), [this] {
                    return stopping_.load() || !events_.empty();
                });
                if (!events_.empty()) {
                    marker = events_.front();
                    has_marker = true;
                }
            }
            if (!has_marker) {
                maybe_write_running();
                continue;
            }
            const cudaError_t status = cudaEventQuery(marker.event);
            if (status == cudaErrorNotReady) {
                maybe_write_running();
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            {
                std::lock_guard lock(mutex_);
                if (!events_.empty() && events_.front().event == marker.event) {
                    events_.pop_front();
                }
            }
            cudaEventDestroy(marker.event);
            if (status == cudaSuccess) {
                publish_completed(marker.completed_prices);
                maybe_write_running();
            }
        }
    }

    void publish_completed(std::size_t value) noexcept {
        std::size_t previous = completed_prices_.load();
        while (previous < value
               && !completed_prices_.compare_exchange_weak(previous, value)) {}
    }

    void maybe_write_running() noexcept {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard lock(write_mutex_);
        if (now - last_write_ < std::chrono::seconds(10)) return;
        last_write_ = now;
        write_sidecar("running", completed_prices_.load());
    }

    void write_sidecar(
        const char* state, std::size_t completed_prices
    ) noexcept {
        try {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started_
            ).count();
            const double rate = elapsed > 0.0
                ? static_cast<double>(completed_prices) / elapsed : 0.0;
            const double percent = total_prices_ == 0U ? 100.0
                : 100.0 * static_cast<double>(completed_prices)
                    / static_cast<double>(total_prices_);
            const double unix_time = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            std::ostringstream record;
            record << std::setprecision(12)
                   << "{\"event\":\"progress\",\"unix_time\":" << unix_time
                   << ",\"state\":\"" << state << "\""
                   << ",\"completed_prices\":" << completed_prices
                   << ",\"total_prices\":" << total_prices_
                   << ",\"percent\":" << percent
                   << ",\"elapsed_seconds\":" << elapsed
                   << ",\"prices_per_second\":" << rate
                   << ",\"estimated_seconds_remaining\":";
            if (rate > 0.0 && completed_prices < total_prices_) {
                record << static_cast<double>(total_prices_ - completed_prices)
                    / rate;
            } else if (completed_prices == total_prices_) {
                record << 0.0;
            } else {
                record << "null";
            }
            record << "}\n";
            const std::filesystem::path temporary = path_.string() + ".tmp";
            std::filesystem::create_directories(path_.parent_path());
            std::ofstream output(temporary, std::ios::trunc);
            output << record.str();
            output.close();
            if (!output) return;
            std::filesystem::rename(temporary, path_);
            if (!journal_path_.empty()) {
                std::ofstream journal(journal_path_, std::ios::app);
                journal << record.str();
            }
        } catch (...) {
            // Observability must never fail or delay the numerical generation.
        }
    }

    void stop_monitor() noexcept {
        if (!worker_.joinable()) return;
        stopping_.store(true);
        condition_.notify_all();
        worker_.join();
    }

    void clear_events() noexcept {
        std::deque<Marker> remaining;
        {
            std::lock_guard lock(mutex_);
            remaining.swap(events_);
        }
        for (const Marker& marker : remaining) {
            cudaEventDestroy(marker.event);
        }
    }

    std::size_t total_prices_ = 0U;
    std::size_t next_marker_ = 1U;
    std::filesystem::path path_;
    std::filesystem::path journal_path_;
    std::chrono::steady_clock::time_point started_;
    std::chrono::steady_clock::time_point last_write_{};
    std::atomic<std::size_t> completed_prices_{0U};
    std::atomic<bool> stopping_{false};
    bool enabled_ = false;
    bool finished_ = false;
    int device_ = 0;
    std::mutex mutex_;
    std::mutex write_mutex_;
    std::condition_variable condition_;
    std::deque<Marker> events_;
    std::thread worker_;
};

inline thread_local GenerationProgress* active_generation_progress = nullptr;

class ScopedGenerationProgress {
public:
    explicit ScopedGenerationProgress(GenerationProgress& progress) noexcept
        : previous_(std::exchange(active_generation_progress, &progress)) {}

    ~ScopedGenerationProgress() {
        active_generation_progress = previous_;
    }

private:
    GenerationProgress* previous_;
};

inline void record_active_host_progress(std::size_t completed_prices) noexcept {
    if (active_generation_progress != nullptr) {
        active_generation_progress->record_host_progress(completed_prices);
    }
}

}  // namespace ai_factory::workbench::offline::cuda

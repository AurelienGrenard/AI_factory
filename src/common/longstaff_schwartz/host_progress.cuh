// Optional host-side batch completion notification for LSM callers.
#pragma once

#include <cstddef>
#include <utility>

namespace ai_factory::workbench::longstaff_schwartz {

using HostProgressCallback = void (*)(std::size_t, void*) noexcept;

struct HostProgressObserver {
    HostProgressCallback callback = nullptr;
    void* context = nullptr;
};

inline thread_local HostProgressObserver active_host_progress{};

class ScopedHostProgress {
public:
    ScopedHostProgress(HostProgressCallback callback, void* context) noexcept
        : previous_(std::exchange(
              active_host_progress, HostProgressObserver{callback, context}
          )) {}

    ~ScopedHostProgress() noexcept { active_host_progress = previous_; }

    ScopedHostProgress(const ScopedHostProgress&) = delete;
    ScopedHostProgress& operator=(const ScopedHostProgress&) = delete;

private:
    HostProgressObserver previous_;
};

inline void report_completed_batch(std::size_t completed_prices) noexcept {
    const auto observer = active_host_progress;
    if (observer.callback != nullptr) {
        observer.callback(completed_prices, observer.context);
    }
}

}  // namespace ai_factory::workbench::longstaff_schwartz

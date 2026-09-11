// Host RAM available to a sample plan, including Linux reclaimable page cache.
#pragma once

#include <cstddef>
#include <fstream>
#include <istream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace ai_factory::workbench::offline::sampling {

inline std::optional<std::size_t> parse_linux_available_memory_bytes(
    std::istream& meminfo
) {
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key;
        fields >> key;
        if (key != "MemAvailable:") continue;
        long long kibibytes = 0;
        std::string unit;
        if (!(fields >> kibibytes >> unit) || kibibytes < 0 || unit != "kB"
            || static_cast<unsigned long long>(kibibytes)
                > std::numeric_limits<std::size_t>::max() / 1024U) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(kibibytes) * 1024U;
    }
    return std::nullopt;
}

inline std::optional<std::size_t> available_host_memory_bytes() {
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    if (const auto available = parse_linux_available_memory_bytes(meminfo)) {
        return available;
    }
    // Older kernels or inaccessible procfs: retain the conservative free-page
    // estimate. Missing telemetry is not interpreted as unlimited measured RAM.
    const long pages = ::sysconf(_SC_AVPHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages > 0L && page_size > 0L
        && static_cast<unsigned long long>(pages)
            <= std::numeric_limits<std::size_t>::max()
                / static_cast<unsigned long long>(page_size)) {
        return static_cast<std::size_t>(pages)
            * static_cast<std::size_t>(page_size);
    }
#endif
    return std::nullopt;
}

}  // namespace ai_factory::workbench::offline::sampling

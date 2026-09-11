// Check Linux available-RAM parsing without depending on current machine load.
#include "tools/sampling/host_memory.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

int main() {
    using ai_factory::workbench::offline::sampling::parse_linux_available_memory_bytes;
    std::istringstream cached(
        "MemTotal: 16000000 kB\nMemFree: 1000 kB\n"
        "MemAvailable: 7500000 kB\nCached: 7000000 kB\n"
    );
    if (parse_linux_available_memory_bytes(cached) != 7500000ULL * 1024U) {
        throw std::runtime_error("Reclaimable cache was excluded from available RAM.");
    }
    std::istringstream empty_memory("MemAvailable: 0 kB\n");
    if (parse_linux_available_memory_bytes(empty_memory) != 0U) {
        throw std::runtime_error("Zero available RAM must remain a measured zero.");
    }
    for (const std::string text : {
             "", "MemFree: 1000 kB\n", "MemAvailable: -1 kB\n",
             "MemAvailable: nope kB\n", "MemAvailable: 123 MB\n",
             "MemAvailable: 123\n", "MemAvailable: 9223372036854775807 kB\n"}) {
        std::istringstream invalid(text);
        if (parse_linux_available_memory_bytes(invalid)) {
            throw std::runtime_error("Invalid RAM telemetry was accepted: " + text);
        }
    }
}

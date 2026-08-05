// Formatting and deduplication for opt-in CUDA kernel diagnostics.
#include "common/cuda_kernel_diagnostics.cuh"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

namespace ai_factory::workbench {
namespace {

// Interpret the usual explicit truth values and reject accidental activation.
bool environment_flag_enabled(const char* value) {
    if (value == nullptr) return false;
    std::string normalized(value);
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        }
    );
    return normalized == "1" || normalized == "true" || normalized == "on";
}

// Build a stable key without querying the GPU for already-seen batches.
std::string report_key(
    const char* kernel_name,
    const char* variant,
    dim3 grid,
    dim3 block,
    std::size_t dynamic_shared_bytes
) {
    std::ostringstream key;
    key << kernel_name << '\n'
        << variant << '\n'
        << grid.x << ',' << grid.y << ',' << grid.z << '\n'
        << block.x << ',' << block.y << ',' << block.z << '\n'
        << dynamic_shared_bytes;
    return key.str();
}

}  // namespace

bool cuda_kernel_diagnostics_enabled() noexcept {
    static const bool enabled = environment_flag_enabled(
        std::getenv("AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS")
    );
    return enabled;
}

bool reserve_cuda_kernel_launch_diagnostics(
    const char* kernel_name,
    const char* variant,
    dim3 grid,
    dim3 block,
    std::size_t dynamic_shared_bytes
) {
    static std::mutex reservation_mutex;
    static std::set<std::string> reserved_reports;
    const std::lock_guard<std::mutex> lock(reservation_mutex);
    return reserved_reports.insert(
        report_key(
            kernel_name,
            variant,
            grid,
            block,
            dynamic_shared_bytes
        )
    ).second;
}

void emit_cuda_kernel_launch_diagnostics(
    const char* kernel_name,
    const char* variant,
    const CudaKernelLaunchDiagnostics& diagnostics
) {
    static std::mutex output_mutex;
    const std::lock_guard<std::mutex> lock(output_mutex);
    const nlohmann::ordered_json report{
        {"type", "cuda_kernel_launch_diagnostics"},
        {"kernel", kernel_name},
        {"variant", variant},
        {"device", {
            {"index", diagnostics.device_index},
            {"name", diagnostics.device_name},
            {"compute_capability",
                std::to_string(diagnostics.compute_capability_major) + "."
                + std::to_string(diagnostics.compute_capability_minor)},
        }},
        {"launch", {
            {"grid_block_count", diagnostics.grid_block_count},
            {"grid", {
                diagnostics.grid_x,
                diagnostics.grid_y,
                diagnostics.grid_z,
            }},
            {"threads_per_block", diagnostics.threads_per_block},
            {"block", {
                diagnostics.block_x,
                diagnostics.block_y,
                diagnostics.block_z,
            }},
            {"dynamic_shared_bytes_per_block",
                diagnostics.dynamic_shared_bytes_per_block},
        }},
        {"resources", {
            {"registers_per_thread", diagnostics.registers_per_thread},
            {"static_shared_bytes_per_block",
                diagnostics.static_shared_bytes_per_block},
            {"local_bytes_per_thread", diagnostics.local_bytes_per_thread},
            {"maximum_threads_per_block",
                diagnostics.maximum_threads_per_block},
            {"maximum_dynamic_shared_bytes_per_block",
                diagnostics.maximum_dynamic_shared_bytes_per_block},
        }},
        {"occupancy", {
            {"active_blocks_per_multiprocessor",
                diagnostics.active_blocks_per_multiprocessor},
            {"active_warps_per_multiprocessor",
                diagnostics.active_warps_per_multiprocessor},
            {"maximum_warps_per_multiprocessor",
                diagnostics.maximum_warps_per_multiprocessor},
            {"theoretical", diagnostics.theoretical_occupancy},
        }},
        {"code", {
            {"binary_version", diagnostics.binary_version},
            {"ptx_version", diagnostics.ptx_version},
        }},
    };
    std::cerr << report.dump() << '\n';
}

}  // namespace ai_factory::workbench

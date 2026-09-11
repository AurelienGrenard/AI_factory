// Reuse one bounded production FFT workspace across rows and path-count jobs.
std::size_t workspace_bytes = 0U;
for (const auto& job : jobs) {
    workspace_bytes = std::max(workspace_bytes, model_binding::plan_pricing_workspace(
        maximum_maturity_days(products) * kStepsPerDay, job.paths, job.path_chunk
    ).workspace_bytes);
}
require_memory_budget(workspace_bytes);
DeviceBuffer workspace(workspace_bytes, DeviceMemoryRole::caller_workspace);

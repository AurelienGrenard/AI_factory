// Bounded catalogue-input scaling measurements shared by generated public-launcher probes.
#pragma once

#include "tests/performance/benchmark_support.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/tuning_profile.hpp"
#include "tools/datasets/price_dataset.hpp"
#include "common/dataset_validation.hpp"
#include "common/longstaff_schwartz/launch.cuh"

#include <filesystem>
#include <fstream>
#include <set>
#include <type_traits>

namespace ai_factory::workbench::performance::scaling {

using Clock = std::chrono::steady_clock;
inline constexpr float kDayFraction = 1.0f / 252.0f;
inline constexpr std::uint32_t kStepsPerDay = 2U;
inline constexpr float kDt = kDayFraction / kStepsPerDay;
inline constexpr std::size_t kMaximumPriceCount = 10000U;

struct InputPaths {
    std::filesystem::path model, product, curve;
};

inline InputPaths read_input_paths(int argc, char** argv, InputPaths defaults) {
    if (argc == 3) return defaults;
    if (argc != 5 || std::string(argv[3]) != "--inputs") {
        throw std::invalid_argument("Expected --jobs jobs.json [--inputs inputs.json]");
    }
    std::ifstream stream(argv[4]);
    if (!stream) throw std::invalid_argument("Cannot read scaling input paths.");
    const auto paths = nlohmann::ordered_json::parse(stream);
    return {paths.at("model").get<std::string>(), paths.at("product").get<std::string>(),
            paths.value("curve", std::string{})};
}

inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Job {
    std::string id;
    std::size_t rows;
    std::size_t offset;
    std::size_t paths;
    unsigned int threads;
    std::size_t batch_rows;
    std::size_t block_limit;
    std::size_t blocks_per_price;
    std::size_t path_chunk;
    int warmups;
    int repetitions;
    std::size_t operations_per_sample;
    std::filesystem::path publication_directory;
    InputPaths publication_inputs;
};

inline std::vector<Job> read_jobs(int argc, char** argv, bool closed_form) {
    if ((argc != 3 && argc != 5) || std::string(argv[1]) != "--jobs") {
        throw std::invalid_argument("Usage: pricing_scaling_probe --jobs jobs.json");
    }
    std::ifstream stream(argv[2]);
    if (!stream) throw std::invalid_argument("Cannot read scaling job file.");
    const auto document = nlohmann::ordered_json::parse(stream);
    if (!document.is_array() || document.empty()) {
        throw std::invalid_argument("Scaling jobs must be a nonempty array.");
    }
    std::vector<Job> jobs;
    std::set<std::string> ids;
    for (const auto& row : document) {
        Job job{
            row.at("id"), row.at("rows"), row.value("offset", 0U),
            row.value("paths", closed_form ? 0U : offline::cuda_tuning::kProductionPathsPerPrice),
            row.at("threads"), row.at("batch_rows"),
            row.value("block_limit", 4096U), row.value("blocks_per_price", 64U),
            row.value("path_chunk", 65536U),
            row.value("warmups", 1), row.value("repetitions", 3),
            row.value("operations_per_sample", 1U),
            row.value("publication_directory", std::string{}),
            {row.value("publication_model", std::string{}),
             row.value("publication_product", std::string{}),
             row.value("publication_curve", std::string{})},
        };
        if (job.rows == 0U || job.rows > kMaximumPriceCount
            || job.offset > kMaximumPriceCount - job.rows
            || (job.paths < 2U && !(closed_form && job.paths == 0U))
            || job.paths > offline::cuda_tuning::kProductionPathsPerPrice || job.batch_rows == 0U
            || job.block_limit == 0U || job.blocks_per_price == 0U
            || job.blocks_per_price > 4096U || job.path_chunk == 0U
            || job.path_chunk > 65536U || job.threads < 32U
            || job.threads > 1024U || job.threads % 32U != 0U
            || job.warmups < 0 || job.warmups > 5
            || job.repetitions < 1 || job.repetitions > 21
            || job.operations_per_sample == 0U || job.operations_per_sample > 4096U
            || !ids.insert(job.id).second) {
            throw std::invalid_argument("Invalid or duplicate bounded scaling job.");
        }
        jobs.push_back(job);
    }
    return jobs;
}

inline void validate_input_counts(std::size_t models, std::size_t products,
                                  const std::vector<Job>& jobs) {
    if (models == 0U || models > kMaximumPriceCount || products != models) {
        throw std::invalid_argument("Scaling requires between 1 and 10,000 aligned input rows.");
    }
    for (const auto& job : jobs) {
        if (job.offset + job.rows > models) throw std::invalid_argument("Job exceeds inputs.");
    }
}

template<class Products>
std::size_t maximum_maturity_days(const Products& products) {
    std::size_t days = 0;
    for (const auto& product : products) days = std::max(days, std::size_t(product.maturity_days));
    return days;
}

inline void require_memory_budget(std::size_t bytes) {
    std::size_t free = 0, total = 0;
    check_cuda(cudaMemGetInfo(&free, &total), "scaling memory budget");
    if (bytes > free / 2U) {
        throw std::invalid_argument("Scaling workspace would consume more than half of free VRAM.");
    }
}

inline void publish(const Job& job, bool closed_form, bool lsm,
                    const std::filesystem::path& model_path,
                    const std::filesystem::path& product_path,
                    const std::filesystem::path& curve_path,
                    const std::vector<float>& prices,
                    const std::vector<float>& errors, std::uint64_t seed,
                    const char* time_step_description, double host_ms, double gpu_ms) {
    const auto output = job.publication_directory / (job.id + ".json");
    const auto yaml = job.publication_directory / (job.id + ".yaml");
    if (std::filesystem::exists(output) || std::filesystem::exists(yaml)) {
        throw std::invalid_argument("Scaling publication must not overwrite an artifact.");
    }
    const std::string url = "https://datasets.ai-factory.example/performance/" + output.filename().string();
    nlohmann::ordered_json metadata{
        {"threads_per_block", job.threads},
        {"batch_rows", job.batch_rows}, {"scaling_measurement_only", true},
    };
    if (time_step_description[0] != '\0') metadata["simulation_steps_per_day"] = kStepsPerDay;
    if (lsm) metadata["blocks_per_price"] = job.blocks_per_price;
    const char* numerical_method = lsm ? "Published Longstaff-Schwartz launcher (scaling probe)"
                                        : "Published Monte Carlo launcher (scaling probe)";
    if (closed_form) {
        if (curve_path.empty()) {
            datasets::write_analytical_price_dataset(model_path, product_path,
                PriceConstruction::Aligned, prices, output, yaml, url,
                "Published closed-form launcher (scaling probe)", metadata, host_ms / 1000, gpu_ms / 1000);
        } else {
            datasets::write_analytical_price_dataset(model_path, curve_path, product_path,
                PriceConstruction::Aligned, prices, output, yaml, url,
                "Published closed-form launcher (scaling probe)", metadata, host_ms / 1000, gpu_ms / 1000);
        }
    } else if (curve_path.empty()) {
        datasets::write_monte_carlo_price_dataset(model_path, product_path,
            PriceConstruction::Aligned, prices, errors, "Philox", output, yaml, url,
            numerical_method, job.paths, time_step_description,
            metadata, nlohmann::ordered_json::object(), seed, host_ms / 1000, gpu_ms / 1000);
    } else {
        datasets::write_monte_carlo_price_dataset(model_path, curve_path, product_path,
            PriceConstruction::Aligned, prices, errors, "Philox", output, yaml, url,
            numerical_method, job.paths, time_step_description,
            metadata, nlohmann::ordered_json::object(), seed, host_ms / 1000, gpu_ms / 1000);
    }
    datasets::validate_price_dataset_file(output);
}

template<class Launch>
void benchmark(const Job& job, const char* case_id, bool closed_form, bool volterra,
               double preparation_ms, DeviceBuffer& device_prices, DeviceBuffer& device_errors,
               const std::filesystem::path& model_path, const std::filesystem::path& product_path,
               const std::filesystem::path& curve_path, std::uint64_t seed,
               const char* time_step_description, Launch&& launch) {
    constexpr bool lsm = std::is_same_v<std::invoke_result_t<Launch,
        std::size_t, std::size_t, std::size_t>, longstaff_schwartz::LaunchResult>;
    std::cout << nlohmann::ordered_json{{"event", "job_start"}, {"case", case_id}, {"id", job.id},
        {"rows", job.rows}, {"paths", job.paths}, {"preparation_once_ms", preparation_ms}}.dump() << std::endl;
    offline::cuda::Event start, stop;
    std::vector<double> gpu_samples, api_samples, raw_host_samples;
    struct SampleTiming {
        double gpu_ms;
        double api_ms;
        double raw_host_ms;
    };
    const std::size_t batch_rows = volterra ? 1U : job.batch_rows;
    std::size_t transient_peak_bytes = 0U;
    nlohmann::ordered_json batch_metrics;
    const auto run = [&] {
        batch_metrics = nlohmann::ordered_json::array();
        double lsm_gpu_ms = 0.0;
        const auto host_start = Clock::now();
        if constexpr (!lsm) check_cuda(cudaEventRecord(start.get()), "scaling start event");
        for (std::size_t operation = 0; operation < job.operations_per_sample; ++operation) {
            for (std::size_t done = 0; done < job.rows;) {
                const std::size_t count = std::min(batch_rows, job.rows - done);
                const std::size_t blocks = std::min(job.block_limit,
                    closed_form ? 1U + (count - 1U) / job.threads : count);
                if constexpr (lsm) {
                    const auto result = launch(job.offset + done, count, blocks);
                    longstaff_schwartz::validate_regression_diagnostics(result, case_id);
                    lsm_gpu_ms += result.kernel_seconds * 1000.0;
                    transient_peak_bytes = std::max(transient_peak_bytes, result.workspace_bytes);
                    const auto& diagnostics = result.regression_diagnostics;
                    batch_metrics.push_back({{"offset", job.offset + done}, {"rows", count},
                        {"gpu_ms", result.kernel_seconds * 1000.0},
                        {"native_batch_count", result.batch_count},
                        {"maximum_prices_per_batch", result.maximum_prices_per_batch},
                        {"kernel_launch_count", result.kernel_launch_count},
                        {"blocks_per_price", result.blocks_per_price},
                        {"workspace_bytes", result.workspace_bytes},
                        {"successful_regressions", diagnostics.successful_regression_count},
                        {"no_candidates", diagnostics.no_candidate_count},
                        {"insufficient_candidates", diagnostics.insufficient_candidate_count},
                        {"fatal_regressions", diagnostics.affected_result_count}});
                } else {
                    launch(job.offset + done, count, blocks);
                }
                done += count;
            }
        }
        float gpu_ms = 0;
        if constexpr (!lsm) {
            check_cuda(cudaEventRecord(stop.get()), "scaling stop event");
            check_cuda(cudaEventSynchronize(stop.get()), "scaling synchronize sample");
            check_cuda(cudaEventElapsedTime(&gpu_ms, start.get(), stop.get()), "scaling elapsed");
        }
        const double operations = static_cast<double>(job.operations_per_sample);
        const double normalized_gpu_ms =
            (lsm ? lsm_gpu_ms : static_cast<double>(gpu_ms)) / operations;
        const double normalized_host_ms = elapsed_ms(host_start) / operations;
        return SampleTiming{
            normalized_gpu_ms,
            std::max(normalized_gpu_ms, normalized_host_ms),
            normalized_host_ms
        };
    };
    for (int warmup = 0; warmup < job.warmups; ++warmup) run();
    std::vector<float> prices(job.rows), errors(closed_form ? 0U : job.rows);
    std::vector<float> first_prices, first_errors;
    std::vector<double> copies;
    for (int repetition = 0; repetition < job.repetitions; ++repetition) {
        const auto [gpu_ms, api_ms, raw_host_ms] = run();
        gpu_samples.push_back(gpu_ms);
        api_samples.push_back(api_ms);
        raw_host_samples.push_back(raw_host_ms);
        const auto copy_start = Clock::now();
        check_cuda(cudaMemcpy(prices.data(), device_prices.as<float>() + job.offset,
            prices.size() * sizeof(float), cudaMemcpyDeviceToHost), "scaling prices D2H");
        if (!closed_form) check_cuda(cudaMemcpy(errors.data(), device_errors.as<float>() + job.offset,
            errors.size() * sizeof(float), cudaMemcpyDeviceToHost), "scaling errors D2H");
        copies.push_back(elapsed_ms(copy_start));
        for (std::size_t row = 0; row < prices.size(); ++row) {
            if (!std::isfinite(prices[row]) || (!closed_form && (!std::isfinite(errors[row]) || errors[row] < 0))) {
                throw std::runtime_error("Non-finite price/error at row " + std::to_string(job.offset + row));
            }
        }
        if (repetition == 0) { first_prices = prices; first_errors = errors; }
        else if (prices != first_prices || errors != first_errors) {
            throw std::runtime_error("Scaling replay changed numerical outputs.");
        }
        std::cout << nlohmann::ordered_json{{"event", "repetition"}, {"id", job.id},
            {"repetition", repetition}, {"gpu_ms", gpu_ms}, {"api_ms", api_ms},
            {"raw_host_ms", raw_host_ms},
            {"lsm_batches", batch_metrics}}.dump() << std::endl;
    }
    nlohmann::ordered_json publication = nullptr;
    if (!job.publication_directory.empty()) {
        if (job.offset != 0U) {
            throw std::invalid_argument("Native publication requires an aligned input prefix.");
        }
        const auto publication_start = Clock::now();
        const bool explicit_inputs = !job.publication_inputs.model.empty();
        publish(job, closed_form, lsm,
                explicit_inputs ? job.publication_inputs.model : model_path,
                explicit_inputs ? job.publication_inputs.product : product_path,
                explicit_inputs ? job.publication_inputs.curve : curve_path, prices, errors, seed,
                time_step_description, api_samples.back() + preparation_ms + copies.back(), gpu_samples.back());
        publication = {{"wall_ms", elapsed_ms(publication_start)}, {"native_writer", true}};
    }
    std::cout << nlohmann::ordered_json{
        {"event", "scaling_result"}, {"case", case_id}, {"id", job.id},
        {"environment", environment_json()},
        {"configuration", {{"rows", job.rows}, {"offset", job.offset},
            {"paths_per_price", closed_form ? 0U : job.paths},
            {"requested_threads", volterra ? nlohmann::ordered_json(nullptr)
                                            : nlohmann::ordered_json(job.threads)},
            {"batch_rows", batch_rows}, {"block_limit", job.block_limit},
            {"blocks_per_price", lsm ? job.blocks_per_price : 0U},
            {"path_chunk", volterra ? job.path_chunk : 0U}, {"warmups", job.warmups},
            {"repetitions", job.repetitions}, {"seed", seed}}},
        {"operations_per_sample", job.operations_per_sample},
        {"fixed_time_step", time_step_description},
        {"gpu", timing_json(summarize(gpu_samples))}, {"public_api", timing_json(summarize(api_samples))},
        {"raw_host_clock", timing_json(summarize(raw_host_samples))},
        {"gpu_samples_ms", gpu_samples}, {"api_samples_ms", api_samples},
        {"raw_host_samples_ms", raw_host_samples},
        {"preparation_once_ms", preparation_ms}, {"output_copy", timing_json(summarize(copies))},
        {"publication", publication}, {"device_memory", device_memory_json(transient_peak_bytes)},
        {"lsm_batches_last_repetition", batch_metrics},
        {"prices", prices}, {"standard_errors", errors}, {"deterministic_replay", true},
        {"timing_scope", lsm
            ? "GPU: sum of native LSM batch events; raw_host_clock: allocation, planning, launches, sync, diagnostics and batch accounting; public_api: max(raw_host_clock, GPU); input preparation once per process"
            : "GPU events enclose all API launches in one sample; raw_host_clock includes final synchronization; public_api: max(raw_host_clock, GPU); input preparation once per process"},
    }.dump() << std::endl;
}

}  // namespace ai_factory::workbench::performance::scaling

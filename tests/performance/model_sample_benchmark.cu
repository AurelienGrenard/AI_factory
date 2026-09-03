// Representative model-sample throughput and streaming-publication baseline.
#include "tests/performance/benchmark_support.cuh"

#include "model/equity/markovian/black_scholes/sample.cuh"
#include "model/equity/markovian/heston/sample.cuh"
#include "model/equity/rough/rough_bergomi/sample.cuh"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/rough_heston/sample.cuh"
#include "tools/cuda/tuning_profile.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ai_factory::workbench;
using performance::DeviceBuffer;
using performance::DeviceMemoryRole;
using performance::Measurement;
using performance::TimingStatistics;
using performance::copy_from_device;
using performance::copy_to_device;
using performance::emit_measurement;
using performance::measure_cuda;
using performance::measure_host;
using performance::timing_json;

namespace black_scholes = model::equity::black_scholes;
namespace heston = model::equity::heston;
namespace rough_bergomi = model::equity::rough_bergomi;
namespace rough_heston = model::equity::rough_heston;

constexpr std::uint32_t kMaturityDays = 504U;
constexpr std::uint64_t kSeed = 940000001ULL;
// Large enough that JSON/YAML publication dominates scheduler and filesystem
// jitter on the reference machine while remaining a reduced dataset payload.
constexpr std::size_t kPublicationRows = 262'144U;
constexpr std::size_t kPublicationBufferBytes = 4U << 20U;
constexpr std::size_t kPublicationBatchSize = 16U;
constexpr std::size_t kExactLaunchesPerMeasurement = 2'048U;

struct Layout {
    const char* id;
    std::size_t parameter_count;
    std::size_t paths_per_parameter;
    std::size_t production_parameter_count;
    std::size_t production_paths_per_parameter;
};

constexpr Layout kUnconditional{
    "3m_x_1", 3'000'000U, 1U, 3'000'000U, 1U
};
constexpr Layout kConditional{
    "12000_x_250", 12'000U, 250U, 12'000U, 250U
};
constexpr Layout kVolterraUnconditional{
    "3m_x_1", 8'192U, 1U, 3'000'000U, 1U
};
constexpr Layout kVolterraConditional{
    "12000_x_250", 256U, 250U, 12'000U, 250U
};

std::size_t sample_count(const Layout& layout) {
    return layout.parameter_count * layout.paths_per_parameter;
}

std::size_t block_count(const Layout& layout, unsigned int threads) {
    const std::size_t work_items = layout.paths_per_parameter == 1U
        ? (sample_count(layout) + threads - 1U) / threads
        : layout.parameter_count;
    return std::max<std::size_t>(
        1U,
        std::min(work_items, offline::cuda_tuning::kSampleBlockCountLimit)
    );
}

void require_finite(const std::vector<float>& values, const char* label) {
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(label) + " is not finite.");
        }
    }
}

TimingStatistics measure_publication(const Layout& layout) {
    std::vector<char> json_buffer(kPublicationBufferBytes);
    const auto publish_once = [&] {
        std::ofstream json;
        json.rdbuf()->pubsetbuf(json_buffer.data(), json_buffer.size());
        json.open(
            "/tmp/ai_factory_model_sample_performance.json", std::ios::trunc
        );
        if (!json) throw std::runtime_error("Cannot open sample benchmark JSON.");
        json << "{\"row_count\":" << kPublicationRows << ",\"samples\":[";
        json << std::setprecision(9);
        for (std::size_t row = 0U; row < kPublicationRows; ++row) {
            if (row != 0U) json << ',';
            const std::size_t parameter = row / layout.paths_per_parameter;
            const float value = 0.75f
                + static_cast<float>(row % 257U) * 1.0e-3f;
            json << "{\"id\":" << row
                 << ",\"parameter_index\":" << parameter
                 << ",\"maturity_days\":504,\"T\":2.0,"
                    "\"values\":{\"spot\":" << value << "}}";
        }
        json << "]}\n";
        json.close();
        if (!json) throw std::runtime_error("Cannot write sample benchmark JSON.");

        std::ofstream yaml(
            "/tmp/ai_factory_model_sample_performance.yaml",
            std::ios::trunc
        );
        yaml << "row_count: " << kPublicationRows << '\n'
             << "layout: " << layout.id << '\n'
             << "format: streamed-json-plus-yaml\n";
        yaml.close();
        if (!yaml) throw std::runtime_error("Cannot write sample benchmark YAML.");
    };
    TimingStatistics statistics = measure_host([&] {
        for (std::size_t pass = 0U; pass < kPublicationBatchSize; ++pass) {
            publish_once();
        }
    });
    const double divisor = static_cast<double>(kPublicationBatchSize);
    statistics.minimum_ms /= divisor;
    statistics.median_ms /= divisor;
    statistics.p95_ms /= divisor;
    statistics.mean_ms /= divisor;
    statistics.standard_deviation_ms /= divisor;
    return statistics;
}

const TimingStatistics& publication_statistics(const Layout& layout) {
    if (layout.paths_per_parameter == 1U) {
        static const TimingStatistics unconditional = measure_publication(
            kUnconditional
        );
        return unconditional;
    }
    static const TimingStatistics conditional = measure_publication(
        kConditional
    );
    return conditional;
}

nlohmann::ordered_json configuration(
    const char* engine,
    const Layout& layout,
    unsigned int threads,
    std::size_t blocks,
    std::size_t launches_per_measurement
) {
    return {
        {"engine", engine},
        {"layout", layout.id},
        {"production_parameter_count", layout.production_parameter_count},
        {"production_paths_per_parameter",
            layout.production_paths_per_parameter},
        {"benchmark_parameter_count", layout.parameter_count},
        {"benchmark_paths_per_parameter", layout.paths_per_parameter},
        {"benchmark_sample_count", sample_count(layout)},
        {"maturity_days", kMaturityDays},
        {"threads_per_block", threads},
        {"block_count", blocks},
        {"launches_per_measurement", launches_per_measurement},
        {"publication_row_count", kPublicationRows},
        {"publication_batch_size", kPublicationBatchSize},
        {"tuning_profile", offline::cuda_tuning::kProfileId},
        {
            "reduction_rationale",
            sample_count(layout)
                    == layout.production_parameter_count
                        * layout.production_paths_per_parameter
                ? "full production layout"
                : "preserves the production path package and at least 8192 independent FFT paths"
        },
    };
}

void report(
    const char* engine,
    const Layout& layout,
    unsigned int threads,
    std::size_t blocks,
    std::size_t launches_per_measurement,
    const Measurement& measurement,
    const std::vector<float>& values
) {
    require_finite(values, engine);
    emit_measurement(
        "PERF-010",
        "model_sample_pipeline",
        std::string(engine) + "_" + layout.id,
        measurement,
        configuration(
            engine, layout, threads, blocks, launches_per_measurement
        ),
        {
            {"finite", true},
            {
                "samples_per_second",
                static_cast<double>(sample_count(layout))
                    * static_cast<double>(launches_per_measurement) * 1'000.0
                    / measurement.kernel.median_ms
            },
            {"philox_mapping", "parameter row key and path counter unchanged"},
        },
        performance::kDefaultWarmups,
        performance::kDefaultRepetitions,
        {{"publication_wall", timing_json(publication_statistics(layout))}}
    );
}

void benchmark_black_scholes(const Layout& layout) {
    const std::vector<black_scholes::ModelParameters> parameters(
        layout.parameter_count,
        {1.0f, 0.03f, 0.01f, 0.20f}
    );
    DeviceBuffer device_parameters(
        parameters.size() * sizeof(parameters[0]),
        DeviceMemoryRole::persistent_input
    );
    DeviceBuffer output(
        sample_count(layout) * sizeof(float), DeviceMemoryRole::output
    );
    copy_to_device(device_parameters, parameters);
    const unsigned int threads = offline::cuda_tuning::kSampleThreadsPerBlock;
    const std::size_t blocks = block_count(layout, threads);
    const auto launch = [&] {
        for (std::size_t repeat = 0U;
             repeat < kExactLaunchesPerMeasurement;
             ++repeat) {
            black_scholes::launch_black_scholes_terminal_samples_cuda(
                device_parameters.as<black_scholes::ModelParameters>(),
                layout.parameter_count,
                layout.paths_per_parameter,
                kMaturityDays,
                0U,
                sample_count(layout),
                threads,
                blocks,
                kSeed,
                output.as<float>()
            );
        }
    };
    const Measurement measurement = measure_cuda(launch);
    report(
        "exact_markovian",
        layout,
        threads,
        blocks,
        kExactLaunchesPerMeasurement,
        measurement,
        copy_from_device<float>(output, sample_count(layout))
    );
}

void benchmark_heston(const Layout& layout) {
    const std::vector<heston::ModelParameters> parameters(
        layout.parameter_count,
        {1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f}
    );
    DeviceBuffer device_parameters(
        parameters.size() * sizeof(parameters[0]),
        DeviceMemoryRole::persistent_input
    );
    DeviceBuffer spots(
        sample_count(layout) * sizeof(float), DeviceMemoryRole::output
    );
    DeviceBuffer variances(
        sample_count(layout) * sizeof(float), DeviceMemoryRole::output
    );
    copy_to_device(device_parameters, parameters);
    const unsigned int threads = offline::cuda_tuning::kSampleThreadsPerBlock;
    const std::size_t blocks = block_count(layout, threads);
    const auto launch = [&] {
        heston::launch_heston_terminal_samples_cuda(
            device_parameters.as<heston::ModelParameters>(),
            layout.parameter_count,
            layout.paths_per_parameter,
            kMaturityDays,
            0U,
            sample_count(layout),
            threads,
            blocks,
            kSeed,
            spots.as<float>(),
            variances.as<float>()
        );
    };
    const Measurement measurement = measure_cuda(launch, 5, 21, 2U);
    const auto variance_values = copy_from_device<float>(
        variances,
        sample_count(layout)
    );
    require_finite(variance_values, "fixed_step_markovian variance");
    report(
        "fixed_step_markovian",
        layout,
        threads,
        blocks,
        1U,
        measurement,
        copy_from_device<float>(spots, sample_count(layout))
    );
}

void benchmark_rough_heston(const Layout& layout) {
    const rough_heston::ModelParameters model{
        1.0f, 0.03f, 0.01f, 0.04f, 0.30f, 0.02f, 0.30f, 0.10f, -0.70f
    };
    const auto one_prepared = rough_heston::prepare_dynamics<7U>(
        model,
        2.0f,
        1.0f / 504.0f
    );
    const std::vector<rough_heston::PreparedDynamics<7U>> prepared(
        layout.parameter_count,
        one_prepared
    );
    DeviceBuffer device_prepared(
        prepared.size() * sizeof(prepared[0]),
        DeviceMemoryRole::persistent_input
    );
    DeviceBuffer spots(
        sample_count(layout) * sizeof(float), DeviceMemoryRole::output
    );
    copy_to_device(device_prepared, prepared);
    const unsigned int threads = offline::cuda_tuning::kSampleThreadsPerBlock;
    const std::size_t blocks = block_count(layout, threads);
    const auto launch = [&] {
        rough_heston::launch_rough_heston_terminal_samples_cuda<7U>(
            device_prepared.as<rough_heston::PreparedDynamics<7U>>(),
            layout.parameter_count,
            layout.paths_per_parameter,
            kMaturityDays,
            0U,
            sample_count(layout),
            threads,
            blocks,
            kSeed,
            spots.as<float>()
        );
    };
    const Measurement measurement = measure_cuda(launch, 5, 21, 2U);
    report(
        "rough_n_factor_7",
        layout,
        threads,
        blocks,
        1U,
        measurement,
        copy_from_device<float>(spots, sample_count(layout))
    );
}

void benchmark_rough_bergomi(const Layout& layout) {
    const std::vector<rough_bergomi::ModelParameters> parameters(
        layout.parameter_count,
        {1.0f, 0.03f, 0.01f, 0.04f, 1.2f, 0.10f, -0.70f}
    );
    DeviceBuffer device_parameters(
        parameters.size() * sizeof(parameters[0]),
        DeviceMemoryRole::persistent_input
    );
    DeviceBuffer spots(
        sample_count(layout) * sizeof(float), DeviceMemoryRole::output
    );
    copy_to_device(device_parameters, parameters);
    const std::size_t blocks = std::min(
        layout.parameter_count,
        offline::cuda_tuning::kSampleBlockCountLimit
    );
    const auto launch = [&] {
        rough_bergomi::launch_rough_bergomi_terminal_samples_cuda(
            device_parameters.as<rough_bergomi::ModelParameters>(),
            layout.parameter_count,
            layout.paths_per_parameter,
            kMaturityDays,
            0U,
            sample_count(layout),
            blocks,
            kSeed,
            spots.as<float>()
        );
    };
    const Measurement measurement = measure_cuda(launch, 5, 21, 2U);
    report(
        "volterra_fft",
        layout,
        0U,
        blocks,
        1U,
        measurement,
        copy_from_device<float>(spots, sample_count(layout))
    );
}

}  // namespace

int main() {
    benchmark_black_scholes(kUnconditional);
    benchmark_black_scholes(kConditional);
    benchmark_heston(kUnconditional);
    benchmark_heston(kConditional);
    benchmark_rough_heston(kUnconditional);
    benchmark_rough_heston(kConditional);
    benchmark_rough_bergomi(kVolterraUnconditional);
    benchmark_rough_bergomi(kVolterraConditional);
}

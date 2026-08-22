// Side-by-side correctness and timing benchmark for the equity policy layer.
#include "common/check_cuda.cuh"
#include "model/equity/cev/athena_autocall.cuh"
#include "model/equity/cev/athena_autocallbis.cuh"
#include "model/equity/merton/athena_autocall.cuh"
#include "model/equity/merton/athena_autocallbis.cuh"
#include "product/athena_autocall/dataset.hpp"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ai_factory::workbench::check_cuda;

constexpr std::size_t kPathsPerPrice = 16'384U;
constexpr unsigned int kThreadsPerBlock = 512U;
constexpr std::size_t kMaximumBlockCount = 4'096U;
constexpr std::uint64_t kSeed = 900000001ULL;

struct BenchmarkCase {
    const char* name;
    std::size_t model_count;
    std::size_t product_count;
    bool cartesian_product;
    std::size_t result_count;
    std::size_t repetitions;
    bool compare_dataset;
};

constexpr BenchmarkCase kCases[] = {
    {"tiny", 1U, 1U, false, 1U, 12U, false},
    {"small", 64U, 64U, false, 64U, 10U, false},
    {"dataset", 1'000U, 1'000U, false, 1'000U, 12U, true},
    {"large", 64U, 64U, true, 4'096U, 6U, false},
};

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_(size) {
        check_cuda(
            cudaMalloc(&data_, size_ * sizeof(T)),
            "benchmark cudaMalloc"
        );
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

    void copy_from_host(const std::vector<T>& values) {
        if (values.size() > size_) {
            throw std::invalid_argument("DeviceBuffer input is too large.");
        }
        check_cuda(
            cudaMemcpy(
                data_,
                values.data(),
                values.size() * sizeof(T),
                cudaMemcpyHostToDevice
            ),
            "benchmark cudaMemcpy host to device"
        );
    }

    std::vector<T> copy_to_host(std::size_t count) const {
        if (count > size_) {
            throw std::invalid_argument("DeviceBuffer output is too large.");
        }
        std::vector<T> values(count);
        check_cuda(
            cudaMemcpy(
                values.data(),
                data_,
                count * sizeof(T),
                cudaMemcpyDeviceToHost
            ),
            "benchmark cudaMemcpy device to host"
        );
        return values;
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0U;
};

class EventPair {
public:
    EventPair() {
        check_cuda(cudaEventCreate(&start_), "benchmark create start event");
        check_cuda(cudaEventCreate(&stop_), "benchmark create stop event");
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    ~EventPair() {
        if (start_ != nullptr) cudaEventDestroy(start_);
        if (stop_ != nullptr) cudaEventDestroy(stop_);
    }

    cudaEvent_t start() const { return start_; }
    cudaEvent_t stop() const { return stop_; }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

struct TimedRun {
    double cuda_milliseconds;
    double wall_milliseconds;
};

struct TimingSummary {
    double minimum;
    double median;
    double maximum;
};

TimingSummary summarize(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("No benchmark samples.");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    const double median = values.size() % 2U == 0U
        ? 0.5 * (values[middle - 1U] + values[middle])
        : values[middle];
    return {
        values.front(),
        median,
        values.back(),
    };
}

struct ReferenceDataset {
    std::vector<float> prices;
    std::vector<float> standard_errors;
    double wall_seconds;
    double kernel_seconds;
};

ReferenceDataset load_reference_dataset(
    const std::filesystem::path& path
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open reference dataset: " + path.string());
    }
    nlohmann::json document;
    input >> document;

    ReferenceDataset reference;
    const auto& results = document.at("results");
    reference.prices.reserve(results.size());
    reference.standard_errors.reserve(results.size());
    for (const auto& result : results) {
        const auto& outputs = result.at("outputs");
        reference.prices.push_back(outputs.at("price").get<float>());
        reference.standard_errors.push_back(
            outputs.at("standard_error").get<float>()
        );
    }
    reference.wall_seconds =
        document.at("timing").at("wall_seconds").get<double>();
    reference.kernel_seconds =
        document.at("timing").at("kernel_seconds").get<double>();
    return reference;
}

void require_bitwise_equal(
    const std::vector<float>& left,
    const std::vector<float>& right,
    const std::string& label
) {
    if (left.size() != right.size()) {
        throw std::runtime_error(label + ": result sizes differ.");
    }
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const std::uint32_t left_bits = std::bit_cast<std::uint32_t>(left[index]);
        const std::uint32_t right_bits =
            std::bit_cast<std::uint32_t>(right[index]);
        if (left_bits != right_bits) {
            throw std::runtime_error(
                label + ": first mismatch at row " + std::to_string(index)
                + " (" + std::to_string(left[index]) + " versus "
                + std::to_string(right[index]) + ")."
            );
        }
    }
}

struct CevTraits {
    using ModelParameters = ai_factory::workbench::cev::ModelParameters;

    static constexpr const char* name = "CEV";
    static constexpr std::size_t pipeline_repetitions = 10U;
    static constexpr const char* model_path =
        "datasets/model/equity/cev/parameters/cev_01.json";
    static constexpr const char* reference_path =
        "datasets/model/equity/cev/prices/athena_autocalls/"
        "cev_01__athena_autocalls_01__01.json";

    static std::vector<ModelParameters> load_models() {
        return ai_factory::workbench::cev::load_models(model_path);
    }

    static void launch(
        bool candidate,
        const ModelParameters* models,
        std::size_t model_count,
        const ai_factory::workbench::product::AthenaAutocallParameters* products,
        std::size_t product_count,
        bool cartesian_product,
        std::size_t result_count,
        float* prices,
        float* standard_errors
    ) {
        constexpr float dt = 1.0f / 504.0f;
        constexpr std::uint32_t simulation_steps_per_day = 2U;
        const std::size_t blocks = std::min(
            result_count,
            kMaximumBlockCount
        );
        if (candidate) {
            ai_factory::workbench::cev::launch_cev_athena_autocallbis_cuda(
                models, model_count, products, product_count,
                cartesian_product, result_count, 0U, result_count,
                kPathsPerPrice, dt, simulation_steps_per_day,
                kThreadsPerBlock, blocks, kSeed, prices, standard_errors
            );
        } else {
            ai_factory::workbench::cev::launch_cev_athena_autocall_cuda(
                models, model_count, products, product_count,
                cartesian_product, result_count, 0U, result_count,
                kPathsPerPrice, dt, simulation_steps_per_day,
                kThreadsPerBlock, blocks, kSeed, prices, standard_errors
            );
        }
    }
};

struct MertonTraits {
    using ModelParameters = ai_factory::workbench::merton::ModelParameters;

    static constexpr const char* name = "Merton";
    static constexpr std::size_t pipeline_repetitions = 26U;
    static constexpr const char* model_path =
        "datasets/model/equity/merton/parameters/merton_01.json";
    static constexpr const char* reference_path =
        "datasets/model/equity/merton/prices/athena_autocalls/"
        "merton_01__athena_autocalls_01__01.json";

    static std::vector<ModelParameters> load_models() {
        return ai_factory::workbench::merton::load_models(model_path);
    }

    static void launch(
        bool candidate,
        const ModelParameters* models,
        std::size_t model_count,
        const ai_factory::workbench::product::AthenaAutocallParameters* products,
        std::size_t product_count,
        bool cartesian_product,
        std::size_t result_count,
        float* prices,
        float* standard_errors
    ) {
        constexpr float day_fraction = 1.0f / 252.0f;
        const std::size_t blocks = std::min(
            result_count,
            kMaximumBlockCount
        );
        if (candidate) {
            ai_factory::workbench::merton::launch_merton_athena_autocallbis_cuda(
                models, model_count, products, product_count,
                cartesian_product, result_count, 0U, result_count,
                kPathsPerPrice, day_fraction, kThreadsPerBlock, blocks,
                kSeed, prices, standard_errors
            );
        } else {
            ai_factory::workbench::merton::launch_merton_athena_autocall_cuda(
                models, model_count, products, product_count,
                cartesian_product, result_count, 0U, result_count,
                kPathsPerPrice, day_fraction, kThreadsPerBlock, blocks,
                kSeed, prices, standard_errors
            );
        }
    }
};

template<typename Traits>
TimedRun time_launch(
    bool candidate,
    const BenchmarkCase& benchmark_case,
    const typename Traits::ModelParameters* device_models,
    const ai_factory::workbench::product::AthenaAutocallParameters*
        device_products,
    float* device_prices,
    float* device_standard_errors,
    const EventPair& events
) {
    const auto wall_start = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(events.start()), "benchmark record start");
    Traits::launch(
        candidate,
        device_models,
        benchmark_case.model_count,
        device_products,
        benchmark_case.product_count,
        benchmark_case.cartesian_product,
        benchmark_case.result_count,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaEventRecord(events.stop()), "benchmark record stop");
    check_cuda(cudaEventSynchronize(events.stop()), "benchmark synchronize stop");
    const double wall_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start
    ).count();
    float cuda_milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(
            &cuda_milliseconds,
            events.start(),
            events.stop()
        ),
        "benchmark elapsed time"
    );
    return {static_cast<double>(cuda_milliseconds), wall_milliseconds};
}

template<typename Traits>
TimedRun time_complete_pipeline(
    bool candidate,
    const std::vector<typename Traits::ModelParameters>& models,
    const std::vector<ai_factory::workbench::product::AthenaAutocallParameters>&
        products
) {
    constexpr std::size_t result_count = 1'000U;
    constexpr std::size_t warmup_count = 64U;
    double cuda_milliseconds = 0.0;
    const auto wall_start = std::chrono::steady_clock::now();
    {
        DeviceBuffer<typename Traits::ModelParameters> device_models(
            models.size()
        );
        DeviceBuffer<ai_factory::workbench::product::AthenaAutocallParameters>
            device_products(products.size());
        DeviceBuffer<float> device_prices(result_count);
        DeviceBuffer<float> device_errors(result_count);
        device_models.copy_from_host(models);
        device_products.copy_from_host(products);

        Traits::launch(
            candidate,
            device_models.data(),
            warmup_count,
            device_products.data(),
            warmup_count,
            false,
            warmup_count,
            device_prices.data(),
            device_errors.data()
        );
        check_cuda(cudaDeviceSynchronize(), "complete-pipeline warmup");

        EventPair events;
        check_cuda(
            cudaEventRecord(events.start()),
            "complete-pipeline record start"
        );
        Traits::launch(
            candidate,
            device_models.data(),
            result_count,
            device_products.data(),
            result_count,
            false,
            result_count,
            device_prices.data(),
            device_errors.data()
        );
        check_cuda(
            cudaEventRecord(events.stop()),
            "complete-pipeline record stop"
        );
        check_cuda(
            cudaEventSynchronize(events.stop()),
            "complete-pipeline synchronize stop"
        );
        float elapsed = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&elapsed, events.start(), events.stop()),
            "complete-pipeline elapsed time"
        );
        cuda_milliseconds = static_cast<double>(elapsed);
        device_prices.copy_to_host(result_count);
        device_errors.copy_to_host(result_count);
    }
    const double wall_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start
    ).count();
    return {cuda_milliseconds, wall_milliseconds};
}

template<typename Traits>
void benchmark_model(
    const std::vector<ai_factory::workbench::product::AthenaAutocallParameters>&
        products
) {
    const std::vector<typename Traits::ModelParameters> models =
        Traits::load_models();
    if (models.size() < 1'000U || products.size() < 1'000U) {
        throw std::runtime_error("Athena benchmark requires 1000 input rows.");
    }

    DeviceBuffer<typename Traits::ModelParameters> device_models(models.size());
    DeviceBuffer<ai_factory::workbench::product::AthenaAutocallParameters>
        device_products(products.size());
    DeviceBuffer<float> current_prices(4'096U);
    DeviceBuffer<float> current_errors(4'096U);
    DeviceBuffer<float> candidate_prices(4'096U);
    DeviceBuffer<float> candidate_errors(4'096U);
    device_models.copy_from_host(models);
    device_products.copy_from_host(products);

    const ReferenceDataset reference =
        load_reference_dataset(Traits::reference_path);
    std::cout << "\n" << Traits::name
              << " stored dataset: kernel=" << std::fixed
              << std::setprecision(3) << reference.kernel_seconds * 1.0e3
              << " ms, wall=" << reference.wall_seconds * 1.0e3 << " ms\n";
    std::cout
        << "case      rows  repeats  bitwise  current CUDA/wall ms"
        << "  policy CUDA/wall ms  CUDA delta  wall delta\n";

    EventPair events;
    for (const BenchmarkCase& benchmark_case : kCases) {
        Traits::launch(
            false,
            device_models.data(),
            benchmark_case.model_count,
            device_products.data(),
            benchmark_case.product_count,
            benchmark_case.cartesian_product,
            benchmark_case.result_count,
            current_prices.data(),
            current_errors.data()
        );
        Traits::launch(
            true,
            device_models.data(),
            benchmark_case.model_count,
            device_products.data(),
            benchmark_case.product_count,
            benchmark_case.cartesian_product,
            benchmark_case.result_count,
            candidate_prices.data(),
            candidate_errors.data()
        );
        check_cuda(cudaDeviceSynchronize(), "benchmark warmup");

        std::vector<double> current_cuda;
        std::vector<double> current_wall;
        std::vector<double> candidate_cuda;
        std::vector<double> candidate_wall;
        std::vector<double> paired_cuda_ratios;
        std::vector<double> paired_wall_ratios;
        for (std::size_t repetition = 0U;
             repetition < benchmark_case.repetitions;
             ++repetition) {
            const bool candidate_first = repetition % 2U != 0U;
            TimedRun current_run{};
            TimedRun candidate_run{};
            for (int order = 0; order < 2; ++order) {
                const bool candidate = order == 0
                    ? candidate_first
                    : !candidate_first;
                const TimedRun run = time_launch<Traits>(
                    candidate,
                    benchmark_case,
                    device_models.data(),
                    device_products.data(),
                    candidate ? candidate_prices.data() : current_prices.data(),
                    candidate ? candidate_errors.data() : current_errors.data(),
                    events
                );
                (candidate ? candidate_run : current_run) = run;
                (candidate ? candidate_cuda : current_cuda).push_back(
                    run.cuda_milliseconds
                );
                (candidate ? candidate_wall : current_wall).push_back(
                    run.wall_milliseconds
                );
            }
            paired_cuda_ratios.push_back(
                candidate_run.cuda_milliseconds
                    / current_run.cuda_milliseconds
            );
            paired_wall_ratios.push_back(
                candidate_run.wall_milliseconds
                    / current_run.wall_milliseconds
            );
        }

        const std::vector<float> host_current_prices =
            current_prices.copy_to_host(benchmark_case.result_count);
        const std::vector<float> host_current_errors =
            current_errors.copy_to_host(benchmark_case.result_count);
        const std::vector<float> host_candidate_prices =
            candidate_prices.copy_to_host(benchmark_case.result_count);
        const std::vector<float> host_candidate_errors =
            candidate_errors.copy_to_host(benchmark_case.result_count);
        require_bitwise_equal(
            host_current_prices,
            host_candidate_prices,
            std::string(Traits::name) + " " + benchmark_case.name + " prices"
        );
        require_bitwise_equal(
            host_current_errors,
            host_candidate_errors,
            std::string(Traits::name) + " " + benchmark_case.name + " errors"
        );
        if (benchmark_case.compare_dataset) {
            require_bitwise_equal(
                reference.prices,
                host_current_prices,
                std::string(Traits::name) + " stored prices"
            );
            require_bitwise_equal(
                reference.standard_errors,
                host_current_errors,
                std::string(Traits::name) + " stored errors"
            );
        }

        const TimingSummary current_cuda_summary = summarize(current_cuda);
        const TimingSummary current_wall_summary = summarize(current_wall);
        const TimingSummary candidate_cuda_summary = summarize(candidate_cuda);
        const TimingSummary candidate_wall_summary = summarize(candidate_wall);
        const double cuda_delta = summarize(paired_cuda_ratios).median - 1.0;
        const double wall_delta = summarize(paired_wall_ratios).median - 1.0;
        std::cout
            << std::left << std::setw(9) << benchmark_case.name
            << std::right << std::setw(5) << benchmark_case.result_count
            << std::setw(9) << benchmark_case.repetitions
            << std::setw(9) << "yes"
            << std::setw(10) << std::setprecision(3)
            << current_cuda_summary.median << "/"
            << std::setw(7) << current_wall_summary.median
            << std::setw(10) << candidate_cuda_summary.median << "/"
            << std::setw(7) << candidate_wall_summary.median
            << std::setw(11) << std::showpos << std::setprecision(2)
            << 100.0 * cuda_delta << "%"
            << std::setw(11) << 100.0 * wall_delta << "%"
            << std::noshowpos << "\n";
    }

    std::vector<double> current_pipeline_cuda;
    std::vector<double> current_pipeline_wall;
    std::vector<double> candidate_pipeline_cuda;
    std::vector<double> candidate_pipeline_wall;
    std::vector<double> paired_pipeline_cuda_ratios;
    std::vector<double> paired_pipeline_wall_ratios;
    for (std::size_t repetition = 0U;
         repetition < Traits::pipeline_repetitions;
         ++repetition) {
        const bool candidate_first = repetition % 2U != 0U;
        TimedRun current_run{};
        TimedRun candidate_run{};
        for (int order = 0; order < 2; ++order) {
            const bool candidate = order == 0
                ? candidate_first
                : !candidate_first;
            const TimedRun run = time_complete_pipeline<Traits>(
                candidate,
                models,
                products
            );
            (candidate ? candidate_run : current_run) = run;
            (candidate ? candidate_pipeline_cuda : current_pipeline_cuda)
                .push_back(run.cuda_milliseconds);
            (candidate ? candidate_pipeline_wall : current_pipeline_wall)
                .push_back(run.wall_milliseconds);
        }
        paired_pipeline_cuda_ratios.push_back(
            candidate_run.cuda_milliseconds
                / current_run.cuda_milliseconds
        );
        paired_pipeline_wall_ratios.push_back(
            candidate_run.wall_milliseconds
                / current_run.wall_milliseconds
        );
    }
    const TimingSummary current_pipeline_cuda_summary =
        summarize(current_pipeline_cuda);
    const TimingSummary current_pipeline_wall_summary =
        summarize(current_pipeline_wall);
    const TimingSummary candidate_pipeline_cuda_summary =
        summarize(candidate_pipeline_cuda);
    const TimingSummary candidate_pipeline_wall_summary =
        summarize(candidate_pipeline_wall);
    const double pipeline_cuda_delta =
        summarize(paired_pipeline_cuda_ratios).median - 1.0;
    const double pipeline_wall_delta =
        summarize(paired_pipeline_wall_ratios).median - 1.0;
    std::cout
        << "complete pipeline (1000 rows, "
        << Traits::pipeline_repetitions
        << " repeats): current " << std::setprecision(3)
        << current_pipeline_cuda_summary.median << "/"
        << current_pipeline_wall_summary.median << " ms, policy "
        << candidate_pipeline_cuda_summary.median << "/"
        << candidate_pipeline_wall_summary.median << " ms, deltas "
        << std::showpos << std::setprecision(2)
        << 100.0 * pipeline_cuda_delta << "%/"
        << 100.0 * pipeline_wall_delta << "%"
        << std::noshowpos << "\n";
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice || device_count == 0) return 77;
    check_cuda(availability, "Athena policy benchmark cudaGetDeviceCount");

    const auto products =
        ai_factory::workbench::product::load_athena_autocalls(
            "datasets/product/equity/athena_autocalls/"
            "athena_autocalls_01.json"
        );
    benchmark_model<CevTraits>(products);
    benchmark_model<MertonTraits>(products);
    std::cout << "\nAll current/policy outputs are bitwise identical.\n";
    return 0;
}

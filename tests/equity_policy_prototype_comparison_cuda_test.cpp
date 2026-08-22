// Bitwise and timing comparison for representative equity policy prototypes.
#include "common/check_cuda.cuh"
#include "common/option_side.cuh"
#include "model/equity/bates/athena_autocall.cuh"
#include "model/equity/bates/athena_autocallbis.cuh"
#include "model/equity/heston/asian_option.cuh"
#include "model/equity/heston/asian_optionbis.cuh"
#include "model/equity/heston/up_and_out_option.cuh"
#include "model/equity/heston/up_and_out_optionbis.cuh"
#include "model/equity/merton/forward_start_option.cuh"
#include "model/equity/merton/forward_start_optionbis.cuh"
#include "model/equity/variance_gamma/asian_option.cuh"
#include "model/equity/variance_gamma/asian_optionbis.cuh"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ai_factory::workbench::OptionSide;
using ai_factory::workbench::check_cuda;

constexpr std::size_t kBenchmarkPathsPerPrice = 2'048U;
constexpr unsigned int kBenchmarkThreadsPerBlock = 256U;
constexpr std::size_t kProductionPathsPerPrice = 16'384U;
constexpr unsigned int kProductionThreadsPerBlock = 512U;
constexpr std::uint64_t kSeed = 900000001ULL;
constexpr std::size_t kMaximumResultCount = 4'096U;

struct BenchmarkCase {
    const char* name;
    std::size_t model_count;
    std::size_t product_count;
    bool cartesian_product;
    std::size_t result_count;
    std::size_t repetitions;
};

constexpr BenchmarkCase kCases[] = {
    {"tiny", 1U, 1U, false, 1U, 10U},
    {"small", 64U, 64U, false, 64U, 8U},
    {"dataset", 1'000U, 1'000U, false, 1'000U, 8U},
    {"large", 64U, 64U, true, 4'096U, 6U},
};

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_(size) {
        check_cuda(cudaMalloc(&data_, size * sizeof(T)), "prototype cudaMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    T* data() { return data_; }

    void copy_from_host(const std::vector<T>& values) {
        if (values.size() > size_) {
            throw std::invalid_argument("Prototype input is too large");
        }
        check_cuda(
            cudaMemcpy(
                data_,
                values.data(),
                values.size() * sizeof(T),
                cudaMemcpyHostToDevice
            ),
            "prototype input copy"
        );
    }

    std::vector<T> copy_to_host(std::size_t count) const {
        if (count > size_) {
            throw std::invalid_argument("Prototype output is too large");
        }
        std::vector<T> values(count);
        check_cuda(
            cudaMemcpy(
                values.data(),
                data_,
                count * sizeof(T),
                cudaMemcpyDeviceToHost
            ),
            "prototype output copy"
        );
        return values;
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0U;
};

void require_bitwise_equal(
    const std::vector<float>& current,
    const std::vector<float>& candidate,
    const std::string& label
) {
    if (current.size() != candidate.size()) {
        throw std::runtime_error(label + ": result sizes differ");
    }
    for (std::size_t index = 0U; index < current.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(current[index])
            != std::bit_cast<std::uint32_t>(candidate[index])) {
            throw std::runtime_error(
                label + ": mismatch at row " + std::to_string(index)
                + " (" + std::to_string(current[index]) + " versus "
                + std::to_string(candidate[index]) + ", bits "
                + std::to_string(std::bit_cast<std::uint32_t>(current[index]))
                + " versus "
                + std::to_string(std::bit_cast<std::uint32_t>(candidate[index]))
                + ")"
            );
        }
    }
}

struct ReferenceDataset {
    std::vector<float> prices;
    std::vector<float> standard_errors;
};

struct BitwiseDifferenceSummary {
    std::size_t mismatch_count = 0U;
    std::uint32_t maximum_ulp_distance = 0U;
    double maximum_absolute_difference = 0.0;
};

BitwiseDifferenceSummary summarize_bitwise_differences(
    const std::vector<float>& left,
    const std::vector<float>& right
) {
    if (left.size() != right.size()) {
        throw std::runtime_error("Reference result sizes differ");
    }
    BitwiseDifferenceSummary summary;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        const std::uint32_t left_bits =
            std::bit_cast<std::uint32_t>(left[index]);
        const std::uint32_t right_bits =
            std::bit_cast<std::uint32_t>(right[index]);
        if (left_bits == right_bits) continue;
        ++summary.mismatch_count;
        const std::uint32_t ulp_distance = left_bits > right_bits
            ? left_bits - right_bits
            : right_bits - left_bits;
        summary.maximum_ulp_distance = std::max(
            summary.maximum_ulp_distance,
            ulp_distance
        );
        summary.maximum_absolute_difference = std::max(
            summary.maximum_absolute_difference,
            static_cast<double>(fabsf(left[index] - right[index]))
        );
    }
    return summary;
}

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
    return reference;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    return values.size() % 2U == 0U
        ? 0.5 * (values[middle - 1U] + values[middle])
        : values[middle];
}

template<typename Model, typename Product, typename Launcher>
void compare_prototype(
    const char* name,
    const char* reference_path,
    const std::vector<Model>& models,
    const std::vector<Product>& products,
    Launcher launcher
) {
    if (models.size() < 1'000U || products.size() < 1'000U) {
        throw std::runtime_error(std::string(name) + ": requires 1000 rows");
    }

    DeviceBuffer<Model> device_models(models.size());
    DeviceBuffer<Product> device_products(products.size());
    DeviceBuffer<float> current_prices(kMaximumResultCount);
    DeviceBuffer<float> current_errors(kMaximumResultCount);
    DeviceBuffer<float> candidate_prices(kMaximumResultCount);
    DeviceBuffer<float> candidate_errors(kMaximumResultCount);
    device_models.copy_from_host(models);
    device_products.copy_from_host(products);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "prototype event start");
    check_cuda(cudaEventCreate(&stop), "prototype event stop");

    std::cout << '\n' << name << '\n'
              << "case      rows  repeats  bitwise  current CUDA/wall ms"
              << "  policy CUDA/wall ms  CUDA delta  wall delta\n";
    for (const BenchmarkCase& benchmark_case : kCases) {
        const std::size_t block_count = std::min(
            benchmark_case.result_count,
            kMaximumResultCount
        );
        launcher(
            false,
            device_models.data(),
            benchmark_case.model_count,
            device_products.data(),
            benchmark_case.product_count,
            benchmark_case.cartesian_product,
            benchmark_case.result_count,
            kBenchmarkPathsPerPrice,
            kBenchmarkThreadsPerBlock,
            block_count,
            current_prices.data(),
            current_errors.data()
        );
        launcher(
            true,
            device_models.data(),
            benchmark_case.model_count,
            device_products.data(),
            benchmark_case.product_count,
            benchmark_case.cartesian_product,
            benchmark_case.result_count,
            kBenchmarkPathsPerPrice,
            kBenchmarkThreadsPerBlock,
            block_count,
            candidate_prices.data(),
            candidate_errors.data()
        );
        check_cuda(cudaDeviceSynchronize(), "prototype comparison synchronize");

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
            double paired_cuda[2]{};
            double paired_wall[2]{};
            for (int order = 0; order < 2; ++order) {
                const bool candidate = order == 0
                    ? candidate_first
                    : !candidate_first;
                const auto wall_start = std::chrono::steady_clock::now();
                check_cuda(cudaEventRecord(start), "prototype timing start");
                launcher(
                    candidate,
                    device_models.data(),
                    benchmark_case.model_count,
                    device_products.data(),
                    benchmark_case.product_count,
                    benchmark_case.cartesian_product,
                    benchmark_case.result_count,
                    kBenchmarkPathsPerPrice,
                    kBenchmarkThreadsPerBlock,
                    block_count,
                    candidate
                        ? candidate_prices.data() : current_prices.data(),
                    candidate
                        ? candidate_errors.data() : current_errors.data()
                );
                check_cuda(cudaEventRecord(stop), "prototype timing stop");
                check_cuda(
                    cudaEventSynchronize(stop),
                    "prototype timing synchronize"
                );
                float cuda_milliseconds = 0.0f;
                check_cuda(
                    cudaEventElapsedTime(
                        &cuda_milliseconds,
                        start,
                        stop
                    ),
                    "prototype elapsed time"
                );
                const double wall_milliseconds =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - wall_start
                    ).count();
                const std::size_t candidate_index = candidate ? 1U : 0U;
                paired_cuda[candidate_index] = cuda_milliseconds;
                paired_wall[candidate_index] = wall_milliseconds;
                (candidate ? candidate_cuda : current_cuda).push_back(
                    cuda_milliseconds
                );
                (candidate ? candidate_wall : current_wall).push_back(
                    wall_milliseconds
                );
            }
            paired_cuda_ratios.push_back(paired_cuda[1] / paired_cuda[0]);
            paired_wall_ratios.push_back(paired_wall[1] / paired_wall[0]);
        }

        require_bitwise_equal(
            current_prices.copy_to_host(benchmark_case.result_count),
            candidate_prices.copy_to_host(benchmark_case.result_count),
            std::string(name) + " prices/" + benchmark_case.name
        );
        require_bitwise_equal(
            current_errors.copy_to_host(benchmark_case.result_count),
            candidate_errors.copy_to_host(benchmark_case.result_count),
            std::string(name) + " errors/" + benchmark_case.name
        );

        const double current_cuda_median = median(current_cuda);
        const double current_wall_median = median(current_wall);
        const double candidate_cuda_median = median(candidate_cuda);
        const double candidate_wall_median = median(candidate_wall);
        const double cuda_delta = median(paired_cuda_ratios) - 1.0;
        const double wall_delta = median(paired_wall_ratios) - 1.0;
        std::cout
            << std::left << std::setw(9) << benchmark_case.name
            << std::right << std::setw(5) << benchmark_case.result_count
            << std::setw(9) << benchmark_case.repetitions
            << std::setw(9) << "yes"
            << std::setw(10) << std::fixed << std::setprecision(3)
            << current_cuda_median << "/"
            << std::setw(7) << current_wall_median
            << std::setw(10) << candidate_cuda_median << "/"
            << std::setw(7) << candidate_wall_median
            << std::setw(11) << std::showpos << std::setprecision(2)
            << 100.0 * cuda_delta << "%"
            << std::setw(11) << 100.0 * wall_delta << "%"
            << std::noshowpos << '\n';
    }

    if (reference_path != nullptr) {
        launcher(
            false,
            device_models.data(),
            1'000U,
            device_products.data(),
            1'000U,
            false,
            1'000U,
            kProductionPathsPerPrice,
            kProductionThreadsPerBlock,
            1'000U,
            current_prices.data(),
            current_errors.data()
        );
        launcher(
            true,
            device_models.data(),
            1'000U,
            device_products.data(),
            1'000U,
            false,
            1'000U,
            kProductionPathsPerPrice,
            kProductionThreadsPerBlock,
            1'000U,
            candidate_prices.data(),
            candidate_errors.data()
        );
        check_cuda(
            cudaDeviceSynchronize(),
            "prototype production-reference synchronize"
        );
        const ReferenceDataset reference = load_reference_dataset(
            reference_path
        );
        require_bitwise_equal(
            current_prices.copy_to_host(1'000U),
            candidate_prices.copy_to_host(1'000U),
            std::string(name) + " production current/policy prices"
        );
        require_bitwise_equal(
            current_errors.copy_to_host(1'000U),
            candidate_errors.copy_to_host(1'000U),
            std::string(name) + " production current/policy errors"
        );
        const BitwiseDifferenceSummary price_differences =
            summarize_bitwise_differences(
                reference.prices,
                current_prices.copy_to_host(1'000U)
            );
        const BitwiseDifferenceSummary error_differences =
            summarize_bitwise_differences(
                reference.standard_errors,
                current_errors.copy_to_host(1'000U)
            );
        std::cout
            << "stored production dataset: price mismatches="
            << price_differences.mismatch_count
            << " (max " << price_differences.maximum_ulp_distance
            << " ULP, " << std::scientific
            << price_differences.maximum_absolute_difference
            << " abs), error mismatches="
            << error_differences.mismatch_count
            << " (max " << error_differences.maximum_ulp_distance
            << " ULP, " << error_differences.maximum_absolute_difference
            << " abs)" << std::fixed << '\n';
    }
    check_cuda(cudaEventDestroy(start), "prototype destroy start event");
    check_cuda(cudaEventDestroy(stop), "prototype destroy stop event");
}

template<OptionSide Side>
void compare_heston_asian(
    const std::vector<ai_factory::workbench::heston::ModelParameters>& models,
    const std::vector<ai_factory::workbench::product::AsianOptionParameters>&
        products
) {
    using namespace ai_factory::workbench;
    compare_prototype(
        Side == OptionSide::call ? "Heston Asian call" : "Heston Asian put",
        Side == OptionSide::call
            ? "datasets/model/equity/heston/prices/asian_calls/"
              "heston_01__asian_calls_01__01.json"
            : "datasets/model/equity/heston/prices/asian_puts/"
              "heston_01__asian_puts_01__01.json",
        models,
        products,
        [](bool candidate, const heston::ModelParameters* model,
           std::size_t model_count,
           const product::AsianOptionParameters* product,
           std::size_t product_count, bool cartesian_product,
           std::size_t result_count,
           std::size_t paths_per_price, unsigned int threads_per_block,
           std::size_t block_count, float* prices, float* errors) {
            constexpr float dt = 1.0f / 504.0f;
            constexpr std::uint32_t steps_per_day = 2U;
            if (candidate) {
                heston::launch_heston_asian_optionbis_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            } else {
                heston::launch_heston_asian_option_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            }
        }
    );
}

template<OptionSide Side>
void compare_heston_up_and_out(
    const std::vector<ai_factory::workbench::heston::ModelParameters>& models,
    const std::vector<
        ai_factory::workbench::product::UpAndOutOptionParameters
    >& products
) {
    using namespace ai_factory::workbench;
    compare_prototype(
        Side == OptionSide::call
            ? "Heston up-and-out call" : "Heston up-and-out put",
        Side == OptionSide::call
            ? "datasets/model/equity/heston/prices/up_and_out_calls/"
              "heston_01__up_and_out_calls_01__01.json"
            : nullptr,
        models,
        products,
        [](bool candidate, const heston::ModelParameters* model,
           std::size_t model_count,
           const product::UpAndOutOptionParameters* product,
           std::size_t product_count, bool cartesian_product,
           std::size_t result_count,
           std::size_t paths_per_price, unsigned int threads_per_block,
           std::size_t block_count, float* prices, float* errors) {
            constexpr float dt = 1.0f / 504.0f;
            constexpr std::uint32_t steps_per_day = 2U;
            if (candidate) {
                heston::launch_heston_up_and_out_optionbis_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            } else {
                heston::launch_heston_up_and_out_option_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            }
        }
    );
}

template<OptionSide Side>
void compare_merton_forward_start(
    const std::vector<ai_factory::workbench::merton::ModelParameters>& models,
    const std::vector<
        ai_factory::workbench::product::ForwardStartOptionParameters
    >& products
) {
    using namespace ai_factory::workbench;
    compare_prototype(
        Side == OptionSide::call
            ? "Merton forward-start call" : "Merton forward-start put",
        Side == OptionSide::call
            ? "datasets/model/equity/merton/prices/forward_start_calls/"
              "merton_01__forward_start_calls_01__01.json"
            : "datasets/model/equity/merton/prices/forward_start_puts/"
              "merton_01__forward_start_puts_01__01.json",
        models,
        products,
        [](bool candidate, const merton::ModelParameters* model,
           std::size_t model_count,
           const product::ForwardStartOptionParameters* product,
           std::size_t product_count, bool cartesian_product,
           std::size_t result_count,
           std::size_t paths_per_price, unsigned int threads_per_block,
           std::size_t block_count, float* prices, float* errors) {
            constexpr float day_fraction = 1.0f / 252.0f;
            if (candidate) {
                merton::launch_merton_forward_start_optionbis_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    day_fraction, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            } else {
                merton::launch_merton_forward_start_option_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    day_fraction, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            }
        }
    );
}

template<OptionSide Side>
void compare_variance_gamma_asian(
    const std::vector<
        ai_factory::workbench::variance_gamma::ModelParameters
    >& models,
    const std::vector<ai_factory::workbench::product::AsianOptionParameters>&
        products
) {
    using namespace ai_factory::workbench;
    compare_prototype(
        Side == OptionSide::call
            ? "Variance-Gamma Asian call" : "Variance-Gamma Asian put",
        Side == OptionSide::call
            ? "datasets/model/equity/variance_gamma/prices/asian_calls/"
              "variance_gamma_01__asian_calls_01__01.json"
            : "datasets/model/equity/variance_gamma/prices/asian_puts/"
              "variance_gamma_01__asian_puts_01__01.json",
        models,
        products,
        [](bool candidate, const variance_gamma::ModelParameters* model,
           std::size_t model_count,
           const product::AsianOptionParameters* product,
           std::size_t product_count, bool cartesian_product,
           std::size_t result_count,
           std::size_t paths_per_price, unsigned int threads_per_block,
           std::size_t block_count, float* prices, float* errors) {
            constexpr float dt = 1.0f / 504.0f;
            constexpr std::uint32_t steps_per_day = 2U;
            if (candidate) {
                variance_gamma::launch_variance_gamma_asian_optionbis_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            } else {
                variance_gamma::launch_variance_gamma_asian_option_cuda<Side>(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            }
        }
    );
}

void compare_bates_athena(
    const std::vector<ai_factory::workbench::bates::ModelParameters>& models,
    const std::vector<
        ai_factory::workbench::product::AthenaAutocallParameters
    >& products
) {
    using namespace ai_factory::workbench;
    compare_prototype(
        "Bates Athena autocall",
        "datasets/model/equity/bates/prices/athena_autocalls/"
        "bates_01__athena_autocalls_01__01.json",
        models,
        products,
        [](bool candidate, const bates::ModelParameters* model,
           std::size_t model_count,
           const product::AthenaAutocallParameters* product,
           std::size_t product_count, bool cartesian_product,
           std::size_t result_count,
           std::size_t paths_per_price, unsigned int threads_per_block,
           std::size_t block_count, float* prices, float* errors) {
            constexpr float dt = 1.0f / 504.0f;
            constexpr std::uint32_t steps_per_day = 2U;
            if (candidate) {
                bates::launch_bates_athena_autocallbis_cuda(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            } else {
                bates::launch_bates_athena_autocall_cuda(
                    model, model_count, product, product_count,
                    cartesian_product,
                    result_count, 0U, result_count, paths_per_price,
                    dt, steps_per_day, threads_per_block, block_count,
                    kSeed, prices, errors
                );
            }
        }
    );
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice || device_count == 0) return 77;
    check_cuda(availability, "prototype cudaGetDeviceCount");

    using namespace ai_factory::workbench;
    const auto bates_models = bates::load_models(
        "datasets/model/equity/bates/parameters/bates_01.json"
    );
    const auto heston_models = heston::load_models(
        "datasets/model/equity/heston/parameters/heston_01.json"
    );
    const auto merton_models = merton::load_models(
        "datasets/model/equity/merton/parameters/merton_01.json"
    );
    const auto variance_gamma_models = variance_gamma::load_models(
        "datasets/model/equity/variance_gamma/parameters/variance_gamma_01.json"
    );
    const auto athena_products = product::load_athena_autocalls(
        "datasets/product/equity/athena_autocalls/athena_autocalls_01.json"
    );
    const auto asian_products = product::load_asian_options(
        "datasets/product/equity/asian_options/asian_options_01.json"
    );
    const auto barrier_products = product::load_up_and_out_options(
        "datasets/product/equity/up_and_out_options/up_and_out_options_01.json"
    );
    const auto forward_products = product::load_forward_start_options(
        "datasets/product/equity/forward_start_options/forward_start_options_01.json"
    );

    compare_bates_athena(bates_models, athena_products);
    compare_heston_asian<OptionSide::call>(heston_models, asian_products);
    compare_heston_asian<OptionSide::put>(heston_models, asian_products);
    compare_heston_up_and_out<OptionSide::call>(
        heston_models, barrier_products
    );
    compare_heston_up_and_out<OptionSide::put>(
        heston_models, barrier_products
    );
    compare_merton_forward_start<OptionSide::call>(
        merton_models, forward_products
    );
    compare_merton_forward_start<OptionSide::put>(
        merton_models, forward_products
    );
    compare_variance_gamma_asian<OptionSide::call>(
        variance_gamma_models, asian_products
    );
    compare_variance_gamma_asian<OptionSide::put>(
        variance_gamma_models, asian_products
    );
    return 0;
}

// Bounded, row-preserving qualification of the production terminal-forward CIR LSM.
// Reads jobs and catalogue inputs; emits NDJSON only, never publishes datasets.
#include "tests/performance/benchmark_support.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "model/fixed_income/cir/product/bermudan_swaption.cuh"
#include "product/bermudan_swaption/dataset.hpp"

#include <fstream>
#include <bit>

namespace wb = ai_factory::workbench;
namespace perf = wb::performance;
namespace cir = wb::model::fixed_income::cir;
namespace lsm = wb::longstaff_schwartz;
using Json = nlohmann::ordered_json;

template<wb::SwaptionSide Side>
void run_job(const Json& job, const std::vector<cir::ModelParameters>& all_models,
             const std::vector<wb::product::BermudanSwaptionParameters>& all_products) {
    using Product = wb::product::BermudanSwaptionParameters;
    const auto indices = job.at("indices").get<std::vector<std::size_t>>();
    const std::size_t paths = job.at("paths");
    const std::size_t chunk_rows = job.value("chunk_rows", 32U);
    const unsigned threads = job.value("threads", 128U);
    const std::size_t blocks = job.value("blocks", 64U);
    const std::uint64_t seed = job.at("seed");
    const std::string method = job.at("method");
    if (method != "production")
        throw std::invalid_argument("Use production; historical Q/prototype runs are archived in the comparison report");
    if (indices.empty() || indices.size() > 1000 || paths < 2 || paths > (1ULL << 22)
        || !chunk_rows || chunk_rows > 128)
        throw std::invalid_argument("Experiment bounds exceeded");
    std::vector<cir::ModelParameters> models;
    std::vector<Product> products;
    for (auto index : indices) {
        models.push_back(all_models.at(index));
        products.push_back(all_products.at(index));
    }
    perf::DeviceBuffer dm(models.size() * sizeof(models[0]), perf::DeviceMemoryRole::persistent_input);
    perf::DeviceBuffer dp(products.size() * sizeof(products[0]), perf::DeviceMemoryRole::persistent_input);
    perf::DeviceBuffer prices(models.size() * sizeof(float), perf::DeviceMemoryRole::output);
    perf::DeviceBuffer errors(models.size() * sizeof(float), perf::DeviceMemoryRole::output);
    perf::copy_to_device(dm, models);
    perf::copy_to_device(dp, products);
    std::cout << Json{{"event", "start"}, {"job", job}, {"environment", perf::environment_json()}}
        .dump() << std::endl;
    double gpu_seconds = 0.0;
    std::size_t peak_workspace = 0;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t begin = 0; begin < indices.size();) {
        // Contiguous source IDs share one launch. Sparse probes never change row keys.
        std::size_t count = 1;
        while (count < chunk_rows && begin + count < indices.size()
               && indices[begin + count] == indices[begin] + count) ++count;
        wb::validate_row_seed_range(indices[begin] + count, seed);
        const auto* device_models = dm.as<cir::ModelParameters>() + begin;
        const auto* device_products = dp.as<Product>() + begin;
        auto* device_prices = prices.as<float>() + begin;
        auto* device_errors = errors.as<float>() + begin;
        const auto chunk_started = std::chrono::steady_clock::now();
        const lsm::LaunchResult result = cir::launch_cir_bermudan_swaption_cuda<Side>(
            device_models, count, products.data() + begin, device_products, count,
            wb::PriceConstruction::Aligned, count, paths, 1.0f / 252.0f,
            threads, blocks, seed + indices[begin], device_prices, device_errors
        );
        lsm::validate_regression_diagnostics(result, "CIR method comparison");
        gpu_seconds += result.kernel_seconds;
        peak_workspace = std::max(peak_workspace, result.workspace_bytes);
        std::cout << Json{{"event", "chunk"}, {"job_id", job.at("id")},
            {"first_source_index", indices[begin]}, {"count", count},
            {"gpu_seconds", result.kernel_seconds},
            {"host_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - chunk_started).count()},
            {"workspace_bytes", result.workspace_bytes}, {"batches", result.batch_count},
            {"successful_regressions", result.regression_diagnostics.successful_regression_count},
            {"no_candidates", result.regression_diagnostics.no_candidate_count},
            {"insufficient_candidates", result.regression_diagnostics.insufficient_candidate_count}}
            .dump() << std::endl;
        begin += count;
    }
    const double host_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto values = perf::copy_from_device<float>(prices, indices.size());
    const auto standard_errors = perf::copy_from_device<float>(errors, indices.size());
    Json rows = Json::array();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (!std::isfinite(values[i]) || !std::isfinite(standard_errors[i])
            || values[i] < 0 || standard_errors[i] < 0) throw std::runtime_error("Invalid row output");
        rows.push_back({{"source_index", indices[i]}, {"price", values[i]},
            {"standard_error", standard_errors[i]}, {"price_bits", std::bit_cast<std::uint32_t>(values[i])}});
    }
    std::cout << Json{{"event", "result"}, {"job", job}, {"rows", rows},
        {"gpu_seconds", gpu_seconds}, {"raw_host_seconds", host_seconds},
        {"workspace_bytes", peak_workspace}}.dump() << std::endl;
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--jobs")
            throw std::invalid_argument("Usage: cir_forward_measure_probe --jobs jobs.json");
        std::ifstream stream(argv[2]);
        const auto jobs = Json::parse(stream);
        const auto models = cir::load_models("datasets/model/fixed_income/cir/parameters/cir_01.json");
        const auto products = wb::product::load_bermudan_swaptions(
            "datasets/product/bermudan_swaption/bermudan_swaptions_01.json");
        for (const auto& job : jobs) {
            const std::string side = job.value("side", "payer");
            if (side == "payer") run_job<wb::SwaptionSide::payer>(job, models, products);
            else if (side == "receiver") run_job<wb::SwaptionSide::receiver>(job, models, products);
            else throw std::invalid_argument("Unknown swaption side");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

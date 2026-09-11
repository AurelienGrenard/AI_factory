// Verify generated FFT recipe metadata and real replay for both sample layouts.
#include "model/equity/rough/rough_bergomi/sample.cuh"
#include "model/equity/rough/rough_sabr/sample.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/sample.cuh"
#include "model/equity/rough/rough_stein_stein/sample.cuh"
#include "tools/sampling/generated/rough_bergomi_sample_generation.cuh"
#include "tools/sampling/generated/rough_sabr_sample_generation.cuh"
#include "tools/sampling/generated/log_modulated_rough_bergomi_sample_generation.cuh"
#include "tools/sampling/generated/rough_stein_stein_sample_generation.cuh"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
namespace wb = ai_factory::workbench;
namespace sampling = wb::offline::sampling;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_plans() {
    // No production-sized allocation: inspect the exact host plans only.
    const sampling::ModelSampleProfile fft{256U, 4096U, "volterra_samples", dim3(128U)};
    const sampling::ModelSampleProfile markovian{256U, 4096U, "markovian_samples"};
    for (const std::size_t paths : {1U, 250U}) {
        const auto primary = sampling::model_sample_launch(3'000'000U, paths, fft);
        const auto replay = sampling::model_sample_launch(3'000'000U, paths, fft, true);
        require(primary.block_count == 4096U && replay.block_count == 4095U,
                "FFT production replay does not change the effective grid");
        require(primary.requested_threads == replay.requested_threads,
                "FFT replay pretends to change an ignored thread argument");
        const auto metadata = sampling::model_sample_launch_metadata(primary, paths, fft);
        require(metadata.at("threads_per_block") == 128U
                && metadata.at("execution_strategy") == "persistent parameter-block",
                "FFT metadata describes a Markovian launch");
        const auto dynamic = sampling::model_sample_launch(3'000'000U, paths, markovian, true);
        require(dynamic.requested_threads == 128U && dynamic.block.x == 128U,
                "Markovian replay no longer changes its real block dimensions");
    }
    const auto single = sampling::model_sample_launch(250U, 250U, fft, true);
    require(single.block_count == 1U, "Single-block replay became invalid");
}

template<typename Recipe, typename Generate, typename Dimensions>
void check_recipes(const char* name, Recipe recipe, Generate generate, Dimensions dimensions) {
    for (const std::size_t paths : {250U, 1U}) {
        // Use the real preflight pipeline, but bound the fixture to 1,000 rows.
        auto value = recipe(paths == 250U ? "samples_01" : "samples_02",
                            paths == 250U ? 4U : 1000U, paths,
                            wb::datasets::ModelSampleSeeds{8192U, 16384U, 32768U});
        const auto native = dimensions(value.maximum_maturity_days);
        std::ostringstream captured;
        auto* previous = std::cout.rdbuf(captured.rdbuf());
        char program[] = "sample_recipe_test";
        char option[] = "--preflight";
        char* arguments[]{program, option};
        try {
            require(generate(2, arguments, value) == 0, "Sample preflight failed");
        } catch (...) {
            std::cout.rdbuf(previous);
            throw;
        }
        std::cout.rdbuf(previous);
        const std::string prefix = "MODEL_SAMPLE_PREFLIGHT ";
        require(captured.str().starts_with(prefix), "Missing JSON preflight record");
        const auto record = nlohmann::ordered_json::parse(captured.str().substr(prefix.size()));
        const auto& primary = record.at("primary_launch");
        const auto& replay = record.at("replay_launch");
        require(record.at("finite") && record.at("deterministic_replay"), "Replay not qualified");
        require(record.at("replay_scope") == "grid_block_count", "Wrong FFT replay scope");
        require(primary.at("threads_per_block") == native.x * native.y * native.z,
                "Published threads differ from the compiled FFT descriptor");
        require(primary.at("block_dimensions") == nlohmann::ordered_json({native.x, native.y, native.z})
                && primary.at("block_dimensions") == replay.at("block_dimensions"),
                "FFT replay changed or misstated its compile-time block");
        require(primary.at("block_count").template get<std::size_t>()
                    == replay.at("block_count").template get<std::size_t>() + 1U,
                "FFT replay reused its primary grid");
        std::cout << name << " paths=" << paths << ' ' << captured.str();
    }
}
}  // namespace

int main() try {
    check_plans();
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || devices == 0) return 77;
    wb::check_cuda(status, "recipe replay device discovery");
    namespace model = wb::model::equity;
    check_recipes("rough_bergomi", sampling::rough_bergomi::recipe, sampling::rough_bergomi::generate,
                  model::rough_bergomi::rough_bergomi_sample_block_dimensions);
    check_recipes("rough_sabr", sampling::rough_sabr::recipe, sampling::rough_sabr::generate,
                  model::rough_sabr::rough_sabr_sample_block_dimensions);
    check_recipes("log_modulated_rough_bergomi", sampling::log_modulated_rough_bergomi::recipe,
                  sampling::log_modulated_rough_bergomi::generate,
                  model::log_modulated_rough_bergomi::log_modulated_rough_bergomi_sample_block_dimensions);
    check_recipes("rough_stein_stein", sampling::rough_stein_stein::recipe, sampling::rough_stein_stein::generate,
                  model::rough_stein_stein::rough_stein_stein_sample_block_dimensions);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

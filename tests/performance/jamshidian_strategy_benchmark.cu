// Compare existing scalar/cooperative Jamshidian policies without changing pricing.
// Repeated catalogue rows are throughput fixtures, not independent new datasets.
#include "tests/performance/benchmark_support.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "product/european_swaption/pricing_policy.cuh"
#include "product/european_swaption/dataset.hpp"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "model/fixed_income/cir_plus_plus/nelson_siegel/analytics_impl.cuh"
#include "model/fixed_income/cir_plus_plus/svensson/analytics_impl.cuh"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"
#include "model/fixed_income/hull_white/nelson_siegel/analytics_impl.cuh"
#include "model/fixed_income/hull_white/svensson/analytics_impl.cuh"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/analytics_impl.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "model/fixed_income/vasicek/analytics_impl.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"
#include "curve/nelson_siegel/dataset.hpp"
#include "curve/svensson/dataset.hpp"

#include <bit>
#include <fstream>

namespace wb = ai_factory::workbench;
namespace rates = wb::model::fixed_income;
namespace perf = wb::performance;
namespace cf = wb::closed_form;
using Json = nlohmann::ordered_json;
using Product = wb::product::RegularEuropeanSwaptionParameters;
using Source = wb::product::RegularEuropeanSwaptionScheduleSource;
using Buffer = wb::offline::cuda::DeviceBuffer<float>;
using Clock = std::chrono::steady_clock;

constexpr double kAbsoluteTolerance = 2e-6;
constexpr double kRelativeTolerance = 2e-5;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <class T>
std::vector<T> repeat_rows(const std::vector<T> &original, std::size_t count,
                           const std::vector<std::size_t> &selection) {
    std::vector<T> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        result.push_back(original.at(selection[i % selection.size()]));
    return result;
}

// Keep raw host and CUDA samples separate, including very short launch groups.
template <class Launch>
Json measure(Launch &&launch, int warmups, int repetitions, int operations) {
    for (int i = 0; i < warmups; ++i) {
        for (int op = 0; op < operations; ++op)
            launch();
        wb::check_cuda(cudaDeviceSynchronize(), "Jamshidian warmup");
    }
    wb::offline::cuda::Event start, stop;
    std::vector<double> gpu, host, enclosing;
    for (int i = 0; i < repetitions; ++i) {
        const auto begin = Clock::now();
        wb::check_cuda(cudaEventRecord(start.get()), "Jamshidian start");
        for (int op = 0; op < operations; ++op)
            launch();
        wb::check_cuda(cudaEventRecord(stop.get()), "Jamshidian stop");
        wb::check_cuda(cudaEventSynchronize(stop.get()), "Jamshidian synchronize");
        const double host_ms = elapsed_ms(begin) / operations;
        float gpu_ms = 0;
        wb::check_cuda(cudaEventElapsedTime(&gpu_ms, start.get(), stop.get()),
                       "Jamshidian elapsed");
        gpu.push_back(gpu_ms / operations);
        host.push_back(host_ms);
        enclosing.push_back(std::max(gpu.back(), host.back()));
    }
    return {{"kernel", perf::timing_json(perf::summarize(gpu))},
            {"public_api", perf::timing_json(perf::summarize(enclosing))},
            {"raw_host_clock", perf::timing_json(perf::summarize(host))},
            {"gpu_samples_ms", gpu},
            {"raw_host_samples_ms", host},
            {"warmups", warmups},
            {"repetitions", repetitions},
            {"operations_per_sample", operations}};
}

template <class Policy>
void evaluate(const Json &plan, const typename Policy::DeviceInputs &inputs, std::size_t capacity,
              const std::vector<std::size_t> &selection, std::uint32_t payments, float *prices,
              double preparation_ms, std::size_t input_bytes) {
    const typename Policy::TimeConfiguration time{1.f / 252.f};
    const auto scalar = [&](std::size_t count, unsigned threads, std::size_t blocks) {
        cf::launch_closed_form_cuda<Policy>(inputs, capacity, 0, count, time, threads, blocks,
                                            prices, "jamshidian_probe", "scalar",
                                            "Jamshidian scalar benchmark");
    };
    // One reference tile, never a second million-price baseline calculation.
    scalar(selection.size(), 256, (selection.size() + 255) / 256);
    std::vector<float> reference(selection.size());
    wb::check_cuda(cudaMemcpy(reference.data(), prices, reference.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "Jamshidian reference copy");
    Json reference_issues = Json::array();
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (std::isfinite(reference[i]) && reference[i] >= 0)
            continue;
        reference_issues.push_back({{"source_index", selection[i]},
                                    {"value", reference[i]},
                                    {"finite", std::isfinite(reference[i])}});
    }
    // CIR's catalogue launcher is cooperative. Keep an explicit record of
    // scalar failures, then use that production path as comparison reference.
    const bool cooperative_reference = plan.at("model") == "cir";
    if (cooperative_reference) {
        if (!cf::launch_cooperative_closed_form_cuda<Policy>(
                inputs, capacity, 0, selection.size(), time, payments, 128, selection.size(),
                prices, "jamshidian_reference", "cooperative", "Jamshidian cooperative reference"))
            throw std::runtime_error("Cooperative reference cannot launch");
        wb::check_cuda(cudaMemcpy(reference.data(), prices, reference.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "Jamshidian cooperative reference copy");
    }
    Json production_reference_issues = Json::array();
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (std::isfinite(reference[i]) && reference[i] >= 0)
            continue;
        production_reference_issues.push_back({{"source_index", selection[i]},
                                               {"value", reference[i]},
                                               {"finite", std::isfinite(reference[i])}});
    }
    std::cout << Json{{"type", "fixture"},
                      {"model", plan.at("model")},
                      {"curve", plan.value("curve", "")},
                      {"side", plan.value("side", "payer")},
                      {"selection", selection},
                      {"reference_prices", reference},
                      {"capacity", capacity},
                      {"reference_strategy", cooperative_reference ? "cooperative" : "scalar"},
                      {"scalar_reference_issues", reference_issues},
                      {"production_reference_issues", production_reference_issues},
                      {"maximum_payment_count", payments},
                      {"input_bytes", input_bytes},
                      {"output_bytes", capacity * sizeof(float)},
                      {"preparation_ms", preparation_ms},
                      {"environment", perf::environment_json()}}
                     .dump()
              << std::endl;

    for (const auto &job : plan.at("jobs")) {
        const std::string id = job.at("id"), strategy = job.at("strategy");
        const std::size_t count = job.at("rows"), blocks = job.at("blocks");
        const unsigned threads = job.at("threads");
        const bool cooperative = strategy == "cooperative";
        if (!cooperative && strategy != "scalar")
            throw std::invalid_argument("Unknown strategy");
        const auto launch = [&] {
            if (!cooperative) {
                scalar(count, threads, blocks);
                return;
            }
            const bool launched = cf::launch_cooperative_closed_form_cuda<Policy>(
                inputs, capacity, 0, count, time, payments, threads, blocks, prices,
                "jamshidian_probe", "cooperative", "Jamshidian cooperative benchmark");
            if (!launched)
                throw std::runtime_error("Cooperative geometry has no resident block");
        };
        auto diagnostic =
            cooperative
                ? wb::inspect_cuda_kernel_launch(cf::cooperative_closed_form_price_kernel<Policy>,
                                                 dim3(blocks), dim3(threads),
                                                 Policy::required_shared_memory_bytes(payments))
            : count > blocks * threads
                ? wb::inspect_cuda_kernel_launch(cf::closed_form_price_kernel<Policy, true>,
                                                 dim3(blocks), dim3(threads), 0)
                : wb::inspect_cuda_kernel_launch(cf::closed_form_price_kernel<Policy, false>,
                                                 dim3(blocks), dim3(threads), 0);
        wb::emit_cuda_kernel_launch_diagnostics(id.c_str(), strategy.c_str(), diagnostic);
        if (threads > diagnostic.maximum_threads_per_block ||
            diagnostic.active_blocks_per_multiprocessor == 0) {
            std::cout
                << Json{{"type", "result"}, {"job", job}, {"status", "unsupported_geometry"}}.dump()
                << std::endl;
            continue;
        }
        Json result = measure(launch, job.value("warmups", 1), job.value("repetitions", 3),
                              job.value("operations", 1));
        const auto copy_begin = Clock::now();
        std::vector<float> output(count);
        wb::check_cuda(
            cudaMemcpy(output.data(), prices, count * sizeof(float), cudaMemcpyDeviceToHost),
            "Jamshidian output copy");
        result["output_copy_ms"] = elapsed_ms(copy_begin);
        double maximum_error = 0, maximum_ratio = 0;
        std::size_t changed = 0, failures = 0, reference_failures = 0;
        Json failed_examples = Json::array();
        Json reference_invalid_examples = Json::array();
        for (std::size_t i = 0; i < count; ++i) {
            const float expected = reference[i % reference.size()], actual = output[i];
            if (!std::isfinite(expected) || expected < 0) {
                ++reference_failures;
                if (reference_invalid_examples.size() < 16)
                    reference_invalid_examples.push_back(
                        {{"result_index", i},
                         {"source_index", selection[i % reference.size()]},
                         {"actual", actual},
                         {"finite", std::isfinite(actual)}});
                continue;
            }
            const double error = std::abs(static_cast<double>(actual) - expected);
            const double allowance = kAbsoluteTolerance + kRelativeTolerance * std::abs(expected);
            if (!std::isfinite(actual) || actual < 0 || error > allowance) {
                ++failures;
                if (failed_examples.size() < 16)
                    failed_examples.push_back({{"result_index", i},
                                               {"source_index", selection[i % reference.size()]},
                                               {"expected", expected},
                                               {"actual", actual},
                                               {"finite", std::isfinite(actual)}});
            }
            maximum_error = std::max(maximum_error, error);
            maximum_ratio = std::max(maximum_ratio, error / allowance);
            changed +=
                std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected);
        }
        result["type"] = "result";
        result["job"] = job;
        result["status"] = failures             ? "numerical_failure"
                           : reference_failures ? "reference_incomplete"
                                                : "passed";
        result["numerically_eligible"] = failures == 0 && reference_failures == 0;
        result["numerics"] = {{"absolute_tolerance", kAbsoluteTolerance},
                              {"relative_tolerance", kRelativeTolerance},
                              {"maximum_absolute_error", maximum_error},
                              {"maximum_error_ratio", maximum_ratio},
                              {"changed_rows", changed},
                              {"failed_rows", failures},
                              {"reference_invalid_rows", reference_failures},
                              {"failed_examples", failed_examples},
                              {"reference_invalid_examples", reference_invalid_examples}};
        std::cout << result.dump() << std::endl;
        // A numerical failure excludes this candidate, not other configurations.
    }
}

template <wb::SwaptionSide Side, class Model, class Provider>
void standalone(const Json &plan, const std::vector<Model> &native,
                const std::vector<Product> &original, const std::vector<std::size_t> &selected,
                std::size_t count, std::uint32_t payments) {
    const auto start = Clock::now();
    const auto models = repeat_rows(native, count, selected);
    const auto products = repeat_rows(original, count, selected);
    wb::offline::cuda::DeviceBuffer<Model> dm(count);
    wb::offline::cuda::DeviceBuffer<Product> dp(count);
    Buffer output(count);
    dm.copy_from(models.data());
    dp.copy_from(products.data());
    using Policy = wb::fixed_income::CooperativeOneFactorEuropeanSwaptionClosedFormPricingPolicy<
        Side, Provider, Model, Product, Source>;
    evaluate<Policy>(plan,
                     wb::with_device_context(
                         wb::make_model_product_device_inputs(dm.data(), count, dp.data(), count,
                                                              wb::PriceConstruction::Aligned),
                         Source{}),
                     count, selected, payments, output.data(), elapsed_ms(start),
                     count * (sizeof(Model) + sizeof(Product)));
}

template <wb::SwaptionSide Side, class Provider, class Composition, class Model, class Curve>
void fitted(const Json &plan, const std::vector<Model> &native,
            const std::vector<Curve> &native_curves, const std::vector<Product> &original,
            const std::vector<std::size_t> &selected, std::size_t count, std::uint32_t payments) {
    const auto start = Clock::now();
    const auto models = repeat_rows(native, count, selected);
    const auto products = repeat_rows(original, count, selected);
    const auto curves = repeat_rows(native_curves, count, selected);
    wb::offline::cuda::DeviceBuffer<Model> dm(count);
    wb::offline::cuda::DeviceBuffer<Product> dp(count);
    wb::offline::cuda::DeviceBuffer<Curve> dc(count);
    Buffer output(count);
    dm.copy_from(models.data());
    dp.copy_from(products.data());
    dc.copy_from(curves.data());
    using Policy =
        wb::fixed_income::CooperativeFittedOneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side, Provider, Composition, Model, Curve, Product, Source>;
    evaluate<Policy>(plan,
                     wb::with_device_context(wb::make_model_curve_product_device_inputs(
                                                 dm.data(), count, dc.data(), count, dp.data(),
                                                 count, wb::PriceConstruction::Aligned),
                                             Source{}),
                     count, selected, payments, output.data(), elapsed_ms(start),
                     count * (sizeof(Model) + sizeof(Product) + sizeof(Curve)));
}

template <wb::SwaptionSide Side> void dispatch(const Json &plan) {
    const std::string model = plan.at("model"), curve = plan.value("curve", "");
    const auto original = wb::product::load_european_swaptions(
                              "datasets/product/european_swaption/european_swaptions_01.json")
                              .products;
    std::vector<std::size_t> selected;
    std::uint32_t payments = 0;
    const std::string profile = plan.value("profile", "catalogue");
    for (std::size_t i = 0; i < original.size(); ++i) {
        const auto n = original[i].payment_count;
        if (profile == "catalogue" || (profile == "short" && n <= 8) ||
            (profile == "long" && n >= 100)) {
            selected.push_back(i);
            payments = std::max(payments, n);
        }
    }
    if (selected.empty())
        throw std::invalid_argument("Unknown/empty product profile");
    std::size_t count = selected.size();
    for (const auto &job : plan.at("jobs")) {
        const std::size_t rows = job.at("rows");
        if (!rows || rows > (1U << 20U) || job.at("blocks").get<std::size_t>() > rows ||
            job.at("blocks") == 0 || job.at("threads") == 0 || job.value("warmups", 1) < 1 ||
            job.value("repetitions", 3) < 1 || job.value("operations", 1) < 1)
            throw std::invalid_argument("Invalid bounded benchmark job");
        count = std::max(count, rows);
    }
    const std::string path =
        "datasets/model/fixed_income/" + model + "/parameters/" + model + "_01.json";
    if (model == "cir")
        standalone<Side, rates::cir::ModelParameters, rates::cir::AnalyticsProvider>(
            plan, rates::cir::load_models(path), original, selected, count, payments);
    else if (model == "vasicek")
        standalone<Side, rates::vasicek::ModelParameters, rates::vasicek::AnalyticsProvider>(
            plan, rates::vasicek::load_models(path), original, selected, count, payments);
    else if (model == "ornstein_uhlenbeck")
        standalone<Side, rates::ornstein_uhlenbeck::ModelParameters,
                   rates::ornstein_uhlenbeck::AnalyticsProvider>(
            plan, rates::ornstein_uhlenbeck::load_models(path), original, selected, count,
            payments);
    else if (model == "hull_white" || model == "cir_plus_plus") {
        const std::string curve_path = "datasets/curve/" + curve + "/" + curve + "_01.json";
        if (curve == "nelson_siegel") {
            auto curves = wb::curve::nelson_siegel::load_curves(curve_path);
            if (model == "hull_white")
                fitted<Side, rates::hull_white::nelson_siegel::FittedAnalyticsProvider,
                       rates::hull_white::nelson_siegel::FittedModelComposition>(
                    plan, rates::hull_white::load_models(path), curves, original, selected, count,
                    payments);
            else
                fitted<Side, rates::cir_plus_plus::nelson_siegel::FittedAnalyticsProvider,
                       rates::cir_plus_plus::nelson_siegel::FittedModelComposition>(
                    plan, rates::cir_plus_plus::load_models(path), curves, original, selected,
                    count, payments);
        } else if (curve == "svensson") {
            auto curves = wb::curve::svensson::load_curves(curve_path);
            if (model == "hull_white")
                fitted<Side, rates::hull_white::svensson::FittedAnalyticsProvider,
                       rates::hull_white::svensson::FittedModelComposition>(
                    plan, rates::hull_white::load_models(path), curves, original, selected, count,
                    payments);
            else
                fitted<Side, rates::cir_plus_plus::svensson::FittedAnalyticsProvider,
                       rates::cir_plus_plus::svensson::FittedModelComposition>(
                    plan, rates::cir_plus_plus::load_models(path), curves, original, selected,
                    count, payments);
        } else
            throw std::invalid_argument("Unsupported curve");
    } else
        throw std::invalid_argument("Not a supported one-factor model");
}

int main(int argc, char **argv) try {
    if (argc != 2)
        throw std::invalid_argument("Usage: jamshidian_strategy_benchmark plan.json");
    std::ifstream stream(argv[1]);
    Json plan;
    stream >> plan;
    const std::string side = plan.value("side", "payer");
    if (side == "payer")
        dispatch<wb::SwaptionSide::payer>(plan);
    else if (side == "receiver")
        dispatch<wb::SwaptionSide::receiver>(plan);
    else
        throw std::invalid_argument("Unknown side");
    return 0;
} catch (const std::exception &error) {
    std::cerr << "Jamshidian strategy benchmark: " << error.what() << '\n';
    return 1;
}

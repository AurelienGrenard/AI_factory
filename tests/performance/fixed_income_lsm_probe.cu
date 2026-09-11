// Bounded exploratory probe of the production launchers; never publishes datasets.
#include "tests/performance/benchmark_support.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "model/fixed_income/cir/product/bermudan_swaption.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/product/bermudan_swaption.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "model/fixed_income/g2/product/bermudan_swaption.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"
#include "model/fixed_income/vasicek/product/bermudan_swaption.cuh"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "model/fixed_income/hull_white/product/nelson_siegel/bermudan_swaption.cuh"
#include "model/fixed_income/hull_white/product/svensson/bermudan_swaption.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "model/fixed_income/g2_plus_plus/product/nelson_siegel/bermudan_swaption.cuh"
#include "model/fixed_income/g2_plus_plus/product/svensson/bermudan_swaption.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "model/equity/markovian/black_scholes/product/american_option.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/heston/product/american_option.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "model/equity/markovian/bates/product/american_option.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "model/equity/markovian/variance_gamma/product/american_option.cuh"
#include "product/american_option/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "common/fixed_income/bermudan_swaption_continuation_state.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/execution_plan.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "model/fixed_income/cir/dynamics.cuh"
#include "product/bermudan_swaption/pricing_policy.cuh"
#include <cuda_profiler_api.h>
#include <bit>
#include <charconv>
#include <map>

namespace {
namespace wb = ai_factory::workbench;
namespace perf = wb::performance;
namespace rates = wb::model::fixed_income;
namespace equity = wb::model::equity;
using Json = nlohmann::ordered_json;

struct Options {
    std::string model = "cir";
    std::string curve = "nelson_siegel";
    bool catalog = false;
    std::size_t rows = 1, offset = 0, paths = 65536, blocks = 64;
    unsigned threads = 128, warmups = 1, repetitions = 3;
    std::uint32_t first = 126, interval = 126, payments = 10, exercises = 10;
    std::size_t memory_limit_mib = 2048;
    std::size_t chunk_rows = 0; // Independent prices, never pathwise LSM chunks.
    std::uint32_t maturity = 252;
    bool native_batching = false, trace_chunks = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("Expected --option value");
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--model") { o.model = value; continue; }
        if (key == "--curve") { o.curve = value; continue; }
        if (key == "--source") {
            if (value != "catalog" && value != "synthetic")
                throw std::invalid_argument("Source must be catalog or synthetic");
            o.catalog = value == "catalog";
            continue;
        }
        std::uint64_t n = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()
            || n > UINT32_MAX) throw std::invalid_argument("Invalid bounded integer: " + key);
        if (key == "--rows") o.rows = n;
        else if (key == "--offset") o.offset = n;
        else if (key == "--paths") o.paths = n;
        else if (key == "--blocks") o.blocks = n;
        else if (key == "--threads") o.threads = n;
        else if (key == "--warmups") o.warmups = n;
        else if (key == "--repetitions") o.repetitions = n;
        else if (key == "--first") o.first = n;
        else if (key == "--interval") o.interval = n;
        else if (key == "--payments") o.payments = n;
        else if (key == "--exercises") o.exercises = n;
        else if (key == "--memory-limit-mib") o.memory_limit_mib = n;
        else if (key == "--chunk-rows") o.chunk_rows = n;
        else if (key == "--maturity") o.maturity = n;
        else if (key == "--native-batching" && n <= 1) o.native_batching = n;
        else if (key == "--trace-chunks" && n <= 1) o.trace_chunks = n;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (!o.rows || o.rows > 1000 || o.paths < 2 || o.paths > (1U << 20)
        || !o.blocks || o.blocks > 8192 || !o.repetitions || o.repetitions > 21
        || o.warmups > 5 || !o.first || !o.interval || o.exercises < 2
        || o.exercises > o.payments || o.payments > 200 || o.first > 7560
        || o.interval > 252 || !o.memory_limit_mib
        || o.memory_limit_mib > (o.native_batching ? 15360U : 14336U)
        || o.chunk_rows > o.rows || !o.maturity || o.maturity > 2520
        || (o.curve != "nelson_siegel" && o.curve != "svensson"))
        throw std::invalid_argument("Probe safety bounds exceeded");
    if (o.native_batching && (o.model != "cir" || !o.catalog || o.rows != 1000
        || o.offset != 0 || o.chunk_rows != 0))
        throw std::invalid_argument("Native memory batching is restricted to the full CIR catalog");
    return o;
}

// Inspect the exact production planner, without compiling replacement kernels.
auto cir_plan(const Options& o, const wb::product::BermudanSwaptionParameters* products,
              std::size_t budget) {
    using Dynamics = rates::cir::joint::DynamicsPolicy;
    using Schedule = wb::simulation::FixedStepRegularExerciseSchedule<Dynamics>;
    using Pricing = wb::product::StandaloneBermudanSwaptionPricingPolicy<Schedule,
        rates::cir::BermudanSwaptionAnalyticsPolicy, wb::SwaptionSide::payer,
        wb::fixed_income::OneFactorRateContinuationState<Dynamics>>;
    using Regressor = wb::longstaff_schwartz::NormalEquationRegressor<
        wb::longstaff_schwartz::basis::OneFactorHermiteBasis<3U>>;
    return wb::longstaff_schwartz::make_execution_plan<Pricing, Regressor>(
        {products, o.rows, wb::PriceConstruction::Aligned}, o.rows, o.paths,
        o.blocks, budget, {1.0f / 504.0f, 2U}, "CIR full-catalog preflight");
}

template<bool American = false, class Model, class Loader, class Launcher>
void run(const Options& o, Model synthetic_model, Loader load, Launcher launch) {
    using Product = std::conditional_t<American, wb::product::AmericanOptionParameters,
        wb::product::BermudanSwaptionParameters>;
    std::vector<Model> models(o.rows, synthetic_model);
    Product synthetic_product{};
    if constexpr (American) synthetic_product = {1.05f, o.maturity, o.interval};
    else synthetic_product = {1.0f, 0.03f, float(o.interval) / 252.0f,
        o.first, o.interval, o.payments, o.exercises};
    std::vector<Product> products(o.rows, synthetic_product);
    if (o.catalog) {
        const auto all_models = load(std::string("datasets/model/")
            + (American ? "equity/markovian/" : "fixed_income/") + o.model
            + "/parameters/" + o.model + "_01.json");
        const auto all_products = [] {
            if constexpr (American) return wb::product::load_american_options(
                "datasets/product/american_option/american_options_01.json");
            else return wb::product::load_bermudan_swaptions(
                "datasets/product/bermudan_swaption/bermudan_swaptions_01.json");
        }();
        if (o.offset > all_models.size() || o.rows > all_models.size() - o.offset
            || o.offset > all_products.size() || o.rows > all_products.size() - o.offset)
            throw std::invalid_argument("Catalog slice is outside the aligned inputs");
        std::copy_n(all_models.begin() + o.offset, o.rows, models.begin());
        std::copy_n(all_products.begin() + o.offset, o.rows, products.begin());
    }
    std::uint64_t exercise_sum = 0, horizon_sum_days = 0, payoff_bond_sum = 0;
    std::vector<std::size_t> row_exercises;
    Json calendars = Json::array();
    for (const auto& p : products) {
        if constexpr (American) {
            if (!p.maturity_days || !p.exercise_interval_days)
                throw std::invalid_argument("Invalid American calendar");
            const auto count = 1 + (p.maturity_days - 1) / p.exercise_interval_days;
            row_exercises.push_back(count);
            exercise_sum += count;
            horizon_sum_days += p.maturity_days;
            calendars.push_back({p.maturity_days, p.exercise_interval_days, count});
        } else {
        row_exercises.push_back(p.exercise_count);
        exercise_sum += p.exercise_count;
        horizon_sum_days += std::uint64_t(p.first_exercise_time_days)
            + std::uint64_t(p.exercise_count - 1) * p.payment_interval_days;
        // One terminal evaluation and two immediate evaluations per backward level.
        payoff_bond_sum += p.payment_count - p.exercise_count + 1;
        for (unsigned j = 0; j + 1 < p.exercise_count; ++j)
            payoff_bond_sum += 2 * (p.payment_count - j);
        calendars.push_back({p.first_exercise_time_days, p.payment_interval_days,
            p.payment_count, p.exercise_count});
        }
    }
    const bool fixed_step = o.model == "cir" || o.model == "heston" || o.model == "bates";
    const bool fitted = o.model == "hull_white" || o.model == "g2_plus_plus";
    const std::size_t state_bytes = American ? (fixed_step ? 8 : 4)
        : (o.model == "g2" || o.model == "g2_plus_plus" ? 12 : 8);
    const auto path_storage = o.paths * (state_bytes * exercise_sum + 4 * o.rows);
    // Conservative partials/rows allowance for the 4-/6-feature regressors.
    const auto chunk_rows = o.chunk_rows ? o.chunk_rows : o.rows;
    std::size_t estimated_bytes = 0;
    for (std::size_t begin = 0; begin < o.rows; begin += chunk_rows) {
        const auto end = std::min(begin + chunk_rows, o.rows);
        std::size_t bytes = (end - begin) * (o.paths * 4 + o.blocks * 256 + 4096);
        for (auto j = begin; j < end; ++j) bytes += o.paths * state_bytes * row_exercises[j];
        estimated_bytes = std::max(estimated_bytes, bytes);
    }
    const auto logical_estimated_bytes = estimated_bytes;
    Json native_plan = nullptr;
    if constexpr (!American) {
        if (o.native_batching) {
            const auto budget = wb::longstaff_schwartz::query_workspace_budget("CIR probe preflight");
            const auto plan = cir_plan(o, products.data(), budget.available_bytes);
            estimated_bytes = plan.maximum_workspace_bytes;
            native_plan = {{"available_bytes", budget.available_bytes},
                {"free_bytes", budget.free_bytes}, {"safety_margin", budget.safety_margin},
                {"batches", Json::array()}};
            for (const auto& batch : plan.batches)
                native_plan["batches"].push_back({{"offset", batch.result_offset},
                    {"rows", batch.result_count}, {"state_values", batch.state_value_count},
                    {"regression_levels", batch.maximum_regression_count}});
        }
    }
    if (estimated_bytes > o.memory_limit_mib * (1ULL << 20))
        throw std::invalid_argument("Probe workspace estimate exceeds memory limit");
    Json config{{"model", o.model}, {"source", o.catalog ? "catalog" : "synthetic"},
        {"product", American ? "american_put" : "bermudan_payer_swaption"},
        {"curve", fitted ? Json(o.curve) : Json(nullptr)}, {"chunk_rows", chunk_rows},
        {"native_batching", o.native_batching}, {"native_plan_preflight", native_plan},
        {"logical_estimated_workspace_bytes", logical_estimated_bytes},
        {"offset", o.offset}, {"rows", o.rows}, {"paths", o.paths},
        {"threads", o.threads}, {"blocks", o.blocks},
        {"warmups", o.warmups}, {"repetitions", o.repetitions},
        {"days_per_year", 252}, {"steps_per_year", fixed_step ? Json(504) : Json(nullptr)},
        {American ? "calendars_maturity_interval_exercises" : "calendars_first_interval_payments_exercises", calendars},
        {"simulation_transitions", o.paths * (fixed_step ? 2 * horizon_sum_days : exercise_sum)},
        {"payoff_bond_evaluations", o.paths * payoff_bond_sum},
        {"path_storage_bytes", path_storage}, {"estimated_workspace_bytes", estimated_bytes}};
    std::cout << Json{{"event", "start"}, {"config", config},
        {"environment", perf::environment_json()}}.dump() << std::endl;

    perf::DeviceBuffer dm(models.size() * sizeof(Model), perf::DeviceMemoryRole::persistent_input);
    perf::DeviceBuffer dp(products.size() * sizeof(products.front()), perf::DeviceMemoryRole::persistent_input);
    perf::DeviceBuffer prices(o.rows * sizeof(float), perf::DeviceMemoryRole::output);
    perf::DeviceBuffer errors(o.rows * sizeof(float), perf::DeviceMemoryRole::output);
    perf::copy_to_device(dm, models);
    perf::copy_to_device(dp, products);
    const auto seed = 11668829039698640896ULL + o.offset;
    auto once = [&] {
        wb::longstaff_schwartz::LaunchResult total{};
        for (std::size_t begin = 0; begin < o.rows; begin += chunk_rows) {
            const auto n = std::min(chunk_rows, o.rows - begin);
            const auto chunk_start = std::chrono::steady_clock::now();
            if (o.trace_chunks) std::cout << Json{{"event", "chunk_start"},
                {"offset", begin}, {"rows", n}}.dump() << std::endl;
            const auto r = launch(dm.template as<Model>() + begin, products.data() + begin,
                dp.template as<Product>() + begin, prices.template as<float>() + begin,
                errors.template as<float>() + begin, seed + begin, n, begin);
            wb::longstaff_schwartz::validate_regression_diagnostics(r, "LSM probe chunk");
            if (o.trace_chunks) std::cout << Json{{"event", "chunk_end"},
                {"offset", begin}, {"rows", n}, {"gpu_ms", r.kernel_seconds * 1000},
                {"raw_host_ms", std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - chunk_start).count()},
                {"batches", r.batch_count}, {"workspace_bytes", r.workspace_bytes}}
                .dump() << std::endl;
            total.kernel_seconds += r.kernel_seconds;
            total.batch_count += r.batch_count;
            total.kernel_launch_count += r.kernel_launch_count;
            total.maximum_prices_per_batch = std::max(total.maximum_prices_per_batch, r.maximum_prices_per_batch);
            total.workspace_bytes = std::max(total.workspace_bytes, r.workspace_bytes);
            total.blocks_per_price = r.blocks_per_price;
            total.regression_diagnostics.successful_regression_count += r.regression_diagnostics.successful_regression_count;
            total.regression_diagnostics.no_candidate_count += r.regression_diagnostics.no_candidate_count;
            total.regression_diagnostics.insufficient_candidate_count += r.regression_diagnostics.insufficient_candidate_count;
        }
        return total;
    };
    for (unsigned i = 0; i < o.warmups; ++i) once();
    wb::check_cuda(cudaProfilerStart(), "probe profiler start");
    std::vector<double> gpu_samples, host_samples;
    for (unsigned i = 0; i < o.repetitions; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        const auto result = once();
        const double host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        const auto pv = perf::copy_from_device<float>(prices, o.rows);
        const auto ev = perf::copy_from_device<float>(errors, o.rows);
        bool finite = !result.regression_diagnostics.has_fatal_failure();
        std::vector<std::uint32_t> bits;
        for (std::size_t j = 0; j < o.rows; ++j) {
            finite &= std::isfinite(pv[j]) && pv[j] >= 0 && std::isfinite(ev[j]) && ev[j] >= 0;
            bits.push_back(std::bit_cast<std::uint32_t>(pv[j]));
        }
        gpu_samples.push_back(result.kernel_seconds * 1000);
        host_samples.push_back(host_ms);
        std::cout << Json{{"event", "measurement"}, {"repetition", i},
            {"gpu_ms", gpu_samples.back()}, {"raw_host_ms", host_ms},
            {"workspace_bytes", result.workspace_bytes}, {"batch_count", result.batch_count},
            {"maximum_prices_per_batch", result.maximum_prices_per_batch},
            {"kernel_launch_count", result.kernel_launch_count},
            {"finite", finite}, {"prices", pv}, {"price_bits", bits}, {"errors", ev},
            {"fatal_regressions", result.regression_diagnostics.affected_result_count},
            {"no_candidates", result.regression_diagnostics.no_candidate_count},
            {"insufficient_candidates", result.regression_diagnostics.insufficient_candidate_count}}
            .dump() << std::endl;
        if (!finite) throw std::runtime_error("Non-finite/negative output or fatal regression");
    }
    wb::check_cuda(cudaProfilerStop(), "probe profiler stop");
    std::cout << Json{{"event", "summary"}, {"config", config},
        {"gpu", perf::timing_json(perf::summarize(gpu_samples))},
        {"raw_host", perf::timing_json(perf::summarize(host_samples))}}
        .dump() << std::endl;
}

template<class Model, class ModelLoader, class Curve, class CurveLoader, class Launcher>
void run_fitted(const Options& o, Model model, ModelLoader load_models,
                Curve curve, CurveLoader load_curves, Launcher public_launcher) {
    std::vector<Curve> curves(o.rows, curve);
    if (o.catalog) {
        const auto all = load_curves("datasets/curve/" + o.curve + "/" + o.curve + "_01.json");
        if (o.offset > all.size() || o.rows > all.size() - o.offset)
            throw std::invalid_argument("Curve slice is outside the aligned inputs");
        std::copy_n(all.begin() + o.offset, o.rows, curves.begin());
    }
    perf::DeviceBuffer dc(o.rows * sizeof(Curve), perf::DeviceMemoryRole::persistent_input);
    perf::copy_to_device(dc, curves);
    run(o, model, load_models, [&](const auto* m, const auto* hp, const auto* dp,
        float* p, float* e, std::uint64_t seed, std::size_t n, std::size_t begin) {
        return public_launcher(m, n, dc.template as<Curve>() + begin, n, hp, dp, n,
            wb::PriceConstruction::Aligned, n, o.paths, 1.0f / 252.0f,
            o.threads, o.blocks, seed, p, e);
    });
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse(argc, argv);
        const auto launch = [&](auto public_launcher, auto... time) {
            return [&, public_launcher, time...](const auto* m, const auto* hp,
                const auto* dp, float* p, float* e, std::uint64_t seed,
                std::size_t n, std::size_t) {
                return public_launcher(m, n, hp, dp, n,
                    wb::PriceConstruction::Aligned, n, o.paths, time...,
                    o.threads, o.blocks, seed, p, e);
            };
        };
        if (o.model == "cir") {
            run(o, rates::cir::ModelParameters{{0.5f, 0.04f, 0.1f}, 0.04f},
                rates::cir::load_models,
                launch(&rates::cir::launch_cir_bermudan_swaption_cuda<wb::SwaptionSide::payer>,
                    1.0f / 252.0f));
        } else if (o.model == "ornstein_uhlenbeck") {
            run(o, rates::ornstein_uhlenbeck::ModelParameters{{0.35f, 0.02f}, 0.03f},
                rates::ornstein_uhlenbeck::load_models,
                launch(&rates::ornstein_uhlenbeck::launch_ornstein_uhlenbeck_bermudan_swaption_cuda<wb::SwaptionSide::payer>, 1.0f / 252.0f));
        } else if (o.model == "g2") {
            run(o, rates::g2::ModelParameters{{0.20f, 0.005f, 0.75f, 0.012f, -0.40f}, {0.02f, 0.01f}},
                rates::g2::load_models,
                launch(&rates::g2::launch_g2_bermudan_swaption_cuda<wb::SwaptionSide::payer>, 1.0f / 252.0f));
        } else if (o.model == "vasicek") {
            run(o, rates::vasicek::ModelParameters{{0.35f, 0.04f, 0.02f}, 0.03f},
                rates::vasicek::load_models,
                launch(&rates::vasicek::launch_vasicek_bermudan_swaption_cuda<wb::SwaptionSide::payer>, 1.0f / 252.0f));
        } else if (o.model == "hull_white") {
            const rates::hull_white::ModelParameters model{0.35f, 0.02f};
            if (o.curve == "nelson_siegel")
                run_fitted(o, model, rates::hull_white::load_models,
                    wb::curve::nelson_siegel::NelsonSiegelParameters{0.03f, -0.01f, 0.01f, 2.0f},
                    wb::curve::nelson_siegel::load_curves,
                    &rates::hull_white::nelson_siegel::launch_hull_white_nelson_siegel_bermudan_swaption_cuda<wb::SwaptionSide::payer>);
            else run_fitted(o, model, rates::hull_white::load_models,
                    wb::curve::svensson::SvenssonParameters{0.03f, -0.01f, 0.01f, 0.005f, 2.0f, 5.0f},
                    wb::curve::svensson::load_curves,
                    &rates::hull_white::svensson::launch_hull_white_svensson_bermudan_swaption_cuda<wb::SwaptionSide::payer>);
        } else if (o.model == "g2_plus_plus") {
            const rates::g2_plus_plus::ModelParameters model{{0.20f, 0.005f, 0.75f, 0.012f, -0.40f}};
            if (o.curve == "nelson_siegel")
                run_fitted(o, model, rates::g2_plus_plus::load_models,
                    wb::curve::nelson_siegel::NelsonSiegelParameters{0.03f, -0.01f, 0.01f, 2.0f},
                    wb::curve::nelson_siegel::load_curves,
                    &rates::g2_plus_plus::nelson_siegel::launch_g2_plus_plus_nelson_siegel_bermudan_swaption_cuda<wb::SwaptionSide::payer>);
            else run_fitted(o, model, rates::g2_plus_plus::load_models,
                    wb::curve::svensson::SvenssonParameters{0.03f, -0.01f, 0.01f, 0.005f, 2.0f, 5.0f},
                    wb::curve::svensson::load_curves,
                    &rates::g2_plus_plus::svensson::launch_g2_plus_plus_svensson_bermudan_swaption_cuda<wb::SwaptionSide::payer>);
        } else if (o.model == "black_scholes") {
            run<true>(o, equity::black_scholes::ModelParameters{1.0f, 0.03f, 0.01f, 0.25f},
                equity::black_scholes::load_models,
                launch(&equity::black_scholes::launch_black_scholes_american_option_cuda<wb::OptionSide::put>, 1.0f / 252.0f));
        } else if (o.model == "heston") {
            run<true>(o, equity::heston::ModelParameters{1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f},
                equity::heston::load_models,
                launch(&equity::heston::launch_heston_american_option_cuda<wb::OptionSide::put>, 1.0f / 504.0f, 2U));
        } else if (o.model == "bates") {
            run<true>(o, equity::bates::ModelParameters{1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f, 0.2f, -0.10f, 0.15f},
                equity::bates::load_models,
                launch(&equity::bates::launch_bates_american_option_cuda<wb::OptionSide::put>, 1.0f / 504.0f, 2U));
        } else if (o.model == "variance_gamma") {
            run<true>(o, equity::variance_gamma::ModelParameters{1.0f, 0.03f, 0.01f, 0.25f, 0.20f, -0.10f},
                equity::variance_gamma::load_models,
                launch(&equity::variance_gamma::launch_variance_gamma_american_option_cuda<wb::OptionSide::put>, 1.0f / 252.0f));
        } else throw std::invalid_argument("Unsupported probe model");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

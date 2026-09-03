// Fukasawa--Gatheral Figure 6.4 probe using the production CUDA launcher.
#include "common/check_cuda.cuh"
#include "model/equity/rough/rough_sabr/product/european_option.cuh"

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ai_factory::workbench::OptionSide;
using ai_factory::workbench::PriceConstruction;
using ai_factory::workbench::check_cuda;
using ai_factory::workbench::model::equity::rough_sabr::ModelParameters;
using ai_factory::workbench::product::EuropeanOptionParameters;
using nlohmann::ordered_json;

constexpr float day_fraction = 1.0f / 252.0f;
constexpr std::size_t path_chunk_size = 65'536U;
constexpr std::array<std::uint32_t, 4U> maturity_days = {
    32U, 63U, 126U, 252U,
};
constexpr std::array<std::size_t, 3U> time_step_counts = {
    1'024U, 2'048U, 4'096U,
};
constexpr std::array<double, 7U> scaled_log_moneyness = {
    -0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45,
};

struct Estimate {
    float price;
    float standard_error;
    float milliseconds;
};

template<OptionSide Side>
Estimate launch_row(
    const ModelParameters* device_model,
    const EuropeanOptionParameters* device_products,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t path_count,
    std::uint32_t maturity_days_value,
    std::size_t time_steps,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t seed,
    float* device_prices,
    float* device_standard_errors
) {
    namespace rough =
        ai_factory::workbench::model::equity::rough_sabr;
    const float maturity_years =
        static_cast<float>(maturity_days_value) * day_fraction;
    const float target_dt = maturity_years / static_cast<float>(time_steps);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "rough-SABR probe create start");
    check_cuda(cudaEventCreate(&stop), "rough-SABR probe create stop");
    check_cuda(cudaEventRecord(start), "rough-SABR probe record start");
    rough::launch_rough_sabr_european_option_cuda<Side>(
        device_model,
        1U,
        device_products,
        result_count,
        PriceConstruction::Aligned,
        result_count,
        result_index,
        path_count,
        day_fraction,
        target_dt,
        time_steps,
        path_chunk_size,
        device_workspace,
        workspace_bytes,
        seed,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaEventRecord(stop), "rough-SABR probe record stop");
    check_cuda(cudaEventSynchronize(stop), "rough-SABR probe synchronize");
    Estimate result{};
    check_cuda(
        cudaEventElapsedTime(&result.milliseconds, start, stop),
        "rough-SABR probe elapsed time"
    );
    check_cuda(
        cudaMemcpy(
            &result.price,
            device_prices + result_index,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "rough-SABR probe copy price"
    );
    check_cuda(
        cudaMemcpy(
            &result.standard_error,
            device_standard_errors + result_index,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "rough-SABR probe copy standard error"
    );
    check_cuda(cudaEventDestroy(stop), "rough-SABR probe destroy stop");
    check_cuda(cudaEventDestroy(start), "rough-SABR probe destroy start");
    if (!std::isfinite(result.price) || result.price < 0.0f
        || !std::isfinite(result.standard_error)
        || result.standard_error <= 0.0f) {
        throw std::runtime_error("rough-SABR probe returned invalid moments");
    }
    return result;
}

ordered_json estimate_document(const Estimate& estimate) {
    return {
        {"price", estimate.price},
        {"standard_error", estimate.standard_error},
        {"milliseconds", estimate.milliseconds},
    };
}

}  // namespace

int main(int argument_count, char** arguments) {
    namespace rough =
        ai_factory::workbench::model::equity::rough_sabr;
    if (argument_count > 4) {
        throw std::invalid_argument(
            "usage: rough_sabr_cuda_probe [PATH_COUNT] [SEED] [OUTPUT_JSON]"
        );
    }
    const std::size_t path_count = argument_count > 1
        ? std::stoull(arguments[1])
        : 1U << 20U;
    const std::uint64_t seed = argument_count > 2
        ? std::stoull(arguments[2])
        : 913'000'001ULL;
    const std::filesystem::path output_path = argument_count > 3
        ? arguments[3]
        : "validation/volterra/fixtures/"
          "rough_sabr_fukasawa_gatheral_figure_6_4_1m.json";
    if (path_count < 2U || path_count % 2U != 0U) {
        throw std::invalid_argument(
            "path count must be even and at least two"
        );
    }
    if (path_chunk_size > path_count) {
        throw std::invalid_argument(
            "path count must be at least the fixed path chunk size"
        );
    }

    const ModelParameters model = {
        1.0f,
        0.0f,
        0.0f,
        0.04f,
        1.0f,
        0.05f,
        -0.90f,
        0.50f,
    };
    const rough::WorkspacePlan plan = rough::plan_pricing_workspace(
        time_step_counts.back(), path_count, path_chunk_size
    );

    ModelParameters* device_model = nullptr;
    EuropeanOptionParameters* device_products = nullptr;
    void* device_workspace = nullptr;
    float* device_prices = nullptr;
    float* device_standard_errors = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "probe model malloc");
    check_cuda(
        cudaMalloc(
            &device_products,
            scaled_log_moneyness.size() * sizeof(EuropeanOptionParameters)
        ),
        "probe products malloc"
    );
    check_cuda(
        cudaMalloc(&device_workspace, plan.workspace_bytes),
        "probe workspace malloc"
    );
    check_cuda(
        cudaMalloc(
            &device_prices,
            scaled_log_moneyness.size() * sizeof(float)
        ),
        "probe prices malloc"
    );
    check_cuda(
        cudaMalloc(
            &device_standard_errors,
            scaled_log_moneyness.size() * sizeof(float)
        ),
        "probe standard errors malloc"
    );
    check_cuda(
        cudaMemcpy(
            device_model,
            &model,
            sizeof(model),
            cudaMemcpyHostToDevice
        ),
        "probe model copy"
    );

    ordered_json document = {
        {"model", "rough_sabr"},
        {"paper_case", "Fukasawa--Gatheral Figure 6.4"},
        {"paper", {
            {"authors", "Masaaki Fukasawa and Jim Gatheral"},
            {"title", "A rough SABR formula"},
            {"doi", "10.3934/fmf.2021003"},
            {"equations", {"3.4", "5.2", "6.1"}},
        }},
        {"eta_convention", "d_xi_over_xi"},
        {"parameters", {
            {"spot", model.spot},
            {"risk_free_rate", model.risk_free_rate},
            {"dividend_yield", model.dividend_yield},
            {"xi_0", model.xi_0},
            {"eta", model.eta},
            {"hurst_exponent", model.hurst_exponent},
            {"rho", model.rho},
            {"beta", model.beta},
        }},
        {"numerics", {
            {"path_count", path_count},
            {"seed", seed},
            {"day_fraction", day_fraction},
            {"path_chunk_size", path_chunk_size},
            {"hybrid_scheme", "AI_factory BLP kappa=1 L2 far cells"},
            {"paper_finest_time_steps", 4096U},
        }},
        {"runs", ordered_json::array()},
    };

    for (const std::uint32_t maturity_days_value : maturity_days) {
        const double maturity_years =
            static_cast<double>(maturity_days_value) * day_fraction;
        const double kernel = static_cast<double>(model.eta)
            * std::sqrt(2.0 * static_cast<double>(model.hurst_exponent))
            * std::pow(
                maturity_years,
                static_cast<double>(model.hurst_exponent) - 0.5
            );
        std::vector<EuropeanOptionParameters> products;
        products.reserve(scaled_log_moneyness.size());
        std::array<double, scaled_log_moneyness.size()> log_moneyness{};
        for (std::size_t index = 0U;
             index < scaled_log_moneyness.size();
             ++index) {
            log_moneyness[index] = scaled_log_moneyness[index]
                * std::sqrt(static_cast<double>(model.xi_0)) / kernel;
            products.push_back({
                static_cast<float>(std::exp(log_moneyness[index])),
                maturity_days_value,
            });
        }
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(EuropeanOptionParameters),
                cudaMemcpyHostToDevice
            ),
            "probe products copy"
        );

        for (const std::size_t time_steps : time_step_counts) {
            ordered_json run = {
                {"maturity_days", maturity_days_value},
                {"maturity", maturity_years},
                {"time_steps", time_steps},
                {"effective_dt", maturity_years / time_steps},
                {"rows", ordered_json::array()},
            };
            for (std::size_t index = 0U;
                 index < products.size();
                 ++index) {
                const bool use_put = products[index].strike < model.spot;
                const Estimate estimate = use_put
                    ? launch_row<OptionSide::put>(
                        device_model,
                        device_products,
                        products.size(),
                        index,
                        path_count,
                        maturity_days_value,
                        time_steps,
                        device_workspace,
                        plan.workspace_bytes,
                        seed,
                        device_prices,
                        device_standard_errors
                    )
                    : launch_row<OptionSide::call>(
                        device_model,
                        device_products,
                        products.size(),
                        index,
                        path_count,
                        maturity_days_value,
                        time_steps,
                        device_workspace,
                        plan.workspace_bytes,
                        seed,
                        device_prices,
                        device_standard_errors
                    );
                run["rows"].push_back({
                    {
                        "scaled_log_moneyness",
                        scaled_log_moneyness[index]
                    },
                    {"log_moneyness", log_moneyness[index]},
                    {"strike", products[index].strike},
                    {"side", use_put ? "put" : "call"},
                    {"estimate", estimate_document(estimate)},
                });
            }
            document["runs"].push_back(std::move(run));
        }
    }

    check_cuda(
        cudaFree(device_standard_errors),
        "probe standard errors free"
    );
    check_cuda(cudaFree(device_prices), "probe prices free");
    check_cuda(cudaFree(device_workspace), "probe workspace free");
    check_cuda(cudaFree(device_products), "probe products free");
    check_cuda(cudaFree(device_model), "probe model free");

    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error(
            "cannot open rough-SABR probe output: " + output_path.string()
        );
    }
    output << document.dump(2) << '\n';
    if (!output) {
        throw std::runtime_error(
            "cannot write rough-SABR probe output: " + output_path.string()
        );
    }
    return 0;
}

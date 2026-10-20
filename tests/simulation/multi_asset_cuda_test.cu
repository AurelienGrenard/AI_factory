#include "correlation/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/markovian/cev/dynamics_impl.cuh"
#include "model/equity/markovian/bates/dynamics_impl.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"
#include "model/equity/markovian/kou/dynamics_impl.cuh"
#include "model/equity/markovian/heston_3_2/dynamics_impl.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"
#include "model/equity/markovian/stein_stein/dynamics_impl.cuh"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/multi_asset_price_generation.cuh"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wb = ai_factory::workbench;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

wb::offline::pricing::BatchedMonteCarloProfile profile(std::size_t paths) {
    wb::offline::pricing::BatchedMonteCarloProfile result{};
    result.paths_per_price = paths;
    result.results_per_launch = 1U;
    result.block_count_limit = 1U;
    result.threads_per_block = 128U;
    result.seed = 19433U;
    return result;
}

void check_cholesky() {
    const std::vector<double> indefinite{
        1.0, 0.9, 0.9,
        0.9, 1.0, -0.9,
        0.9, -0.9, 1.0
    };
    bool rejected = false;
    try {
        wb::correlation::packed_cholesky(indefinite, 3U);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Indefinite correlation matrix was accepted.");
}

void check_thirty_assets() {
    using Dynamics = wb::model::equity::black_scholes::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<Dynamics, 30U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.0f});
    host.scenario_indices.assign(30U, 0U);
    std::vector<std::string> names;
    std::vector<double> identity(30U * 30U, 0.0);
    for (std::size_t i = 0; i < 30U; ++i) {
        names.push_back("asset_" + std::to_string(i) + ".spot");
        identity[i * 30U + i] = 1.0;
    }
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"identity", names, wb::correlation::packed_cholesky(identity, 30U)}
    };
    const std::vector<Product::ProductParameters> products{{0.0f, 1U}};
    const std::vector<wb::monte_carlo::MultiAssetPriceRow> rows{
        {0U, 0U, 0U, 0U, 1.0f}
    };
    const auto output = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, 1U, products, matrices, names, rows,
                           profile(256U), 1.0f / 252.0f, 1U);
    require(output.prices.size() == 1U && output.prices[0] == 1.0f
            && output.standard_errors[0] == 0.0f,
            "Thirty-asset deterministic worst-of price is incorrect.");

    host.models[0].volatility = 0.2f;
    const std::vector<Product::ProductParameters> stochastic_product{
        {0.7f, 21U}
    };
    const auto p = profile(8192U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, 1U, stochastic_product, matrices,
                           names, rows, p, 1.0f / 252.0f, 1U);
    const auto four = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, 1U, stochastic_product, matrices,
                           names, rows, p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f,
            "Thirty-asset stochastic price is invalid.");
    require(scalar.prices == four.prices
            && scalar.standard_errors == four.standard_errors,
            "Thirty-asset prices depend on path-team size.");
    std::cout << "D=30: TeamSize=1 kernel " << scalar.kernel_seconds
              << " s; TeamSize=4 kernel " << four.kernel_seconds << " s\n";
}

void check_all_team_sizes() {
    using Dynamics = wb::model::equity::black_scholes::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<Dynamics, 2U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models = {{1.0f, 0.0f, 0.0f, 0.2f},
                   {1.2f, 0.0f, 0.0f, 0.3f}};
    host.scenario_indices = {0U, 1U};
    const std::vector<std::string> names{
        "asset_0.spot", "asset_1.spot"
    };
    const std::vector<double> dense{1.0, 0.4, 0.4, 1.0};
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"two_spots", names, wb::correlation::packed_cholesky(dense, 2U)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    const std::vector<wb::monte_carlo::MultiAssetPriceRow> rows{
        {0U, 0U, 0U, 0U, 1.0f}
    };
    const auto p = profile(2053U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    const auto verify = [&](const wb::offline::cuda::MonteCarloRun& run) {
        require(run.prices == scalar.prices
                && run.standard_errors == scalar.standard_errors,
                "Brownian mapping or reduction depends on path-team size.");
    };
    verify(wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 2U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U));
    verify(wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U));
    verify(wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 8U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U));
    verify(wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 16U>(host, 1U, products, matrices, names, rows,
                            p, 1.0f / 252.0f, 1U));
    verify(wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 32U>(host, 1U, products, matrices, names, rows,
                            p, 1.0f / 252.0f, 1U));
}

void check_mixed_volatility() {
    using BS = wb::model::equity::black_scholes::DynamicsPolicy;
    using Heston = wb::model::equity::heston::DynamicsPolicy;
    using SABR = wb::model::equity::sabr::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<BS, 1U>,
        wb::simulation::ModelGroup<Heston, 1U>,
        wb::simulation::ModelGroup<SABR, 1U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.2f});
    host.scenario_indices.push_back(0U);
    host.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.04f,
                                2.0f, 0.04f, 0.5f, -0.6f});
    host.tail.scenario_indices.push_back(0U);
    host.tail.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                                     0.4f, -0.3f, 1.0f});
    host.tail.tail.scenario_indices.push_back(0U);

    // Construct the whole Brownian matrix from independent columns. The
    // internal Heston and SABR correlations are exact by construction.
    constexpr std::size_t d = 5U;
    double l[d][d]{};
    l[0][0] = 1.0;
    l[1][0] = 0.35;
    l[1][1] = std::sqrt(1.0 - 0.35 * 0.35);
    for (std::size_t k = 0; k < d; ++k) {
        l[2][k] = -0.6 * l[1][k];
    }
    l[2][2] = std::sqrt(1.0 - 0.6 * 0.6);
    l[3][0] = 0.2;
    l[3][3] = std::sqrt(1.0 - 0.2 * 0.2);
    for (std::size_t k = 0; k < d; ++k) {
        l[4][k] = -0.3 * l[3][k];
    }
    l[4][4] = std::sqrt(1.0 - 0.3 * 0.3);
    std::vector<double> dense(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            for (std::size_t k = 0; k < d; ++k) {
                dense[i * d + j] += l[i][k] * l[j][k];
            }
        }
    }
    const std::vector<std::string> names{
        "asset_0.spot", "asset_1.spot", "asset_1.variance",
        "asset_2.spot", "asset_2.volatility"
    };
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"mixed", names, wb::correlation::packed_cholesky(dense, d)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    const std::vector<wb::monte_carlo::MultiAssetPriceRow> rows{
        {0U, 0U, 0U, 0U, 1.0f}
    };
    const auto p = profile(4096U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    const auto cooperative = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f
            && std::isfinite(cooperative.prices[0]),
            "Mixed-model worst-of price is invalid.");
    require(scalar.prices[0] == cooperative.prices[0]
            && scalar.standard_errors[0] == cooperative.standard_errors[0],
            "Path-team sizes disagree for the mixed-model price.");
    std::cout << "Mixed BS/Heston/SABR worst-of: " << scalar.prices[0]
              << " (SE " << scalar.standard_errors[0] << ")\n";
}

void check_auxiliary_jump_stream() {
    using CEV = wb::model::equity::cev::DynamicsPolicy;
    using Bates = wb::model::equity::bates::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<CEV, 1U>,
        wb::simulation::ModelGroup<Bates, 1U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.2f, 0.8f});
    host.scenario_indices.push_back(0U);
    host.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.04f,
                                2.0f, 0.04f, 0.5f, -0.4f,
                                0.7f, -0.05f, 0.15f});
    host.tail.scenario_indices.push_back(0U);
    const std::vector<std::string> names{
        "asset_0.spot", "asset_1.spot", "asset_1.variance"
    };
    constexpr double cross = 0.25;
    constexpr double rho = -0.4;
    const double covariance = cross * rho;
    const std::vector<double> dense{
        1.0, cross, covariance,
        cross, 1.0, rho,
        covariance, rho, 1.0
    };
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"jump", names, wb::correlation::packed_cholesky(dense, 3U)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    const std::vector<wb::monte_carlo::MultiAssetPriceRow> rows{
        {0U, 0U, 0U, 0U, 1.0f}
    };
    const auto p = profile(4096U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    const auto cooperative = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f
            && std::isfinite(cooperative.prices[0]),
            "CEV/Bates worst-of price is invalid.");
    require(scalar.prices[0] == cooperative.prices[0]
            && scalar.standard_errors[0] == cooperative.standard_errors[0],
            "Auxiliary QE/jump stream depends on path-team size.");
}

void check_jump_diffusions() {
    using Merton = wb::model::equity::merton::DynamicsPolicy;
    using Kou = wb::model::equity::kou::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<Merton, 1U>,
        wb::simulation::ModelGroup<Kou, 1U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                           0.8f, -0.05f, 0.15f});
    host.scenario_indices.push_back(0U);
    host.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                                0.8f, 0.4f, 5.0f, 6.0f});
    host.tail.scenario_indices.push_back(0U);
    const std::vector<std::string> names{
        "asset_0.spot", "asset_1.spot"
    };
    const std::vector<double> dense{1.0, 0.3, 0.3, 1.0};
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"jumps", names, wb::correlation::packed_cholesky(dense, 2U)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    const std::vector<wb::monte_carlo::MultiAssetPriceRow> rows{
        {0U, 0U, 0U, 0U, 1.0f}
    };
    const auto p = profile(4096U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    const auto cooperative = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f,
            "Merton/Kou worst-of price is invalid.");
    require(scalar.prices == cooperative.prices
            && scalar.standard_errors == cooperative.standard_errors,
            "Merton/Kou auxiliary jumps depend on path-team size.");
}

void check_exact_ou_bridges() {
    using Stein = wb::model::equity::stein_stein::DynamicsPolicy;
    using Schobel = wb::model::equity::schobel_zhu::DynamicsPolicy;
    using Heston32 = wb::model::equity::heston_3_2::DynamicsPolicy;
    using Pack = wb::simulation::MultiAssetPack<
        wb::simulation::ModelGroup<Stein, 1U>,
        wb::simulation::ModelGroup<Schobel, 1U>,
        wb::simulation::ModelGroup<Heston32, 1U>>;
    using Product = wb::product::WorstOfCallPathPolicy;
    wb::offline::pricing::MultiAssetHostModels<Pack> host;
    host.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                           2.0f, 0.3f, -0.3f});
    host.scenario_indices.push_back(0U);
    host.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.2f,
                                2.0f, 0.2f, 0.3f, -0.25f});
    host.tail.scenario_indices.push_back(0U);
    host.tail.tail.models.push_back({1.0f, 0.0f, 0.0f, 0.04f,
                                     2.0f, 0.04f, 0.3f, -0.5f});
    host.tail.tail.scenario_indices.push_back(0U);
    const std::vector<std::string> names{
        "asset_0.spot", "asset_0.volatility",
        "asset_1.spot", "asset_1.volatility",
        "asset_2.spot", "asset_2.volatility"
    };
    std::vector<double> dense(36U, 0.0);
    for (std::size_t i = 0; i < 6U; ++i) dense[i * 6U + i] = 1.0;
    for (std::size_t asset = 0; asset < 3U; ++asset) {
        const double rho = asset == 0U ? -0.3
            : asset == 1U ? -0.25 : -0.5;
        dense[(2U * asset) * 6U + 2U * asset + 1U] = rho;
        dense[(2U * asset + 1U) * 6U + 2U * asset] = rho;
    }
    const std::vector<wb::correlation::MatrixRow> matrices{
        {"ou", names, wb::correlation::packed_cholesky(dense, 6U)}
    };
    const std::vector<Product::ProductParameters> products{{0.95f, 21U}};
    const std::vector<wb::monte_carlo::MultiAssetPriceRow> rows{
        {0U, 0U, 0U, 0U, 1.0f}
    };
    const auto p = profile(4096U);
    const auto scalar = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 1U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    const auto cooperative = wb::offline::pricing::execute_multi_asset_prices<
        Pack, Product, 4U>(host, 1U, products, matrices, names, rows,
                           p, 1.0f / 252.0f, 1U);
    require(std::isfinite(scalar.prices[0]) && scalar.prices[0] > 0.0f,
            "OU/Heston 3/2 worst-of price is invalid.");
    require(scalar.prices == cooperative.prices
            && scalar.standard_errors == cooperative.standard_errors,
            "OU bridge or Heston 3/2 depends on path-team size.");
}

int main() {
    check_cholesky();
    check_thirty_assets();
    check_all_team_sizes();
    check_mixed_volatility();
    check_auxiliary_jump_stream();
    check_jump_diffusions();
    check_exact_ou_bridges();
}

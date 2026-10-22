// Eleven-factor positive-definite correlations for five equities and CIR.
#include "correlation/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    constexpr std::size_t dimension=11U;
    constexpr std::size_t latent_count=12U;
    constexpr std::array<const char*,dimension> factors{
        "heston.spot","heston.variance",
        "bates.spot","bates.variance",
        "sabr.spot","sabr.volatility",
        "rough_heston.spot","rough_heston.variance",
        "rough_bergomi.spot","rough_bergomi.variance",
        "cir.rate"};
    constexpr std::array<double,10U> common_shares{
        0.0,0.05,0.12,0.20,0.30,0.42,0.55,0.68,0.78,0.85};
    constexpr std::array<double,10U> rate_loadings{
        -0.35,-0.25,-0.15,-0.05,0.0,0.05,0.15,0.25,0.35,0.45};
    nlohmann::ordered_json rows=nlohmann::ordered_json::array();
    for(std::size_t scenario=0U;scenario<10U;++scenario) {
        std::array<std::array<double,latent_count>,dimension> loadings{};
        const double common=std::sqrt(common_shares[scenario]);
        const double specific=std::sqrt(1.0-common_shares[scenario]);
        for(std::size_t asset=0U;asset<5U;++asset) {
            auto& spot=loadings[2U*asset];
            auto& volatility=loadings[2U*asset+1U];
            spot[0U]=common;
            spot[1U+asset]=specific;
            const double own_rho=-0.55-0.4*static_cast<double>(
                (scenario+2U*asset)%10U)/9.0;
            for(std::size_t factor=0U;factor<latent_count;++factor)
                volatility[factor]=own_rho*spot[factor];
            volatility[6U+asset]=std::sqrt(1.0-own_rho*own_rho);
        }
        loadings[10U][0U]=rate_loadings[scenario];
        loadings[10U][11U]=std::sqrt(
            1.0-rate_loadings[scenario]*rate_loadings[scenario]);
        std::vector<double> dense(dimension*dimension,0.0);
        nlohmann::ordered_json matrix=nlohmann::ordered_json::array();
        for(std::size_t row=0U;row<dimension;++row) {
            nlohmann::ordered_json values=nlohmann::ordered_json::array();
            for(std::size_t column=0U;column<dimension;++column) {
                double value=0.0;
                for(std::size_t latent=0U;latent<latent_count;++latent)
                    value+=loadings[row][latent]*loadings[column][latent];
                dense[row*dimension+column]=value;
                values.push_back(value);
            }
            matrix.push_back(std::move(values));
        }
        correlation::packed_cholesky(dense,dimension);
        rows.push_back({{"id",datasets::format_row_id(scenario)},
            {"factors",factors},{"matrix",std::move(matrix)}});
    }
    constexpr auto path=
        "datasets/correlation/five_equity_cir_01.json";
    constexpr auto catalog=
        "catalog/correlation/five_equity_cir_01";
    datasets::write_json_file(path,{
        {"database_id","five_equity_cir_01"},
        {"catalog",catalog},
        {"row_count",rows.size()},
        {"correlations",rows}});
    correlation::load_matrices(path);
    datasets::write_yaml_document(
        "catalog/correlation/five_equity_cir_01/recipe.yaml",{
            {"schema_version",1},
            {"kind","correlation_parameters"},
            {"dataset_id","five_equity_cir_01"},
            {"generator","catalog/correlation/five_equity_cir_01/generator.cpp"},
            {"output",{{"path",path},{"format","json"}}},
            {"generation_details",
                "catalog/correlation/five_equity_cir_01/generation.yaml"},
            {"row_count",10},
            {"factor_order",factors},
            {"construction",{
                {"method","latent common spot driver and independent residuals"},
                {"spot_common_variance_shares",common_shares},
                {"rate_common_driver_loadings",rate_loadings},
                {"own_spot_volatility_correlation_range",{-0.95,-0.55}},
                {"positive_definite","verified by Cholesky per matrix"}}}});
    datasets::write_generation_receipt(
        "catalog/correlation/five_equity_cir_01/generation.yaml",
        rows.size(),{{"method","latent-factor Gram matrix"}},0.0,0.0);
}

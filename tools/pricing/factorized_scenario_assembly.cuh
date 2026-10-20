// Bind factor-only parameter rows to a complete correlated pricing scenario.
// This runs once per distinct (model selection, correlation row), never per path.
#pragma once

#include "correlation/dataset.hpp"
#include "tools/pricing/heterogeneous_rough_price_generation.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ai_factory::workbench::offline::pricing {

inline float correlation_from_factor(
    const correlation::MatrixRow& matrix,
    std::size_t first, std::size_t second
) {
    if (first >= matrix.dimension() || second >= matrix.dimension())
        throw std::invalid_argument("Factor index exceeds correlation matrix");
    double value = 0.0;
    const auto& lower = matrix.lower;
    for (std::size_t column = 0U;
         column <= (first < second ? first : second); ++column) {
        value += static_cast<double>(
            lower[first * (first + 1U) / 2U + column])
            * static_cast<double>(
                lower[second * (second + 1U) / 2U + column]);
    }
    if (!std::isfinite(value) || std::abs(value) > 1.0 + 2.0e-5)
        throw std::invalid_argument("Invalid recovered Brownian correlation");
    return static_cast<float>(value);
}

template<typename Pack>
struct FactorEquityBinder;

template<>
struct FactorEquityBinder<simulation::MultiAssetPack<>> {
    static void append(
        MultiAssetHostModels<simulation::MultiAssetPack<>>&,
        const MultiAssetHostModels<simulation::MultiAssetPack<>>&,
        std::uint32_t, const correlation::MatrixRow&, std::size_t
    ) {}
};

template<typename Head, typename... Tail>
struct FactorEquityBinder<simulation::MultiAssetPack<Head, Tail...>> {
    using Pack = simulation::MultiAssetPack<Head, Tail...>;
    using Rest = simulation::MultiAssetPack<Tail...>;

    static void append(
        MultiAssetHostModels<Pack>& output,
        const MultiAssetHostModels<Pack>& source,
        std::uint32_t scenario,
        const correlation::MatrixRow& matrix,
        std::size_t factor_offset
    ) {
        for (std::size_t asset = 0U; asset < Head::kAssetCount; ++asset) {
            auto model = source.models[source.scenario_indices[
                static_cast<std::size_t>(scenario) * Head::kAssetCount + asset]];
            model.risk_free_rate = 0.0f;
            if constexpr (Head::Dynamics::kBrownianDimension == 2U) {
                const auto first = factor_offset + 2U * asset;
                const float rho = correlation_from_factor(
                    matrix, first, first + 1U);
                if constexpr (requires { model.rho; }) model.rho = rho;
                else if constexpr (requires { model.correlation; })
                    model.correlation = rho;
                else static_assert(Head::Dynamics::kBrownianDimension != 2U,
                                   "Two-factor model needs a rho field");
            }
            if (output.models.size() >=
                std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Too many factor equity rows");
            output.scenario_indices.push_back(
                static_cast<std::uint32_t>(output.models.size()));
            output.models.push_back(model);
        }
        FactorEquityBinder<Rest>::append(
            output.tail, source.tail, scenario, matrix,
            factor_offset + Head::kFactorCount);
    }
};

template<typename Pack>
struct FactorRoughBinder;

template<>
struct FactorRoughBinder<simulation::MultiRoughPack<>> {
    static void append(
        MultiRoughHostModels<simulation::MultiRoughPack<>>&,
        const MultiRoughHostModels<simulation::MultiRoughPack<>>&,
        std::uint32_t, const correlation::MatrixRow&,
        std::size_t, std::size_t, std::size_t, std::size_t
    ) {}
};

template<typename Head, typename... Tail>
struct FactorRoughBinder<simulation::MultiRoughPack<Head, Tail...>> {
    using Pack = simulation::MultiRoughPack<Head, Tail...>;
    using Rest = simulation::MultiRoughPack<Tail...>;

    static void append(
        MultiRoughHostModels<Pack>& output,
        const MultiRoughHostModels<Pack>& source,
        std::uint32_t scenario,
        const correlation::MatrixRow& matrix,
        std::size_t rough_count,
        std::size_t markov_and_rate_factors,
        std::size_t variance_offset,
        std::size_t spot_offset
    ) {
        auto model = source.models[source.scenario_indices[scenario]];
        model.risk_free_rate = 0.0f;
        if constexpr (Head::kBrownianDimension == 2U) {
            model.rho = correlation_from_factor(
                matrix, variance_offset,
                rough_count + markov_and_rate_factors + spot_offset);
        }
        if (output.models.size() >=
            std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Too many factor rough rows");
        output.scenario_indices.push_back(
            static_cast<std::uint32_t>(output.models.size()));
        output.models.push_back(model);
        FactorRoughBinder<Rest>::append(
            output.tail, source.tail, scenario, matrix,
            rough_count, markov_and_rate_factors,
            variance_offset + 1U,
            spot_offset + (Head::kBrownianDimension == 2U ? 1U : 0U));
    }
};

template<typename EquityPack, typename RoughPack, typename RatePolicy>
struct FactorizedScenarios {
    MultiAssetHostModels<EquityPack> equities;
    MultiRoughHostModels<RoughPack> rough;
    std::vector<typename RatePolicy::Parameters> rates;
    std::vector<std::uint32_t> rate_indices; // indices into expanded device rates
    std::vector<std::uint32_t> source_rate_indices; // IDs in the source rate dataset
    std::vector<std::uint32_t> model_indices;
    std::vector<monte_carlo::MultiAssetPriceRow> rows;
    std::size_t scenario_count = 0U;
};

template<typename EquityPack, typename RoughPack, typename RatePolicy>
FactorizedScenarios<EquityPack, RoughPack, RatePolicy>
assemble_factorized_scenarios(
    const MultiAssetHostModels<EquityPack>& equities,
    const MultiRoughHostModels<RoughPack>& rough,
    const std::vector<typename RatePolicy::Parameters>& rates,
    const std::vector<std::uint32_t>& rate_indices,
    const std::vector<std::uint32_t>& model_indices,
    std::size_t source_scenarios,
    const std::vector<correlation::MatrixRow>& correlations,
    const std::vector<std::string>& external_factor_order,
    const std::vector<std::string>& internal_factor_order,
    const std::vector<monte_carlo::MultiAssetPriceRow>& rows
) {
    constexpr std::size_t asset_count =
        EquityPack::kAssetCount + RoughPack::kAssetCount;
    constexpr std::size_t rate_factor =
        RoughPack::kAssetCount + EquityPack::kFactorCount;
    constexpr std::size_t dimension = rate_factor
        + RatePolicy::kBrownianDimension + RoughPack::kTwoFactorCount;
    if (source_scenarios == 0U || rates.empty() || correlations.empty()
        || rows.empty() || rate_indices.size() != source_scenarios
        || model_indices.size() != source_scenarios * asset_count
        || external_factor_order.size() != dimension
        || internal_factor_order.size() != dimension)
        throw std::invalid_argument("Invalid factorized scenario dimensions");
    equities.validate(source_scenarios);
    rough.validate(source_scenarios);
    std::vector<correlation::MatrixRow> ordered;
    ordered.reserve(correlations.size());
    for (const auto& matrix : correlations) {
        if (matrix.factors != external_factor_order)
            throw std::invalid_argument("Unexpected factor order");
        ordered.push_back(correlation::reorder_factors(
            matrix, internal_factor_order));
    }
    FactorizedScenarios<EquityPack, RoughPack, RatePolicy> output;
    output.rows.reserve(rows.size());
    std::map<std::pair<std::uint32_t, std::uint32_t>,
             std::uint32_t> scenario_map;
    for (const auto& row : rows) {
        if (row.scenario_index >= source_scenarios
            || row.correlation_index >= ordered.size()
            || row.discount_factor != 1.0f)
            throw std::invalid_argument("Invalid factorized result row");
        const auto key = std::pair{row.scenario_index, row.correlation_index};
        auto [entry, inserted] = scenario_map.try_emplace(
            key, static_cast<std::uint32_t>(scenario_map.size()));
        if (inserted) {
            const auto& matrix = ordered[row.correlation_index];
            FactorEquityBinder<EquityPack>::append(
                output.equities, equities, row.scenario_index,
                matrix, RoughPack::kAssetCount);
            FactorRoughBinder<RoughPack>::append(
                output.rough, rough, row.scenario_index, matrix,
                RoughPack::kAssetCount,
                EquityPack::kFactorCount + RatePolicy::kBrownianDimension,
                0U, 0U);
            const auto source_rate_index = rate_indices[row.scenario_index];
            if (source_rate_index >= rates.size())
                throw std::invalid_argument("Rate model index exceeds dataset");
            auto rate = rates[source_rate_index];
            if constexpr (RatePolicy::kBrownianDimension == 2U) {
                static_assert(requires { rate.process.correlation; });
                rate.process.correlation = correlation_from_factor(
                    matrix, rate_factor, rate_factor + 1U);
            }
            output.rate_indices.push_back(
                static_cast<std::uint32_t>(output.rates.size()));
            output.source_rate_indices.push_back(source_rate_index);
            output.rates.push_back(rate);
            const auto offset =
                static_cast<std::size_t>(row.scenario_index) * asset_count;
            output.model_indices.insert(
                output.model_indices.end(),
                model_indices.begin() + offset,
                model_indices.begin() + offset + asset_count);
        }
        auto expanded_row = row;
        expanded_row.scenario_index = entry->second;
        output.rows.push_back(expanded_row);
    }
    output.scenario_count = scenario_map.size();
    return output;
}

}  // namespace ai_factory::workbench::offline::pricing

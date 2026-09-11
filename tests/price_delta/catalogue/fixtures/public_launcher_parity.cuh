// Public-launcher parity fixture; small path counts test contracts, not launch tuning.
#pragma once

#include "tests/price_delta/cuda_test_support.cuh"
#include "common/equity/price_delta/spot_bump.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/price_construction.cuh"
#include <array>
#include <bit>
#include <cmath>
#include <iostream>

namespace price_delta_test {
using namespace ai_factory::workbench;

template<bool Stochastic, bool EarlyExercise, typename Model, typename Product,
         typename Price, typename Delta, typename... Time>
void compare_public(const char* name, Model model, Product product, std::size_t paths,
                    Price price, Delta delta, Time... time) {
    constexpr std::size_t rows = 2;
    constexpr unsigned threads = Stochastic && !EarlyExercise ? 256 : 128;
    const std::size_t blocks = EarlyExercise ? 128 : 2;
    constexpr std::uint64_t seed = 719;
    const equity::price_delta::SpotBumpConfiguration bump{};
    std::array<Model, rows> models{model, model};
    models[1].spot *= .9f;
    const std::array<Product, rows> products{product, product};
    DeviceArray<Model> dm(rows);
    DeviceArray<Product> dp(rows);
    DeviceArray<float> output(6 * rows);
    check_cuda(cudaMemcpy(dm.data, models.data(), sizeof(models), cudaMemcpyHostToDevice), name);
    check_cuda(cudaMemcpy(dp.data, products.data(), sizeof(products), cudaMemcpyHostToDevice), name);
    auto* base = output.data;
    auto* base_error = base + rows;
    auto* central = base + 2 * rows;
    auto* error = base + 3 * rows;
    auto* gradient = base + 4 * rows;
    auto* gradient_error = base + 5 * rows;
    if constexpr (EarlyExercise) {
        const auto p = price(dm.data, rows, products.data(), dp.data, rows,
            PriceConstruction::Aligned, rows, paths, time..., threads, blocks, seed, base, base_error);
        const auto d = delta(models.data(), dm.data, rows, products.data(), dp.data, rows,
            PriceConstruction::Aligned, rows, paths, time..., threads, blocks, seed, bump,
            central, error, gradient, gradient_error);
        longstaff_schwartz::validate_regression_diagnostics(p, name);
        longstaff_schwartz::validate_regression_diagnostics(d, name);
        require(d.kernel_launch_count == p.kernel_launch_count + 2 * p.batch_count,
                "Unexpected LSM refit or pass count");
    } else if constexpr (Stochastic) {
        price(dm.data, rows, products.data(), dp.data, rows, PriceConstruction::Aligned,
            rows, 0, rows, paths, time..., threads, blocks, seed, base, base_error);
        for (std::size_t offset = 0; offset < rows; ++offset)
            delta(models.data(), dm.data, rows, products.data(), dp.data, rows,
                PriceConstruction::Aligned, rows, offset, 1, paths, time..., threads,
                1U, seed, bump, central, error, gradient, gradient_error);
    } else if constexpr (sizeof...(Time) == 2) {
        price(dm.data, rows, products.data(), dp.data, rows, PriceConstruction::Aligned,
            rows, 0, rows, time..., threads, blocks, base);
        delta(models.data(), dm.data, rows, products.data(), dp.data, rows,
            PriceConstruction::Aligned, rows, 0, rows, time..., threads, blocks, bump, central, gradient);
    } else {
        price(dm.data, rows, dp.data, rows, PriceConstruction::Aligned,
            rows, 0, rows, time..., threads, blocks, base);
        delta(models.data(), dm.data, rows, dp.data, rows, PriceConstruction::Aligned,
            rows, 0, rows, time..., threads, blocks, bump, central, gradient);
    }
    std::array<float, 6 * rows> values{};
    check_cuda(cudaMemcpy(values.data(), output.data, sizeof(values), cudaMemcpyDeviceToHost), name);
    for (std::size_t row = 0; row < rows; ++row) {
        if (std::bit_cast<unsigned>(values[row]) != std::bit_cast<unsigned>(values[2 * rows + row]))
            throw std::runtime_error(std::string(name) + ": central price bits differ");
        require(std::isfinite(values[row]) && std::isfinite(values[4 * rows + row]), "Non-finite price/delta");
        if constexpr (Stochastic) {
            if (std::bit_cast<unsigned>(values[rows + row]) != std::bit_cast<unsigned>(values[3 * rows + row]))
                throw std::runtime_error(std::string(name) + ": central error bits differ");
            require(std::isfinite(values[5 * rows + row]) && values[5 * rows + row] >= 0,
                    "Invalid paired standard error");
        }
    }
    std::cout << name << ": central bits identical; " << paths << " paths\n";
}
}  // namespace price_delta_test

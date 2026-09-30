// Flat fitted scalar bindings: rows sharing warps match isolated launches.
#include "model/fixed_income/cir_plus_plus/product/flat/rate_option_price_gradients.cuh"
#include "tests/price_gradients/diagonal_cuda_test_support.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace fitted = model::fixed_income::cir_plus_plus::flat;
namespace pg = price_gradients;

constexpr std::size_t kRowCount = 65U;
constexpr std::size_t kSensitivityCount = 2U;
constexpr pg::SensitivityOrders kOrders =
    pg::SensitivityOrders::first_and_second;

pg::LaunchConfiguration launch_configuration(std::size_t rows) {
    return {
        pg::PricingMethod::closed_form,
        0U,
        rows,
        0U,
        128U,
        rows,
        0U,
        1U,
    };
}

auto execute(
    const std::vector<model::fixed_income::cir_plus_plus::ModelParameters>& models,
    const std::vector<curve::flat::FlatCurveParameters>& curves,
    const std::vector<product::RateOptionParameters>& products
) {
    const pg::PriceGradientConfiguration selection{{
        {"curve.rate", {0.0005f, pg::BumpScale::absolute}},
        {"product.strike", {0.0005f, pg::BumpScale::absolute}},
    }};
    const auto plan = fitted::prepare_cir_plus_plus_flat_rate_option_sensitivities(
        models,
        curves,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        {kOrders}
    );
    return price_gradient_test::execute_diagonal<kOrders>(
        plan,
        launch_configuration(plan.result_count),
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            fitted::launch_cir_plus_plus_flat_rate_option_diagonal_sensitivities_cuda<
                OptionSide::call,
                kOrders
            >(host, inputs, stencils, launch, outputs);
        }
    );
}

void check_row(
    const price_gradient_test::DiagonalResults& packed,
    std::size_t row,
    const model::fixed_income::cir_plus_plus::ModelParameters& model,
    const curve::flat::FlatCurveParameters& curve,
    const product::RateOptionParameters& product
) {
    const auto isolated = execute({model}, {curve}, {product});
    price_gradient_test::require_same_bits(
        {packed.price[row]}, isolated.price, "Flat packed scalar price"
    );
    price_gradient_test::require_same_bits(
        {packed.price_error[row]}, isolated.price_error,
        "Flat packed scalar price error"
    );
    for (std::size_t sensitivity = 0U;
         sensitivity < kSensitivityCount;
         ++sensitivity) {
        const auto index = row * kSensitivityCount + sensitivity;
        price_gradient_test::require_same_bits(
            {packed.gradient[index]},
            {isolated.gradient[sensitivity]},
            "Flat packed scalar gradient"
        );
        price_gradient_test::require_same_bits(
            {packed.gradient_error[index]},
            {isolated.gradient_error[sensitivity]},
            "Flat packed scalar gradient error"
        );
        price_gradient_test::require_same_bits(
            {packed.diagonal_hessian[index]},
            {isolated.diagonal_hessian[sensitivity]},
            "Flat packed scalar diagonal Hessian"
        );
        price_gradient_test::require_same_bits(
            {packed.diagonal_hessian_error[index]},
            {isolated.diagonal_hessian_error[sensitivity]},
            "Flat packed scalar diagonal-Hessian error"
        );
        price_gradient_test::require_same_stencils(
            {packed.stencils[index]},
            {isolated.stencils[sensitivity]},
            "Flat packed scalar stencil"
        );
    }
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return 77;
    }

    std::vector<model::fixed_income::cir_plus_plus::ModelParameters> models;
    std::vector<curve::flat::FlatCurveParameters> curves;
    std::vector<product::RateOptionParameters> products;
    models.reserve(kRowCount);
    curves.reserve(kRowCount);
    products.reserve(kRowCount);
    for (std::size_t row = 0U; row < kRowCount; ++row) {
        const float coordinate = static_cast<float>(row);
        models.push_back({
            {0.20f + 0.002f * coordinate,
             0.025f + 0.0001f * coordinate,
             0.08f + 0.0002f * coordinate},
            0.02f + 0.0001f * coordinate,
        });
        curves.push_back({-0.005f + 0.00075f * coordinate});
        products.push_back({
            1.0f + 0.01f * coordinate,
            0.015f + 0.0004f * coordinate,
            126U,
            252U,
            126U,
        });
    }

    const auto packed = execute(models, curves, products);
    constexpr std::array selected_rows{0U, 15U, 16U, 31U, 32U, 63U, 64U};
    for (const auto row : selected_rows) {
        check_row(packed, row, models[row], curves[row], products[row]);
    }
    std::cout << "Flat scalar bindings preserve isolated results across warp boundaries\n";
    return 0;
}

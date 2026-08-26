// Compare CIR rate-option launchers with independent QuantLib 1.43 values.
#include "common/check_cuda.cuh"
#include "model/fixed_income/cir/rate_option.cuh"
#include "model/fixed_income/cir/zero_coupon_bond_option.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

namespace cir = ai_factory::workbench::model::fixed_income::cir;
namespace product = ai_factory::workbench::product;
using ai_factory::workbench::OptionSide;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Product, typename Launcher>
std::vector<float> launch_prices(
    const std::vector<ai_factory::workbench::model::fixed_income::cir::ModelParameters>& models,
    const std::vector<Product>& products,
    Launcher launcher
) {
    using ai_factory::workbench::check_cuda;
    ai_factory::workbench::model::fixed_income::cir::ModelParameters* device_models = nullptr;
    Product* device_products = nullptr;
    float* device_prices = nullptr;
    std::vector<float> prices(models.size());
    try {
        check_cuda(
            cudaMalloc(
                &device_models, models.size() * sizeof(models.front())
            ),
            "CIR rate-option test cudaMalloc models"
        );
        check_cuda(
            cudaMalloc(
                &device_products, products.size() * sizeof(products.front())
            ),
            "CIR rate-option test cudaMalloc products"
        );
        check_cuda(
            cudaMalloc(&device_prices, prices.size() * sizeof(float)),
            "CIR rate-option test cudaMalloc prices"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            "CIR rate-option test cudaMemcpy models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            "CIR rate-option test cudaMemcpy products"
        );
        launcher(
            device_models,
            models.size(),
            device_products,
            products.size(),
            false,
            prices.size(),
            0U,
            prices.size(),
            1.0f / 252.0f,
            32U,
            1U,
            device_prices
        );
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                prices.size() * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "CIR rate-option test cudaMemcpy prices"
        );
        check_cuda(
            cudaFree(device_models),
            "CIR rate-option test cudaFree models"
        );
        device_models = nullptr;
        check_cuda(
            cudaFree(device_products),
            "CIR rate-option test cudaFree products"
        );
        device_products = nullptr;
        check_cuda(
            cudaFree(device_prices),
            "CIR rate-option test cudaFree prices"
        );
        device_prices = nullptr;
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }
    return prices;
}

void compare(
    const std::vector<float>& prices,
    const std::vector<double>& expected,
    const char* message
) {
    require(prices.size() == expected.size(), "CIR test size mismatch");
    for (std::size_t index = 0U; index < prices.size(); ++index) {
        const double error = std::fabs(
            static_cast<double>(prices[index]) - expected[index]
        );
        if (!std::isfinite(prices[index])
            || prices[index] < 0.0f
            || error > 5.0e-6) {
            std::cerr << message << " at row " << index
                      << ": actual=" << prices[index]
                      << ", expected=" << expected[index]
                      << ", absolute_error=" << error << '\n';
            throw std::runtime_error(message);
        }
    }
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "CIR rate-option test cudaGetDeviceCount");

    // Row two deliberately violates Feller; row three has a narrow diffusion.
    const std::vector<ai_factory::workbench::model::fixed_income::cir::ModelParameters> models = {
        {{0.60f, 0.04f, 0.15f}, 0.03f},
        {{0.05f, 0.015f, 0.12f}, 0.001f},
        {{1.50f, 0.10f, 0.05f}, 0.20f},
    };
    const std::vector<product::ZeroCouponBondOptionParameters> bond_options = {
        {1.0f, 0.97f, 126U, 504U},
        {2.0f, 0.97f, 504U, 1260U},
        {1.5f, 0.95f, 252U, 504U},
    };
    const std::vector<product::RateOptionParameters> rate_options = {
        {1.0f, 0.04f, 126U, 252U, 126U},
        {2.0f, 0.02f, 504U, 756U, 126U},
        {1.5f, 0.15f, 252U, 504U, 252U},
    };

    const std::vector<float> calls = launch_prices(
        models,
        bond_options,
        ai_factory::workbench::model::fixed_income::cir::launch_cir_zero_coupon_bond_option_cuda<OptionSide::call>
    );
    const std::vector<float> puts = launch_prices(
        models,
        bond_options,
        ai_factory::workbench::model::fixed_income::cir::launch_cir_zero_coupon_bond_option_cuda<OptionSide::put>
    );
    const std::vector<float> caplets = launch_prices(
        models,
        rate_options,
        ai_factory::workbench::model::fixed_income::cir::launch_cir_rate_option_cuda<OptionSide::call>
    );
    const std::vector<float> floorlets = launch_prices(
        models,
        rate_options,
        ai_factory::workbench::model::fixed_income::cir::launch_cir_rate_option_cuda<OptionSide::put>
    );

    compare(
        calls,
        {0.000139872875123716, 0.0455935442734126, 8.63795473484784e-131},
        "CIR bond-call price differs from QuantLib"
    );
    compare(
        puts,
        {0.0207236489712082, 0.00422955846218609, 0.0715353974806218},
        "CIR bond-put price differs from QuantLib"
    );
    compare(
        caplets,
        {0.00162889439636455, 0.00160689495376182, 2.32979164294100e-9},
        "CIR caplet price differs from QuantLib"
    );
    compare(
        floorlets,
        {0.00465377163401236, 0.0163504191445522, 0.0369468924249311},
        "CIR floorlet price differs from QuantLib"
    );
    return 0;
}

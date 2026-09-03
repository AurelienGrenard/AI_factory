// Compare beta=0 SABR with the killed arithmetic-Brownian reference law.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/sabr/european_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

namespace sabr = ai_factory::workbench::model::equity::sabr;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double normal_cdf(double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double normal_pdf(double value) {
    constexpr double inverse_sqrt_two_pi = 0.39894228040143267794;
    return inverse_sqrt_two_pi * std::exp(-0.5 * value * value);
}

double normal_call(double mean, double standard_deviation, double strike) {
    const double d = (mean - strike) / standard_deviation;
    return (mean - strike) * normal_cdf(d)
        + standard_deviation * normal_pdf(d);
}

double killed_arithmetic_brownian_call(
    double spot,
    double alpha,
    double strike,
    double maturity
) {
    const double standard_deviation = alpha * std::sqrt(maturity);
    return normal_call(spot, standard_deviation, strike)
        - normal_call(-spot, standard_deviation, strike);
}

struct Run {
    std::array<float, 2> prices;
    std::array<float, 2> standard_errors;
};

}  // namespace

int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "SABR boundary cudaGetDeviceCount");

    constexpr std::array<sabr::ModelParameters, 2> models{{
        {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
        {0.05f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f},
    }};
    constexpr std::array<product::EuropeanOptionParameters, 2> products{{
        {1.0f, 1U},
        {0.05f, 1U},
    }};
    constexpr std::size_t path_count = 1U << 20U;
    constexpr unsigned int threads_per_block = 256U;
    constexpr std::size_t block_count = models.size();
    constexpr std::uint64_t seed = 902000001ULL;

    sabr::ModelParameters* device_models = nullptr;
    product::EuropeanOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    float* device_standard_errors = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_models, sizeof(models)),
            "SABR boundary model allocation"
        );
        check_cuda(
            cudaMalloc(&device_products, sizeof(products)),
            "SABR boundary product allocation"
        );
        check_cuda(
            cudaMalloc(&device_prices, 2U * sizeof(float)),
            "SABR boundary price allocation"
        );
        check_cuda(
            cudaMalloc(&device_standard_errors, 2U * sizeof(float)),
            "SABR boundary error allocation"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                sizeof(models),
                cudaMemcpyHostToDevice
            ),
            "SABR boundary model copy"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                sizeof(products),
                cudaMemcpyHostToDevice
            ),
            "SABR boundary product copy"
        );

        const auto launch = [&](std::uint32_t steps_per_day) {
            sabr::launch_sabr_european_option_cuda<OptionSide::call>(
                device_models,
                models.size(),
                device_products,
                products.size(),
                PriceConstruction::Aligned,
                models.size(),
                0U,
                models.size(),
                path_count,
                1.0f / static_cast<float>(steps_per_day),
                steps_per_day,
                threads_per_block,
                block_count,
                seed,
                device_prices,
                device_standard_errors
            );
            check_cuda(cudaDeviceSynchronize(), "SABR boundary synchronize");
            Run result{};
            check_cuda(
                cudaMemcpy(
                    result.prices.data(),
                    device_prices,
                    sizeof(result.prices),
                    cudaMemcpyDeviceToHost
                ),
                "SABR boundary price copy"
            );
            check_cuda(
                cudaMemcpy(
                    result.standard_errors.data(),
                    device_standard_errors,
                    sizeof(result.standard_errors),
                    cudaMemcpyDeviceToHost
                ),
                "SABR boundary error copy"
            );
            return result;
        };

        const Run coarse = launch(1U);
        const Run fine = launch(256U);
        for (std::size_t index = 0U; index < models.size(); ++index) {
            const double spot = models[index].spot;
            const double alpha = models[index].initial_volatility * spot;
            const double strike = products[index].strike;
            const double killed_reference = killed_arithmetic_brownian_call(
                spot, alpha, strike, 1.0
            );
            const double unrestricted_reference = normal_call(
                spot, alpha, strike
            );
            const double coarse_error = std::fabs(
                static_cast<double>(coarse.prices[index]) - killed_reference
            );
            const double fine_error = std::fabs(
                static_cast<double>(fine.prices[index]) - killed_reference
            );
            const double joint_statistical_error = 5.0 * std::hypot(
                coarse.standard_errors[index], fine.standard_errors[index]
            );
            std::cerr
                << "SABR beta=0 row " << index
                << ": reference=" << killed_reference
                << ", unrestricted=" << unrestricted_reference
                << ", coarse=" << coarse.prices[index]
                << " +/- " << coarse.standard_errors[index]
                << ", fine=" << fine.prices[index]
                << " +/- " << fine.standard_errors[index]
                << '\n';
            require(
                std::fabs(
                    static_cast<double>(coarse.prices[index])
                    - unrestricted_reference
                ) <= 5.0 * coarse.standard_errors[index] + 0.002 * spot,
                "one-step beta=0 SABR does not match arithmetic Brownian motion"
            );
            require(
                fine_error < coarse_error,
                "beta=0 absorbing SABR does not converge toward the killed law"
            );
            require(
                fine_error <= 0.03 * spot + 5.0 * fine.standard_errors[index],
                "refined beta=0 SABR exceeds its killed-law error bound"
            );
            require(
                fine.prices[index]
                    <= coarse.prices[index] + joint_statistical_error,
                "absorbing-grid refinement raises the beta=0 call price"
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        if (device_standard_errors != nullptr) {
            cudaFree(device_standard_errors);
        }
        throw;
    }
    check_cuda(cudaFree(device_models), "SABR boundary model free");
    check_cuda(cudaFree(device_products), "SABR boundary product free");
    check_cuda(cudaFree(device_prices), "SABR boundary price free");
    check_cuda(
        cudaFree(device_standard_errors),
        "SABR boundary error free"
    );
    return 0;
}

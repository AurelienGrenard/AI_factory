// Exact Variance-Gamma and NIG terminal gradients over shared primitives.
#include "model/equity/markovian/normal_inverse_gaussian/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/european_option_price_delta.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/variance_gamma/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/product/european_option_price_delta.cuh"
#include "model/equity/markovian/variance_gamma/price_gradients/coupled_dynamics_impl.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>

using namespace price_gradient_test;
namespace nig = model::equity::normal_inverse_gaussian;
namespace vg = model::equity::variance_gamma;
namespace philox = ai_factory::workbench::philox;

namespace {

__device__ bool same_bits(float first, float second) {
    return __float_as_uint(first) == __float_as_uint(second);
}

struct CouplingViolations {
    int vg = 0;
    int nig = 0;
};

__global__ void coupling_contract_kernel(
    vg::ModelParameters vg_base,
    nig::ModelParameters nig_base,
    std::uint64_t seed,
    std::size_t path_count,
    CouplingViolations* violations
) {
    for (std::size_t path = std::size_t(blockIdx.x) * blockDim.x
             + threadIdx.x;
         path < path_count;
         path += std::size_t(gridDim.x) * blockDim.x) {
        using VgCoupling = vg::price_gradients::CoupledDynamics;
        auto vg_nu = vg_base;
        auto vg_theta = vg_base;
        auto vg_maturity = vg_base;
        vg_nu.nu *= 1.1f;
        vg_theta.theta += 0.02f;
        const VgCoupling::Prepared vg_nodes[4U]{
            VgCoupling::prepare(vg_base, 0.5f),
            VgCoupling::prepare(vg_nu, 0.5f),
            VgCoupling::prepare(vg_theta, 0.5f),
            VgCoupling::prepare(vg_maturity, 0.75f),
        };
        philox::DomainRandomContext vg_coupled_random(
            philox::make_key(seed), path
        );
        VgCoupling::Innovations vg_coupled[4U]{};
        VgCoupling::draw_coupled(
            vg_coupled_random, vg_nodes, 4U, vg_coupled
        );
        philox::DomainRandomContext vg_standalone_random(
            philox::make_key(seed), path
        );
        const auto vg_standalone = VgCoupling::draw(
            vg_standalone_random, vg_nodes[0U]
        );
        philox::DomainRandomContext vg_prefix_random(
            philox::make_key(seed), path
        );
        VgCoupling::Innovations vg_prefix[3U]{};
        const VgCoupling::Prepared vg_prefix_nodes[3U]{
            vg_nodes[0U], vg_nodes[1U], vg_nodes[2U]
        };
        VgCoupling::draw_coupled(
            vg_prefix_random, vg_prefix_nodes, 3U, vg_prefix
        );
        bool vg_invalid = !same_bits(
                vg_standalone.gamma_increment,
                vg_coupled[0U].gamma_increment
            )
            || !same_bits(
                vg_standalone.brownian_normal,
                vg_coupled[0U].brownian_normal
            )
            || !same_bits(
                vg_coupled[0U].gamma_increment,
                vg_coupled[2U].gamma_increment
            );
        for (unsigned int node = 0U; node < 3U; ++node) {
            vg_invalid = vg_invalid
                || !same_bits(
                    vg_prefix[node].gamma_increment,
                    vg_coupled[node].gamma_increment
                )
                || !same_bits(
                    vg_prefix[node].brownian_normal,
                    vg_coupled[node].brownian_normal
                );
        }

        using NigCoupling = nig::price_gradients::CoupledDynamics;
        auto nig_alpha = nig_base;
        auto nig_beta = nig_base;
        nig_alpha.alpha *= 1.1f;
        nig_beta.beta += 0.1f;
        const NigCoupling::Prepared nig_nodes[4U]{
            NigCoupling::prepare(nig_base, 0.5f),
            NigCoupling::prepare(nig_alpha, 0.5f),
            NigCoupling::prepare(nig_beta, 0.5f),
            NigCoupling::prepare(nig_base, 0.75f),
        };
        philox::NormalRandomContext nig_coupled_random(
            philox::make_key(seed + 1U), path
        );
        NigCoupling::Innovations nig_coupled[4U]{};
        NigCoupling::draw_coupled(
            nig_coupled_random, nig_nodes, 4U, nig_coupled
        );
        philox::NormalRandomContext nig_standalone_random(
            philox::make_key(seed + 1U), path
        );
        const auto nig_standalone = NigCoupling::draw(
            nig_standalone_random, nig_nodes[0U]
        );
        philox::NormalRandomContext nig_prefix_random(
            philox::make_key(seed + 1U), path
        );
        NigCoupling::Innovations nig_prefix[3U]{};
        const NigCoupling::Prepared nig_prefix_nodes[3U]{
            nig_nodes[0U], nig_nodes[1U], nig_nodes[2U]
        };
        NigCoupling::draw_coupled(
            nig_prefix_random, nig_prefix_nodes, 3U, nig_prefix
        );
        bool nig_invalid = !same_bits(
                nig_standalone.inverse_gaussian_increment,
                nig_coupled[0U].inverse_gaussian_increment
            )
            || !same_bits(
                nig_standalone.brownian_normal,
                nig_coupled[0U].brownian_normal
            );
        for (unsigned int node = 0U; node < 3U; ++node) {
            nig_invalid = nig_invalid
                || !same_bits(
                    nig_prefix[node].inverse_gaussian_increment,
                    nig_coupled[node].inverse_gaussian_increment
                )
                || !same_bits(
                    nig_prefix[node].brownian_normal,
                    nig_coupled[node].brownian_normal
                );
        }
        if (vg_invalid) atomicAdd(&violations->vg, 1);
        if (nig_invalid) atomicAdd(&violations->nig, 1);
    }
}

void coupling_contract() {
    constexpr std::size_t path_count = 1U << 14U;
    constexpr unsigned int threads = 256U;
    DeviceArray<CouplingViolations> violations(1U);
    coupling_contract_kernel<<<
        static_cast<unsigned int>((path_count + threads - 1U) / threads),
        threads
    >>>(
        {1.0f, 0.03f, 0.01f, 0.2f, 0.2f, -0.1f},
        {1.0f, 0.03f, 0.01f, 8.0f, -2.0f, 0.5f},
        9017U,
        path_count,
        violations.data
    );
    check_cuda(cudaDeviceSynchronize(), "Levy coupling contract");
    const auto result = violations.read()[0U];
    require(result.vg == 0, "VG primitive replay or projectivity failed.");
    require(result.nig == 0, "NIG primitive replay or projectivity failed.");
}

std::complex<double> characteristic(
    const vg::ModelParameters& model,
    double maturity,
    std::complex<double> argument
) {
    constexpr std::complex<double> imaginary{0.0, 1.0};
    const double variance = double(model.sigma) * model.sigma;
    const double omega = std::log(
        1.0 - model.theta * model.nu - 0.5 * variance * model.nu
    ) / model.nu;
    const double location = std::log(double(model.spot))
        + (model.risk_free_rate - model.dividend_yield + omega) * maturity;
    const auto gamma_characteristic = 1.0
        - imaginary * double(model.theta) * double(model.nu) * argument
        + 0.5 * variance * double(model.nu) * argument * argument;
    return std::exp(imaginary * argument * location)
        * std::pow(gamma_characteristic, -maturity / model.nu);
}

std::complex<double> characteristic(
    const nig::ModelParameters& model,
    double maturity,
    std::complex<double> argument
) {
    constexpr std::complex<double> imaginary{0.0, 1.0};
    const double alpha2 = double(model.alpha) * model.alpha;
    const double gamma = std::sqrt(
        alpha2 - double(model.beta) * model.beta
    );
    const double shifted = model.beta + 1.0;
    const double omega = model.delta
        * (std::sqrt(alpha2 - shifted * shifted) - gamma);
    const double location = std::log(double(model.spot))
        + (model.risk_free_rate - model.dividend_yield + omega) * maturity;
    const auto tilted = double(model.beta) + imaginary * argument;
    return std::exp(
        imaginary * argument * location
        + double(model.delta) * maturity
            * (gamma - std::sqrt(alpha2 - tilted * tilted))
    );
}

template<OptionSide Side, typename Scenario>
double independent_price(const Scenario& scenario) {
    constexpr std::complex<double> imaginary{0.0, 1.0};
    constexpr double upper = 160.0;
    constexpr double spacing = 0.01;
    const auto& model = scenario.model;
    const double maturity = scenario.maturity_years;
    const double log_strike = std::log(double(scenario.product.strike));
    const auto stock_expectation = characteristic(
        model, maturity, -imaginary
    );
    double first_integral = 0.0;
    double second_integral = 0.0;
    for (double u = 0.5 * spacing; u < upper; u += spacing) {
        const auto phase = std::exp(-imaginary * u * log_strike);
        first_integral += std::real(
            phase * characteristic(model, maturity, u - imaginary)
            / (imaginary * u * stock_expectation)
        );
        second_integral += std::real(
            phase * characteristic(model, maturity, u)
            / (imaginary * u)
        );
    }
    const double pi = std::acos(-1.0);
    const double first_probability = 0.5 + spacing * first_integral / pi;
    const double second_probability = 0.5 + spacing * second_integral / pi;
    const double discount = std::exp(-model.risk_free_rate * maturity);
    const double carry = std::exp(-model.dividend_yield * maturity);
    const double call = model.spot * carry * first_probability
        - scenario.product.strike * discount * second_probability;
    if constexpr (Side == OptionSide::call) return call;
    return call - model.spot * carry + scenario.product.strike * discount;
}

template<OptionSide Side, typename Plan>
void compare_independent(
    const char* model_name,
    const Plan& plan,
    const Results& result
) {
    const auto sensitivity_count = plan.sensitivity_count();
    for (std::size_t row = 0U; row < plan.result_count; ++row) {
        const double central = independent_price<Side>(
            central_scenario(plan, row)
        );
        if (std::abs(result.price[row] - central)
            > 7.0 * result.price_error[row] + 1.5e-3) {
            std::cerr << model_name << " independent price mismatch row="
                      << row << " estimate=" << result.price[row]
                      << " reference=" << central
                      << " se=" << result.price_error[row] << '\n';
            throw std::runtime_error("Independent Levy price comparison failed.");
        }
        for (std::size_t sensitivity = 0U;
             sensitivity < sensitivity_count;
             ++sensitivity) {
            const auto task = sensitivity_task<pg::SensitivityOrders::first>(
                plan, row, sensitivity
            );
            const double first = independent_price<Side>(task.nodes[1U]);
            const double second = independent_price<Side>(task.nodes[2U]);
            const auto& stencil = task.stencil;
            const double reference = stencil.kind == pg::StencilKind::centered
                ? (second - first) / stencil.represented_width
                : stencil.first_endpoint_weights[0U] * (first - central)
                    + stencil.first_endpoint_weights[1U]
                        * (second - central);
            const auto index = row * sensitivity_count + sensitivity;
            const double estimate = result.gradient[index];
            const double standard_error = result.gradient_error[index];
            const double numerical_budget = 1.5e-3
                + 1.5e-2 * std::abs(reference);
            if (std::abs(estimate - reference)
                > 7.0 * standard_error + numerical_budget) {
                std::cerr << model_name << " independent gradient mismatch row="
                          << row << " coordinate=" << sensitivity
                          << " estimate=" << estimate
                          << " reference=" << reference
                          << " se=" << standard_error << '\n';
                throw std::runtime_error(
                    "Independent Levy represented-stencil comparison failed."
                );
            }
        }
    }
}

template<OptionSide Side>
void check_variance_gamma(std::size_t paths) {
    const std::vector<vg::ModelParameters> models{
        {1.05f, 0.03f, 0.01f, 0.2f, 0.2f, -0.1f},
        {1.0f, -0.01f, 0.0f, 0.35f, 0.5f, -0.2f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 126U}, {1.1f, 504U},
    };
    const pg::Sensitivity spot{"model.spot", {0.005f}};
    const pg::PriceGradientConfiguration full{{
        spot,
        {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
        {"model.sigma", {0.005f}},
        {"model.nu", {0.005f}},
        {"model.theta", {0.002f, pg::BumpScale::absolute}},
        {"product.strike", {0.005f}},
        {"product.maturity_years", {
            1.0f / 504.0f, pg::BumpScale::absolute
        }},
    }};
    auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return vg::prepare_variance_gamma_european_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selection
        );
    };
    auto launcher =
        vg::launch_variance_gamma_european_option_price_gradients_cuda<Side>;
    const auto plan = prepare(full);
    DeviceArray<vg::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    for (unsigned int threads : {128U, 256U}) {
        pg::LaunchConfiguration launch{
            pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
            threads, models.size(), 19001U
        };
        const auto solo = execute(prepare({{spot}}), launch, launcher);
        const auto price_only = execute(prepare({}), launch, launcher);
        DeviceArray<float> legacy(4U * models.size());
        vg::launch_variance_gamma_european_option_price_delta_cuda<Side>(
            models.data(), device_models.data, models.size(),
            products.data(), device_products.data, products.size(),
            PriceConstruction::Aligned, models.size(), 0U, models.size(),
            paths, 1.0f / 252.0f, threads, models.size(), 19001U,
            {0.01f}, legacy.data, legacy.data + models.size(),
            legacy.data + 2U * models.size(),
            legacy.data + 3U * models.size()
        );
        const auto reference = legacy.read();
        for (std::size_t row = 0U; row < models.size(); ++row) {
            same(price_only.price[row], solo.price[row],
                 "VG empty selection changed price");
            same(price_only.price_error[row], solo.price_error[row],
                 "VG empty selection changed price error");
            same(solo.price[row], reference[row], "VG legacy price");
            same(solo.price_error[row], reference[models.size() + row],
                 "VG legacy price error");
            same(solo.gradient[row], reference[2U * models.size() + row],
                 "VG legacy delta");
            same(solo.gradient_error[row],
                 reference[3U * models.size() + row],
                 "VG legacy delta error");
        }
        const auto all = execute(plan, launch, launcher);
        compare_independent<Side>("VG", plan, all);
        if (threads == 256U) {
            selected_prefixes(full, all, launch, prepare, launcher);
        }
    }

    const auto diagonal_plan =
        vg::prepare_variance_gamma_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, full,
            {pg::SensitivityOrders::first_and_second}
        );
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        256U, models.size() * full.sensitivities.size(), 19001U, 1U
    };
    const auto diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan,
        launch,
        vg::launch_variance_gamma_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto first = execute(plan, launch, launcher);
    for (std::size_t index = 0U; index < diagonal.gradient.size(); ++index) {
        same_or_one_ulp(
            diagonal.gradient[index], first.gradient[index],
            "VG derivative order changed gradient"
        );
        same_or_one_ulp(
            diagonal.gradient_error[index], first.gradient_error[index],
            "VG derivative order changed gradient error"
        );
        require(std::isfinite(diagonal.diagonal_hessian[index])
                && std::isfinite(diagonal.diagonal_hessian_error[index]),
                "VG diagonal Hessian is non-finite.");
    }
}

template<OptionSide Side>
void check_nig(std::size_t paths) {
    const std::vector<nig::ModelParameters> models{
        {1.05f, 0.03f, 0.01f, 8.0f, -2.0f, 0.5f},
        {1.0f, -0.01f, 0.0f, 15.0f, -5.0f, 1.0f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 126U}, {1.1f, 504U},
    };
    const pg::Sensitivity spot{"model.spot", {0.005f}};
    const pg::PriceGradientConfiguration full{{
        spot,
        {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
        {"model.alpha", {0.005f}},
        {"model.beta", {0.002f, pg::BumpScale::absolute}},
        {"model.delta", {0.005f}},
        {"product.strike", {0.005f}},
        {"product.maturity_years", {
            1.0f / 504.0f, pg::BumpScale::absolute
        }},
    }};
    auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return nig::prepare_normal_inverse_gaussian_european_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selection
        );
    };
    auto launcher =
        nig::launch_normal_inverse_gaussian_european_option_price_gradients_cuda<
            Side
        >;
    const auto plan = prepare(full);
    DeviceArray<nig::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    for (unsigned int threads : {128U, 256U}) {
        pg::LaunchConfiguration launch{
            pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
            threads, models.size(), 19101U
        };
        const auto solo = execute(prepare({{spot}}), launch, launcher);
        const auto price_only = execute(prepare({}), launch, launcher);
        DeviceArray<float> legacy(4U * models.size());
        nig::launch_normal_inverse_gaussian_european_option_price_delta_cuda<
            Side
        >(
            models.data(), device_models.data, models.size(),
            products.data(), device_products.data, products.size(),
            PriceConstruction::Aligned, models.size(), 0U, models.size(),
            paths, 1.0f / 252.0f, threads, models.size(), 19101U,
            {0.01f}, legacy.data, legacy.data + models.size(),
            legacy.data + 2U * models.size(),
            legacy.data + 3U * models.size()
        );
        const auto reference = legacy.read();
        for (std::size_t row = 0U; row < models.size(); ++row) {
            same(price_only.price[row], solo.price[row],
                 "NIG empty selection changed price");
            same(price_only.price_error[row], solo.price_error[row],
                 "NIG empty selection changed price error");
            same(solo.price[row], reference[row], "NIG legacy price");
            same(solo.price_error[row], reference[models.size() + row],
                 "NIG legacy price error");
            same(solo.gradient[row], reference[2U * models.size() + row],
                 "NIG legacy delta");
            same(solo.gradient_error[row],
                 reference[3U * models.size() + row],
                 "NIG legacy delta error");
        }
        const auto all = execute(plan, launch, launcher);
        compare_independent<Side>("NIG", plan, all);
        if (threads == 256U) {
            selected_prefixes(full, all, launch, prepare, launcher);
        }
    }

    const auto diagonal_plan =
        nig::prepare_normal_inverse_gaussian_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, full,
            {pg::SensitivityOrders::first_and_second}
        );
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        256U, models.size() * full.sensitivities.size(), 19101U, 1U
    };
    const auto diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan,
        launch,
        nig::launch_normal_inverse_gaussian_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto first = execute(plan, launch, launcher);
    for (std::size_t index = 0U; index < diagonal.gradient.size(); ++index) {
        same_or_one_ulp(
            diagonal.gradient[index], first.gradient[index],
            "NIG derivative order changed gradient"
        );
        same_or_one_ulp(
            diagonal.gradient_error[index], first.gradient_error[index],
            "NIG derivative order changed gradient error"
        );
        require(std::isfinite(diagonal.diagonal_hessian[index])
                && std::isfinite(diagonal.diagonal_hessian_error[index]),
                "NIG diagonal Hessian is non-finite.");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::size_t paths = 65537U;
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            paths = 513U;
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: test [--sanitizer]");
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        coupling_contract();
        check_variance_gamma<OptionSide::call>(paths);
        check_variance_gamma<OptionSide::put>(paths);
        check_nig<OptionSide::call>(paths);
        check_nig<OptionSide::put>(paths);
        std::cout << "VG and NIG eight-coordinate gradients passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

// Kou event replay, analytical prices and selected diagonal sensitivities.
#include "model/equity/markovian/kou/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/kou/product/european_option_price_delta.cuh"
#include "model/equity/markovian/kou/price_gradients/coupled_dynamics_impl.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>

using namespace price_gradient_test;
namespace kou = model::equity::kou;
namespace philox = ai_factory::workbench::philox;

namespace {

using KouCoupling = kou::price_gradients::CoupledDynamics;

__device__ bool same_bits(float first, float second) {
    return __float_as_uint(first) == __float_as_uint(second);
}

__global__ void event_coupling_contract_kernel(
    kou::ModelParameters base,
    std::uint64_t seed,
    std::size_t path_count,
    int* violations
) {
    for (std::size_t path = std::size_t(blockIdx.x) * blockDim.x
             + threadIdx.x;
         path < path_count;
         path += std::size_t(gridDim.x) * blockDim.x) {
        kou::ModelParameters lower_intensity = base;
        kou::ModelParameters upper_intensity = base;
        lower_intensity.jump_intensity *= 0.75f;
        upper_intensity.jump_intensity *= 1.25f;
        const KouCoupling::Prepared intensity_nodes[3]{
            KouCoupling::prepare(base, 0.5f),
            KouCoupling::prepare(lower_intensity, 0.5f),
            KouCoupling::prepare(upper_intensity, 0.5f),
        };
        philox::DomainRandomContext coupled_random(
            philox::make_key(seed), path
        );
        KouCoupling::Innovations coupled[3]{};
        KouCoupling::draw_coupled(
            coupled_random, intensity_nodes, 3U, coupled
        );
        philox::DomainRandomContext standalone_random(
            philox::make_key(seed), path
        );
        const auto standalone = KouCoupling::draw(
            standalone_random, intensity_nodes[0U]
        );
        bool invalid = coupled[1U].jump_count > coupled[0U].jump_count
            || coupled[0U].jump_count > coupled[2U].jump_count
            || standalone.jump_count != coupled[0U].jump_count
            || !same_bits(
                standalone.diffusion_normal,
                coupled[0U].diffusion_normal
            )
            || !same_bits(standalone.jump_log_sum, coupled[0U].jump_log_sum);

        const KouCoupling::Prepared maturity_nodes[3]{
            KouCoupling::prepare(base, 0.5f),
            KouCoupling::prepare(base, 0.25f),
            KouCoupling::prepare(base, 0.75f),
        };
        philox::DomainRandomContext maturity_random(
            philox::make_key(seed + 1U), path
        );
        KouCoupling::Innovations maturity[3]{};
        KouCoupling::draw_coupled(
            maturity_random, maturity_nodes, 3U, maturity
        );
        invalid = invalid || maturity[1U].jump_count > maturity[0U].jump_count
            || maturity[0U].jump_count > maturity[2U].jump_count;

        kou::ModelParameters zero = base;
        kou::ModelParameters low = base;
        kou::ModelParameters middle = base;
        kou::ModelParameters high = base;
        zero.jump_intensity = 0.0f;
        low.jump_intensity = 0.1f;
        middle.jump_intensity = 0.2f;
        high.jump_intensity = 0.3f;
        const KouCoupling::Prepared boundary_nodes[4]{
            KouCoupling::prepare(zero, 0.5f),
            KouCoupling::prepare(low, 0.5f),
            KouCoupling::prepare(middle, 0.5f),
            KouCoupling::prepare(high, 0.5f),
        };
        philox::DomainRandomContext boundary_random(
            philox::make_key(seed + 2U), path
        );
        KouCoupling::Innovations boundary[4]{};
        KouCoupling::draw_coupled(
            boundary_random, boundary_nodes, 4U, boundary
        );
        invalid = invalid || boundary[0U].jump_count != 0U
            || boundary[0U].jump_count > boundary[1U].jump_count
            || boundary[1U].jump_count > boundary[2U].jump_count
            || boundary[2U].jump_count > boundary[3U].jump_count;
        if (invalid) atomicAdd(violations, 1);
    }
}

void event_coupling_contract() {
    constexpr std::size_t path_count = 1U << 14U;
    constexpr unsigned threads = 256U;
    DeviceArray<int> violations(1U);
    event_coupling_contract_kernel<<<
        static_cast<unsigned>((path_count + threads - 1U) / threads),
        threads
    >>>(
        {1.05f, 0.03f, 0.01f, 0.2f, 2.0f, 0.4f, 4.0f, 5.0f},
        2017U, path_count, violations.data
    );
    check_cuda(cudaDeviceSynchronize(), "Kou event coupling contract");
    require(
        violations.read()[0U] == 0,
        "Kou event coupling violated nesting or central replay."
    );
}

std::complex<double> characteristic(
    const kou::ModelParameters& model,
    double maturity,
    std::complex<double> argument
) {
    constexpr std::complex<double> imaginary{0.0, 1.0};
    const double jump_martingale = model.up_probability
            / (model.positive_jump_rate - 1.0)
        - (1.0 - model.up_probability)
            / (model.negative_jump_rate + 1.0);
    const double variance = double(model.volatility) * model.volatility;
    const double positive_rate = model.positive_jump_rate;
    const double negative_rate = model.negative_jump_rate;
    const double drift = std::log(double(model.spot))
        + (model.risk_free_rate - model.dividend_yield
           - model.jump_intensity * jump_martingale - 0.5 * variance)
            * maturity;
    const auto jump_characteristic = model.up_probability
            * positive_rate
            / (positive_rate - imaginary * argument)
        + (1.0 - model.up_probability) * negative_rate
            / (negative_rate + imaginary * argument);
    return std::exp(
        imaginary * argument * drift
        - 0.5 * variance * argument * argument * maturity
        + model.jump_intensity * maturity
            * (jump_characteristic - 1.0)
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
void compare_independent_gradients(const Plan& plan, const Results& result) {
    const auto sensitivity_count = plan.sensitivity_count();
    for (std::size_t row = 0U; row < plan.result_count; ++row) {
        const double central = independent_price<Side>(
            central_scenario(plan, row)
        );
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
            const double numerical_budget = 8e-4
                + 7e-3 * std::abs(reference);
            if (std::abs(estimate - reference)
                > 6.0 * standard_error + numerical_budget) {
                std::cerr << "Kou independent gradient mismatch row=" << row
                          << " coordinate=" << sensitivity
                          << " estimate=" << estimate
                          << " reference=" << reference
                          << " se=" << standard_error << '\n';
                throw std::runtime_error(
                    "Kou independent represented-stencil comparison failed."
                );
            }
        }
    }
}

template<OptionSide Side>
void check(std::size_t paths) {
    const std::vector<kou::ModelParameters> models{
        {1.05f, 0.03f, 0.01f, 0.2f, 0.5f, 0.4f, 4.0f, 5.0f},
        {1.0f, -0.01f, 0.0f, 0.35f, 5.0f, 0.65f, 6.0f, 3.0f},
        {100.0f, 0.08f, 0.02f, 0.1f, 0.0f, 0.3f, 5.0f, 8.0f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 126U}, {1.1f, 126U}, {95.0f, 504U}
    };
    const pg::Sensitivity spot{"model.spot", {0.005f}};
    const pg::PriceGradientConfiguration full{{
        spot,
        {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
        {"model.volatility", {0.005f}},
        {"model.jump_intensity", {0.05f, pg::BumpScale::absolute}},
        {"model.up_probability", {0.002f, pg::BumpScale::absolute}},
        {"model.positive_jump_rate", {0.005f}},
        {"model.negative_jump_rate", {0.005f}},
        {"product.strike", {0.005f}},
        {"product.maturity_years", {
            1.0f / 504.0f, pg::BumpScale::absolute
        }},
    }};
    auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return kou::prepare_kou_european_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selection
        );
    };
    auto launcher = kou::launch_kou_european_option_price_gradients_cuda<
        Side
    >;
    const auto plan = prepare(full);
    DeviceArray<kou::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    for (unsigned threads : {128U, 256U}) {
        pg::LaunchConfiguration launch{
            pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
            threads, models.size(), 1777U
        };
        const auto solo = execute(prepare({{spot}}), launch, launcher);
        const auto price_only = execute(prepare({}), launch, launcher);
        DeviceArray<float> legacy(4U * models.size());
        kou::launch_kou_european_option_price_delta_cuda<Side>(
            models.data(), device_models.data, models.size(),
            products.data(), device_products.data, products.size(),
            PriceConstruction::Aligned, models.size(), 0U, models.size(),
            paths, 1.0f / 252.0f, threads, models.size(), 1777U,
            {0.01f}, legacy.data, legacy.data + models.size(),
            legacy.data + 2U * models.size(),
            legacy.data + 3U * models.size()
        );
        const auto reference = legacy.read();
        for (std::size_t row = 0U; row < models.size(); ++row) {
            same(price_only.price[row], solo.price[row],
                 "Kou empty selection changed price");
            same(price_only.price_error[row], solo.price_error[row],
                 "Kou empty selection changed price error");
            same(solo.price[row], reference[row], "Kou legacy price");
            same(solo.price_error[row], reference[models.size() + row],
                 "Kou legacy price error");
            same(solo.gradient[row], reference[2U * models.size() + row],
                 "Kou legacy delta");
            same(
                solo.gradient_error[row],
                reference[3U * models.size() + row],
                "Kou legacy delta error"
            );
        }
        const auto all = execute(plan, launch, launcher);
        compare_independent_gradients<Side>(plan, all);
        selected_prefixes(full, all, launch, prepare, launcher);
    }

    const auto diagonal_plan = kou::prepare_kou_european_option_sensitivities(
        models, products, PriceConstruction::Aligned, {}, full,
        {pg::SensitivityOrders::first_and_second}
    );
    const pg::LaunchConfiguration diagonal_launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        256U, models.size() * full.sensitivities.size(), 1777U, 1U
    };
    const auto diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan, diagonal_launch,
        kou::launch_kou_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto first = execute(plan, diagonal_launch, launcher);
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(diagonal.price[row], first.price[row],
             "Kou diagonal changed central price");
        same(diagonal.price_error[row], first.price_error[row],
             "Kou diagonal changed central price error");
        for (std::size_t sensitivity = 0U;
             sensitivity < full.sensitivities.size();
             ++sensitivity) {
            const auto index = row * full.sensitivities.size() + sensitivity;
            same_or_one_ulp(
                diagonal.gradient[index], first.gradient[index],
                "Kou derivative order changed gradient"
            );
            same_or_one_ulp(
                diagonal.gradient_error[index], first.gradient_error[index],
                "Kou derivative order changed gradient error"
            );
            require(
                std::isfinite(diagonal.diagonal_hessian[index])
                    && std::isfinite(
                        diagonal.diagonal_hessian_error[index]
                    ),
                "Kou diagonal Hessian is non-finite."
            );
            const auto& parameter = full.sensitivities[sensitivity].parameter;
            if (models[row].jump_intensity == 0.0f
                && parameter.starts_with("model.")
                && parameter != "model.jump_intensity"
                && (parameter == "model.up_probability"
                    || parameter == "model.positive_jump_rate"
                    || parameter == "model.negative_jump_rate")) {
                require(
                    std::abs(diagonal.gradient[index]) < 1e-5f
                        && std::abs(diagonal.diagonal_hessian[index]) < 1e-5f,
                    "Zero-intensity Kou mark sensitivity is non-zero."
                );
            }
        }
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
        event_coupling_contract();
        check<OptionSide::call>(paths);
        check<OptionSide::put>(paths);
        std::cout << "Kou ten-coordinate gradients and diagonal Hessians passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

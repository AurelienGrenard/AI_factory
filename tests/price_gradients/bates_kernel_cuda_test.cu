// Bates terminal aggregation, event coupling and diagonal sensitivities.
#include "model/equity/markovian/bates/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/bates/product/european_option_price_delta.cuh"
#include "model/equity/markovian/bates/price_gradients/coupled_dynamics_impl.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <cmath>

using namespace price_gradient_test;
namespace bates = model::equity::bates;
namespace philox = ai_factory::workbench::philox;

namespace {

using BatesCoupling = bates::price_gradients::CoupledDynamics;

__global__ void terminal_coupling_contract_kernel(
    bates::ModelParameters base,
    std::uint32_t central_steps,
    std::uint64_t seed,
    std::size_t path_count,
    int* violations
) {
    for (std::size_t path = std::size_t(blockIdx.x) * blockDim.x
             + threadIdx.x;
         path < path_count;
         path += std::size_t(gridDim.x) * blockDim.x) {
        bates::ModelParameters lower_intensity = base;
        bates::ModelParameters upper_intensity = base;
        lower_intensity.jump_intensity *= 0.75f;
        upper_intensity.jump_intensity *= 1.25f;
        constexpr float dt = 1.0f / 504.0f;
        const BatesCoupling::Prepared prepared[3]{
            BatesCoupling::prepare(base, dt),
            BatesCoupling::prepare(lower_intensity, dt),
            BatesCoupling::prepare(upper_intensity, dt),
        };
        BatesCoupling::State coupled_states[3]{
            BatesCoupling::initial(prepared[0U]),
            BatesCoupling::initial(prepared[1U]),
            BatesCoupling::initial(prepared[2U]),
        };
        philox::DomainRandomContext coupled_random(
            philox::make_key(seed), path
        );
        BatesCoupling::simulate_coupled_terminal(
            coupled_random,
            prepared,
            3U,
            central_steps + 1U,
            [] (unsigned int) { return true; },
            [=] (unsigned int node) {
                return node == 1U ? central_steps - 1U
                    : node == 2U ? central_steps + 1U
                    : central_steps;
            },
            [&] (unsigned int node)
                -> BatesCoupling::State& { return coupled_states[node]; }
        );

        auto canonical_state = bates::initial_state(prepared[0U]);
        philox::DomainRandomContext canonical_random(
            philox::make_key(seed), path
        );
        bates::DynamicsPolicy::advance(
            prepared[0U], central_steps, canonical_random, canonical_state
        );
        if (__float_as_uint(canonical_state.log_spot)
                != __float_as_uint(coupled_states[0U].log_spot)
            || __float_as_uint(canonical_state.variance)
                != __float_as_uint(coupled_states[0U].variance)) {
            atomicAdd(violations, 1);
        }

        bates::ModelParameters zero_variance = base;
        bates::ModelParameters first_variance = base;
        bates::ModelParameters second_variance = base;
        bates::ModelParameters third_variance = base;
        zero_variance.initial_variance = 0.0f;
        first_variance.initial_variance = 0.001f;
        second_variance.initial_variance = 0.002f;
        third_variance.initial_variance = 0.003f;
        const BatesCoupling::Prepared boundary_prefix[3]{
            BatesCoupling::prepare(zero_variance, dt),
            BatesCoupling::prepare(first_variance, dt),
            BatesCoupling::prepare(second_variance, dt),
        };
        const BatesCoupling::Prepared boundary_full[4]{
            boundary_prefix[0U], boundary_prefix[1U], boundary_prefix[2U],
            BatesCoupling::prepare(third_variance, dt),
        };
        BatesCoupling::State prefix_states[3]{
            BatesCoupling::initial(boundary_prefix[0U]),
            BatesCoupling::initial(boundary_prefix[1U]),
            BatesCoupling::initial(boundary_prefix[2U]),
        };
        BatesCoupling::State full_states[4]{
            BatesCoupling::initial(boundary_full[0U]),
            BatesCoupling::initial(boundary_full[1U]),
            BatesCoupling::initial(boundary_full[2U]),
            BatesCoupling::initial(boundary_full[3U]),
        };
        philox::DomainRandomContext prefix_random(
            philox::make_key(seed + 1U), path
        );
        philox::DomainRandomContext full_random(
            philox::make_key(seed + 1U), path
        );
        BatesCoupling::simulate_coupled_terminal(
            prefix_random, boundary_prefix, 3U, central_steps,
            [] (unsigned int) { return true; },
            [=] (unsigned int) { return central_steps; },
            [&] (unsigned int node) -> BatesCoupling::State& {
                return prefix_states[node];
            }
        );
        BatesCoupling::simulate_coupled_terminal(
            full_random, boundary_full, 4U, central_steps,
            [] (unsigned int) { return true; },
            [=] (unsigned int) { return central_steps; },
            [&] (unsigned int node) -> BatesCoupling::State& {
                return full_states[node];
            }
        );
        for (unsigned int node = 0U; node < 3U; ++node) {
            if (__float_as_uint(prefix_states[node].log_spot)
                    != __float_as_uint(full_states[node].log_spot)
                || __float_as_uint(prefix_states[node].variance)
                    != __float_as_uint(full_states[node].variance)) {
                atomicAdd(violations, 1);
            }
        }
    }
}

void terminal_coupling_contract() {
    constexpr std::size_t path_count = 1U << 13U;
    constexpr unsigned threads = 256U;
    DeviceArray<int> violations(1U);
    terminal_coupling_contract_kernel<<<
        static_cast<unsigned>((path_count + threads - 1U) / threads),
        threads
    >>>(
        {1.05f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f,
         2.0f, -0.1f, 0.25f},
        126U, 2063U, path_count, violations.data
    );
    check_cuda(cudaDeviceSynchronize(), "Bates terminal coupling contract");
    require(
        violations.read()[0U] == 0,
        "Bates coupled central differs from canonical terminal advance."
    );
}

template<OptionSide Side>
void check(std::size_t paths) {
    const std::vector<bates::ModelParameters> models{
        {1.05f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f,
         0.5f, -0.1f, 0.25f},
        {1.0f, -0.01f, 0.0f, 0.06f, 0.8f, 0.04f, 0.4f, -0.3f,
         5.0f, 0.02f, 0.4f},
        {100.0f, 0.08f, 0.02f, 0.0f, 1.0f, 0.04f, 0.3f, 1.0f,
         0.0f, -0.2f, 0.1f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 126U}, {1.1f, 126U}, {95.0f, 126U}
    };
    const pg::Sensitivity spot{"model.spot", {0.005f}};
    const pg::PriceGradientConfiguration full{{
        spot,
        {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
        {"model.initial_variance", {0.001f, pg::BumpScale::absolute}},
        {"model.kappa", {0.005f}},
        {"model.theta", {0.005f}},
        {"model.gamma", {0.005f}},
        {"model.rho", {0.002f, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05f, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002f, pg::BumpScale::absolute}},
        {"model.jump_log_volatility", {0.005f}},
        {"product.strike", {0.005f}},
        {"product.maturity_years", {
            1.0f / 504.0f, pg::BumpScale::absolute
        }},
    }};
    auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return bates::prepare_bates_european_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selection
        );
    };
    auto launcher = bates::launch_bates_european_option_price_gradients_cuda<
        Side
    >;
    const auto plan = prepare(full);
    DeviceArray<bates::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    for (unsigned threads : {128U, 256U}) {
        pg::LaunchConfiguration launch{
            pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
            threads, models.size(), 1879U
        };
        const auto solo = execute(prepare({{spot}}), launch, launcher);
        const auto price_only = execute(prepare({}), launch, launcher);
        DeviceArray<float> legacy(4U * models.size());
        bates::launch_bates_european_option_price_delta_cuda<Side>(
            models.data(), device_models.data, models.size(),
            products.data(), device_products.data, products.size(),
            PriceConstruction::Aligned, models.size(), 0U, models.size(),
            paths, 1.0f / 504.0f, 2U, threads, models.size(), 1879U,
            {0.01f}, legacy.data, legacy.data + models.size(),
            legacy.data + 2U * models.size(),
            legacy.data + 3U * models.size()
        );
        const auto reference = legacy.read();
        for (std::size_t row = 0U; row < models.size(); ++row) {
            same(price_only.price[row], solo.price[row],
                 "Bates empty selection changed price");
            same(price_only.price_error[row], solo.price_error[row],
                 "Bates empty selection changed price error");
            same(solo.price[row], reference[row], "Bates legacy price");
            same(solo.price_error[row], reference[models.size() + row],
                 "Bates legacy price error");
            same(solo.gradient[row], reference[2U * models.size() + row],
                 "Bates legacy delta");
            same(
                solo.gradient_error[row],
                reference[3U * models.size() + row],
                "Bates legacy delta error"
            );
        }
        const auto all = execute(plan, launch, launcher);
        selected_prefixes(full, all, launch, prepare, launcher);
        for (float value : all.gradient) {
            require(std::isfinite(value), "Bates gradient is non-finite.");
        }
    }

    const auto diagonal_plan =
        bates::prepare_bates_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, full,
            {pg::SensitivityOrders::first_and_second}
        );
    const pg::LaunchConfiguration diagonal_launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        256U, models.size() * full.sensitivities.size(), 1879U, 1U
    };
    const auto diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan, diagonal_launch,
        bates::launch_bates_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto first = execute(plan, diagonal_launch, launcher);
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(diagonal.price[row], first.price[row],
             "Bates diagonal changed central price");
        same(diagonal.price_error[row], first.price_error[row],
             "Bates diagonal changed central price error");
        for (std::size_t sensitivity = 0U;
             sensitivity < full.sensitivities.size();
             ++sensitivity) {
            const auto index = row * full.sensitivities.size() + sensitivity;
            same_within_ulps(
                diagonal.gradient[index], first.gradient[index],
                32U,
                "Bates derivative order changed gradient"
            );
            same_within_ulps(
                diagonal.gradient_error[index], first.gradient_error[index],
                32U,
                "Bates derivative order changed gradient error"
            );
            require(
                std::isfinite(diagonal.diagonal_hessian[index])
                    && std::isfinite(
                        diagonal.diagonal_hessian_error[index]
                    ),
                "Bates diagonal Hessian is non-finite."
            );
            const auto& parameter = full.sensitivities[sensitivity].parameter;
            if (models[row].jump_intensity == 0.0f
                && parameter != "model.jump_intensity"
                && (parameter == "model.jump_log_mean"
                    || parameter == "model.jump_log_volatility")) {
                require(
                    std::abs(diagonal.gradient[index]) < 1e-5f
                        && std::abs(diagonal.diagonal_hessian[index]) < 1e-5f,
                    "Zero-intensity Bates mark sensitivity is non-zero."
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
            paths = 257U;
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: test [--sanitizer]");
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        terminal_coupling_contract();
        check<OptionSide::call>(paths);
        check<OptionSide::put>(paths);
        std::cout << "Bates thirteen-coordinate gradients and diagonal "
                     "Hessians passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

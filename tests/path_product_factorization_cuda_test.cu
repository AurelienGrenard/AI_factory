// Enforce canonical factorization for every public equity product policy.
#include "common/check_cuda.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "product/asian_option/pricing_policy.cuh"
#include "product/asset_or_nothing_option/pricing_policy.cuh"
#include "product/athena_autocall/pricing_policy.cuh"
#include "product/cliquet/pricing_policy.cuh"
#include "product/digital_option/pricing_policy.cuh"
#include "product/double_knock_out_option/pricing_policy.cuh"
#include "product/down_and_in_option/pricing_policy.cuh"
#include "product/down_and_out_option/pricing_policy.cuh"
#include "product/european_option/pricing_policy.cuh"
#include "product/forward_start_option/pricing_policy.cuh"
#include "product/gap_option/pricing_policy.cuh"
#include "product/geometric_asian_option/pricing_policy.cuh"
#include "product/lookback_option/pricing_policy.cuh"
#include "product/phoenix_autocall/pricing_policy.cuh"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"
#include "product/range_accrual/pricing_policy.cuh"
#include "product/straddle/pricing_policy.cuh"
#include "product/up_and_in_option/pricing_policy.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"
#include "product/up_no_touch/pricing_policy.cuh"
#include "product/up_one_touch/pricing_policy.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace policy_contract_probe {

namespace equity = ai_factory::workbench::equity;
namespace heston = ai_factory::workbench::model::equity::heston;
namespace monte_carlo = ai_factory::workbench::monte_carlo;
namespace product = ai_factory::workbench::product;
namespace simulation = ai_factory::workbench::simulation;
using ai_factory::workbench::OptionSide;
using ai_factory::workbench::PriceConstruction;
using ai_factory::workbench::check_cuda;

using Dynamics = heston::DynamicsPolicy;
using TerminalSchedule = simulation::FixedStepTerminalSchedule<Dynamics>;
using DenseSchedule = simulation::FixedStepDenseSchedule<Dynamics>;
using RegularSchedule = simulation::FixedStepRegularSchedule<Dynamics>;
using TwoDateSchedule = simulation::FixedStepCalendarSchedule<Dynamics, 2U>;

struct SpotOnlyStatePolicy {
    struct Parameters {};
    struct State {
        float spot;
    };

    [[maybe_unused]] __device__ __forceinline__ static float spot(
        const State& state
    ) {
        return state.spot;
    }
};

struct IncompleteHandlerProductPolicy {
    using ProductParameters = product::EuropeanOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    [[maybe_unused]] static constexpr equity::ObservationCoordinate
    kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {};
    struct Handler {};

    [[maybe_unused]] __host__ __device__ static Calendar calendar(
        const ProductParameters& parameters
    ) {
        return Calendar{parameters.maturity_days};
    }

    [[maybe_unused]] __device__ static PreparedProduct prepare_product(
        const SpotOnlyStatePolicy::Parameters&,
        const ProductParameters&,
        equity::ProductPreparationContext
    ) {
        return {};
    }

    [[maybe_unused]] __device__ static Handler make_handler(
        const PreparedProduct&
    ) {
        return {};
    }

    template<typename StatePolicy>
    __device__ static float finalize(
        const PreparedProduct&,
        const typename StatePolicy::State&,
        const Handler&
    ) {
        return 0.0f;
    }
};

static_assert(equity::SpotStatePolicy<SpotOnlyStatePolicy>);
static_assert(!equity::LogSpotStatePolicy<SpotOnlyStatePolicy>);
static_assert(equity::EquityPathProductPolicy<
    product::EuropeanOptionPathPolicy<OptionSide::call>,
    SpotOnlyStatePolicy
>);
static_assert(!equity::EquityPathProductPolicy<
    product::GeometricAsianOptionPathPolicy<OptionSide::call>,
    SpotOnlyStatePolicy
>);
static_assert(!equity::EquityPathProductPolicy<
    IncompleteHandlerProductPolicy,
    SpotOnlyStatePolicy
>);

}  // namespace policy_contract_probe

namespace {

namespace equity = ai_factory::workbench::equity;
namespace heston = ai_factory::workbench::model::equity::heston;
namespace monte_carlo = ai_factory::workbench::monte_carlo;
namespace product = ai_factory::workbench::product;
namespace simulation = ai_factory::workbench::simulation;
using ai_factory::workbench::OptionSide;
using ai_factory::workbench::PriceConstruction;
using ai_factory::workbench::check_cuda;

using Dynamics = heston::DynamicsPolicy;
using TerminalSchedule = simulation::FixedStepTerminalSchedule<Dynamics>;
using DenseSchedule = simulation::FixedStepDenseSchedule<Dynamics>;
using RegularSchedule = simulation::FixedStepRegularSchedule<Dynamics>;
using TwoDateSchedule = simulation::FixedStepCalendarSchedule<Dynamics, 2U>;

constexpr simulation::FixedStepTimeConfiguration kTimeConfiguration{
    1.0f / 504.0f,
    2U,
};
constexpr std::size_t kPathsPerPrice = 4'096U;
constexpr unsigned int kThreadsPerBlock = 256U;
constexpr std::size_t kBlockCount = 1U;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename PublicPolicy, typename ProductPathPolicy>
void compare_policy(
    const heston::ModelParameters* device_model,
    const typename PublicPolicy::ProductParameters& product_parameters,
    std::uint64_t seed,
    const char* product_name,
    std::size_t& maximum_public_row_bytes,
    std::size_t& maximum_canonical_row_bytes
) {
    using CanonicalPolicy = equity::PathProductMonteCarloPricingPolicy<
        typename PublicPolicy::Schedule,
        ProductPathPolicy
    >;
    static_assert(std::same_as<PublicPolicy, CanonicalPolicy>);
    static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<PublicPolicy>);

    using ProductParameters = typename PublicPolicy::ProductParameters;
    ProductParameters* device_product = nullptr;
    float* device_prices = nullptr;
    float* device_standard_errors = nullptr;
    check_cuda(
        cudaMalloc(&device_product, sizeof(ProductParameters)),
        "allocate factorization product"
    );
    check_cuda(
        cudaMalloc(&device_prices, 2U * sizeof(float)),
        "allocate factorization prices"
    );
    check_cuda(
        cudaMalloc(&device_standard_errors, 2U * sizeof(float)),
        "allocate factorization standard errors"
    );
    check_cuda(
        cudaMemcpy(
            device_product,
            &product_parameters,
            sizeof(ProductParameters),
            cudaMemcpyHostToDevice
        ),
        "copy factorization product"
    );

    const auto launch = [&]<typename Policy>(
        std::size_t output_index,
        const char* variant
    ) {
        monte_carlo::launch_monte_carlo_cuda<Policy>(
            ai_factory::workbench::make_model_product_device_inputs(
                device_model,
                1U,
                device_product,
                1U,
                PriceConstruction::Aligned
            ),
            typename Policy::HostInputs{
                &product_parameters,
                1U,
                PriceConstruction::Aligned,
            },
            1U,
            0U,
            1U,
            kPathsPerPrice,
            kTimeConfiguration,
            kThreadsPerBlock,
            kBlockCount,
            seed,
            device_prices + output_index,
            device_standard_errors + output_index,
            "path_product_factorization",
            variant,
            "path-product factorization parity"
        );
    };
    launch.template operator()<PublicPolicy>(0U, "public_alias");
    launch.template operator()<CanonicalPolicy>(1U, "canonical");
    check_cuda(
        cudaDeviceSynchronize(),
        "synchronize factorization parity"
    );
    float prices[2]{};
    float standard_errors[2]{};
    check_cuda(
        cudaMemcpy(
            prices,
            device_prices,
            sizeof(prices),
            cudaMemcpyDeviceToHost
        ),
        "copy factorization prices"
    );
    check_cuda(
        cudaMemcpy(
            standard_errors,
            device_standard_errors,
            sizeof(standard_errors),
            cudaMemcpyDeviceToHost
        ),
        "copy factorization standard errors"
    );
    require(
        prices[0] == prices[1] && standard_errors[0] == standard_errors[1],
        std::string("Public/canonical replay failed for ") + product_name
    );
    maximum_public_row_bytes = std::max(
        maximum_public_row_bytes,
        sizeof(typename PublicPolicy::PreparedRow)
    );
    maximum_canonical_row_bytes = std::max(
        maximum_canonical_row_bytes,
        sizeof(typename CanonicalPolicy::PreparedRow)
    );

    check_cuda(
        cudaFree(device_standard_errors),
        "free factorization standard errors"
    );
    check_cuda(cudaFree(device_prices), "free factorization prices");
    check_cuda(cudaFree(device_product), "free factorization product");
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t availability = cudaGetDeviceCount(&device_count);
        if (availability == cudaErrorNoDevice || device_count == 0) return 77;
        check_cuda(availability, "query CUDA devices for product parity");

        const heston::ModelParameters model{
            1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f,
        };
        heston::ModelParameters* device_model = nullptr;
        check_cuda(
            cudaMalloc(&device_model, sizeof(model)),
            "allocate factorization model"
        );
        check_cuda(
            cudaMemcpy(
                device_model,
                &model,
                sizeof(model),
                cudaMemcpyHostToDevice
            ),
            "copy factorization model"
        );

        std::size_t maximum_public_row_bytes = 0U;
        std::size_t maximum_canonical_row_bytes = 0U;
        std::uint64_t seed = 9'410'000'000ULL;
#define COMPARE_PRODUCT_4(Direct, Path, Parameters, Name) \
        compare_policy<Direct, Path>( \
            device_model, Parameters, seed++, Name, \
            maximum_public_row_bytes, maximum_canonical_row_bytes \
        )
#define COMPARE_PRODUCT_5(DirectHead, DirectTail, Path, Parameters, Name) \
        compare_policy<DirectHead, DirectTail, Path>( \
            device_model, Parameters, seed++, Name, \
            maximum_public_row_bytes, maximum_canonical_row_bytes \
        )
#define SELECT_COMPARE_PRODUCT(_1, _2, _3, _4, _5, Selected, ...) Selected
#define COMPARE_PRODUCT(...) \
        SELECT_COMPARE_PRODUCT( \
            __VA_ARGS__, COMPARE_PRODUCT_5, COMPARE_PRODUCT_4 \
        )(__VA_ARGS__)

        COMPARE_PRODUCT(
            product::AsianOptionPricingPolicy<
                DenseSchedule, OptionSide::call
            >,
            product::AsianOptionPathPolicy<OptionSide::call>,
            (product::AsianOptionParameters{1.0f, 63U}),
            "asian_option"
        );
        COMPARE_PRODUCT(
            product::AssetOrNothingOptionPricingPolicy<
                TerminalSchedule, OptionSide::call
            >,
            product::AssetOrNothingOptionPathPolicy<OptionSide::call>,
            (product::AssetOrNothingOptionParameters{1.0f, 63U}),
            "asset_or_nothing_option"
        );
        COMPARE_PRODUCT(
            product::AthenaAutocallPricingPolicy<RegularSchedule>,
            product::AthenaAutocallPathPolicy,
            (product::AthenaAutocallParameters{
                63U, 21U, 1.10f, 0.75f, 0.12f,
            }),
            "athena_autocall"
        );
        COMPARE_PRODUCT(
            product::CliquetPricingPolicy<RegularSchedule>,
            product::CliquetPathPolicy,
            (product::CliquetParameters{
                63U, 21U, 1.0f, -0.10f, 0.10f, -0.20f, 0.20f,
            }),
            "cliquet"
        );
        COMPARE_PRODUCT(
            product::DigitalOptionPricingPolicy<
                TerminalSchedule, OptionSide::call
            >,
            product::DigitalOptionPathPolicy<OptionSide::call>,
            (product::DigitalOptionParameters{1.0f, 63U, 1.0f}),
            "digital_option"
        );
        COMPARE_PRODUCT(
            product::DoubleKnockOutOptionPricingPolicy<
                DenseSchedule, OptionSide::call
            >,
            product::DoubleKnockOutOptionPathPolicy<OptionSide::call>,
            (product::DoubleKnockOutOptionParameters{
                1.0f, 0.70f, 1.30f, 63U,
            }),
            "double_knock_out_option"
        );
        COMPARE_PRODUCT(
            product::DownAndInOptionPricingPolicy<
                DenseSchedule, OptionSide::call
            >,
            product::DownAndInOptionPathPolicy<OptionSide::call>,
            (product::DownAndInOptionParameters{1.0f, 0.75f, 63U}),
            "down_and_in_option"
        );
        COMPARE_PRODUCT(
            product::DownAndOutOptionPricingPolicy<
                DenseSchedule, OptionSide::call
            >,
            product::DownAndOutOptionPathPolicy<OptionSide::call>,
            (product::DownAndOutOptionParameters{1.0f, 0.75f, 63U}),
            "down_and_out_option"
        );
        COMPARE_PRODUCT(
            product::EuropeanOptionPricingPolicy<
                TerminalSchedule, OptionSide::call
            >,
            product::EuropeanOptionPathPolicy<OptionSide::call>,
            (product::EuropeanOptionParameters{1.0f, 63U}),
            "european_option"
        );
        COMPARE_PRODUCT(
            product::ForwardStartOptionPricingPolicy<
                TwoDateSchedule, OptionSide::call
            >,
            product::ForwardStartOptionPathPolicy<OptionSide::call>,
            (product::ForwardStartOptionParameters{1.0f, 21U, 63U}),
            "forward_start_option"
        );
        COMPARE_PRODUCT(
            product::GapOptionPricingPolicy<
                TerminalSchedule, OptionSide::call
            >,
            product::GapOptionPathPolicy<OptionSide::call>,
            (product::GapOptionParameters{1.0f, 0.95f, 63U}),
            "gap_option"
        );
        COMPARE_PRODUCT(
            product::GeometricAsianOptionPricingPolicy<
                DenseSchedule, OptionSide::call
            >,
            product::GeometricAsianOptionPathPolicy<OptionSide::call>,
            (product::GeometricAsianOptionParameters{1.0f, 63U}),
            "geometric_asian_option"
        );
        COMPARE_PRODUCT(
            product::LookbackOptionPricingPolicy<DenseSchedule>,
            product::LookbackOptionPathPolicy,
            (product::LookbackOptionParameters{1.0f, 63U}),
            "lookback_option"
        );
        COMPARE_PRODUCT(
            product::PhoenixAutocallPricingPolicy<RegularSchedule>,
            product::PhoenixAutocallPathPolicy,
            (product::PhoenixAutocallParameters{
                63U, 21U, 1.10f, 0.90f, 0.70f, 0.12f,
            }),
            "phoenix_autocall"
        );
        COMPARE_PRODUCT(
            product::PhoenixMemoryAutocallPricingPolicy<RegularSchedule>,
            product::PhoenixMemoryAutocallPathPolicy,
            (product::PhoenixMemoryAutocallParameters{
                63U, 21U, 1.10f, 0.90f, 0.70f, 0.12f,
            }),
            "phoenix_memory_autocall"
        );
        COMPARE_PRODUCT(
            product::RangeAccrualPricingPolicy<RegularSchedule>,
            product::RangeAccrualPathPolicy,
            (product::RangeAccrualParameters{
                63U, 21U, 0.75f, 1.25f, 0.10f,
            }),
            "range_accrual"
        );
        COMPARE_PRODUCT(
            product::StraddlePricingPolicy<TerminalSchedule>,
            product::StraddlePathPolicy,
            (product::StraddleParameters{1.0f, 63U}),
            "straddle"
        );
        COMPARE_PRODUCT(
            product::UpAndInOptionPricingPolicy<
                DenseSchedule, OptionSide::call
            >,
            product::UpAndInOptionPathPolicy<OptionSide::call>,
            (product::UpAndInOptionParameters{1.0f, 1.25f, 63U}),
            "up_and_in_option"
        );
        COMPARE_PRODUCT(
            product::UpAndOutOptionPricingPolicy<
                DenseSchedule, OptionSide::call
            >,
            product::UpAndOutOptionPathPolicy<OptionSide::call>,
            (product::UpAndOutOptionParameters{1.0f, 1.25f, 63U}),
            "up_and_out_option"
        );
        COMPARE_PRODUCT(
            product::UpNoTouchPricingPolicy<DenseSchedule>,
            product::UpNoTouchPathPolicy,
            (product::UpNoTouchParameters{1.25f, 1.0f, 63U}),
            "up_no_touch"
        );
        COMPARE_PRODUCT(
            product::UpOneTouchPricingPolicy<DenseSchedule>,
            product::UpOneTouchPathPolicy,
            (product::UpOneTouchParameters{1.25f, 1.0f, 63U}),
            "up_one_touch"
        );
#undef COMPARE_PRODUCT
#undef SELECT_COMPARE_PRODUCT
#undef COMPARE_PRODUCT_5
#undef COMPARE_PRODUCT_4

        std::cout << "PATH_PRODUCT_FACTORIZATION products=21"
                  << " paths_per_product=" << kPathsPerPrice
                  << " maximum_public_prepared_row_bytes="
                  << maximum_public_row_bytes
                  << " maximum_canonical_prepared_row_bytes="
                  << maximum_canonical_row_bytes << '\n';
        check_cuda(cudaFree(device_model), "free factorization model");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

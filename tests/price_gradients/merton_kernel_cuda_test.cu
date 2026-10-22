// Fixed-horizon Merton CRN coupling and spot-delta compatibility within the current RNG mapping.
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_delta.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"
#include "model/equity/markovian/merton/price_gradients/coupled_dynamics_impl.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"
#include <algorithm>
#include <cmath>

using namespace price_gradient_test;
namespace merton = model::equity::merton;
namespace philox = ai_factory::workbench::philox;

namespace {

using MertonCoupling = merton::price_gradients::CoupledDynamics;

__device__ bool same_bits(float first, float second) {
    return __float_as_uint(first) == __float_as_uint(second);
}

template<std::size_t NodeCapacity>
__device__ void record_event_coupling_violations(
    const MertonCoupling::Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    std::uint64_t seed,
    std::uint64_t path,
    int* violations
) {
    philox::DomainRandomContext standalone_random(
        philox::make_key(seed), path
    );
    const auto standalone = MertonCoupling::draw(
        standalone_random, prepared[0U]
    );
    philox::DomainRandomContext coupled_random(philox::make_key(seed), path);
    MertonCoupling::Innovations coupled[NodeCapacity]{};
    MertonCoupling::draw_coupled(
        coupled_random, prepared, node_count, coupled
    );

    bool invalid = standalone.jump_count != coupled[0U].jump_count
        || !same_bits(
            standalone.diffusion_normal,
            coupled[0U].diffusion_normal
        )
        || !same_bits(
            standalone.jump_standard_normal_sum,
            coupled[0U].jump_standard_normal_sum
        );
    for (std::uint8_t node = 1U; node < node_count; ++node) {
        invalid = invalid
            || coupled[node - 1U].jump_count > coupled[node].jump_count;
    }
    if (invalid) atomicAdd(violations, 1);
}

__global__ void event_coupling_contract_kernel(
    merton::ModelParameters base,
    std::uint64_t seed,
    std::size_t path_count,
    int* violations
) {
    for (std::size_t path = std::size_t(blockIdx.x) * blockDim.x
             + threadIdx.x;
         path < path_count;
         path += std::size_t(gridDim.x) * blockDim.x) {
        merton::ModelParameters lower_intensity = base;
        merton::ModelParameters upper_intensity = base;
        lower_intensity.jump_intensity *= 0.75f;
        upper_intensity.jump_intensity *= 1.25f;
        const MertonCoupling::Prepared intensity_nodes[3]{
            MertonCoupling::prepare(base, 0.5f),
            MertonCoupling::prepare(lower_intensity, 0.5f),
            MertonCoupling::prepare(upper_intensity, 0.5f),
        };
        philox::DomainRandomContext intensity_random(
            philox::make_key(seed), path
        );
        MertonCoupling::Innovations intensity_innovations[3]{};
        MertonCoupling::draw_coupled(
            intensity_random, intensity_nodes, 3U, intensity_innovations
        );
        bool invalid = intensity_innovations[1U].jump_count
                > intensity_innovations[0U].jump_count
            || intensity_innovations[0U].jump_count
                > intensity_innovations[2U].jump_count;
        philox::DomainRandomContext standalone_intensity_random(
            philox::make_key(seed), path
        );
        const auto standalone_intensity = MertonCoupling::draw(
            standalone_intensity_random, intensity_nodes[0U]
        );
        invalid = invalid
            || standalone_intensity.jump_count
                != intensity_innovations[0U].jump_count
            || !same_bits(
                standalone_intensity.diffusion_normal,
                intensity_innovations[0U].diffusion_normal
            )
            || !same_bits(
                standalone_intensity.jump_standard_normal_sum,
                intensity_innovations[0U].jump_standard_normal_sum
            );
        if (invalid) atomicAdd(violations, 1);

        const MertonCoupling::Prepared maturity_nodes[3]{
            MertonCoupling::prepare(base, 0.5f),
            MertonCoupling::prepare(base, 0.25f),
            MertonCoupling::prepare(base, 0.75f),
        };
        philox::DomainRandomContext maturity_random(
            philox::make_key(seed + 1U), path
        );
        MertonCoupling::Innovations maturity_innovations[3]{};
        MertonCoupling::draw_coupled(
            maturity_random, maturity_nodes, 3U, maturity_innovations
        );
        if (maturity_innovations[1U].jump_count
                > maturity_innovations[0U].jump_count
            || maturity_innovations[0U].jump_count
                > maturity_innovations[2U].jump_count) {
            atomicAdd(violations, 1);
        }

        merton::ModelParameters zero_intensity = base;
        zero_intensity.jump_intensity = 0.0f;
        merton::ModelParameters first_positive = base;
        merton::ModelParameters second_positive = base;
        merton::ModelParameters third_positive = base;
        first_positive.jump_intensity = 0.1f;
        second_positive.jump_intensity = 0.2f;
        third_positive.jump_intensity = 0.3f;
        const MertonCoupling::Prepared boundary_nodes[4]{
            MertonCoupling::prepare(zero_intensity, 0.5f),
            MertonCoupling::prepare(first_positive, 0.5f),
            MertonCoupling::prepare(second_positive, 0.5f),
            MertonCoupling::prepare(third_positive, 0.5f),
        };
        const MertonCoupling::Prepared boundary_prefix[3]{
            boundary_nodes[0U], boundary_nodes[1U], boundary_nodes[2U]
        };
        philox::DomainRandomContext boundary_prefix_random(
            philox::make_key(seed + 2U), path
        );
        philox::DomainRandomContext boundary_full_random(
            philox::make_key(seed + 2U), path
        );
        MertonCoupling::Innovations boundary_prefix_innovations[3]{};
        MertonCoupling::Innovations boundary_full_innovations[4]{};
        MertonCoupling::draw_coupled(
            boundary_prefix_random, boundary_prefix, 3U,
            boundary_prefix_innovations
        );
        MertonCoupling::draw_coupled(
            boundary_full_random, boundary_nodes, 4U,
            boundary_full_innovations
        );
        for (std::uint8_t node = 0U; node < 3U; ++node) {
            if (boundary_prefix_innovations[node].jump_count
                    != boundary_full_innovations[node].jump_count
                || !same_bits(
                    boundary_prefix_innovations[node].diffusion_normal,
                    boundary_full_innovations[node].diffusion_normal
                )
                || !same_bits(
                    boundary_prefix_innovations[node]
                        .jump_standard_normal_sum,
                    boundary_full_innovations[node]
                        .jump_standard_normal_sum
                )) {
                atomicAdd(violations, 1);
            }
        }
        record_event_coupling_violations(
            boundary_nodes, 4U, seed + 2U, path, violations
        );
    }
}

void event_coupling_contract() {
    constexpr std::size_t path_count = 1U << 14U;
    constexpr unsigned threads = 256U;
    const unsigned blocks = static_cast<unsigned>(
        (path_count + threads - 1U) / threads
    );
    DeviceArray<int> violations(1U);
    event_coupling_contract_kernel<<<blocks, threads>>>(
        {1.05f, 0.03f, 0.01f, 0.2f, 2.0f, -0.1f, 0.25f},
        1987U, path_count, violations.data
    );
    check_cuda(cudaDeviceSynchronize(), "Merton event coupling contract");
    if (violations.read()[0U] != 0) {
        throw std::runtime_error(
            "Merton event coupling violated nesting or central replay."
        );
    }
}

}  // namespace

__global__ void legacy_factorization_kernel(merton::ModelParameters parameters,float maturity,
                                             std::uint64_t seed,float* output,std::size_t paths) {
    for (std::size_t path=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;path<paths;
         path+=std::size_t(gridDim.x)*blockDim.x) {
        const auto prepared_model=merton::prepare_model(parameters);
        const auto prepared_transition=merton::prepare_transition(prepared_model,maturity);
        auto state=merton::initial_state(prepared_model);
        philox::NormalRandomContext random(philox::make_key(seed),path);
        constexpr float threshold=10.f;
        const std::uint32_t count=prepared_transition.poisson_mean<threshold
            ? philox::poisson_from_uniform(random.uniforms.next(),prepared_transition.poisson_mean,
                                           prepared_transition.zero_jump_probability)
            : philox::poisson_from_uniform_sequence(random.uniforms,prepared_transition.poisson_mean);
        const float diffusion=philox::next_normal(random.uniforms,random.normals);
        const float jump=count==0U ? 0.f : philox::next_normal(random.uniforms,random.normals);
        merton::one_step_transition(
            prepared_model, prepared_transition, count, diffusion,
            sqrtf(static_cast<float>(count)) * jump, state
        );
        output[path]=state.log_spot;
    }
}

__global__ void canonical_factorization_kernel(merton::ModelParameters parameters,float maturity,
                                                std::uint64_t seed,float* output,std::size_t paths) {
    for (std::size_t path=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;path<paths;
         path+=std::size_t(gridDim.x)*blockDim.x) {
        const auto prepared_model=merton::prepare_model(parameters);
        const auto prepared_transition=merton::prepare_transition(prepared_model,maturity);
        auto state=merton::initial_state(prepared_model);
        philox::DomainRandomContext random(philox::make_key(seed),path);
        const auto innovations=merton::draw_transition_innovations(prepared_transition,random);
        merton::one_step_transition(
            prepared_model, prepared_transition, innovations.jump_count,
            innovations.diffusion_normal,
            innovations.jump_standard_normal_sum, state
        );
        output[path]=state.log_spot;
    }
}

void factorization_benchmark() {
    constexpr std::size_t paths=1U<<20U;
    constexpr unsigned threads=256U, iterations=20U, rounds=10U;
    const unsigned blocks=unsigned((paths+threads-1U)/threads);
    const merton::ModelParameters parameters{1.05f,.03f,.01f,.2f,25.f,-.1f,.25f};
    DeviceArray<float> legacy(paths),canonical(paths);
    legacy_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U,legacy.data,paths);
    canonical_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U,canonical.data,paths);
    check_cuda(cudaDeviceSynchronize(),"Merton factorization warmup");
    const auto before=legacy.read(),after=canonical.read();
    double old_sum=0.,new_sum=0.,old_square=0.,new_square=0.;
    for (std::size_t i=0;i<paths;++i) {
        old_sum+=before[i];new_sum+=after[i];
        old_square+=double(before[i])*before[i];
        new_square+=double(after[i])*after[i];
    }
    const double old_mean=old_sum/paths,new_mean=new_sum/paths;
    const double old_variance=old_square/paths-old_mean*old_mean;
    const double new_variance=new_square/paths-new_mean*new_mean;
    const double mean_budget=6.*std::sqrt((old_variance+new_variance)/paths);
    const double variance_budget=9.*std::sqrt(2./paths)*std::max(old_variance,new_variance);
    if (!(std::abs(old_mean-new_mean)<mean_budget
          && std::abs(old_variance-new_variance)<variance_budget)) {
        throw std::runtime_error("Merton domain mapping changed the terminal law");
    }
    cudaEvent_t start{},stop{};check_cuda(cudaEventCreate(&start),"factorization event");
    check_cuda(cudaEventCreate(&stop),"factorization event");
    auto elapsed=[&](bool use_legacy) {
        check_cuda(cudaEventRecord(start),"factorization start");
        for (unsigned i=0;i<iterations;++i) {
            if (use_legacy) legacy_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U+i,legacy.data,paths);
            else canonical_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U+i,canonical.data,paths);
        }
        check_cuda(cudaEventRecord(stop),"factorization stop");check_cuda(cudaEventSynchronize(stop),"factorization sync");
        float milliseconds=0;check_cuda(cudaEventElapsedTime(&milliseconds,start,stop),"factorization time");
        return milliseconds/iterations;
    };
    float legacy_ms=0.f,canonical_ms=0.f;
    for (unsigned round=0;round<rounds;++round) {
        if ((round&1U)==0U) { legacy_ms+=elapsed(true);canonical_ms+=elapsed(false); }
        else { canonical_ms+=elapsed(false);legacy_ms+=elapsed(true); }
    }
    legacy_ms/=rounds;canonical_ms/=rounds;
    cudaFuncAttributes legacy_attributes{},canonical_attributes{};
    check_cuda(cudaFuncGetAttributes(&legacy_attributes,legacy_factorization_kernel),"legacy attributes");
    check_cuda(cudaFuncGetAttributes(&canonical_attributes,canonical_factorization_kernel),"canonical attributes");
    cudaEventDestroy(start);cudaEventDestroy(stop);
    std::cout << "{\"paths\":" << paths << ",\"iterations_per_round\":" << iterations
              << ",\"rounds\":" << rounds
              << ",\"legacy_ms\":" << legacy_ms << ",\"canonical_ms\":" << canonical_ms
              << ",\"legacy_registers\":" << legacy_attributes.numRegs
              << ",\"canonical_registers\":" << canonical_attributes.numRegs
              << ",\"legacy_local_bytes\":" << legacy_attributes.localSizeBytes
              << ",\"canonical_local_bytes\":" << canonical_attributes.localSizeBytes
              << ",\"old_mean\":" << old_mean << ",\"new_mean\":" << new_mean
              << ",\"old_variance\":" << old_variance
              << ",\"new_variance\":" << new_variance << "}\n";
}

double normal_cdf(double value) { return .5 * std::erfc(-value/std::sqrt(2.)); }

template<OptionSide Side,typename Scenario>
double independent_price(const Scenario& scenario) {
    const auto& model=scenario.model;
    const double t=scenario.maturity_years, strike=scenario.product.strike;
    const double jump_variance=double(model.jump_log_volatility)*model.jump_log_volatility;
    const double compensator=std::exp(double(model.jump_log_mean)+.5*jump_variance)-1.;
    const double poisson_mean=double(model.jump_intensity)*t;
    double weight=std::exp(-poisson_mean), total=0., probability=0.;
    for (unsigned count=0;count<1024;++count) {
        const double variance=double(model.volatility)*model.volatility*t+count*jump_variance;
        const double mean=std::log(double(model.spot))
            +(double(model.risk_free_rate)-model.dividend_yield-model.jump_intensity*compensator
              -.5*double(model.volatility)*model.volatility)*t+count*model.jump_log_mean;
        const double root=std::sqrt(variance);
        double call;
        if (root==0.) call=std::max(std::exp(mean)-strike,0.);
        else {
            const double d2=(mean-std::log(strike))/root, d1=d2+root;
            call=std::exp(mean+.5*variance)*normal_cdf(d1)-strike*normal_cdf(d2);
        }
        const double discount=std::exp(-double(model.risk_free_rate)*t);
        const double conditional = Side==OptionSide::call ? discount*call
            : discount*(call-std::exp(mean+.5*variance)+strike);
        total+=weight*conditional; probability+=weight;
        if (count>poisson_mean && 1.-probability<1e-14) break;
        weight*=poisson_mean/double(count+1);
    }
    return total;
}

template<OptionSide Side,typename Plan>
std::pair<double,double> independent_gradients(const Plan& plan,const Results& result) {
    const auto k=plan.sensitivity_count();
    double maximum_absolute=0.,maximum_budget_fraction=0.;
    for (std::size_t row=0;row<plan.result_count;++row) {
        const auto central_scenario_value = central_scenario(plan, row);
        const double central=independent_price<Side>(central_scenario_value);
        for (std::size_t i=0;i<k;++i) {
            const auto task = sensitivity_task<pg::SensitivityOrders::first>(
                plan, row, i
            );
            const double first=independent_price<Side>(task.nodes[1U]);
            const double second=independent_price<Side>(task.nodes[2U]);
            const auto& stencil=task.stencil;
            const double reference=stencil.kind==pg::StencilKind::centered
                ? (second-first)/stencil.represented_width
                : stencil.first_endpoint_weights[0U]*(first-central)
                    + stencil.first_endpoint_weights[1U]*(second-central);
            const double estimate=result.gradient[row*k+i], se=result.gradient_error[row*k+i];
            const double numerical=5e-4+5e-3*std::abs(reference);
            const double error=std::abs(estimate-reference),budget=6.*se+numerical;
            maximum_absolute=std::max(maximum_absolute,error);
            maximum_budget_fraction=std::max(maximum_budget_fraction,error/budget);
            if (error>budget) {
                std::cerr << "Merton independent gradient mismatch row=" << row << " coordinate=" << i
                          << " estimate=" << estimate << " reference=" << reference << " se=" << se << '\n';
                throw std::runtime_error("Merton independent represented-stencil comparison failed");
            }
            if (plan.configuration.sensitivities[i].parameter.starts_with(
                    "model.jump_"
                )
                && plan.configuration.sensitivities[i].parameter
                    != "model.jump_intensity"
                && central_scenario_value.model.jump_intensity==0.f) {
                if (!(std::abs(estimate)<1e-5 && std::abs(reference)<1e-9)) {
                    std::cerr << "Zero-intensity jump gradient estimate=" << estimate
                              << " reference=" << reference << " coordinate=" << i << '\n';
                    throw std::runtime_error("Zero-intensity Merton jump gradient is not zero");
                }
            }
        }
    }
    return {maximum_absolute,maximum_budget_fraction};
}

template<OptionSide Side> void check(std::size_t paths) {
    const std::vector<merton::ModelParameters> models{
        {1.05f,.03f,.01f,.2f,.5f,-.1f,.25f},
        {1.f,-.01f,0.f,.35f,25.f,.02f,.4f},
        {100.f,.08f,.02f,.1f,0.f,-.2f,.1f}};
    const std::vector<product::EuropeanOptionParameters> products{{1.f,126U},{1.1f,126U},{95.f,504U}};
    const pg::Sensitivity spot{"model.spot",{.005f}};
    const pg::PriceGradientConfiguration full{{spot,
        {"model.risk_free_rate",{.0005,pg::BumpScale::absolute}},
        {"model.dividend_yield",{.0005,pg::BumpScale::absolute}}, {"model.volatility",{.005}},
        {"model.jump_intensity",{.05f,pg::BumpScale::absolute}},
        {"model.jump_log_mean",{.002,pg::BumpScale::absolute}},
        {"model.jump_log_volatility",{.005}}, {"product.strike",{.005}},
        {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
    auto prepare=[&](const pg::PriceGradientConfiguration& selection) {
        return merton::prepare_merton_european_option_price_gradients(
            models,products,PriceConstruction::Aligned,{},selection);
    };
    auto launcher=merton::launch_merton_european_option_price_gradients_cuda<Side>;
    const auto plan=prepare(full);
    double maximum_independent_error=0.,maximum_independent_budget_fraction=0.;
    DeviceArray<merton::ModelParameters> dm(models);
    DeviceArray<product::EuropeanOptionParameters> dp(products);
    for (unsigned threads : {128U,256U}) {
        pg::LaunchConfiguration launch{pg::PricingMethod::monte_carlo,0U,3U,paths,threads,3U,719U};
        const auto solo=execute(prepare({{spot}}),launch,launcher);
        const auto price_only=execute(prepare({}),launch,launcher);
        DeviceArray<float> old(12);
        merton::launch_merton_european_option_price_delta_cuda<Side>(models.data(),dm.data,3U,
            products.data(),dp.data,3U,PriceConstruction::Aligned,3U,0U,3U,paths,
            1.f/252.f,threads,3U,719U,{.01f},old.data,old.data+3,old.data+6,old.data+9);
        const auto reference=old.read();
        for (unsigned row=0;row<3;++row) {
            same(price_only.price[row],solo.price[row],
                "Merton price-only price");
            same(price_only.price_error[row],solo.price_error[row],
                "Merton price-only price SE");
            same(solo.price[row],reference[row],"Merton legacy price");
            same(solo.price_error[row],reference[3+row],"Merton legacy price SE");
            same(solo.gradient[row],reference[6+row],"Merton legacy delta");
            same(solo.gradient_error[row],reference[9+row],"Merton legacy delta SE");
        }
        const auto all=execute(plan,launch,launcher);
        const auto independent=independent_gradients<Side>(plan,all);
        maximum_independent_error=std::max(maximum_independent_error,independent.first);
        maximum_independent_budget_fraction=std::max(maximum_independent_budget_fraction,independent.second);
        selected_prefixes(full,all,launch,prepare,launcher);
        for (std::size_t i=0;i<full.sensitivities.size();++i) {
            const auto one=execute(prepare({{full.sensitivities[i]}}),launch,launcher);
            for (std::size_t row=0;row<3;++row) {
                same(all.gradient[row*full.sensitivities.size()+i],one.gradient[row],
                    "Merton selected coordinate");
                same(all.gradient_error[row*full.sensitivities.size()+i],one.gradient_error[row],
                    "Merton selected coordinate SE");
            }
        }
    }
    const auto diagonal_plan =
        merton::prepare_merton_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, full,
            {pg::SensitivityOrders::first_and_second}
        );
    const pg::LaunchConfiguration diagonal_launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        256U, models.size()*full.sensitivities.size(), 719U, 1U
    };
    const auto diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan, diagonal_launch,
        merton::launch_merton_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto first = execute(prepare(full), diagonal_launch, launcher);
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(diagonal.price[row], first.price[row],
             "Merton diagonal changed central price");
        same(diagonal.price_error[row], first.price_error[row],
             "Merton diagonal changed central price error");
        for (std::size_t i = 0U; i < full.sensitivities.size(); ++i) {
            const auto index = row*full.sensitivities.size()+i;
            same_or_one_ulp(
                diagonal.gradient[index], first.gradient[index],
                "Merton derivative order changed gradient"
            );
            same_or_one_ulp(
                diagonal.gradient_error[index], first.gradient_error[index],
                "Merton derivative order changed gradient error"
            );
            require(std::isfinite(diagonal.diagonal_hessian[index])
                    && std::isfinite(
                        diagonal.diagonal_hessian_error[index]
                    ),
                    "Merton diagonal Hessian is non-finite.");
            if (models[row].jump_intensity == 0.0f
                && full.sensitivities[i].parameter.starts_with(
                    "model.jump_"
                )
                && full.sensitivities[i].parameter
                    != "model.jump_intensity") {
                if (std::abs(diagonal.diagonal_hessian[index]) >= 1e-5f) {
                    std::cerr << "Zero-intensity Merton jump Hessian row="
                              << row << " coordinate=" << i << " estimate="
                              << diagonal.diagonal_hessian[index] << '\n';
                    throw std::runtime_error(
                        "Zero-intensity Merton jump Hessian is not zero."
                    );
                }
            }
        }
    }
    std::cout << "Merton " << option_side_name(Side)
              << ": nine gradients and diagonal Hessians passed; max_abs_error="
              << maximum_independent_error << ", max_budget_fraction="
              << maximum_independent_budget_fraction
              << "; variable-consumption parity and batches passed\n";
}

int main(int argc,char** argv) {
    try {
        std::size_t paths=65537;
        if (argc==2 && std::string_view(argv[1])=="--sanitizer") paths=513;
        else if (argc==2 && std::string_view(argv[1])=="--factorization-benchmark") {
            factorization_benchmark();return 0;
        }
        else if (argc!=1) throw std::invalid_argument("Usage: test [--sanitizer]");
        int devices=0;
        if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        event_coupling_contract();
        check<OptionSide::call>(paths);
        check<OptionSide::put>(paths);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

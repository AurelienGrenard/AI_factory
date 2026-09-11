// Qualify shared terminal MC for G2 and both fitted curves on catalogue core/stress rows.
// Checks batching/geometry replay and explicit calendars; emits prices for independent diagnostics.
#include "model/fixed_income/g2/product/european_swaption.cuh"
#include "model/fixed_income/g2_plus_plus/product/nelson_siegel/european_swaption.cuh"
#include "model/fixed_income/g2_plus_plus/product/svensson/european_swaption.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/nelson_siegel/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace wb = ai_factory::workbench;
namespace product = wb::product;
namespace rates = wb::model::fixed_income;
namespace cuda = wb::offline::cuda;
using Json = nlohmann::ordered_json;
constexpr std::array<std::size_t, 8> indices{0, 100, 499, 899, 900, 950, 975, 999};
std::size_t paths = 1U << 18U;
constexpr std::uint64_t seed = 9382026000ULL;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct G2 {
    using Model = rates::g2::ModelParameters;
    using Curve = std::monostate;
    static constexpr const char* model_name = "g2";
    static constexpr const char* curve_name = "";
    static auto models() { return rates::g2::load_models("datasets/model/fixed_income/g2/parameters/g2_01.json"); }
    static auto curves() { return std::vector<Curve>(1000); }
    template<wb::SwaptionSide Side, typename Product>
    static void launch(const Model* models, const Curve* curves, const Product* host_products,
                       const Product* products, const std::uint32_t* host_dates, const float* host_accruals,
                       const std::uint32_t* dates, const float* accruals, std::size_t pool_size,
                       std::size_t count, std::size_t offset, std::size_t batch,
                       unsigned threads, std::size_t blocks, float* prices, float* errors,
                       wb::PriceConstruction construction = wb::PriceConstruction::Aligned) {
        const auto result_count = wb::price_row_count(count, count, construction);
        if constexpr (std::is_same_v<Product, product::RegularEuropeanSwaptionParameters>) {
            rates::g2::launch_g2_european_swaption_cuda<Side>(models, count, host_products, products, count,
                construction, result_count, offset, batch, paths, 1.0f/252.0f,
                threads, blocks, seed, prices, errors);
        } else {
            rates::g2::launch_g2_european_swaption_cuda<Side>(models, count, host_products, products,
                host_dates, host_accruals, dates, accruals, pool_size, count,
                construction, result_count, offset, batch, paths, 1.0f/252.0f,
                threads, blocks, seed, prices, errors);
        }
    }
};

struct G2NelsonSiegel {
    using Model = rates::g2_plus_plus::ModelParameters;
    using Curve = wb::curve::nelson_siegel::NelsonSiegelParameters;
    static constexpr const char* model_name = "g2_plus_plus";
    static constexpr const char* curve_name = "nelson_siegel";
    static auto models() { return rates::g2_plus_plus::load_models("datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json"); }
    static auto curves() { return wb::curve::nelson_siegel::load_curves("datasets/curve/nelson_siegel/nelson_siegel_01.json"); }
    template<wb::SwaptionSide Side, typename Product>
    static void launch(const Model* models, const Curve* curves, const Product* host_products,
                       const Product* products, const std::uint32_t* host_dates, const float* host_accruals,
                       const std::uint32_t* dates, const float* accruals, std::size_t pool_size,
                       std::size_t count, std::size_t offset, std::size_t batch,
                       unsigned threads, std::size_t blocks, float* prices, float* errors,
                       wb::PriceConstruction construction = wb::PriceConstruction::Aligned) {
        const auto result_count = wb::price_row_count(count, count, count, construction);
        if constexpr (std::is_same_v<Product, product::RegularEuropeanSwaptionParameters>) {
            rates::g2_plus_plus::nelson_siegel::launch_g2_plus_plus_nelson_siegel_european_swaption_cuda<Side>(models, count, curves, count, host_products, products, count,
                construction, result_count, offset, batch, paths, 1.0f/252.0f,
                threads, blocks, seed, prices, errors);
        } else {
            rates::g2_plus_plus::nelson_siegel::launch_g2_plus_plus_nelson_siegel_european_swaption_cuda<Side>(models, count, curves, count, host_products, products,
                host_dates, host_accruals, dates, accruals, pool_size, count,
                construction, result_count, offset, batch, paths, 1.0f/252.0f,
                threads, blocks, seed, prices, errors);
        }
    }
};

struct G2Svensson {
    using Model = rates::g2_plus_plus::ModelParameters;
    using Curve = wb::curve::svensson::SvenssonParameters;
    static constexpr const char* model_name = "g2_plus_plus";
    static constexpr const char* curve_name = "svensson";
    static auto models() { return rates::g2_plus_plus::load_models("datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json"); }
    static auto curves() { return wb::curve::svensson::load_curves("datasets/curve/svensson/svensson_01.json"); }
    template<wb::SwaptionSide Side, typename Product>
    static void launch(const Model* models, const Curve* curves, const Product* host_products,
                       const Product* products, const std::uint32_t* host_dates, const float* host_accruals,
                       const std::uint32_t* dates, const float* accruals, std::size_t pool_size,
                       std::size_t count, std::size_t offset, std::size_t batch,
                       unsigned threads, std::size_t blocks, float* prices, float* errors,
                       wb::PriceConstruction construction = wb::PriceConstruction::Aligned) {
        const auto result_count = wb::price_row_count(count, count, count, construction);
        if constexpr (std::is_same_v<Product, product::RegularEuropeanSwaptionParameters>) {
            rates::g2_plus_plus::svensson::launch_g2_plus_plus_svensson_european_swaption_cuda<Side>(models, count, curves, count, host_products, products, count,
                construction, result_count, offset, batch, paths, 1.0f/252.0f,
                threads, blocks, seed, prices, errors);
        } else {
            rates::g2_plus_plus::svensson::launch_g2_plus_plus_svensson_european_swaption_cuda<Side>(models, count, curves, count, host_products, products,
                host_dates, host_accruals, dates, accruals, pool_size, count,
                construction, result_count, offset, batch, paths, 1.0f/252.0f,
                threads, blocks, seed, prices, errors);
        }
    }
};

template<typename Values>
auto selected(const Values& all) {
    Values result;
    for (auto index : indices) result.push_back(all.at(index));
    return result;
}

template<typename Adapter, wb::SwaptionSide Side>
void validate_side() {
    const auto models = selected(Adapter::models());
    const auto curves = selected(Adapter::curves());
    auto products = selected(product::load_european_swaptions(
        "datasets/product/european_swaption/european_swaptions_01.json").products);
    // Include a single-coupon bond-option identity in every model/side sweep.
    products[0].payment_count = 1U;
    constexpr auto count = indices.size();
    std::uint32_t maximum_count = 0;
    for (const auto& p : products) maximum_count = std::max(maximum_count, p.payment_count);
    std::vector<product::ExplicitEuropeanSwaptionParameters> explicit_products;
    std::vector<std::uint32_t> dates(count*maximum_count);
    std::vector<float> accruals(dates.size());
    for (std::size_t i=0; i<count; ++i) {
        const auto& p = products[i];
        explicit_products.push_back({p.notional,p.strike,p.exercise_time_days,p.payment_count,i});
        for (std::size_t j=0; j<p.payment_count; ++j) {
            dates[j*count+i] = p.exercise_time_days + (j+1)*p.payment_interval_days;
            accruals[j*count+i] = p.accrual_fraction;
        }
    }
    cuda::DeviceBuffer<typename Adapter::Model> dm(count);
    cuda::DeviceBuffer<typename Adapter::Curve> dc(count);
    cuda::DeviceBuffer<product::RegularEuropeanSwaptionParameters> dp(count);
    cuda::DeviceBuffer<product::ExplicitEuropeanSwaptionParameters> de(count);
    cuda::DeviceBuffer<std::uint32_t> dd(dates.size());
    cuda::DeviceBuffer<float> da(accruals.size()), output(count), errors(count);
    dm.copy_from(models.data()); dc.copy_from(curves.data()); dp.copy_from(products.data());
    de.copy_from(explicit_products.data()); dd.copy_from(dates.data()); da.copy_from(accruals.data());
    std::vector<float> prices(count), se(count), replay(count), replay_se(count);
    const auto run = [&]<typename Product>(const Product* hp, const Product* device,
        unsigned threads, std::size_t offset, std::size_t batch, std::size_t blocks) {
        Adapter::template launch<Side>(dm.data(),dc.data(),hp,device,dates.data(),accruals.data(),
            dd.data(),da.data(),dates.size(),count,offset,batch,threads,blocks,output.data(),errors.data());
    };
    run(products.data(),dp.data(),128U,0,count,count);
    output.copy_to(prices.data()); errors.copy_to(se.data());
    // Split caller batches and a persistent one-block grid without changing row keys.
    for (std::size_t i=0; i<count; ++i) run(products.data(),dp.data(),512U,i,1,1);
    output.copy_to(replay.data()); errors.copy_to(replay_se.data());
    for (std::size_t i=0; i<count; ++i) {
        require(std::isfinite(prices[i]) && prices[i]>=0 && std::isfinite(se[i]) && se[i]>=0,
            "Invalid European-swaption MC output");
        require(std::bit_cast<std::uint32_t>(prices[i]) == std::bit_cast<std::uint32_t>(replay[i])
            && std::bit_cast<std::uint32_t>(se[i]) == std::bit_cast<std::uint32_t>(replay_se[i]),
            "European-swaption batching or geometry changed output bits");
    }
    run(explicit_products.data(),de.data(),256U,0,count,2);
    output.copy_to(replay.data()); errors.copy_to(replay_se.data());
    for (std::size_t i=0; i<count; ++i) {
        require(std::abs(prices[i]-replay[i]) <= 2e-6f*std::max(1.0f,prices[i]),
            "Equivalent explicit and regular swaption calendars disagree");
        require(std::abs(se[i]-replay_se[i]) <= 2e-6f*std::max(1.0f,se[i]),
            "Equivalent calendar standard errors disagree");
        const auto& p=products[i];
        std::cout << Json{{"model",Adapter::model_name},{"curve",Adapter::curve_name},
            {"side",Side==wb::SwaptionSide::payer?"payer":"receiver"},{"source_index",indices[i]},
            {"paths",paths},{"price",prices[i]},{"standard_error",se[i]},
            {"payment_count",p.payment_count},{"seed",seed+i}}.dump() << '\n';
    }
    auto bad = products;
    bad[0].payment_interval_days=0;
    bool rejected=false;
    try { run(bad.data(),dp.data(),128U,0,count,count); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Invalid regular payment calendar was accepted");
    dates[explicit_products[0].schedule_offset] = products[0].exercise_time_days;
    rejected=false;
    try { run(explicit_products.data(),de.data(),128U,0,count,count); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"Invalid explicit payment calendar was accepted");
}

template<typename Adapter>
void validate_cartesian() {
    constexpr std::size_t curves_count = std::is_same_v<Adapter, G2> ? 1U : 2U;
    constexpr std::size_t result_count = 2U*curves_count*2U;
    auto models = Adapter::models(); models.resize(2);
    auto curves = Adapter::curves(); curves.resize(2);
    auto products = product::load_european_swaptions(
        "datasets/product/european_swaption/european_swaptions_01.json").products;
    products.resize(2);
    std::vector<typename Adapter::Model> expanded_models;
    std::vector<typename Adapter::Curve> expanded_curves;
    std::vector<product::RegularEuropeanSwaptionParameters> expanded_products;
    // Independent explicit expansion verifies the public Cartesian index mapping.
    for (const auto& model : models) {
        for (std::size_t c=0; c<curves_count; ++c) {
            for (const auto& product : products) {
                expanded_models.push_back(model);
                expanded_curves.push_back(curves[c]);
                expanded_products.push_back(product);
            }
        }
    }
    cuda::DeviceBuffer<typename Adapter::Model> dm(result_count);
    cuda::DeviceBuffer<typename Adapter::Curve> dc(result_count);
    cuda::DeviceBuffer<product::RegularEuropeanSwaptionParameters> dp(result_count);
    cuda::DeviceBuffer<float> output(result_count), errors(result_count);
    const auto run = [&](const auto& m, const auto& c, const auto& p, wb::PriceConstruction construction) {
        wb::check_cuda(cudaMemcpy(dm.data(), m.data(), m.size()*sizeof(m[0]), cudaMemcpyHostToDevice), "copy models");
        wb::check_cuda(cudaMemcpy(dc.data(), c.data(), c.size()*sizeof(c[0]), cudaMemcpyHostToDevice), "copy curves");
        wb::check_cuda(cudaMemcpy(dp.data(), p.data(), p.size()*sizeof(p[0]), cudaMemcpyHostToDevice), "copy products");
        Adapter::template launch<wb::SwaptionSide::payer>(dm.data(),dc.data(),p.data(),dp.data(),
            nullptr,nullptr,nullptr,nullptr,0,m.size(),0,result_count,256U,2U,
            output.data(),errors.data(),construction);
        std::vector<float> prices(result_count), se(result_count);
        output.copy_to(prices.data()); errors.copy_to(se.data());
        return std::pair{prices,se};
    };
    const auto cartesian = run(models,curves,products,wb::PriceConstruction::CartesianProduct);
    const auto aligned = run(expanded_models,expanded_curves,expanded_products,wb::PriceConstruction::Aligned);
    require(cartesian == aligned, "Cartesian MC mapping differs from explicit aligned expansion");
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--smoke-test") paths = 4096U;
    else if (argc != 1) throw std::invalid_argument("Expected no arguments or --smoke-test");
    int devices=0;
    const auto status=cudaGetDeviceCount(&devices);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || devices==0) return 77;
    wb::check_cuda(status,"test cudaGetDeviceCount");
    validate_side<G2,wb::SwaptionSide::payer>();
    validate_side<G2,wb::SwaptionSide::receiver>();
    validate_side<G2NelsonSiegel,wb::SwaptionSide::payer>();
    validate_side<G2NelsonSiegel,wb::SwaptionSide::receiver>();
    validate_side<G2Svensson,wb::SwaptionSide::payer>();
    validate_side<G2Svensson,wb::SwaptionSide::receiver>();
    paths = 4096U;
    validate_cartesian<G2>();
    validate_cartesian<G2NelsonSiegel>();
    validate_cartesian<G2Svensson>();
}

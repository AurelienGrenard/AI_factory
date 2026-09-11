// Diagnostic only: isolate CIR rate-chain bias from FP32 discount accumulation.
// Production transitions are unchanged; FP64 accumulation is a paired control,
// not a proposed hot-path implementation or a performance benchmark.
#include "tests/performance/benchmark_support.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "model/fixed_income/cir/dynamics_impl.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include <filesystem>
#include <fstream>

namespace wb = ai_factory::workbench;
namespace perf = wb::performance;
namespace cir = wb::model::fixed_income::cir;
using Json = nlohmann::ordered_json;

struct ChainSample {
    float rate;
    float discount_fp32;
    double discount_fp64;
    float bond_fp32;
    double bond_fp64;
};

__global__ void sample_chain(cir::ModelParameters model, unsigned steps,
                             std::size_t paths, std::uint64_t seed,
                             ChainSample* output) {
    constexpr float horizon = 10.0f;
    const auto dynamics = cir::joint::DynamicsPolicy::prepare_dynamics(model, horizon/steps);
    const float log_a = cir::log_A(model, 0, 2);
    const float b = cir::B(model, 0, 2);
    for (std::size_t path = blockIdx.x*blockDim.x+threadIdx.x; path < paths;
         path += static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        wb::philox::NormalRandomContext random(wb::philox::make_key(seed), path);
        auto state = cir::joint::DynamicsPolicy::initial_state(dynamics);
        double integral = 0;
        for (unsigned step=0; step<steps; ++step) {
            const float previous = state.state;
            cir::joint::DynamicsPolicy::simulate_one_step(dynamics, random, state);
            integral += (static_cast<double>(previous)+state.state)*(0.5*horizon/steps);
        }
        const float discount = expf(-state.state_integral);
        const double control = exp(-integral);
        const float bond = expf(log_a-b*state.state);
        output[path] = {state.state, discount, control, discount*bond, control*bond};
    }
}

template<class Projection>
Json moments(const std::vector<ChainSample>& samples, Projection projection) {
    long double sum=0, sum2=0;
    for (const auto& sample : samples) {
        const long double value=projection(sample);
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite diagnostic sample");
        sum += value; sum2 += value*value;
    }
    const long double mean=sum/samples.size();
    const long double variance=std::max(0.0L, (sum2-samples.size()*mean*mean)/(samples.size()-1));
    return {{"mean",static_cast<double>(mean)},
            {"standard_error",static_cast<double>(std::sqrt(variance/samples.size()))}};
}

int main(int argc, char** argv) {
    try {
        if (argc!=3 || std::string(argv[1])!="--output")
            throw std::invalid_argument("Expected --output new-file.ndjson");
        if (std::filesystem::exists(argv[2])) throw std::invalid_argument("Output already exists");
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("Cannot open output");
        out << Json{{"kind","environment"},{"environment",perf::environment_json()}}.dump() << '\n';
        constexpr std::size_t paths=1U<<18U;
        const auto models=cir::load_models("datasets/model/fixed_income/cir/parameters/cir_01.json");
        perf::DeviceBuffer output(paths*sizeof(ChainSample),perf::DeviceMemoryRole::output);
        for (std::size_t index : {362U,570U,590U,676U,784U}) {
            for (unsigned steps : {1U,2520U,5040U,20160U}) {
                wb::report_cuda_kernel_launch_if_enabled("cir_forward.discount_chain", "diagnostic",
                    sample_chain,dim3(256),dim3(256),0);
                sample_chain<<<256,256>>>(models[index],steps,paths,9082026999ULL+index,output.as<ChainSample>());
                wb::check_cuda(cudaGetLastError(),"CIR discount chain");
                const auto samples=perf::copy_from_device<ChainSample>(output,paths);
                const auto row=Json{{"kind","result"},{"source_index",index},{"steps",steps},
                    {"horizon",10},{"paths",paths},
                    {"rate",moments(samples,[](auto s){return s.rate;})},
                    {"rate_squared",moments(samples,[](auto s){return static_cast<double>(s.rate)*s.rate;})},
                    {"discount_fp32",moments(samples,[](auto s){return s.discount_fp32;})},
                    {"discount_fp64",moments(samples,[](auto s){return s.discount_fp64;})},
                    {"discount_difference",moments(samples,[](auto s){return s.discount_fp64-s.discount_fp32;})},
                    {"bond_fp32",moments(samples,[](auto s){return s.bond_fp32;})},
                    {"bond_fp64",moments(samples,[](auto s){return s.bond_fp64;})}};
                out << row.dump() << '\n'; out.flush();
                std::cout << "row " << index+1 << " steps " << steps << '\n' << std::flush;
            }
        }
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n'; return 1;}
}

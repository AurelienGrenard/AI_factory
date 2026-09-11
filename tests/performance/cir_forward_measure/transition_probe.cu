// GPU transition-law and numeraire checks, independent of the LSM regression.
#include "model/fixed_income/cir/forward_measure_impl.cuh"
#include "tests/performance/benchmark_support.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include <fstream>

namespace wb = ai_factory::workbench;
namespace perf = wb::performance;
namespace probe = wb::model::fixed_income::cir::terminal_forward;
namespace cir = wb::model::fixed_income::cir;
using Json = nlohmann::ordered_json;

struct Sample { float rate; float bond_value; float call_value; };

__global__ void sample_transition(cir::ModelParameters model, float interval, float remaining,
                                  std::size_t count, std::uint64_t seed, Sample* output) {
    __shared__ probe::PreparedTransition transition;
    __shared__ float degrees, log_initial_numeraire, log_a_t, b_t, log_a_u, b_u;
    if (threadIdx.x == 0) {
        transition = probe::prepare_transition(model, interval, remaining);
        degrees = 4 * model.process.mean_reversion * model.process.long_term_mean
            / (model.process.volatility * model.process.volatility);
        log_initial_numeraire = cir::log_zero_coupon_bond(model, model.initial_state, 0, interval+remaining);
        log_a_t = cir::log_A(model, 0, remaining);
        b_t = cir::B(model, 0, remaining);
        log_a_u = cir::log_A(model, 0, remaining+2);
        b_u = cir::B(model, 0, remaining+2);
    }
    __syncthreads();
    for (std::size_t path = blockIdx.x*blockDim.x+threadIdx.x; path < count;
         path += static_cast<std::size_t>(gridDim.x)*blockDim.x) {
        wb::philox::NormalRandomContext random(wb::philox::make_key(seed), path);
        const float rate = wb::philox::scaled_noncentral_chi_square(
            random.uniforms, random.normals, degrees,
            transition.state_loading*model.initial_state/transition.scale, transition.scale);
        const float ratio = expf(log_initial_numeraire-log_a_t+b_t*rate);
        const float bond = expf(log_a_u-b_u*rate);
        output[path] = {rate, ratio*bond, ratio*fmaxf(bond-.8f, 0)};
    }
}

Json moments(const std::vector<double>& values) {
    long double sum=0, sum2=0;
    for (double value : values) { sum += value; sum2 += static_cast<long double>(value)*value; }
    const long double mean = sum/values.size();
    const long double variance = std::max(0.0L, (sum2-values.size()*mean*mean)/(values.size()-1));
    return {{"mean", static_cast<double>(mean)},
            {"standard_error", static_cast<double>(std::sqrt(variance/values.size()))}};
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--output") throw std::invalid_argument("Expected --output file.json");
        constexpr std::size_t paths = 1U << 18U;
        const auto models = cir::load_models("datasets/model/fixed_income/cir/parameters/cir_01.json");
        perf::DeviceBuffer output(paths*sizeof(Sample), perf::DeviceMemoryRole::output);
        Json results = Json::array();
        for (std::size_t index : {0U, 100U, 499U, 899U, 900U, 950U, 975U, 999U}) {
            for (auto times : {std::pair{1.0f/504.0f,20.0f}, std::pair{.5f,2.0f}, std::pair{10.0f,0.0f}}) {
                wb::report_cuda_kernel_launch_if_enabled("cir_forward.transition_check", "exact", sample_transition,
                                                        dim3(256), dim3(256), 0);
                sample_transition<<<256,256>>>(models[index], times.first, times.second, paths, 9082026000ULL+index, output.as<Sample>());
                wb::check_cuda(cudaGetLastError(), "CIR transition check");
                const auto samples = perf::copy_from_device<Sample>(output, paths);
                std::vector<double> rates(paths), squares(paths), bonds(paths), calls(paths);
                for (std::size_t path=0; path<paths; ++path) {
                    const auto s=samples[path];
                    if (!std::isfinite(s.rate) || !std::isfinite(s.bond_value) || !std::isfinite(s.call_value)
                        || s.rate<0 || s.bond_value<0 || s.call_value<0) throw std::runtime_error("Invalid transition sample");
                    rates[path]=s.rate; squares[path]=static_cast<double>(s.rate)*s.rate;
                    bonds[path]=s.bond_value; calls[path]=s.call_value;
                }
                results.push_back({{"source_index",index},{"interval",times.first},{"remaining",times.second},
                    {"paths",paths},{"rate",moments(rates)},{"rate_squared",moments(squares)},
                    {"bond",moments(bonds)},{"call",moments(calls)}});
            }
        }
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("Cannot open transition evidence output");
        out << Json{{"environment",perf::environment_json()},{"samples",results}}.dump(2) << '\n';
        std::cout << results.size() << " transition cases written\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

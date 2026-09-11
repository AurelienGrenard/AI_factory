// Verify the common Philox counter layout and path-local sequence contract.
#include "common/check_cuda.cuh"
#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using ai_factory::workbench::philox::PhiloxCounter;
namespace philox = ai_factory::workbench::philox;

struct PhiloxResults {
    PhiloxCounter addressed_bits;
    PhiloxCounter direct_bits;
    float sequence_values[8];
    float first_group_values[4];
    float second_group_values[4];
    float cached_normal_values[2];
    float direct_normal_values[2];
    float other_path_first;
    float replay_first;
    std::uint32_t zero_mean_poisson;
    std::uint32_t small_mean_poisson;
    std::uint32_t small_mean_direct_poisson;
    std::uint32_t large_mean_poisson;
    std::uint32_t large_mean_poisson_replay;
    float scaled_noncentral_chi_square;
    float scaled_noncentral_chi_square_composition;
};

__global__ void exercise_philox_kernel(PhiloxResults* output) {
    using namespace ai_factory::workbench;

    constexpr std::uint64_t path_index = 0x00000001'00000002ULL;
    constexpr std::uint64_t local_group_index = 0x00000003'00000004ULL;
    const philox::PhiloxKey key = philox::make_key(900000001ULL);
    const PhiloxCounter addressed_bits = philox::random_bits(
        key, path_index, local_group_index
    );
    const PhiloxCounter direct_bits = philox::philox4x32_10(
        key,
        {0x00000002U, 0x00000001U, 0x00000004U, 0x00000003U}
    );

    constexpr std::uint64_t sequence_path = 17ULL;
    philox::UniformSequence sequence(key, sequence_path);
    float sequence_values[8];
    #pragma unroll
    for (std::uint32_t index = 0U; index < 8U; ++index) {
        sequence_values[index] = sequence.next();
    }
    const philox::RandomQuad first_group = philox::uniform_quad(
        key, sequence_path, 0ULL
    );
    const philox::RandomQuad second_group = philox::uniform_quad(
        key, sequence_path, 1ULL
    );
    const philox::RandomQuad other_path = philox::uniform_quad(
        key, sequence_path + 1ULL, 0ULL
    );
    philox::UniformSequence normal_uniforms(key, sequence_path);
    philox::NormalPairCache normal_cache;
    const float first_cached_normal =
        philox::next_normal(normal_uniforms, normal_cache);
    const float second_cached_normal =
        philox::next_normal(normal_uniforms, normal_cache);
    const philox::NormalPair direct_normals = philox::box_muller(
        first_group.first, first_group.second
    );
    philox::UniformSequence replay(key, sequence_path);
    constexpr std::uint64_t poisson_path = 41ULL;
    philox::UniformSequence zero_poisson_uniforms(key, poisson_path);
    const std::uint32_t zero_mean_poisson =
        philox::poisson_from_uniform_sequence(
            zero_poisson_uniforms, 0.0f
        );
    philox::UniformSequence small_poisson_uniforms(key, poisson_path);
    constexpr float small_poisson_mean = 4.0f;
    const std::uint32_t small_mean_poisson =
        philox::poisson_from_uniform_sequence(
            small_poisson_uniforms, small_poisson_mean
        );
    const float small_poisson_uniform = philox::uniform_quad(
        key, poisson_path, 0ULL
    ).first;
    const std::uint32_t small_mean_direct_poisson =
        philox::poisson_from_uniform(
            small_poisson_uniform,
            small_poisson_mean,
            expf(-small_poisson_mean)
        );
    philox::UniformSequence large_poisson_uniforms(key, poisson_path);
    const std::uint32_t large_mean_poisson =
        philox::poisson_from_uniform_sequence(
            large_poisson_uniforms, 1000.0f
        );
    philox::UniformSequence large_poisson_replay(key, poisson_path);
    const std::uint32_t large_mean_poisson_replay =
        philox::poisson_from_uniform_sequence(
            large_poisson_replay, 1000.0f
        );
    constexpr std::uint64_t chi_square_path = 42ULL;
    constexpr float degrees_of_freedom = 1.75f;
    constexpr float noncentrality = 37.0f;
    constexpr float scale = 0.125f;
    philox::UniformSequence chi_square_uniforms(key, chi_square_path);
    philox::NormalPairCache chi_square_normal_cache;
    const float scaled_noncentral_chi_square =
        philox::scaled_noncentral_chi_square(
            chi_square_uniforms,
            chi_square_normal_cache,
            degrees_of_freedom,
            noncentrality,
            scale
        );
    philox::UniformSequence composition_uniforms(key, chi_square_path);
    philox::NormalPairCache composition_normal_cache;
    const std::uint32_t mixture_poisson =
        philox::poisson_from_uniform_sequence(
            composition_uniforms, 0.5f * noncentrality
        );
    const float scaled_noncentral_chi_square_composition =
        philox::marsaglia_tsang_gamma(
            composition_uniforms,
            composition_normal_cache,
            0.5f * degrees_of_freedom
                + static_cast<float>(mixture_poisson),
            2.0f * scale
        );

    output->addressed_bits = addressed_bits;
    output->direct_bits = direct_bits;
    #pragma unroll
    for (std::uint32_t index = 0U; index < 8U; ++index) {
        output->sequence_values[index] = sequence_values[index];
    }
    output->first_group_values[0] = first_group.first;
    output->first_group_values[1] = first_group.second;
    output->first_group_values[2] = first_group.third;
    output->first_group_values[3] = first_group.fourth;
    output->second_group_values[0] = second_group.first;
    output->second_group_values[1] = second_group.second;
    output->second_group_values[2] = second_group.third;
    output->second_group_values[3] = second_group.fourth;
    output->cached_normal_values[0] = first_cached_normal;
    output->cached_normal_values[1] = second_cached_normal;
    output->direct_normal_values[0] = direct_normals.first;
    output->direct_normal_values[1] = direct_normals.second;
    output->other_path_first = other_path.first;
    output->replay_first = replay.next();
    output->zero_mean_poisson = zero_mean_poisson;
    output->small_mean_poisson = small_mean_poisson;
    output->small_mean_direct_poisson = small_mean_direct_poisson;
    output->large_mean_poisson = large_mean_poisson;
    output->large_mean_poisson_replay = large_mean_poisson_replay;
    output->scaled_noncentral_chi_square = scaled_noncentral_chi_square;
    output->scaled_noncentral_chi_square_composition =
        scaled_noncentral_chi_square_composition;
}

// Draw one independent Poisson value per Philox path for moment checks.
__global__ void sample_poisson_kernel(
    std::uint64_t seed,
    float poisson_mean,
    std::size_t sample_count,
    std::uint32_t* samples
) {
    const std::size_t path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= sample_count) return;
    const philox::PhiloxKey key = philox::make_key(seed);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    samples[path] = philox::poisson_from_uniform_sequence(
        uniforms, poisson_mean
    );
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool equal_counter(const PhiloxCounter& lhs, const PhiloxCounter& rhs) {
    return lhs.v0 == rhs.v0 && lhs.v1 == rhs.v1
        && lhs.v2 == rhs.v2 && lhs.v3 == rhs.v3;
}

// Check deterministic empirical moments tightly enough to catch a broken law.
void check_poisson_moments(
    const std::vector<std::uint32_t>& samples,
    double expected_mean,
    double mean_tolerance,
    double variance_tolerance
) {
    long double sum = 0.0L;
    long double squared_sum = 0.0L;
    for (const std::uint32_t sample : samples) {
        const long double value = static_cast<long double>(sample);
        sum += value;
        squared_sum += value * value;
    }
    const long double count =
        static_cast<long double>(samples.size());
    const long double empirical_mean = sum / count;
    const long double empirical_variance =
        squared_sum / count - empirical_mean * empirical_mean;
    require(
        std::fabs(static_cast<double>(empirical_mean) - expected_mean)
            <= mean_tolerance,
        "Philox adaptive Poisson mean is outside tolerance"
    );
    require(
        std::fabs(static_cast<double>(empirical_variance) - expected_mean)
            <= variance_tolerance,
        "Philox adaptive Poisson variance is outside tolerance"
    );
}

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
    check_cuda(availability, "Philox test cudaGetDeviceCount");

    PhiloxResults* device_results = nullptr;
    check_cuda(
        cudaMalloc(&device_results, sizeof(PhiloxResults)),
        "Philox test cudaMalloc"
    );
    exercise_philox_kernel<<<1, 1>>>(device_results);
    check_cuda(cudaGetLastError(), "Philox test kernel launch");
    PhiloxResults results{};
    check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(results),
            cudaMemcpyDeviceToHost
        ),
        "Philox test cudaMemcpy"
    );
    check_cuda(cudaFree(device_results), "Philox test cudaFree");

    require(
        equal_counter(results.addressed_bits, results.direct_bits),
        "Philox path/local-group counter layout is incorrect"
    );
    for (std::uint32_t index = 0U; index < 4U; ++index) {
        require(
            results.sequence_values[index] == results.first_group_values[index],
            "Philox sequence does not begin at local group zero"
        );
        require(
            results.sequence_values[index + 4U]
                == results.second_group_values[index],
            "Philox sequence does not advance to the next local group"
        );
        require(
            results.sequence_values[index] > 0.0f
                && results.sequence_values[index] < 1.0f,
            "Philox uniform lies outside the open unit interval"
        );
    }
    require(
        results.sequence_values[0] == results.replay_first,
        "Philox path replay is not deterministic"
    );
    require(
        results.cached_normal_values[0] == results.direct_normal_values[0]
            && results.cached_normal_values[1]
                == results.direct_normal_values[1],
        "Philox normal cache does not reuse its Box-Muller pair"
    );
    require(
        results.sequence_values[0] != results.other_path_first,
        "Two adjacent Philox paths unexpectedly share their first value"
    );
    require(
        results.zero_mean_poisson == 0U,
        "Philox adaptive Poisson does not preserve the zero-mean law"
    );
    require(
        results.small_mean_poisson == results.small_mean_direct_poisson,
        "Philox adaptive Poisson did not use small-mean inversion"
    );
    require(
        results.large_mean_poisson
            == results.large_mean_poisson_replay,
        "Philox large-mean Poisson replay is not deterministic"
    );
    require(
        results.scaled_noncentral_chi_square > 0.0f
            && results.scaled_noncentral_chi_square
                == results.scaled_noncentral_chi_square_composition,
        "Philox scaled non-central chi-square mixture is incorrect"
    );

    constexpr std::size_t sample_count = 1U << 18U;
    constexpr unsigned int threads_per_block = 256U;
    constexpr unsigned int block_count = static_cast<unsigned int>(
        (sample_count + threads_per_block - 1U) / threads_per_block
    );
    std::uint32_t* device_samples = nullptr;
    check_cuda(
        cudaMalloc(&device_samples, sample_count * sizeof(std::uint32_t)),
        "Philox Poisson moment test cudaMalloc"
    );
    std::vector<std::uint32_t> samples(sample_count);
    try {
        sample_poisson_kernel<<<block_count, threads_per_block>>>(
            900000101ULL, 4.0f, sample_count, device_samples
        );
        check_cuda(
            cudaGetLastError(),
            "Philox small-mean Poisson moment kernel"
        );
        check_cuda(
            cudaMemcpy(
                samples.data(),
                device_samples,
                sample_count * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost
            ),
            "Philox small-mean Poisson moment cudaMemcpy"
        );
        check_poisson_moments(samples, 4.0, 0.03, 0.06);

        sample_poisson_kernel<<<block_count, threads_per_block>>>(
            900000101ULL, 1000.0f, sample_count, device_samples
        );
        check_cuda(
            cudaGetLastError(),
            "Philox large-mean Poisson moment kernel"
        );
        check_cuda(
            cudaMemcpy(
                samples.data(),
                device_samples,
                sample_count * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost
            ),
            "Philox large-mean Poisson moment cudaMemcpy"
        );
        check_poisson_moments(samples, 1000.0, 0.5, 10.0);
        check_cuda(
            cudaFree(device_samples),
            "Philox Poisson moment test cudaFree"
        );
        device_samples = nullptr;
    } catch (...) {
        if (device_samples != nullptr) cudaFree(device_samples);
        throw;
    }
    return 0;
}

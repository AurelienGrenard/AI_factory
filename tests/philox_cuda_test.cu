// Verify the common Philox counter layout and path-local sequence contract.
#include "common/check_cuda.cuh"
#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

using ai_factory::workbench::philox::PhiloxCounter;

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
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool equal_counter(const PhiloxCounter& lhs, const PhiloxCounter& rhs) {
    return lhs.v0 == rhs.v0 && lhs.v1 == rhs.v1
        && lhs.v2 == rhs.v2 && lhs.v3 == rhs.v3;
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
    return 0;
}

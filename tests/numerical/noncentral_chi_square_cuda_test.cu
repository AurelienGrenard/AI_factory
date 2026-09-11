// Compare device Gamma and non-central chi-square tails with SciPy references.
#include "common/check_cuda.cuh"
#include "common/noncentral_chi_square.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using ai_factory::workbench::DistributionProbabilities;

struct GammaCase {
    float shape;
    float value;
    double expected_cdf;
    double expected_survival;
};

struct NoncentralCase {
    float degrees_of_freedom;
    float noncentrality;
    float value;
    double expected_cdf;
    double expected_survival;
};

__global__ void gamma_probabilities_kernel(
    const GammaCase* cases,
    std::size_t count,
    DistributionProbabilities* probabilities
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    probabilities[index] = ai_factory::workbench::
        regularized_gamma_probabilities(
            cases[index].shape, cases[index].value
        );
}

__global__ void noncentral_probabilities_kernel(
    const NoncentralCase* cases,
    std::size_t count,
    DistributionProbabilities* probabilities
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    probabilities[index] = ai_factory::workbench::
        noncentral_chi_square_probabilities(
            cases[index].degrees_of_freedom,
            cases[index].noncentrality,
            cases[index].value
        );
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Case, typename Launcher>
void check_cases(
    const std::vector<Case>& cases,
    Launcher launcher,
    double absolute_tolerance,
    const char* mismatch_message
) {
    using ai_factory::workbench::check_cuda;
    Case* device_cases = nullptr;
    DistributionProbabilities* device_probabilities = nullptr;
    std::vector<DistributionProbabilities> probabilities(cases.size());
    try {
        check_cuda(
            cudaMalloc(&device_cases, cases.size() * sizeof(Case)),
            "distribution test cudaMalloc cases"
        );
        check_cuda(
            cudaMalloc(
                &device_probabilities,
                cases.size() * sizeof(DistributionProbabilities)
            ),
            "distribution test cudaMalloc probabilities"
        );
        check_cuda(
            cudaMemcpy(
                device_cases,
                cases.data(),
                cases.size() * sizeof(Case),
                cudaMemcpyHostToDevice
            ),
            "distribution test cudaMemcpy cases"
        );
        launcher(device_cases, cases.size(), device_probabilities);
        check_cuda(cudaGetLastError(), "distribution test kernel launch");
        check_cuda(
            cudaMemcpy(
                probabilities.data(),
                device_probabilities,
                probabilities.size() * sizeof(DistributionProbabilities),
                cudaMemcpyDeviceToHost
            ),
            "distribution test cudaMemcpy probabilities"
        );
        check_cuda(cudaFree(device_cases), "distribution test cudaFree cases");
        device_cases = nullptr;
        check_cuda(
            cudaFree(device_probabilities),
            "distribution test cudaFree probabilities"
        );
        device_probabilities = nullptr;
    } catch (...) {
        if (device_cases != nullptr) cudaFree(device_cases);
        if (device_probabilities != nullptr) cudaFree(device_probabilities);
        throw;
    }

    for (std::size_t index = 0U; index < cases.size(); ++index) {
        const DistributionProbabilities probability = probabilities[index];
        require(
            std::isfinite(probability.cdf)
                && std::isfinite(probability.survival)
                && probability.cdf >= 0.0f
                && probability.cdf <= 1.0f
                && probability.survival >= 0.0f
                && probability.survival <= 1.0f,
            "distribution probabilities left [0,1]"
        );
        require(
            std::fabs(
                static_cast<double>(probability.cdf)
                - cases[index].expected_cdf
            ) <= absolute_tolerance
                && std::fabs(
                    static_cast<double>(probability.survival)
                    - cases[index].expected_survival
                ) <= absolute_tolerance
                && std::fabs(
                    static_cast<double>(probability.cdf)
                    + probability.survival - 1.0
                ) <= 2.5e-6,
            mismatch_message
        );
    }
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "distribution test cudaGetDeviceCount");

    // Expected values were generated with scipy.special.gammainc/gammaincc.
    const std::vector<GammaCase> gamma_cases = {
        {1.0f / 6.0f, 1.0e-4f, 0.2322258177713355, 0.7677741822286646},
        {1.0f / 6.0f, 0.1f, 0.7241581224863861, 0.2758418775136138},
        {1.0f / 6.0f, 1.0f, 0.9574702878549342, 0.0425297121450656},
        {0.5f, 0.5f, 0.6826894921370859, 0.3173105078629112},
        {1.0f, 1.0f, 0.6321205588285577, 0.3678794411714424},
        {5.0f, 2.0f, 0.0526530173437111, 0.9473469826562889},
        {10.0f, 10.0f, 0.5420702855281478, 0.4579297144718523},
        {500.0f, 500.0f, 0.5059471461707603, 0.4940528538292396},
        {500.0f, 450.0f, 0.0107172380912897, 0.9892827619087102},
        {500.0f, 550.0f, 0.9853855918737048, 0.0146144081262952},
    };
    check_cases(
        gamma_cases,
        [](const GammaCase* cases,
           std::size_t count,
           DistributionProbabilities* output) {
            gamma_probabilities_kernel<<<1U, 32U>>>(cases, count, output);
        },
        3.0e-6,
        "regularized Gamma device probability differs from SciPy"
    );

    // These cases cross the mixture/saddlepoint boundary and reach lambda=1e8.
    const std::vector<NoncentralCase> noncentral_cases = {
        {0.2f, 0.0f, 0.01f, 0.6185276650753577, 0.3814723349246423},
        {1.0f / 3.0f, 0.1f, 0.2f, 0.6916817313498085, 0.3083182686501915},
        {1.0f, 10.0f, 5.0f, 0.1771684771494461, 0.8228315228505542},
        {2.0f, 200.0f, 180.0f, 0.2230136734704103, 0.7769863265295893},
        {20.0f, 1024.0f, 1044.0f, 0.5061840056203377, 0.4938159943796631},
        {0.2f, 1025.0f, 1025.199951171875f,
         0.5062303811410120, 0.4937696188589874},
        {32.0f, 10000.0f, 10232.16015625f,
         0.8413509896226880, 0.1586490103773152},
        {0.2f, 100000000.0f, 100000000.0f,
         0.5000159576911314, 0.4999840423088960},
        {2.0f, 1000.0f, 622.3369750976562f,
         1.0868120651883932e-11, 0.9999999999891326},
        {2.0f, 1000.0f, 1381.6629638671875f,
         0.9999999842941705, 1.5705830681458537e-8},
        {2.0f, 1000000.0f, 990002.0f,
         2.692532717710062e-7, 0.9999997307467670},
        {2.0f, 1000000.0f, 1010002.0f,
         0.9999996950543566, 3.0494568387210936e-7},
    };
    check_cases(
        noncentral_cases,
        [](const NoncentralCase* cases,
           std::size_t count,
           DistributionProbabilities* output) {
            noncentral_probabilities_kernel<<<1U, 32U>>>(
                cases, count, output
            );
        },
        2.0e-6,
        "non-central chi-square device probability differs from SciPy"
    );
    return 0;
}

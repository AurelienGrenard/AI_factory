// Host/device-compilation and stress checks for reusable Volterra kernels.
#include "common/volterra/fractional_hybrid_kernel.cuh"
#include "common/volterra/fractional_resolvent_hybrid_kernel.cuh"
#include "common/volterra/log_modulated_hybrid_kernel.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool relative_close(double left, double right, double tolerance) {
    return std::abs(left - right)
        <= tolerance * std::max(std::abs(right), 1.0e-15);
}

}  // namespace

int main() {
    using namespace ai_factory::workbench::volterra;

    const auto fractional = FractionalHybridKernelPolicy::prepare(
        0.10f, 1.0f / 360.0f
    );
    require(
        relative_close(
            FractionalHybridKernelPolicy::volterra_variance(fractional, 1.0f),
            1.0,
            2.0e-6
        ),
        "Normalized fractional kernel has incorrect one-year variance."
    );

    const auto log_modulated = LogModulatedHybridKernelPolicy::prepare(
        {0.10f, 0.10f, 2.0f},
        1.0f / 360.0f
    );
    require(
        std::isfinite(log_modulated.normalization)
            && log_modulated.normalization > 0.0f
            && relative_close(log_modulated.normalization, 0.4675985864, 8.0e-3)
            && relative_close(
                LogModulatedHybridKernelPolicy::volterra_variance(
                    log_modulated, 1.0f
                ),
                1.0,
                3.0e-5
            ),
        "Log-modulated kernel normalization is incorrect."
    );
    const auto logarithmic_stress = LogModulatedHybridKernelPolicy::prepare(
        {0.0f, 0.05f, 1.05f},
        1.0f / 252.0f
    );
    require(
        std::isfinite(logarithmic_stress.normalization)
            && relative_close(
                logarithmic_stress.normalization,
                0.1618347187,
                2.0e-4
            ),
        "The H=0 logarithmic kernel stress case is not integrable."
    );
    const auto narrow_logarithmic_tail =
        LogModulatedHybridKernelPolicy::prepare(
            {0.0080062263f, 0.8938079476f, 4.8189830780f},
            1.0f / 252.0f
        );
    require(
        relative_close(
            narrow_logarithmic_tail.normalization,
            0.8995616395,
            2.0e-4
        ),
        "The narrow log-modulated stress tail is under-resolved."
    );

    const double mittag_high = FractionalResolventHybridKernelPolicy::
        mittag_leffler_alpha_alpha(0.95, -10.0);
    const double mittag_low = FractionalResolventHybridKernelPolicy::
        mittag_leffler_alpha_alpha(0.51, -10.0);
    if (!(relative_close(mittag_high, 0.00082191088, 6.0e-4)
          && relative_close(mittag_low, 0.0027981165, 1.0e-4))) {
        std::fprintf(
            stderr,
            "Mittag-Leffler values: %.17g %.17g\n",
            mittag_high,
            mittag_low
        );
    }
    require(
        relative_close(
            mittag_high,
            0.00082191088,
            6.0e-4
        )
            && relative_close(
                mittag_low,
                0.0027981165,
                1.0e-4
            ),
        "Mittag-Leffler negative-axis continuation is inaccurate."
    );
    const auto resolvent = FractionalResolventHybridKernelPolicy::prepare(
        {0.45f, 8.0f},
        1.0f / 252.0f
    );
    for (const float time : {1.0f, 5.0f, 7.0f}) {
        const float value = FractionalResolventHybridKernelPolicy::kernel(
            resolvent, time
        );
        require(
            std::isfinite(value) && value > 0.0f,
            "Fractional resolvent is not finite and positive in the stress tail."
        );
    }
    return 0;
}

// Small value types shared by parameter and schedule sampling policies.
#pragma once

namespace ai_factory::workbench::sample {

struct UniformBounds {
    float minimum;
    float maximum;
};

}  // namespace ai_factory::workbench::sample

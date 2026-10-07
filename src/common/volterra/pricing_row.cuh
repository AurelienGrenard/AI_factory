// Model, product and calendar data shared by Volterra pricing engines.
#pragma once

#include "common/philox.cuh"

namespace ai_factory::workbench::volterra {

template<typename KernelPolicy, typename ModelPathPolicy,
         typename ProductPolicy, typename SchedulePolicy>
struct PricingRow {
    using Kernel = KernelPolicy;
    using Path = ModelPathPolicy;
    using Product = ProductPolicy;
    using Schedule = SchedulePolicy;

    typename Kernel::PreparedKernel kernel;
    typename Path::PreparedModel model;
    typename Product::PreparedProduct product;
    typename Schedule::PreparedSchedule schedule;
    philox::PhiloxKey key;
    float sqrt_time_step;
};

}  // namespace ai_factory::workbench::volterra

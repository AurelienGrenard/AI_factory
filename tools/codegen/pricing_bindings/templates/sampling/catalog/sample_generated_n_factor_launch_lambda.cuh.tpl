// Adapts GPU-drawn and GPU-prepared N-factor rows to the typed sampler.
[bounds, parameter_seed](
        auto* device_parameters,
        std::size_t parameter_count,
        std::size_t paths_per_parameter,
        std::uint32_t minimum_maturity_days,
        std::uint32_t maximum_maturity_days,
        std::size_t sample_offset,
        std::size_t launch_sample_count,
        unsigned int threads_per_block,
        std::size_t block_count,
        std::uint64_t schedule_seed,
        std::uint64_t dynamics_seed,
        std::uint32_t* device_maturity_days,
        std::span<float*> outputs
    ) {
        if (outputs.size() != $output_count) {
            throw std::invalid_argument("$display sample output arity mismatch.");
        }
        $launcher<factor_count>(
            device_parameters, bounds,
            parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days,
            sample_offset, launch_sample_count, threads_per_block,
            block_count, parameter_seed, schedule_seed, dynamics_seed,
            device_maturity_days, $output_pointers
        );
    }

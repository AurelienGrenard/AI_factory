// Prepares quadratic rough-Heston factors on the shared Hurst grid.
[](const std::vector<ModelParameters>& parameters,
   std::uint32_t maximum_maturity_days) {
    return model_binding::prepare_dynamics_on_hurst_grid<
        factor_count, 257U
    >(
        parameters,
        static_cast<float>(maximum_maturity_days) / 252.0f,
        1.0f / 504.0f,
        0.01f,
        0.20f
    );
},

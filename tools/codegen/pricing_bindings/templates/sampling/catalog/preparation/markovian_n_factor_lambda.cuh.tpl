// Prepares the fixed seven-factor Markovian lift for sample generation.
[](const std::vector<ModelParameters>& parameters,
   std::uint32_t maximum_maturity_days) {
    return model_binding::prepare_dynamics<factor_count>(
        parameters,
        static_cast<float>(maximum_maturity_days) / 252.0f,
        1.0f / 504.0f
    );
},

// Builds accepted model-parameter rows from deterministic host uniforms.
inline std::vector<ModelParameters> generate_core_parameters(
    std::size_t parameter_count,
    std::uint64_t seed
) {
    std::vector<ModelParameters> parameters;
    parameters.reserve(parameter_count);
    std::size_t proposal = 0U;
    while (parameters.size() < parameter_count) {
        HostUniformSequence uniforms(seed, proposal++);
        $uniforms$derived
        if (!($acceptance)) continue;
        parameters.push_back($constructor);
    }
    return parameters;
}

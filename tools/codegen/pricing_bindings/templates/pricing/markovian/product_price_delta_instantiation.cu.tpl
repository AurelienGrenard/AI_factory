// Explicit option-side instantiation for the paired-output launcher.
template void launch_${model}_${product}_price_delta_cuda<OptionSide::${side}>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::${product_type}Parameters*, const product::${product_type}Parameters*,
    std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    ${time_types}, unsigned int, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,
    float*, float*, float*, float*);

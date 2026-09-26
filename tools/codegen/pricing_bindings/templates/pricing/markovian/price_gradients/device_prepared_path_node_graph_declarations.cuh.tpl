// Public workspace and launch surface for path node-graph sensitivities.
${sensitivity_template}
std::size_t ${model}_${product}_node_graph_workspace_bytes(
    const ${product_type}PriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

${sensitivity_template}
void launch_${model}_${product}_node_graph_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    void* workspace,
    std::size_t workspace_bytes
);


// Selected mixed derivatives use the node graph; request shape lives in host.
${side_template}
std::size_t ${model}_${product}_mixed_node_graph_workspace_bytes(
    const ${product_type}PriceGradientPlan& host,
    const pg::LaunchConfiguration& configuration
);

${side_template}
void launch_${model}_${product}_mixed_node_graph_sensitivities_cuda(
    const ${product_type}PriceGradientPlan& host,
    ${product_type}PriceGradientPlan::DeviceInputs device,
    ${product_type}PriceGradientPlan::DiagonalStencilOutputs stencil_outputs,
    ${product_type}PriceGradientPlan::MixedStencilOutputs mixed_stencil_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    void* workspace,
    std::size_t workspace_bytes
);

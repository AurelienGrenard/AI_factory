// Shared execution policies for direct and node-graph sensitivities.
#pragma once

#include "common/price_gradients/sensitivity_strategy.hpp"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ai_factory::workbench::offline::pricing::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<typename Launch>
class DirectSensitivityExecution {
public:
    explicit DirectSensitivityExecution(Launch launch)
        : launch_(std::move(launch)) {}

    template<typename Plan>
    std::size_t workspace_bytes(
        const Plan&,
        const pg::LaunchConfiguration&
    ) const {
        return 0U;
    }

    template<typename Plan, typename Inputs, typename Stencils, typename Outputs>
    void launch(
        const Plan& plan,
        Inputs inputs,
        Stencils stencils,
        const pg::LaunchConfiguration& configuration,
        Outputs outputs,
        void*,
        std::size_t
    ) const {
        std::invoke(
            launch_, plan, inputs, stencils, configuration, outputs
        );
    }

private:
    Launch launch_;
};

template<typename Launch>
auto direct_sensitivity_execution(Launch launch) {
    return DirectSensitivityExecution<Launch>(std::move(launch));
}

template<typename MonoLaunch, typename GraphWorkspaceBytes, typename GraphLaunch>
class NodeGraphSensitivityExecution {
public:
    NodeGraphSensitivityExecution(
        pg::SensitivityStrategy strategy,
        MonoLaunch mono_launch,
        GraphWorkspaceBytes graph_workspace_bytes,
        GraphLaunch graph_launch
    )
        : strategy_(strategy),
          mono_launch_(std::move(mono_launch)),
          graph_workspace_bytes_(std::move(graph_workspace_bytes)),
          graph_launch_(std::move(graph_launch)) {}

    template<typename Plan>
    std::size_t workspace_bytes(
        const Plan& plan,
        const pg::LaunchConfiguration& configuration
    ) const {
        if (strategy_ == pg::SensitivityStrategy::mono) return 0U;
        return std::invoke(graph_workspace_bytes_, plan, configuration);
    }

    template<typename Plan, typename Inputs, typename Stencils, typename Outputs>
    void launch(
        const Plan& plan,
        Inputs inputs,
        Stencils stencils,
        const pg::LaunchConfiguration& configuration,
        Outputs outputs,
        void* workspace,
        std::size_t workspace_bytes
    ) const {
        if (strategy_ == pg::SensitivityStrategy::mono) {
            std::invoke(
                mono_launch_,
                plan,
                inputs,
                stencils,
                configuration,
                outputs
            );
            return;
        }
        std::invoke(
            graph_launch_,
            plan,
            inputs,
            stencils,
            configuration,
            outputs,
            workspace,
            workspace_bytes
        );
    }

    std::string strategy_name() const {
        return std::string(pg::to_string(strategy_));
    }

    pg::SensitivityStrategy strategy() const noexcept {
        return strategy_;
    }

private:
    pg::SensitivityStrategy strategy_;
    MonoLaunch mono_launch_;
    GraphWorkspaceBytes graph_workspace_bytes_;
    GraphLaunch graph_launch_;
};

template<typename MonoLaunch, typename GraphWorkspaceBytes, typename GraphLaunch>
auto node_graph_sensitivity_execution(
    pg::SensitivityStrategy strategy,
    MonoLaunch mono_launch,
    GraphWorkspaceBytes graph_workspace_bytes,
    GraphLaunch graph_launch
) {
    return NodeGraphSensitivityExecution<
        MonoLaunch, GraphWorkspaceBytes, GraphLaunch
    >(
        strategy,
        std::move(mono_launch),
        std::move(graph_workspace_bytes),
        std::move(graph_launch)
    );
}

template<typename GraphWorkspaceBytes, typename GraphLaunch>
class MixedNodeGraphSensitivityExecution {
public:
    MixedNodeGraphSensitivityExecution(
        GraphWorkspaceBytes graph_workspace_bytes,
        GraphLaunch graph_launch
    )
        : graph_workspace_bytes_(std::move(graph_workspace_bytes)),
          graph_launch_(std::move(graph_launch)) {}

    template<typename Plan>
    std::size_t workspace_bytes(
        const Plan& plan,
        const pg::LaunchConfiguration& configuration
    ) const {
        return std::invoke(graph_workspace_bytes_, plan, configuration);
    }

    template<
        typename Plan,
        typename Inputs,
        typename Stencils,
        typename MixedStencils,
        typename Outputs,
        typename MixedOutputs>
    void launch(
        const Plan& plan,
        Inputs inputs,
        Stencils stencils,
        MixedStencils mixed_stencils,
        const pg::LaunchConfiguration& configuration,
        Outputs outputs,
        MixedOutputs mixed_outputs,
        void* workspace,
        std::size_t workspace_bytes
    ) const {
        std::invoke(
            graph_launch_,
            plan,
            inputs,
            stencils,
            mixed_stencils,
            configuration,
            outputs,
            mixed_outputs,
            workspace,
            workspace_bytes
        );
    }

    std::string strategy_name() const {
        return "mixed_node_graph";
    }

private:
    GraphWorkspaceBytes graph_workspace_bytes_;
    GraphLaunch graph_launch_;
};

template<typename GraphWorkspaceBytes, typename GraphLaunch>
auto mixed_node_graph_sensitivity_execution(
    GraphWorkspaceBytes graph_workspace_bytes,
    GraphLaunch graph_launch
) {
    return MixedNodeGraphSensitivityExecution<
        GraphWorkspaceBytes, GraphLaunch
    >(
        std::move(graph_workspace_bytes),
        std::move(graph_launch)
    );
}

inline pg::SensitivityStrategy sensitivity_strategy_from_arguments(
    int argument_count,
    const char* const* arguments
) {
    if (argument_count == 1) return pg::SensitivityStrategy::mono;
    if (argument_count == 3
        && std::string_view(arguments[1]) == "--sensitivity-strategy") {
        return pg::sensitivity_strategy(arguments[2]);
    }
    throw std::invalid_argument(
        "Usage: generator [--sensitivity-strategy mono|node_graph]"
    );
}

}  // namespace ai_factory::workbench::offline::pricing::price_gradients

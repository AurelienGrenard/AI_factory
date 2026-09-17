// American spot delta: unchanged central regression, frozen exercise, paired payoffs.
#pragma once

#include "common/equity/price_delta/frozen_exercise_paths.cuh"
#include "common/longstaff_schwartz/frozen_date_delta_kernels.cuh"
#include "product/american_option/frozen_exercise_value.cuh"
#include "product/american_option/pricing_policy.cuh"

namespace ai_factory::workbench::product {

template<typename SchedulePolicy, OptionSide Side, typename Continuation,
         typename FrozenPathPolicy>
struct AmericanOptionPriceDeltaPolicy
    : AmericanOptionPricingPolicy<SchedulePolicy, Side, Continuation> {
    using Base = AmericanOptionPricingPolicy<SchedulePolicy, Side, Continuation>;
    using Schedule = SchedulePolicy;
    using Dynamics = typename Base::Dynamics;
    using ModelParameters = typename Base::ModelParameters;
    using ProductParameters = typename Base::ProductParameters;
    using TimeConfiguration = typename Base::TimeConfiguration;
    using BumpConfiguration = equity::price_delta::SpotBumpConfiguration;
    using Exercise = equity::price_delta::FrozenExercise;
    using InitialDecision = equity::price_delta::InitialExerciseDecision;

    struct HostInputs : Base::HostInputs {
        const ModelParameters* models;
        std::size_t model_count;
        BumpConfiguration bump;
        void validate(std::size_t results) const {
            Base::HostInputs::validate(results);
            if (models == nullptr) throw std::invalid_argument("LSM delta host models are null.");
            validate_model_product_construction(model_count, this->product_count,
                                                this->construction, results);
            for (std::size_t i = 0; i < model_count; ++i) {
                equity::price_delta::validate_spot_bump(models[i].spot, bump);
            }
        }
    };

    struct DeviceInputs {
        typename Base::DeviceInputs primary;
        BumpConfiguration bump;
        float* deltas;
        float* delta_errors;
        void validate(std::size_t results) const {
            primary.validate(results);
            equity::price_delta::validate_device_context(bump);
            validate_device_pointer(deltas, "device_deltas");
            validate_device_pointer(delta_errors, "device_delta_errors");
        }
        template<typename Policy, typename Time, typename... Inputs>
        __device__ __forceinline__ typename Policy::PreparedRow prepare_row(std::size_t index, const Time& time,
                                    const Inputs&... inputs) const {
            return primary.template prepare_row<Policy>(index, time, inputs..., bump);
        }
    };

    struct PreparedRow : Base::PreparedRow {
        typename FrozenPathPolicy::Prepared frozen_path;
        equity::price_delta::SpotBump bump;
        Exercise* exercises;
        InitialDecision* initial_exercise;
    };
    struct StateView : Base::StateView {
        Exercise* exercises;
        InitialDecision* initial_exercises;
    };

    static auto path_field_descriptors() -> std::vector<longstaff_schwartz::StateFieldDescriptor> {
        return {{sizeof(Exercise), alignof(Exercise)}};
    }
    static auto row_field_descriptors() -> std::vector<longstaff_schwartz::StateFieldDescriptor> {
        return {{sizeof(InitialDecision), alignof(InitialDecision)}};
    }
    static StateView make_state_view(unsigned char* workspace,
                                     const longstaff_schwartz::WorkspaceLayout& layout) {
        return {Base::make_state_view(workspace, layout),
            longstaff_schwartz::workspace_pointer<Exercise>(workspace, layout.path_fields.at(0)),
            longstaff_schwartz::workspace_pointer<InitialDecision>(workspace, layout.row_fields.at(0))};
    }

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model, const ProductParameters& product,
        philox::PhiloxKey key, std::size_t result, std::size_t offset,
        std::size_t paths, BumpConfiguration configuration, const TimeConfiguration& time
    ) {
        const auto bump = equity::price_delta::prepare_spot_bump(model.spot, configuration);
        return {Base::prepare_row(model, product, key, result, offset, paths, time),
                FrozenPathPolicy::prepare(model, product, time, bump), bump, nullptr, nullptr};
    }
    __device__ __forceinline__ static void prepare_path_outputs(PreparedRow& row, const StateView& states,
                                                std::size_t batch_price, std::size_t paths) {
        row.exercises = states.exercises + batch_price * paths;
        row.initial_exercise = states.initial_exercises + batch_price;
        *row.initial_exercise = InitialDecision::continuation;
    }
    __device__ __forceinline__ static float simulate_path(const PreparedRow& row, std::size_t path,
                                          std::size_t paths, const StateView& states) {
        return simulate_frozen_exercise_path<Schedule, Side, Continuation>(
            row.schedule,
            row.key,
            row.state_offset,
            row.regression_count,
            row.strike,
            row.exercises,
            path,
            paths,
            states
        );
    }
    __device__ __forceinline__ static void record_exercise(const PreparedRow& row, const StateView& states,
        std::size_t observation, std::size_t path, std::uint32_t backward_level) {
        record_frozen_exercise<Continuation>(
            row.exercises,
            states,
            observation,
            path,
            row.regression_count,
            backward_level
        );
    }
    __device__ __forceinline__ static void record_initial_exercise(const PreparedRow& row, bool exercise, bool valid) {
        *row.initial_exercise = !valid ? InitialDecision::invalid
            : exercise ? InitialDecision::exercise : InitialDecision::continuation;
    }
    __device__ __forceinline__ static float initial_delta(const PreparedRow& row) {
        return (payoff::vanilla_option_payoff<Side>(row.bump.upper, row.strike)
              - payoff::vanilla_option_payoff<Side>(row.bump.lower, row.strike)) / row.bump.width;
    }
    __device__ __forceinline__ static float path_delta(const PreparedRow& row, std::size_t path) {
        const Exercise exercise = row.exercises[path];
        const auto spots = FrozenPathPolicy::evaluate(row.frozen_path, exercise, row.key, path);
        return centered_frozen_exercise_gradient<Side>(
            exercise,
            spots.lower,
            spots.upper,
            row.strike,
            row.initial_discount,
            row.exercise_discount,
            row.bump.width
        );
    }
    static std::size_t finish_batch(const DeviceInputs& inputs, const PreparedRow* rows,
        dim3 grid, unsigned threads, std::size_t paths, std::size_t blocks,
        double* partials, const longstaff_schwartz::RegressionDiagnostics* diagnostics,
        const char* name) {
        return longstaff_schwartz::finish_frozen_date_delta_batch<AmericanOptionPriceDeltaPolicy>(
            inputs, rows, grid, threads, paths, blocks, partials, diagnostics, name);
    }
};

}  // namespace ai_factory::workbench::product

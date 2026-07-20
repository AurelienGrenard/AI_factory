"""black scholes down and in calls result pipeline."""

from tools.registry.result.common.barrier_calls import build_pipeline

_PIPELINE = build_pipeline("black_scholes", "down_and_in_calls", "black_scholes_01")

generate_production_cpp_gpu_result = _PIPELINE.generate_production_cpp_gpu_result
generate_validation_reprice_result = _PIPELINE.generate_validation_reprice_result

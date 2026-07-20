"""Black-Scholes European-call result pipeline."""
from tools.registry.result.common.equity_terminal import AUDIT_ROW_COUNT, PRODUCTION_ROW_COUNT, generate_result as _generate
def generate_result(**kwargs):
    return _generate(model_family="black_scholes", product_family="european_calls", **kwargs)

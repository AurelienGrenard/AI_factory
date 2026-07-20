from tools.registry.result.common.rates import AUDIT_ROW_COUNT, PRODUCTION_ROW_COUNT, RateConfig, generate_result as _generate_result
CONFIG=RateConfig("cir","interest_rate_swaps",False,False)
def generate_result(**kwargs):return _generate_result(CONFIG,**kwargs)

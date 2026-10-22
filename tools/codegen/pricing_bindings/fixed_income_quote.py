"""Read the five fixed-income product quote conventions for pricing recipes."""

from functools import lru_cache
from pathlib import Path
import yaml

_PRODUCT_DATASETS = {
    'zero_coupon_bond_option': 'zero_coupon_bond_options_01',
    'zero_coupon_bond_up_and_out': 'zero_coupon_bond_up_and_outs_01',
    'rate_option': 'rate_options_01',
    'european_swaption': 'european_swaptions_01',
    'bermudan_swaption': 'bermudan_swaptions_01',
}


@lru_cache(maxsize=None)
def quote_convention(product: str) -> dict:
    root = Path(__file__).resolve().parents[3]
    recipe = (root / 'catalog/product' / product
              / _PRODUCT_DATASETS[product] / 'recipe.yaml')
    return yaml.safe_load(recipe.read_text())['quote_convention']

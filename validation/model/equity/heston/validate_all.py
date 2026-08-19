"""Regenerate every persisted Heston validation artifact."""

from validation.model.equity.heston.validation import SPEC
from validation.model.equity.validate_model import run_all_validations


if __name__ == "__main__":
    raise SystemExit(run_all_validations("heston", SPEC.product_kinds))

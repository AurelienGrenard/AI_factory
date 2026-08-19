"""Regenerate every persisted Bates validation artifact."""

from validation.model.equity.bates.validation import SPEC
from validation.model.equity.validate_model import run_all_validations


if __name__ == "__main__":
    raise SystemExit(run_all_validations("bates", SPEC.product_kinds))

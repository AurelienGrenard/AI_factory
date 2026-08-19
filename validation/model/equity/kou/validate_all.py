"""Regenerate every persisted Kou validation artifact."""

from validation.model.equity.kou.validation import SPEC
from validation.model.equity.validate_model import run_all_validations


if __name__ == "__main__":
    raise SystemExit(run_all_validations("kou", SPEC.product_kinds))

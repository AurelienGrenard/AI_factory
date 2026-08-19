"""Regenerate every persisted Merton validation artifact."""

from validation.model.equity.merton.validation import SPEC
from validation.model.equity.validate_model import run_all_validations


if __name__ == "__main__":
    raise SystemExit(run_all_validations("merton", SPEC.product_kinds))

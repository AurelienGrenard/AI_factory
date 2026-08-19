"""Backward-compatible alias for Black-Scholes reference regeneration."""

from validation.model.equity.black_scholes.generate_all_references import main


if __name__ == "__main__":
    raise SystemExit(main())

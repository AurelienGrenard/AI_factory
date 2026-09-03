# Independent price-validation pipeline

This document defines the publication contract for persistent independent
price references. It currently applies to every fixed-income dataset and every
Black-Scholes dataset. Other equity models are migrated separately; their
legacy report/notebook artifacts must not be copied into a new implementation.

## Objective

A published price dataset contains 1,000 ordered rows:

- rows 1--900 form the `core` regime;
- rows 901--1,000 form the `stress` regime.

Both regimes require an independent reference and both must pass before the
catalogue may publish `verified: true`. External engines run only during an
explicit regeneration. Routine tests compare the source dataset with an
immutable JSON cache and therefore require neither Premia, QuantLib, nor Wine.

The reference hierarchy is always:

1. Premia;
2. QuantLib specialized;
3. QuantLib Monte Carlo.

Selection is row-wise. A technical failure falls through to the next compatible
engine for that row only. A successful, financially admissible reference that
fails the declared comparison is not replaced retrospectively by a closer
price.

European swaptions illustrate the distinction. Premia is primary for the
Vasicek, centered-OU, and Hull-White rows inside its audited contract and
operational domain. A distinct coupon accrual, zero strike, 50-year expiry, or
Hull-White 50-year swap tenor is declared unsupported before pricing and uses
the specialized QuantLib Jamshidian reference. CIR does not use row-wise
selection: both available Premia finite-difference methods failed representative
payer and receiver audits, so Premia is globally recorded as
`available but not reliable` and QuantLib supplies all CIR swaption rows.

## Repository layout

Source prices remain under:

```text
datasets/model/<asset_class>/<model>/prices/.../<database_id>.json
```

Independent prices use the same relative hierarchy under:

```text
validation/datasets/price/<asset_class>/<model>/.../<database_id>.json
```

The catalogue directory contains `dataset.yaml` and `generator.cpp`, but no
`validation_report.json` and no `validation.ipynb` for a migrated dataset.

The YAML validation block is deliberately compact:

```yaml
validation:
  status: "available"
  verified: true
  dataset: "validation/datasets/price/<asset_class>/.../<database_id>.json"
```

Engine names, methods, versions, row provenance, tolerances, metrics, and bias
rules belong in the reference JSON, not in YAML.

## Reference JSON contract

The top-level identity mirrors the source dataset:

```json
{
  "database_id": "...",
  "catalog": "catalog/model/<asset_class>/[<family>/]<model>/prices/...",
  "url": "https://datasets.ai-factory.example/v1/validation/price/...json",
  "row_count": 1000,
  "model_dataset": {"id": "...", "catalog": "...", "url": "..."},
  "product_dataset": {"id": "...", "catalog": "...", "url": "..."},
  "source_fingerprints": {"price_results": "sha256:..."},
  "validation_policy_fingerprint": "sha256:..."
}
```

Fitted models also copy `curve_dataset`. `source_fingerprints` hashes the
semantic source prices, model parameters, product parameters, and curve
parameters when present. Timing fields are intentionally excluded, so a
benchmark-only change does not invalidate a correct price reference.

### Two independent fingerprint layers

The fingerprints answer two different questions:

- `source_fingerprints`: are these cached references still aligned with the
  exact generated prices and model, product, and curve parameters?
- `validation_policy_fingerprint`: were they accepted under the exact policy
  implemented today?

The policy fingerprint hashes a canonical document containing the core/stress
row counts, numerical tolerances, accepted-bias policy, and formatting-neutral
Python ASTs of the functions that compute row allowances, systematic bias,
verification metrics, and final publication status. A change to those rules
therefore makes the old cache fail closed even when all source datasets are
unchanged. Fixed-income references require this field; a missing, malformed,
or stale value is a validation failure.

SHA-256 provides stale-policy and accidental-tampering detection inside the
repository. It is not an authenticity signature: a party allowed to modify
both the validator and every reference cache can recompute the hashes. External
authenticity would additionally require a signature or trusted manifest.

### Reference-pricer hierarchy

`reference_pricers` contains ordered `core` and `stress` sections. Each section
contains exactly:

```json
{
  "row_count": 900,
  "premia": {"status": "..."},
  "quantlib_specialized": {"status": "..."},
  "quantlib_monte_carlo": {"status": "..."}
}
```

Supported statuses are:

- `available`: the engine is compatible with the exact validation contract;
- `available but not reliable`: the engine is callable but known not to provide
  a trustworthy reference for this implementation;
- `not_available`: no compatible engine exists.

Only an engine that actually produced rows receives details:

```json
{
  "status": "available",
  "id": "premia_black_scholes_european_call_specialized_pricer",
  "backend": "Premia",
  "backend_version": "19",
  "kind": "specialized_pricer",
  "method": "CF_Call",
  "row_priced": 900
}
```

An available but unused fallback contains only `{"status": "available"}`.
`row_priced` counts must exactly cover the regime and must agree with the
`reference_pricer_id` stored on every result row.

### Result rows

Every result is aligned with the source row and preserves `model_id`,
`product_id`, and `curve_id` when applicable:

```json
{
  "id": "000001",
  "model_id": "000001",
  "product_id": "000001",
  "reference_pricer_id": "...",
  "outputs": {
    "price": 0.055,
    "standard_error": 0.0001
  }
}
```

`standard_error` is omitted for deterministic references. Black-Scholes rows
also persist their exact comparison contract:

```json
"comparison": {
  "relation": "generated_at_least_reference",
  "allowance": 0.00017
}
```

The relation is one of `absolute`, `generated_at_least_reference`, or
`generated_at_most_reference`. Persisting the allowance preserves analytical
bounds and Monte-Carlo uncertainty without rerunning an external backend.

## Tolerances and verification

The default absolute comparison uses:

```text
absolute tolerance
+ relative tolerance * abs(reference price)
+ standard-error multiplier * hypot(source SE, reference SE)
```

The current defaults are `5e-7` absolute, `5e-5` relative, and five combined
standard errors. Aggregate signed errors are tested independently for a
material systematic bias. The JSON persists the policy and the exact core and
stress metrics; cache validation recomputes them and rejects any mismatch. It
also compares the persisted tolerances with the current expected tolerances,
so an old, more permissive threshold cannot remain silently authoritative.

A dataset passes only if every row passes and no unexpected systematic bias is
detected.

### Continuous Premia versus discrete CUDA monitoring

A compatible Premia pricer must still be called when it prices the continuous
version of a payoff monitored discretely by CUDA. The continuous price is a
useful independent analytical reference, and the comparison applies the proven
ordering:

- a discretely monitored knock-out is worth at least its continuous version;
- a discretely monitored knock-in is worth at most its continuous version;
- a discrete running maximum is at most the continuous running maximum;
- discrete no-touch and one-touch prices follow the corresponding missed-hit
  ordering;
- arithmetic Asians use a row-wise bound for the difference between the daily
  average and the continuous-time average.

These families naturally exhibit a systematic signed gap. Their verification
contains:

```json
"systematic_bias_policy": {
  "status": "accepted_expected",
  "explanation": "expected; discrete monitoring knocks out fewer paths than continuous monitoring"
}
```

The bias remains visible as `systematic_bias: true` in each affected regime.
It is accepted only because every row satisfies a documented mathematical
relation and the JSON gives a non-empty explanation. This mechanism must never
be used to excuse an unexplained pricing drift.

## Technical failures and fallbacks

The following are technical engine failures:

- backend process or protocol failure;
- non-finite price or standard error;
- documented engine-domain failure;
- finite output violating a no-arbitrage bound;
- an inconclusive rare-event comparison.

The failed row moves to the next engine and keeps its final
`reference_pricer_id`. The metadata counts how many rows each engine actually
priced. A numerical comparison failure is not a technical failure and cannot
fall through.

For Black-Scholes, Premia is primary on 25 product families. QuantLib Monte
Carlo supplies the four structured families for which Premia has no compatible
pricer and the few individual fallback rows rejected technically by Premia.

For CIR, Premia is explicitly `available but not reliable`; the specialized
QuantLib `CoxIngersollRoss.discountBondOption` implementation is therefore the
selected reference. This is different from a continuous/discrete bound: an
unreliable formula is not used as a comparison reference.

## Commands

Regenerate every fixed-income reference:

```bash
python -m validation.model.fixed_income.generate_all_references
```

Regenerate every Black-Scholes reference:

```bash
python -m validation.model.equity.black_scholes.generate_all_references
```

Both commands accept repeatable family selectors (`--model` for fixed income,
`--product` for Black-Scholes). Regeneration requires the declared external
backends and updates the cache and compact YAML only after comparison.

Routine validation is cache-only. For example:

```bash
python -m validation.model.equity.black_scholes.european_call \
  datasets/model/equity/markovian/black_scholes/prices/european_calls/black_scholes_01__european_calls_01__01.json \
  validation/datasets/price/equity/black_scholes/european_calls/black_scholes_01__european_calls_01__01.json \
  --require-verified
```

Add `--generate` to that product command only when an explicit backend rerun is
intended.

When only the validation implementation or tolerances change, do not rerun the
external pricers. Revalidate the immutable fixed-income reference prices and
publish the new policy fingerprints with:

```bash
python -m validation.model.fixed_income.refresh_policy_fingerprints
```

This command is cache-only. It first verifies the source fingerprints, then
recomputes all row decisions, core/stress metrics, and bias checks under the
current policy. It writes the new verification block and policy fingerprint
only when both regimes pass. If source prices or parameters changed, or if the
selected reference engine/method must change, this command is insufficient:
the affected reference must be regenerated explicitly with `--generate`.

CTest registers migrated datasets with the `cached_reference` label:

```bash
ctest --test-dir build -L cached_reference --output-on-failure
```

These tests have a short timeout because they only validate JSON, fingerprints,
metrics, provenance, and YAML consistency.

## Fail-closed publication checks

Routine validation rejects publication if any of the following occurs:

- the cache is absent or malformed;
- source identity or row order differs;
- a semantic fingerprint is stale;
- the validation-policy fingerprint is absent, malformed, or stale;
- persisted tolerances differ from the current expected policy;
- core or stress is not exactly 900/100 rows;
- pricer hierarchy order or `row_priced` provenance is inconsistent;
- a price, standard error, allowance, or metric is invalid;
- persisted metrics differ from the current comparison;
- expected bias has no explanation;
- either regime fails;
- YAML status, verification flag, or cache path disagrees with the JSON.

## Adding or migrating a validator

1. Reconstruct the exact model and product conventions in Premia and QuantLib.
2. Declare every hierarchy slot without inventing an unavailable pricer.
3. Use Premia first whenever it supplies a valid exact price or a proven
   continuous/discrete bound.
4. Fall back row-wise only on technical failures.
5. Persist all 1,000 aligned references with stable pricer IDs and backend
   versions.
6. Explain every accepted systematic bias in English.
7. Publish only after core and stress both pass.
8. Replace adjacent reports/notebooks with the compact YAML cache link.
9. Add cache-only unit and CTest coverage that blocks external-backend imports.
10. Require `validation_policy_fingerprint` and test missing, malformed, and
    stale-policy failures.
11. Keep direct Premia and QuantLib modules as regeneration and diagnostic
    tools, not routine publication tests.

## Standard policy-change procedure

1. Change the comparison implementation or current tolerances in source.
2. Run the policy and fail-closed unit tests.
3. Run `refresh_policy_fingerprints`; never edit hashes or verification metrics
   by hand.
4. Review the JSON diff: only the policy fingerprint and policy-dependent
   verification fields may change.
5. Run `ctest --test-dir build -L cached_reference --output-on-failure`.
6. If any cached price fails the new policy, stop and investigate. Regenerate
   with the independent backend only when a fresh external reference is
   genuinely required.

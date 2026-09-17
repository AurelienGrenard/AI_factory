# Deep-pricing learning contract

This contract defines reproducible neural pricing experiments over published
AI Factory price and price-gradient datasets. Simulation and pricing remain
owned by C++/CUDA. `learning/` contains only reusable implementation. Concrete
dataset choices, campaign configurations, notebooks and reports live below
`experiments/`. Re-creatable caches and run outputs use the directory selected
by the experiment and remain ignored by Git; ad hoc commands default to
`artifacts/learning/`.

## Dataset-independent boundary

The pricing-data adapter resolves every `*_dataset` reference in a price JSON,
joins result `model_id`, `curve_id` and `product_id` fields to their parameter
rows, and emits one ordered tensor schema. Feature names are role-qualified,
such as `model.spot` and `product.maturity`. The network, trainer and universal
metrics receive tensors and this schema; they do not branch on a concrete
model or product name.

Scalar numeric parameters are the first supported feature encoding. Boolean,
categorical, fixed-vector and variable-sequence inputs require a named encoder
before use. The adapter fails on these values rather than silently inventing an
encoding. Feature order is sorted within the stable role order model, curve,
product. Every row must expose exactly the same fields and finite values.

Price-delta publications declare one `sensitivity.parameter` and an
`outputs.delta`. Price-gradient publications declare an ordered
`sensitivity.parameters` list and matching `outputs.gradients` and optional
`outputs.gradient_standard_errors` objects. The adapter resolves every
parameter to exactly one role-qualified input. Missing, duplicate, reordered,
zero-match or multiple-match coordinates are errors. The training engine sees
both formats through the same ordered derivative list.

## Prepared data

Large pretty-printed JSON is parsed incrementally. Referenced parameter bases
are small lookup tables; price results are converted once to NumPy `.npy`
arrays and subsequently opened read-only with memory mapping. The default cache
is:

```text
artifacts/learning/cache/<content-fingerprint>/
```

Its key covers the price JSON bytes, every referenced input JSON and the cache
schema version. Moving identical files does not change their content identity.
The manifest records physical source paths for inspection but paths do not
define experiment equality.

The cache contains features, values, optional derivatives, optional sampling
standard errors and source-row ordinals. It is recreatable and must not be
committed.

## Fixed evaluation and nested training samples

For Cartesian pricing, complete model-parameter groups first receive stable
train, validation and test membership from a seeded SplitMix64 priority. A
one-million-row model/product Cartesian product therefore uses 800 complete
model surfaces for training, 100 for validation and 100 for final testing.
The 90/10 core/stress model regime is preserved in each partition. No held-out
Heston parameter vector appears in training, and every held-out vector retains
its complete 1,000-product surface for calibration tests. A row split remains
available explicitly for datasets without a suitable grouping role.

Within the training candidates, a second seeded priority selects the requested
number of rows. Selection takes the `N` lowest priorities. With the same seeds,
30,000 rows are consequently a subset of 100,000, which is a subset of 500,000.
The physical indices and SHA-256 digest of every split are stored in the run.
The selection is uniformly distributed over physical result rows and never
takes the first `N` rows of a Cartesian product.

`train_size` is the exact number of optimizer-visible rows. `null` or CLI
`all` selects every candidate left after validation and test. The standard
one-million-row experiment consequently scales through 30k, 100k and 500k to
an exact 800k maximum. An exact one-million-row training experiment would need
a separate compatible validation/test publication.

## Transform and derivative units

Means and standard deviations are fitted only on the selected training rows.
Constant coordinates keep scale one. For normalized input and value

```text
x_norm = (x - mean_x) / scale_x
y_norm = (y - mean_y) / scale_y
```

a raw derivative target is converted as

```text
dy_norm / dx_norm = (scale_x / scale_y) * dy / dx.
```

This conversion belongs to the shared transform and is tested independently.
In the current equity bases, spot is constant at one. Delta supervision can
constrain the local derivative on that hyperplane, but does not by itself
qualify predictions at distant spot levels.

## Composable methods

Networks and loss terms use explicit registries. A configuration composes a
weighted list such as:

```yaml
loss:
  terms:
    - name: value_mse
      weight: 1.0
    - name: gradient_mse
      weight: 0.1
    - name: l2_parameters
      weight: 1.0e-6
```

Adding a penalty means implementing and registering a component that consumes
`LossContext`; the training loop remains unchanged. Gradient-consuming terms
declare that need so the engine creates the required higher-order autograd
graph only for methods that use it. Network builders follow the same rule.
The context exposes raw and normalized inputs, prices and derivatives under
distinct names, plus epoch and global step. Penalties therefore declare their
units explicitly and may implement schedules without reaching into the loop.
Experiment `extensions` list importable Python modules that can register these
components, so a researcher need not edit the common registry or trainer.

For example, an experiment module may contain:

```python
@register_loss("my_penalty")
def build_my_penalty(specification):
    return MyPenalty(specification)
```

and its YAML selects it with:

```yaml
extensions: [my_experiments.penalties]
loss:
  terms:
    - {name: value_mse, weight: 1.0}
    - {name: my_penalty, weight: 0.01}
```

GANs will reuse data, schemas, transforms, networks, registries and run
evidence. Their alternating generator/discriminator optimization will own a
separate engine rather than adding GAN-specific branches to the deep-pricing
loop.

The controlled price-versus-Sobolev comparison uses the same price-delta
publication, selected rows, transform, architecture, initialization, optimizer
and budget. The supervised run ignores published delta during optimization;
both runs are still evaluated against the same held-out price and delta rows.

## Run evidence and comparison

Each run writes below its configured output directory. The ad hoc CLI defaults
to `artifacts/learning/runs/`; a versioned experiment configuration may use an
ignored `output/` directory beside the campaign:

```text
initial_model.pt
checkpoint.pt
best_model.pt
last_model.pt
history.json
state.json
run.json
```

`run.json` records content fingerprints, the tensor schema, split counts and
digests, transform, full configuration, PyTorch/device information, elapsed
time, throughput, parameter count, peak GPU memory and held-out metrics. The
checkpoint is replaced atomically after every epoch, so an interrupted run can
resume at the next epoch. The comparison command rejects runs whose dataset
fingerprint or validation/test digests differ.

Universal V1 metrics cover price and derivative MAE, RMSE, absolute-error
quantiles, maximum error and mean relative error. When Monte Carlo standard
errors are published, errors are also expressed in standard-error units. The
derivative report includes sign accuracy. Product-specific financial
constraints belong to separately registered evaluation suites; they must not
introduce product branches into the common trainer.

## Future deep-calibration benchmark

The grouped test split deliberately retains 100 complete, unseen Heston
surfaces. A calibration benchmark will select a test model row, take its
published prices over the common product grid, and optimize model parameters
through each frozen deep pricer. It will compare price-only and Sobolev models
on final surface residual, normalized parameter error, convergence rate,
iterations, runtime and robustness across several starting points.

Parameters will be optimized through constraint-preserving coordinates:
positive quantities through logarithmic variables and correlation through a
bounded transform. Spot, rates or dividend yield may be fixed when the chosen
surface does not identify them separately. The benchmark will report such
identifiability choices instead of interpreting one low price residual as
unique parameter recovery. Test surfaces and all starting points remain common
to every pricer; no test model group may enter training or hyperparameter
selection. The present Sobolev target is spot delta, whereas calibration uses
derivatives with respect to Heston parameters. Better calibration from delta
supervision is therefore a hypothesis to measure, not an automatic consequence
of the Sobolev loss.

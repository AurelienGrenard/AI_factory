# Learning

`learning/` owns Python/PyTorch training and evaluation. It sits beside `src/`:
the C++/CUDA code in `src/` remains responsible for simulation, pricing and
published datasets. Training reads those datasets without changing their
generators or numerical contracts. A C++/CUDA PyTorch extension belongs here
only if a measured training bottleneck justifies it.

The first generative task is **terminal sampling**. For a model with parameter
vector `theta`, business-day maturity `maturity_days`, year fraction `T` and
terminal observable vector `Y_T`, learn a conditional distribution
`G(theta, T, z) -> Y_T`. `T` is an explicit input, not inferred from a fixed
dataset name. The observable names and semantics come from each model's sample
contract and adjacent YAML; for example, the CIR++ `state` is its unshifted
factor, not a zero-coupon price or curve-adjusted rate.

`common/terminal_samples.py` is the first shared brick. It reads the existing
large, line-streamed `samples_01` and `samples_02` JSON without loading three
million rows into memory, checks the row schema, and exposes stable
train/validation/test assignments. For `samples_01`, all 250 paths sharing one
parameter row go to the same split. Consumers must preserve this grouping and
must keep `T` in the conditioning vector. The reader is independent of
PyTorch so that data-contract checks can run without a GPU.

Conditional terminal GANs are implemented under `gan/`. The generic
WGAN-GP engine learns `G(theta, T, z) -> Y_T` with separate generator and
critic learning rates, configurable critic steps, configurable Gaussian or
bounded-uniform latent vectors, logistic or WGAN-GP objectives, constant, cosine
or exponential scheduling, an exponential moving average of the generator,
deterministic checkpoints and live JSON progress.
Generator and critic registries infer their dimensions from the terminal
schema. A configuration may select any non-empty subset of published
observables, so one source dataset can train separate scalar and joint
generators without model-specific code.

Optional, model-independent anti-collapse terms include mode-seeking
regularization (output sensitivity to paired latent draws) and normalized batch
moment matching. Validation reports generated-to-real standard-deviation ratios
and can penalize collapsed checkpoints during selection. These mechanisms are
disabled by default and must be qualified by an experiment under `work/`.

The terminal cache streams the large JSON once into content-addressed NumPy
memory maps. Published parameter-group splits are preserved, transformations
are fitted on training rows only, and observable constraints are declared as
configuration transforms rather than inferred from a model name. Universal
evaluation covers moments, quantiles, covariance, multiscale MMD, sliced
Wasserstein distance, conditional slices and a direct noise-diversity
diagnostic. Repricing via Monte Carlo still requires a payoff whose observables
are present; arbitrary path-dependent payoffs cannot be repriced from `Y_T`
alone.

Train, monitor and sample a configured run with:

```sh
python3 -m learning.gan.train --config my_terminal_gan.yaml
python3 -m learning.gan.watch artifacts/learning/gan-runs/<run>
python3 -m learning.gan.sample artifacts/learning/gan-runs/<run> conditions.json
```

A completed run can be evaluated on an independent compatible publication
without refitting its preprocessing:

```sh
python3 -m learning.gan.evaluate   artifacts/learning/gan-runs/<run> datasets/model/.../samples_02.json
```

Deep pricing is implemented under `deep_pricing/`. Its reader joins published
price rows to their model, curve and product parameters, creates a content-
addressed memory-mapped cache, and presents one model-independent tensor
contract. Price-only, price-plus-gradient and price-plus-gradient-plus-diagonal-
Hessian methods compose registered loss terms over the same training engine.
Diagonal second derivatives use exact PyTorch autograd on the published
coordinates and are reported in raw units. The normative rules, including nested
training subsets and fixed held-out rows, are in the
[deep-pricing learning contract](../docs/deep-pricing-learning-contract.md).

The complete configurable example is part of the optional local Heston study at
`work/experiments/equity/heston/european_call_price_delta_v1/training/single_run_example/config.yaml`.
Run 30,000-row
supervised and Sobolev experiments on the same price-delta publication with:

```sh
python3 -m learning.deep_pricing.train \
  --dataset datasets/model/equity/markovian/heston/price_delta/european_calls/heston_01__european_calls_01__01_cartesian_price_delta.json \
  --method supervised --train-size 30000

python3 -m learning.deep_pricing.train \
  --dataset datasets/model/equity/markovian/heston/price_delta/european_calls/heston_01__european_calls_01__01_cartesian_price_delta.json \
  --method sobolev --train-size 30000
```

For fine control, copy that experiment configuration and run:

```sh
python3 -m learning.deep_pricing.train --config my_experiment.yaml
```

`--sample-size` is accepted as a CLI alias of `--train-size`. The validation
and test rows remain fixed when training size changes. `all` selects every
training candidate left after those holdouts; it does not reuse held-out rows.

Compare completed runs with:

```sh
python3 -m learning.deep_pricing.compare \
  artifacts/learning/runs/<supervised-run> \
  artifacts/learning/runs/<sobolev-run>
```

The reproducible Heston comparison is split into an architecture pilot and a
dataset-size campaign:

```sh
python3 -m learning.deep_pricing.campaign \
  work/experiments/equity/heston/european_call_price_delta_v1/training/architecture_pilot/campaign.yaml

python3 -m learning.deep_pricing.campaign \
  work/experiments/equity/heston/european_call_price_delta_v1/training/dataset_scaling/campaign.yaml
```

Each price-only run is paired with Sobolev runs at gradient weights 0.1 and 1.
Campaigns skip completed runs and restart an interrupted run from its latest
epoch checkpoint. The generated local notebook at
`work/experiments/equity/heston/european_call_price_delta_v1/analysis/deep_pricing_results.ipynb`
uses validation surfaces for model selection and reserves test surfaces for the
final report.

The focused spot/strike ablation compares the raw ten-input map with the
homogeneous European-call representation `P(S,K,z) = S p(K/S,z)` on the same
30,000 training rows and held-out surfaces:

```sh
python3 -m learning.deep_pricing.campaign \
  work/experiments/equity/heston/european_call_price_delta_v1/training/spot_strike_ablation/campaign.yaml
```

The homogeneous wrapper is deliberately restricted to datasets whose selected
training spots all equal one. It removes spot from the inner MLP, replaces
strike by moneyness, reconstructs the price outside the MLP and lets PyTorch
differentiate that complete expression. No price or delta row is regenerated.
The executed local notebook at
`work/experiments/equity/heston/european_call_price_delta_v1/training/spot_strike_ablation/analysis.ipynb`
reports the paired validation and held-out test results.

Path generation and signature GANs are later work; they require a trajectory
dataset with an explicit observation grid.

From the repository root, inspect a published terminal sample dataset with:

```sh
python3 -m learning.common.terminal_samples \
  datasets/model/fixed_income/cir_plus_plus/samples/samples_01.json \
  --limit 1000
```

This reads 1,000 rows and reports the schema and split counts. It resolves the
sample shape from the `recipe.yaml` referenced by the JSON `catalog` field; the
public JSON keeps only its minimal envelope and `results`. It does not train a
model or alter the dataset.

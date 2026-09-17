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

The next vertical slice will add a PyTorch conditional generator and
discriminator, reusable networks and training utilities under `common/`, and
an evaluation protocol under `learning/`. The protocol will fix dataset
identity, splits, preprocessing, seeds, compute budget, and metrics before
comparing methods. Terminal-distribution tests should compare conditional
moments, quantiles and model-specific constraints; repricing via Monte Carlo
requires a payoff whose required observables are present. Arbitrary path
dependent payoffs cannot be repriced from `Y_T` alone.

Deep pricing is implemented under `deep_pricing/`. Its reader joins published
price rows to their model, curve and product parameters, creates a content-
addressed memory-mapped cache, and presents one model-independent tensor
contract. Price-only and price-plus-gradient methods compose registered loss
terms over the same training engine. The normative rules, including nested
training subsets and fixed held-out rows, are in the
[deep-pricing learning contract](../docs/deep-pricing-learning-contract.md).

The complete configurable example is part of the corresponding Heston study:
[`config.yaml`](../experiments/equity/heston/european_call_price_delta_v1/training/single_run_example/config.yaml).
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
  experiments/equity/heston/european_call_price_delta_v1/training/architecture_pilot/campaign.yaml

python3 -m learning.deep_pricing.campaign \
  experiments/equity/heston/european_call_price_delta_v1/training/dataset_scaling/campaign.yaml
```

Each price-only run is paired with Sobolev runs at gradient weights 0.1 and 1.
Campaigns skip completed runs and restart an interrupted run from its latest
epoch checkpoint. The generated
[`deep_pricing_results.ipynb`](../experiments/equity/heston/european_call_price_delta_v1/analysis/deep_pricing_results.ipynb)
uses validation surfaces for model selection and reserves test surfaces for the
final report.

The focused spot/strike ablation compares the raw ten-input map with the
homogeneous European-call representation `P(S,K,z) = S p(K/S,z)` on the same
30,000 training rows and held-out surfaces:

```sh
python3 -m learning.deep_pricing.campaign \
  experiments/equity/heston/european_call_price_delta_v1/training/spot_strike_ablation/campaign.yaml
```

The homogeneous wrapper is deliberately restricted to datasets whose selected
training spots all equal one. It removes spot from the inner MLP, replaces
strike by moneyness, reconstructs the price outside the MLP and lets PyTorch
differentiate that complete expression. No price or delta row is regenerated.
The executed
[`analysis.ipynb`](../experiments/equity/heston/european_call_price_delta_v1/training/spot_strike_ablation/analysis.ipynb)
reports the paired validation and held-out test results.

Path generation and signature GANs are later work; they require a trajectory
dataset with an explicit observation grid.

From the repository root, inspect a published terminal sample dataset with:

```sh
python3 -m learning.common.terminal_samples \
  datasets/model/fixed_income/cir_plus_plus/samples/samples_01.json \
  --limit 1000
```

This reads 1,000 rows and reports the schema and split counts. It does not
train a model or alter the dataset.

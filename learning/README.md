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

Deep pricing will likewise live under `learning/`, with separate price-only
and price-plus-gradient training on the same price-delta dataset and a common
held-out evaluation. Path generation and signature GANs are later work; they
require a trajectory dataset with an explicit observation grid.

From the repository root, inspect a published terminal sample dataset with:

```sh
python3 -m learning.common.terminal_samples \
  datasets/model/fixed_income/cir_plus_plus/samples/samples_01.json \
  --limit 1000
```

This reads 1,000 rows and reports the schema and split counts. It does not
train a model or alter the dataset.

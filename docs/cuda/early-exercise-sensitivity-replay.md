# Early-exercise sensitivity replay

American options and Bermudan swaptions expose two finite-difference
estimators. Both reuse the central Philox innovations and the central
Longstaff--Schwartz fit. They differ only in the stopping rule replayed on a
bumped node.

## Frozen exercise time

Let \(\tau_0(\omega)\) be the exercise time selected on the central path. A
bumped node \(n\) is valued at that same stopping time:

\[
  V_n^{\mathrm{FET}}(\omega)
  = D_n(0,\tau_0(\omega))
    g_n(X_{n,\tau_0(\omega)}(\omega)).
\]

The stored central state is therefore the exercise trace, one exercise index
per path (plus the central initial-exercise decision for American products).
The bumped node never evaluates a continuation regression.

## Frozen regression policy

At exercise date \(t_k\), let the central regression snapshot be

\[
  \widehat C_{0,k}(x)
  = \sum_{j=1}^{p} \widehat\beta_{0,k,j}
      \phi_j(N_{0,k}(x)).
\]

The coefficients \(\widehat\beta_{0,k}\), solve status, and feature
normalization \(N_{0,k}\) are captured from the central LSM pass. For every
bumped node, the state and immediate payoff remain node-specific:

\[
  I_{n,k}(\omega)
  = \mathbf 1_{\{g_n(X_{n,k}(\omega))
      > \widehat C_{0,k}(X_{n,k}(\omega))\}}.
\]

The strict inequality is the common project convention. The node stops at the
first date for which \(I_{n,k}=1\). No bumped regression is fitted.

For an American decision at \(t_0\), the central continuation threshold
\(C_{0,0}\) is captured once, while the bumped payoff is recomputed:

\[
  I_{n,0}=\mathbf 1_{\{g_n(X_{n,0})>C_{0,0}\}},\qquad
  V_n=I_{n,0}g_n(X_{n,0})+(1-I_{n,0})C_n^{\mathrm{FP}}.
\]

Thus one bumped node may exercise initially while another continues. Bermudan
products whose schedule starts after \(t_0\) need no special initial branch.

## Execution strategies

Both replay conventions are available through both CUDA execution strategies:

| Execution | Work assignment | Main memory retained |
|---|---|---|
| Mono-kernel | One path thread evaluates all local stencil nodes for one sensitivity | Central LSM workspace plus the replay state |
| Node graph | Unique nodes are prepared, evaluated in parallel, then derivatives are reconstructed | Deduplicated graph, prepared node rows, node values and reconstruction moments |

The mixed node graph extends the second strategy to the complete selected
Hessian. Shared axis and cross nodes are evaluated only once.

For frozen exercise time, the graph evaluator replays the captured exercise
trace. For frozen regression policy, each graph node replays the captured
coefficient/status snapshots and the central normalizations against its own
state and payoff. Device kernels are template-specialized; the replay choice is
dispatched on the host and does not add a per-path dynamic branch.

## Public API and provenance

Existing launchers without a replay argument retain the historical
`frozen_exercise_time` behavior. The corresponding `*_with_replay` launchers
and workspace planners accept `ExerciseReplayStrategy` and support:

- `frozen_exercise_time`;
- `frozen_regression_policy`.

The same value must be passed to the workspace planner and launcher. Dataset
recipes record it as the top-level `exercise_replay` field. The generated
dataset repeats it in its public metadata, and `generation.yaml` records it
under `execution.exercise_replay`. Frozen-policy datasets require a distinct
`*_frozen_policy` identity; changing the estimator cannot silently overwrite a
frozen-exercise dataset.

These are finite-difference estimators of two frozen stopping constructions.
The implementation does not claim a general theorem guaranteeing an unbiased
second derivative of the fully refitted optimal-stopping value.
## Performance qualification

The reproducible matrix is driven by
`tools/performance/run_early_exercise_replay.py`. It compares both replay
rules and both diagonal execution strategies on Black--Scholes, Heston, Bates,
CIR, G2 and G2++/Svensson. The versioned SM89 evidence is available in the
[frozen replay report](../../tests/performance/reports/frozen-replay-sm89-2026-09-30-final/summary.md).

The reported public-API medians use 4,096 paths, 20 warmups, 11 samples and 64
complete calls per sample. The separate capture/replay times come from Nsight
Systems, while `cuobjdump` and runtime diagnostics provide registers, stack,
local memory and theoretical occupancy for the exact launched symbols.

On this workload frozen policy makes mono replay more expensive than frozen
exercise, especially for Heston, G2 and G2++/Svensson. The node graph recovers
most or all of that cost by parallelizing nodes and lowering register pressure:
G2 falls from 246 to 138 registers per thread and G2++/Svensson from 254 to
150. The tradeoff is explicit caller workspace: graph peak memory is roughly
2.4--4.8 times the frozen-exercise mono baseline in this matrix. These figures
qualify the implementation architecture; they are not a production-throughput
claim.

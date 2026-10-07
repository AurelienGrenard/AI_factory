# Rough Monte Carlo sensitivity graph

The rough sensitivity graph uses the same row-local finite-difference stencils,
CRN keys, pathwise moment reduction and finalization as the Markovian node
graph. Each output row builds a central scenario and the required bumped nodes.
The graph evaluates all nodes with the same Philox key and path index before
reconstructing each derivative sample. Standard errors therefore include the
covariance between bumped prices.

## Pricing families

- Rough Heston and quadratic rough Heston use host-prepared N-factor lifts.
  Model-node dynamics are cached by represented model parameters. Positive
  exponential kernel fits are cached by H and now fitted in parallel across
  distinct H endpoints before row execution.
- Rough Bergomi, rough SABR, rough Stein–Stein and log-modulated rough Bergomi
  use the Gaussian hybrid FFT. A spectrum is cached by kernel parameters,
  fixed numerical `dt` and FFT-length bucket. Convolution groups also include
  the exact step count: bumped maturity nodes may share a spectrum, but must
  have their own path-length convolution.
- The terminal and path payoff variants share graph construction, workspace
  sizing and moment reconstruction. Path handlers with unchanged model
  dynamics share one path simulation in bounded groups of eight. Individual
  handlers retain their own barrier or exercise stopping state.

Both engines use `dt = 1/504` with two simulation steps per contractual day
when the caller supplies that time configuration. Dense, regular and static
calendar path policies are supported. Product-specific headers expose graph
aliases for the rough equity products, including the legacy delta products.

## Resource strategy

The Gaussian engine prepares each distinct FFT spectrum once, then processes
one row and a bounded path chunk at a time. N-factor execution batches up to
16 rows and 8192 paths per chunk under its workspace limit. The path observer
batch caps per-thread handler storage at eight; additional equivalent nodes
are handled in another path traversal. Product and multiplicative-spot bumps
reuse central dynamics and the central FFT spectrum.

Representative single-run SM89 measurements (16 Cartesian rows, 8192 paths):

| Product and grid | Graph, GPU | Price only, GPU | Graph workspace |
| --- | ---: | ---: | ---: |
| Rough Bergomi Asian, 126 steps, spot and strike first + diagonal second | 33.2 ms | 24.0 ms | 1.21 MB |
| Rough Heston 7-factor Asian, 126 steps, same sensitivities | 4.1 ms | 6.0 ms | 4.01 MB |
| Rough Heston 7-factor European, 504 steps, four sensitivities | 14.6 ms | 16.2 ms | 7.43 MB |

The N-factor European graph's three distinct H fits took about 156 ms after
parallelization, versus 382 ms sequentially on the same machine. These are
indicative timings from one run; they are not throughput distributions. The
price-only lift geometry launches fewer path blocks, which explains why a
graph with additional outputs can be faster in this small comparison.

## Current boundary

The C++ graph currently supports dense first derivatives and diagonal second
derivatives. It rejects selected subsets and mixed Hessians explicitly. The
experimental causal FFT Rough Heston pricer has no graph sensitivity path yet.
The historical rough spot-delta launchers remain as compatibility entry
points and delegate to a first-order rough graph with a single spot stencil.
The lift adapter reuses already prepared dynamics; FFT compatibility calls
reuse the supplied workspace when it is large enough for the graph plan. The
old dedicated FFT delta path engine has been removed. The public launcher
surface and capability manifest still expose the compatibility bindings for
existing experiments and dataset generation.

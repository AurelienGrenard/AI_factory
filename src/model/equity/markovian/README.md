# Markovian equity models

This folder groups equity model families whose defining state is Markovian.
The folder is an organizational boundary only: public C++ namespaces, CMake
target names, catalog identifiers and dataset paths do not include
`markovian`.

Numerical lifts are not classified here merely because their simulator is
Markovian. In particular, the fixed-factor lift of rough Heston remains under
`../rough/rough_heston` because it approximates a rough model.

SABR equity evolves the risk-neutral spot in its Lamperti coordinate and
stores initial log-return volatility, converting it once to the dimensional
SABR alpha with `alpha_0=sigma_0*S0^(1-beta)`. Heston 3/2 steps the reciprocal
CIR process instead of the super-linear variance. The original Stein--Stein
model uses a zero-mean arithmetic OU volatility and an exact OU endpoint
coupled to the Euler equity increment.

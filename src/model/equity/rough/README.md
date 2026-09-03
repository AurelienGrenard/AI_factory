# Rough equity models

This folder groups equity models defined by rough or Volterra dynamics,
independently of the numerical scheme used to simulate them. It therefore
contains both Gaussian-Volterra hybrid-FFT implementations and Markovian
multi-factor approximations such as rough Heston.

Quadratic rough Heston belongs here as well, even though its production
simulator uses a finite-factor Markovian lift. Public C++
namespaces, CMake target names, catalog identifiers and dataset paths do not
include the `rough` organizational component.

The Gaussian-Volterra engine currently supports rough Bergomi,
log-modulated rough Bergomi, rough SABR and rough Stein--Stein. Their model
policies provide only the driver parameters and the transformation from the
Gaussian driver to the equity state; schedules, observation handlers and all
21 non-American product payoffs are shared. Rough Heston and quadratic rough
Heston use the same product layer through prepared 2/3/7-factor dynamics.

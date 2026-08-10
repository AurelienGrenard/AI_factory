# Shared mean-reverting Gaussian formulas

## Role and reference

This directory contains model-level mathematics shared by several fixed-income
models. It is distinct from repository-wide `src/common`: these helpers are
specific to mean-reverting Gaussian factors.

## Files

[`mean_reverting_gaussian.cuh`](mean_reverting_gaussian.cuh) supplies stable conditional moments for one
centered Ornstein–Uhlenbeck factor. It is used by Ornstein–Uhlenbeck, Vasicek,
and both factors of G2/G2++.

## Dataset row

There is no dataset row in this directory. Callers pass mean reversion,
volatility, interval length, and—when already available—the exponential decay.

## Prepared parameters and state

The helpers return scalar loadings and variances rather than owning a model or
state. This keeps each calling model's public structures explicit while
centralizing only identical mathematics.

## Dynamics interface

The file provides the decay, exact endpoint variance, integral loading,
integral variance, and state/integral covariance needed to assemble exact
Gaussian transitions. Variants accepting precomputed decay let callers reuse
one `exp(-a dt)` across several moments.

## Random-number strategy

These functions are deterministic and consume no random numbers. The calling
model applies the resulting loadings to normals from its single path-local
Philox sequence.

## Pricing kernels

There are no kernels or launchers here. OU, Vasicek, G2, Hull–White, and G2++
consume the formulas through their own dynamics or analytics.

## Memory and numerical policy

Small-time series are used where direct expressions would subtract nearly
equal numbers, especially for integral variances of order `dt^3`. Results stay
FP32 because they are per-path coefficients; the formulas avoid extra model
state and repeated transcendental work. Fast-math is forbidden.

## American and Bermudan options

Not applicable; this directory only supplies transition moments.

Related navigation: [Ornstein–Uhlenbeck](../ornstein_uhlenbeck/),
[Vasicek](../vasicek/), [G2](../g2/), [Hull–White](../hull_white/), and
[G2++](../g2_plus_plus/).

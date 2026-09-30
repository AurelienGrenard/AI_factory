# Flat curve

`FlatCurveParameters` represents a continuously compounded zero curve with
one finite coordinate, `rate`:

```math
z(0,T)=f(0,T)=r,\qquad P(0,T)=\exp(-rT).
```

The implementation exposes the same analytical and dataset contracts as the
Nelson--Siegel and Svensson curve families. Negative rates are valid. The
`curve.rate` sensitivity is a parallel initial-curve shift; it is not an
independent bump of a fitted short-rate model's initial stochastic factor.

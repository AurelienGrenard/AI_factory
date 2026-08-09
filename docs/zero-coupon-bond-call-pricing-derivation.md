# Zero-coupon bond call pricing under OU and Hull-White

Let \(t\) be the valuation date, \(S\) the option expiry, \(T>S\) the bond
maturity, and \(K\) the strike. The call payoff at \(S\) is

\[
H_S=(P(S,T)-K)^+.
\]

The derivation first treats the standalone Ornstein-Uhlenbeck short-rate
model, then shows why Hull-White has the same option formula.

## 1. Use The Expiry Bond As Numeraire

Under the money-market risk-neutral measure \(\mathbb Q\), with
\(B_u=\exp(\int_0^u r_v\,dv)\),

\[
C_t=\mathbb E_t^{\mathbb Q}
\left[\frac{B_t}{B_S}H_S\right].
\]

Choose \(P(u,S)\) as numeraire. Its normalized discounted price defines the
\(S\)-forward measure:

\[
L_u=
\left.\frac{d\mathbb Q^S}{d\mathbb Q}\right|_{\mathcal F_u}
=\frac{P(u,S)/B_u}{P(0,S)/B_0}.
\]

Conditional Bayes moves the random discount into the new path weights:

\[
\boxed{
C_t=P(t,S)\,\mathbb E_t^{\mathbb Q^S}
\left[(P(S,T)-K)^+\right].
}
\]

This change of numeraire is model-independent.

## 2. Identify The Density Dynamics

In the standalone OU model,

\[
dr_u=-a r_u\,du+\sigma\,dW_u^{\mathbb Q},
\qquad
\beta(u,U)=\frac{1-e^{-a(U-u)}}{a}.
\]

The bond is exponential-affine:

\[
P(u,S)=\exp\bigl(A(u,S)-\beta(u,S)r_u\bigr).
\]

Ito's formula gives the relative bond diffusion
\(-\sigma\beta(u,S)dW_u^{\mathbb Q}\). Dividing by the money-market account
removes the risk-neutral drift, hence

\[
\frac{dL_u}{L_u}=-\sigma\beta(u,S)dW_u^{\mathbb Q},
\]

or

\[
L_u=\mathcal E\left(
-\int_0^u\sigma\beta(v,S)dW_v^{\mathbb Q}
\right).
\]

## 3. Apply Girsanov

The density above implies that

\[
dW_u^{\mathbb Q}
=dW_u^{\mathbb Q^S}-\sigma\beta(u,S)du.
\]

Therefore, under the \(S\)-forward measure,

\[
dr_u=
\left[-ar_u-\sigma^2\beta(u,S)\right]du
+\sigma dW_u^{\mathbb Q^S}.
\]

The measure change modifies only the drift: the equation remains linear and
Gaussian.

## 4. Obtain The Conditional Lognormal Law

Conditionally on \(\mathcal F_t\), \(r_S\) is Gaussian with variance

\[
q_{t,S}=\sigma^2\frac{1-e^{-2a(S-t)}}{2a}.
\]

Define the bond forward

\[
F_u=\frac{P(u,T)}{P(u,S)},
\qquad F_S=P(S,T).
\]

At expiry, the affine bond map gives

\[
\log F_S=A(S,T)-\beta(S,T)r_S.
\]

Thus \(F_S\mid\mathcal F_t\) is lognormal, with conditional log-variance

\[
\boxed{
\nu^2
=\beta(S,T)^2q_{t,S}
=\beta(S,T)^2\sigma^2
  \frac{1-e^{-2a(S-t)}}{2a}.
}
\]

## 5. Use Martingality To Fix The Mean

Because \(P(u,S)\) is the numeraire, \(F_u\) is a
\(\mathbb Q^S\)-martingale. Hence

\[
\mathbb E_t^{\mathbb Q^S}[F_S]=F_t,
\qquad
F_t=\frac{P(t,T)}{P(t,S)}.
\]

The conditional law is therefore completely characterized as

\[
\boxed{
F_S=F_t\exp\left(-\frac12\nu^2+\nu Z\right),
\qquad Z\sim\mathcal N(0,1),
}
\]

where \(Z\) depends only on Brownian increments after \(t\) and is independent
of \(\mathcal F_t\).

## 6. Integrate The Payoff

The remaining expectation is the standard Black lognormal integral:

\[
\boxed{
C_t=P(t,T)\Phi(d_1)-K P(t,S)\Phi(d_2),
}
\]

with

\[
d_1=
\frac{\log\left(P(t,T)/(K P(t,S))\right)+\nu^2/2}{\nu},
\qquad d_2=d_1-\nu.
\]

For \(\nu=0\), its continuous limit is

\[
C_t=\bigl(P(t,T)-K P(t,S)\bigr)^+.
\]

The logic is sequential: the payoff date selects the numeraire, Girsanov gives
the state law under the new weights, Gaussian affinity gives lognormality,
martingality fixes the mean, and one normal integral gives the price.

## 7. Hull-White: The Same Derivation With A Fitted Curve

Hull-White uses the same centered OU factor,

\[
dx_u=-ax_u\,du+\sigma dW_u^{\mathbb Q},
\qquad x_0=0,
\]

but reconstructs the short rate through a deterministic shift:

\[
r_u=x_u+\phi(u),
\]

where

\[
\phi(u)=f^{\mathrm{market}}(0,u)
+\frac{\sigma^2}{2a^2}(1-e^{-au})^2.
\]

The stochastic integral is unchanged. If

\[
V(\tau)=\operatorname{Var}
\left(\int_t^{t+\tau}x_s\,ds\mid\mathcal F_t\right),
\]

then

\[
P^{HW}(t,U)=\exp\left(
-\beta(t,U)x_t
-\int_t^U\phi(s)ds
+\frac12V(U-t)
\right).
\]

This is exactly the OU zero-coupon formula with the additional deterministic
integral of \(\phi\). The shift is chosen so that

\[
\boxed{P^{HW}(0,U)=P^{\mathrm{market}}(0,U).}
\]

The deterministic shift changes the bond level but not its diffusion. Thus:

- \(dP(u,S)/P(u,S)\) still has diffusion
  \(-\sigma\beta(u,S)dW_u^{\mathbb Q}\);
- the numeraire density and Girsanov correction are unchanged;
- \(x_S\mid\mathcal F_t\) has the same variance \(q_{t,S}\);
- the bond-forward log-volatility \(\nu\) is unchanged;
- the Black formula for the call is unchanged.

Only the bonds inserted into that formula differ:

\[
C_t^{HW}
=P^{HW}(t,T)\Phi(d_1)
-K P^{HW}(t,S)\Phi(d_2).
\]

For equal \(a\), \(\sigma\), and dates, OU and Hull-White therefore have the
same option volatility. Hull-White replaces the standalone OU bond levels by
levels consistent with the observed initial term structure.

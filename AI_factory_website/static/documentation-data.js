(function () {
  "use strict";

  const notation = `
    <div class="definition-grid">
      <div><code>t</code><span>Valuation time</span></div>
      <div><code>S</code><span>Option expiry or fixing time</span></div>
      <div><code>T</code><span>Bond maturity or payment time, T &gt; S</span></div>
      <div><code>K</code><span>Strike</span></div>
      <div><code>P(t,T)</code><span>Zero-coupon bond paying one at T</span></div>
      <div><code>&Phi;</code><span>Standard normal cumulative distribution</span></div>
    </div>`;

  function codeReference(entries) {
    return `<div class="code-reference"><span>CUDA</span><div>${entries.map((entry) =>
      `<p><code>${entry.functions}</code><small>${entry.file}</small></p>`
    ).join("")}</div></div>`;
  }

  function processReference(options) {
    const entries = [{
      functions: "prepare_model · one_step_transition · simulate_terminal_state",
      file: `${options.dynamicsRoot}/dynamics.cu`
    }];
    if (options.fitted) {
      entries.push({
        functions: "compose_model · short_rate_shift · short_rate",
        file: `${options.analyticsRoot}/analytics.cu`
      });
    }
    return codeReference(entries);
  }

  function productFormulas(analyticsRoot) {
    return `
      <h3>Bond options</h3>
      <p>For a call or put expiring at <code>S</code> on <code>P(S,T)</code>, define</p>
      <div class="formula">d<sub>1</sub> = [log(P(t,T)/(K P(t,S))) + &nu;<sup>2</sup>/2] / &nu;, &nbsp; d<sub>2</sub> = d<sub>1</sub> - &nu;.</div>
      <div class="formula">Call = P(t,T)&Phi;(d<sub>1</sub>) - K P(t,S)&Phi;(d<sub>2</sub>).</div>
      ${codeReference([{
        functions: "zero_coupon_bond_call_price",
        file: `${analyticsRoot}/analytics.cu`
      }, {
        functions: "prepare_row · evaluate_price",
        file: `${analyticsRoot}/zero_coupon_bond_call.cu`
      }])}
      <div class="formula">Put = K P(t,S)&Phi;(-d<sub>2</sub>) - P(t,T)&Phi;(-d<sub>1</sub>).</div>
      ${codeReference([{
        functions: "zero_coupon_bond_put_price",
        file: `${analyticsRoot}/analytics.cu`
      }, {
        functions: "prepare_row · evaluate_price",
        file: `${analyticsRoot}/zero_coupon_bond_put.cu`
      }])}

      <h3>Caplets and floorlets</h3>
      <p>Let <code>c = 1 + &delta;K</code> and <code>K<sub>B</sub> = 1/c</code>. Both prices are built from zero-coupon bond options.</p>
      <div class="formula">Caplet(t) = Nc &times; Put<sub>ZCB</sub>(t; T<sub>1</sub>, T<sub>2</sub>, K<sub>B</sub>).</div>
      ${codeReference([{
        functions: "zero_coupon_bond_put_price",
        file: `${analyticsRoot}/analytics.cu`
      }, {
        functions: "prepare_row · evaluate_price",
        file: `${analyticsRoot}/caplet.cu`
      }])}
      <div class="formula">Floorlet(t) = Nc &times; Call<sub>ZCB</sub>(t; T<sub>1</sub>, T<sub>2</sub>, K<sub>B</sub>).</div>
      ${codeReference([{
        functions: "zero_coupon_bond_call_price",
        file: `${analyticsRoot}/analytics.cu`
      }, {
        functions: "prepare_row · evaluate_price",
        file: `${analyticsRoot}/floorlet.cu`
      }])}

      <h3>Forward and swap rates</h3>
      <div class="formula">L(t;T<sub>1</sub>,T<sub>2</sub>) = [P(t,T<sub>1</sub>)/P(t,T<sub>2</sub>)-1]/&delta;.</div>
      ${codeReference([{
        functions: "forward_rate",
        file: `${analyticsRoot}/analytics.cu`
      }])}
      <div class="formula">R(t) = [P(t,T<sub>0</sub>)-P(t,T<sub>n</sub>)] / [&Sigma;<sub>i=1</sub><sup>n</sup>&delta;<sub>i</sub>P(t,T<sub>i</sub>)].</div>
      ${codeReference([{
        functions: "swap_rate",
        file: `${analyticsRoot}/analytics.cu`
      }])}`;
  }

  function oneFactorSection(options) {
    return `
      <p class="eyebrow">${options.family}</p>
      <h2>${options.name}</h2>
      <p>${options.description}</p>
      ${notation}
      <h3>Process and zero-coupon bond</h3>
      <div class="formula">${options.dynamics}</div>
      ${processReference(options)}
      <div class="formula">B<sub>a</sub>(&tau;) = (1 - exp(-a&tau;)) / a.</div>
      ${codeReference([{
        functions: "integral_state_loading",
        file: `${options.dynamicsRoot}/dynamics.cu`
      }])}
      <div class="formula">V<sub>I</sub>(&tau;) = (&sigma;<sup>2</sup>/a<sup>2</sup>)[&tau; - 2B<sub>a</sub>(&tau;) + (1-exp(-2a&tau;))/(2a)].</div>
      ${codeReference([{
        functions: "integral_variance · integral_moments",
        file: `${options.dynamicsRoot}/dynamics.cu`
      }])}
      <div class="formula">${options.bond}</div>
      ${codeReference([{
        functions: "zero_coupon_bond",
        file: `${options.analyticsRoot}/analytics.cu`
      }])}
      <p>${options.bondNote}</p>
      <h3>Volatility used by bond options</h3>
      <div class="formula">q<sub>a</sub>(h) = &sigma;<sup>2</sup>(1-exp(-2ah))/(2a), &nbsp; &nu; = B<sub>a</sub>(T-S)&radic;q<sub>a</sub>(S-t).</div>
      ${codeReference([{
        functions: "zero_coupon_bond_option_price",
        file: `${options.analyticsRoot}/analytics.cu`
      }])}
      ${productFormulas(options.analyticsRoot)}`;
  }

  function twoFactorSection(options) {
    return `
      <p class="eyebrow">${options.family}</p>
      <h2>${options.name}</h2>
      <p>${options.description}</p>
      ${notation}
      <h3>Process and zero-coupon bond</h3>
      <div class="formula">dx<sub>t</sub> = -a x<sub>t</sub>dt + &sigma;dW<sub>t</sub><sup>x</sup>, &nbsp; dy<sub>t</sub> = -b y<sub>t</sub>dt + &eta;dW<sub>t</sub><sup>y</sup>, &nbsp; d&lang;W<sup>x</sup>,W<sup>y</sup>&rang;<sub>t</sub> = &rho;dt.</div>
      ${processReference(options)}
      <div class="formula">B<sub>a</sub>(&tau;) = (1-exp(-a&tau;))/a, &nbsp; B<sub>b</sub>(&tau;) = (1-exp(-b&tau;))/b.</div>
      <div class="formula">V<sub>I</sub>(&tau;) = V<sub>x</sub>(&tau;) + V<sub>y</sub>(&tau;) + 2C<sub>xy</sub>(&tau;), &nbsp; C<sub>xy</sub> = (&rho;&sigma;&eta;/ab)[&tau;-B<sub>a</sub>-B<sub>b</sub>+(1-exp(-(a+b)&tau;))/(a+b)].</div>
      ${codeReference([{
        functions: "integral_moments",
        file: `${options.dynamicsRoot}/dynamics.cu`
      }])}
      <div class="formula">${options.bond}</div>
      ${codeReference([{
        functions: "zero_coupon_bond",
        file: `${options.analyticsRoot}/analytics.cu`
      }])}
      <p>${options.bondNote}</p>
      <h3>Volatility used by bond options</h3>
      <div class="formula">&nu;<sup>2</sup> = B<sub>a</sub>(T-S)<sup>2</sup>q<sub>xx</sub> + B<sub>b</sub>(T-S)<sup>2</sup>q<sub>yy</sub> + 2B<sub>a</sub>(T-S)B<sub>b</sub>(T-S)q<sub>xy</sub>.</div>
      <div class="formula">q<sub>xx</sub> = &sigma;<sup>2</sup>[1-exp(-2a(S-t))]/(2a), &nbsp; q<sub>yy</sub> = &eta;<sup>2</sup>[1-exp(-2b(S-t))]/(2b).</div>
      <div class="formula">q<sub>xy</sub> = &rho;&sigma;&eta;[1-exp(-(a+b)(S-t))]/(a+b).</div>
      ${codeReference([{
        functions: "bond_option_total_volatility",
        file: `${options.analyticsRoot}/analytics.cu`
      }])}
      ${productFormulas(options.analyticsRoot)}`;
  }

  const sections = [
    {
      id: "ornstein-uhlenbeck",
      label: "Ornstein-Uhlenbeck",
      body: oneFactorSection({
        family: "One-factor model",
        name: "Ornstein-Uhlenbeck",
        description: "The current short rate is r(t). Future rates move back toward zero and follow a normal distribution.",
        dynamics: "dr<sub>t</sub> = -a r<sub>t</sub>dt + &sigma;dW<sub>t</sub>.",
        bond: "P(t,T) = exp[-B<sub>a</sub>(&tau;)r<sub>t</sub> + V<sub>I</sub>(&tau;)/2], &nbsp; &tau;=T-t.",
        bondNote: "The rate integral from t to T has average B_a(τ)r(t) and variance V_I(τ).",
        dynamicsRoot: "src/model/fixed_income/ornstein_uhlenbeck",
        analyticsRoot: "src/model/fixed_income/ornstein_uhlenbeck"
      })
    },
    {
      id: "vasicek",
      label: "Vasicek",
      body: oneFactorSection({
        family: "One-factor model",
        name: "Vasicek",
        description: "Vasicek follows the same structure as OU, but the short rate moves toward the long-run level b instead of zero.",
        dynamics: "dr<sub>t</sub> = a(b-r<sub>t</sub>)dt + &sigma;dW<sub>t</sub>.",
        bond: "P(t,T) = exp[-B<sub>a</sub>(&tau;)r<sub>t</sub> - b(&tau;-B<sub>a</sub>(&tau;)) + V<sub>I</sub>(&tau;)/2].",
        bondNote: "Only the average rate integral changes relative to OU. The bond-option volatility is unchanged for equal a and σ.",
        dynamicsRoot: "src/model/fixed_income/vasicek",
        analyticsRoot: "src/model/fixed_income/vasicek"
      })
    },
    {
      id: "g2",
      label: "G2",
      body: twoFactorSection({
        family: "Two-factor model",
        name: "G2",
        description: "The G2 short rate is r(t)=x(t)+y(t). Both factors move back toward zero and their random movements are linked by ρ.",
        bond: "P(t,T) = exp[-B<sub>a</sub>(&tau;)x<sub>t</sub> - B<sub>b</sub>(&tau;)y<sub>t</sub> + V<sub>I</sub>(&tau;)/2].",
        bondNote: "V_I combines the variances of both factor integrals and their covariance.",
        dynamicsRoot: "src/model/fixed_income/g2",
        analyticsRoot: "src/model/fixed_income/g2"
      })
    },
    {
      id: "hull-white",
      label: "Hull-White",
      body: oneFactorSection({
        family: "One-factor model fitted to a curve",
        name: "Hull-White",
        description: "Hull-White writes r(t)=x(t)+φ(t). The OU factor starts at zero and the time-dependent adjustment φ matches the selected starting curve.",
        dynamics: "dx<sub>t</sub> = -a x<sub>t</sub>dt + &sigma;dW<sub>t</sub>, &nbsp; r<sub>t</sub>=x<sub>t</sub>+&phi;(t).",
        bond: "P(t,T) = exp[-B<sub>a</sub>(&tau;)x<sub>t</sub> - &int;<sub>t</sub><sup>T</sup>&phi;(u)du + V<sub>I</sub>(&tau;)/2].",
        bondNote: "The adjustment changes the bond price, while the OU bond-option volatility remains the same.",
        dynamicsRoot: "src/model/fixed_income/ornstein_uhlenbeck",
        analyticsRoot: "src/model/fixed_income/hull_white/&lt;curve&gt;",
        fitted: true
      })
    },
    {
      id: "cir-plus-plus",
      label: "CIR++",
      body: `
        <h2>CIR++: a shifted square-root factor</h2>
        <p>The model writes r(t)=y(t)+φ(t). The factor y follows the exact CIR law
        and starts at its declared nonnegative initial_state. Samples contain y,
        not the curve-shifted short rate.</p>
        <div class="formula">dy = κ(θ−y)dt + σ√y dW, &nbsp; φ(t)=f_market(0,t)−f_CIR(0,t).</div>
        <p>Nelson–Siegel and Svensson supply the independent initial curve.
        Conditional bonds equal the CIR bond times exp(−∫φ).
        Bond options use the same deterministic scaling and an adjusted strike,
        retaining the CIR noncentral-chi-square formula.</p>
        <p>Caplets and floorlets reuse bond-option identities. European swaptions
        use Jamshidian; Bermudans reuse exact terminal-forward CIR transitions
        and the common Longstaff–Schwartz engine, without a fine time grid.</p>
        <p>Source: src/model/fixed_income/cir_plus_plus/README.md.
        Product bindings: product/&lt;curve&gt;/. Curve-independent formulas:
        fitted_analytics_impl.cuh. Publication certification remains pending.</p>`
    },
    {
      id: "g2-plus-plus",
      label: "G2++",
      body: twoFactorSection({
        family: "Two-factor model fitted to a curve",
        name: "G2++",
        description: "G2++ writes r(t)=x(t)+y(t)+φ(t). The adjustment φ matches the starting curve without changing the relationship between x and y.",
        bond: "P(t,T) = exp[-B<sub>a</sub>(&tau;)x<sub>t</sub> - B<sub>b</sub>(&tau;)y<sub>t</sub> - &int;<sub>t</sub><sup>T</sup>&phi;(u)du + V<sub>I</sub>(&tau;)/2].",
        bondNote: "The selected curve changes P(t,T), while the bond-option volatility remains the same as in G2.",
        dynamicsRoot: "src/model/fixed_income/g2",
        analyticsRoot: "src/model/fixed_income/g2_plus_plus/&lt;curve&gt;",
        fitted: true
      })
    }
  ];

  const curveNotation = `
    <div class="definition-grid">
      <div><code>T</code><span>Maturity in years</span></div>
      <div><code>x=T/&tau;</code><span>Scaled maturity</span></div>
      <div><code>z(0,T)</code><span>Continuously compounded zero rate</span></div>
      <div><code>P(0,T)</code><span>Discount factor to maturity T</span></div>
      <div><code>f(0,T)</code><span>Instantaneous forward rate</span></div>
      <div><code>L(x)</code><span>Nelson-Siegel level loading</span></div>
    </div>`;

  const svenssonNotation = `
    <div class="definition-grid">
      <div><code>T</code><span>Maturity in years</span></div>
      <div><code>x<sub>i</sub>=T/&tau;<sub>i</sub></code><span>Two scaled maturities</span></div>
      <div><code>z(0,T)</code><span>Continuously compounded zero rate</span></div>
      <div><code>P(0,T)</code><span>Discount factor to maturity T</span></div>
      <div><code>f(0,T)</code><span>Instantaneous forward rate</span></div>
      <div><code>C(x)</code><span>Curvature loading L(x)-e<sup>-x</sup></span></div>
    </div>`;

  const curveSections = [{
    id: "nelson-siegel",
    label: "Nelson-Siegel",
    body: `
      <p class="eyebrow">Four-parameter yield curve</p>
      <h2>Nelson-Siegel</h2>
      <p>Nelson-Siegel describes the starting yield curve with a long-term level, a short-term slope, one curvature term, and one decay scale.</p>
      ${curveNotation}

      <h3>Parameters</h3>
      <div class="definition-grid">
        <div><code>&beta;<sub>0</sub></code><span>Long-term rate level</span></div>
        <div><code>&beta;<sub>1</sub></code><span>Short-term slope relative to the level</span></div>
        <div><code>&beta;<sub>2</sub></code><span>Size and direction of the curve hump</span></div>
        <div><code>&tau;</code><span>Decay scale in years</span></div>
      </div>
      ${codeReference([{
        functions: "NelsonSiegelParameters",
        file: "src/curve/nelson_siegel/dataset.hpp"
      }])}

      <h3>Zero rate</h3>
      <div class="formula">L(x) = (1-e<sup>-x</sup>)/x, &nbsp; z(0,T) = &beta;<sub>0</sub> + &beta;<sub>1</sub>L(x) + &beta;<sub>2</sub>[L(x)-e<sup>-x</sup>].</div>
      ${codeReference([{
        functions: "zero_rate",
        file: "src/curve/nelson_siegel/term_structure.cu"
      }])}

      <h3>Discount factor</h3>
      <div class="formula">log P(0,T) = -Tz(0,T), &nbsp; P(0,T) = exp[-Tz(0,T)].</div>
      ${codeReference([{
        functions: "log_discount_factor · discount_factor",
        file: "src/curve/nelson_siegel/term_structure.cu"
      }])}

      <h3>Instantaneous forward</h3>
      <div class="formula">f(0,T) = &beta;<sub>0</sub> + e<sup>-x</sup>(&beta;<sub>1</sub> + &beta;<sub>2</sub>x).</div>
      ${codeReference([{
        functions: "instantaneous_forward",
        file: "src/curve/nelson_siegel/term_structure.cu"
      }])}
      <div class="formula">&part;<sub>T</sub>f(0,T) = e<sup>-x</sup>[-&beta;<sub>1</sub> + &beta;<sub>2</sub>(1-x)]/&tau;.</div>
      ${codeReference([{
        functions: "forward_derivative",
        file: "src/curve/nelson_siegel/term_structure.cu"
      }])}

      <h3>Forward over a period</h3>
      <div class="formula">F(0;T<sub>1</sub>,T<sub>2</sub>) = [log P(0,T<sub>1</sub>) - log P(0,T<sub>2</sub>)]/(T<sub>2</sub>-T<sub>1</sub>).</div>
      ${codeReference([{
        functions: "forward_rate",
        file: "src/curve/nelson_siegel/term_structure.cu"
      }])}`
  }, {
    id: "svensson",
    label: "Svensson",
    body: `
      <p class="eyebrow">Six-parameter yield curve</p>
      <h2>Svensson</h2>
      <p>Svensson extends Nelson-Siegel with a second curvature term and a second decay scale.</p>
      ${svenssonNotation}

      <h3>Parameters</h3>
      <div class="definition-grid">
        <div><code>&beta;<sub>0</sub></code><span>Long-term rate level</span></div>
        <div><code>&beta;<sub>1</sub></code><span>Short-term slope relative to the level</span></div>
        <div><code>&beta;<sub>2</sub></code><span>First curvature component</span></div>
        <div><code>&beta;<sub>3</sub></code><span>Second curvature component</span></div>
        <div><code>&tau;<sub>1</sub></code><span>First decay scale in years</span></div>
        <div><code>&tau;<sub>2</sub></code><span>Second decay scale in years</span></div>
      </div>
      ${codeReference([{
        functions: "SvenssonParameters",
        file: "src/curve/svensson/dataset.hpp"
      }])}

      <h3>Zero rate</h3>
      <div class="formula">z(0,T) = &beta;<sub>0</sub> + &beta;<sub>1</sub>L(x<sub>1</sub>) + &beta;<sub>2</sub>C(x<sub>1</sub>) + &beta;<sub>3</sub>C(x<sub>2</sub>).</div>
      ${codeReference([{
        functions: "zero_rate",
        file: "src/curve/svensson/term_structure.cu"
      }])}

      <h3>Discount factor</h3>
      <div class="formula">log P(0,T) = -Tz(0,T), &nbsp; P(0,T) = exp[-Tz(0,T)].</div>
      ${codeReference([{
        functions: "log_discount_factor · discount_factor",
        file: "src/curve/svensson/term_structure.cu"
      }])}

      <h3>Instantaneous forward</h3>
      <div class="formula">f(0,T) = &beta;<sub>0</sub> + e<sup>-x<sub>1</sub></sup>(&beta;<sub>1</sub> + &beta;<sub>2</sub>x<sub>1</sub>) + &beta;<sub>3</sub>x<sub>2</sub>e<sup>-x<sub>2</sub></sup>.</div>
      ${codeReference([{
        functions: "instantaneous_forward",
        file: "src/curve/svensson/term_structure.cu"
      }])}
      <div class="formula">&part;<sub>T</sub>f(0,T) = e<sup>-x<sub>1</sub></sup>[-&beta;<sub>1</sub>+&beta;<sub>2</sub>(1-x<sub>1</sub>)]/&tau;<sub>1</sub> + &beta;<sub>3</sub>e<sup>-x<sub>2</sub></sup>(1-x<sub>2</sub>)/&tau;<sub>2</sub>.</div>
      ${codeReference([{
        functions: "forward_derivative",
        file: "src/curve/svensson/term_structure.cu"
      }])}

      <h3>Forward over a period</h3>
      <div class="formula">F(0;T<sub>1</sub>,T<sub>2</sub>) = [log P(0,T<sub>1</sub>) - log P(0,T<sub>2</sub>)]/(T<sub>2</sub>-T<sub>1</sub>).</div>
      ${codeReference([{
        functions: "forward_rate",
        file: "src/curve/svensson/term_structure.cu"
      }])}`
  }];

  window.AI_FACTORY_DOCUMENTATION = {
    pages: {
      "fixed-income-models": {
        slug: "fixed-income-models",
        title: "Fixed-income pricing formulas",
        summary: "Pricing formulas for each model, followed by the matching CUDA functions and files.",
        navigationLabel: "Models",
        sections
      },
      "yield-curves": {
        slug: "yield-curves",
        title: "Yield curves",
        summary: "Curve parameters and formulas, followed by the matching CUDA functions and files.",
        navigationLabel: "Curves",
        sections: curveSections
      }
    }
  };
})();

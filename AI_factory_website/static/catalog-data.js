(function () {
  "use strict";

  const repository = "https://github.com/AurelienGrenard/AI_factory";
  const storage = "https://datasets.ai-factory.example/v1";
  const documentationSlug = "fixed-income-models";
  const repositoryFolder = (path) => `${repository}/tree/main/${path}`;

  const categories = {
    equity: {
      slug: "equity",
      name: "Equity",
      summary: "Equity models and option price datasets.",
      models: [
        "black-scholes", "heston", "bates", "merton", "kou", "cev",
        "schobel-zhu", "variance-gamma", "normal-inverse-gaussian"
      ]
    },
    "fixed-income": {
      slug: "fixed-income",
      name: "Fixed Income",
      summary: "Interest-rate models, yield curves, and option price datasets.",
      models: ["ornstein-uhlenbeck", "vasicek", "g2", "hull-white", "g2-plus-plus"]
    }
  };

  const models = {
    "black-scholes": {
      slug: "black-scholes",
      category: "equity",
      name: "Black-Scholes",
      family: "Constant-volatility diffusion model",
      image: "static/black_scholes.png",
      summary: "A lognormal equity model with constant volatility and exact transitions.",
      description: "Black-Scholes evolves the asset with constant rates and volatility; every simulated interval is sampled from its exact Gaussian log-price law.",
      products: []
    },
    heston: {
      slug: "heston",
      category: "equity",
      name: "Heston",
      family: "Stochastic volatility model",
      image: "static/heston.png",
      summary: "An equity model where the asset price and its variance move together.",
      description: "Heston models the asset price and its changing variance with two linked sources of randomness.",
      products: [
        "heston-european-calls", "heston-european-puts",
        "heston-asian-calls", "heston-asian-puts",
        "heston-digital-calls", "heston-digital-puts",
        "heston-asset-or-nothing-calls", "heston-asset-or-nothing-puts",
        "heston-gap-calls", "heston-gap-puts", "heston-straddles",
        "heston-lookback-options",
        "heston-geometric-asian-calls", "heston-geometric-asian-puts",
        "heston-forward-start-calls", "heston-forward-start-puts",
        "heston-up-and-out-calls", "heston-down-and-out-puts",
        "heston-up-and-in-calls", "heston-down-and-in-puts",
        "heston-up-one-touches", "heston-up-no-touches",
        "heston-double-knock-out-calls", "heston-double-knock-out-puts",
        "heston-american-calls", "heston-american-puts"
      ]
    },
    bates: {
      slug: "bates",
      category: "equity",
      name: "Bates",
      family: "Stochastic volatility jump-diffusion model",
      image: "static/bates.png",
      summary: "Heston stochastic volatility with compound-Poisson price jumps.",
      description: "Bates adds independent lognormal jumps to the Heston asset process while preserving its stochastic variance.",
      products: []
    },
    merton: {
      slug: "merton",
      category: "equity",
      name: "Merton",
      family: "Lognormal jump-diffusion model",
      image: "static/merton.png",
      summary: "Black-Scholes diffusion with compound-Poisson lognormal jumps.",
      description: "Merton adds independent Gaussian log-jumps whose sum is sampled exactly conditional on the Poisson count.",
      products: []
    },
    kou: {
      slug: "kou",
      category: "equity",
      name: "Kou",
      family: "Double-exponential jump-diffusion model",
      image: "static/kou.png",
      summary: "A diffusion with asymmetric double-exponential price jumps.",
      description: "Kou combines a Gaussian diffusion with exact compound-Poisson upward and downward exponential log-jumps.",
      products: []
    },
    cev: {
      slug: "cev",
      category: "equity",
      name: "CEV",
      family: "Local-volatility diffusion model",
      image: "static/cev.png",
      summary: "The instantaneous volatility changes as a power of spot.",
      description: "CEV is simulated with an absorbed Milstein scheme that retains the local-volatility derivative correction.",
      products: []
    },
    "schobel-zhu": {
      slug: "schobel-zhu",
      category: "equity",
      name: "Schöbel-Zhu",
      family: "Gaussian stochastic-volatility model",
      image: "static/schobel_zhu.png",
      summary: "Volatility follows a correlated mean-reverting Gaussian factor.",
      description: "Schöbel-Zhu uses exact OU volatility endpoints coupled consistently to log-spot Euler transitions.",
      products: []
    },
    "variance-gamma": {
      slug: "variance-gamma",
      category: "equity",
      name: "Variance-Gamma",
      shortName: "VG",
      family: "Pure-jump Lévy model",
      image: "static/variance_gamma.png",
      summary: "A Brownian return evaluated on an independent Gamma clock.",
      description: "Variance-Gamma creates skew and heavy tails through a Gamma-subordinated Brownian motion with a risk-neutral martingale correction.",
      products: []
    },
    "normal-inverse-gaussian": {
      slug: "normal-inverse-gaussian",
      category: "equity",
      name: "Normal-Inverse-Gaussian",
      shortName: "NIG",
      family: "Pure-jump Lévy model",
      image: "static/normal_inverse_gaussian.png",
      summary: "A Brownian return evaluated on an inverse-Gaussian clock.",
      description: "Normal-Inverse-Gaussian combines asymmetric heavy tails with exact inverse-Gaussian subordination and a risk-neutral martingale correction.",
      products: []
    },
    "ornstein-uhlenbeck": {
      slug: "ornstein-uhlenbeck",
      category: "fixed-income",
      name: "Ornstein-Uhlenbeck",
      shortName: "OU",
      family: "One-factor short-rate model",
      image: "static/ornstein_uhlenbeck.png",
      summary: "A short rate that moves back toward zero over time.",
      description: "Ornstein-Uhlenbeck describes the short rate with one factor and can be evaluated exactly.",
      products: []
    },
    vasicek: {
      slug: "vasicek",
      category: "fixed-income",
      name: "Vasicek",
      shortName: "Vasicek",
      family: "One-factor short-rate model",
      image: "static/vasicek.png",
      summary: "A short rate that moves toward a long-run level.",
      description: "Vasicek adds a long-run rate level to OU and can also be evaluated exactly.",
      products: []
    },
    g2: {
      slug: "g2",
      category: "fixed-income",
      name: "G2",
      shortName: "G2",
      family: "Two-factor short-rate model",
      image: "static/g2.png",
      summary: "Two linked factors drive the short rate.",
      description: "G2 combines two OU factors to represent rate movements over different time horizons.",
      products: []
    },
    "hull-white": {
      slug: "hull-white",
      category: "fixed-income",
      name: "Hull-White One Factor",
      shortName: "Hull-White",
      family: "One-factor model fitted to a curve",
      image: "static/hull_white.png",
      summary: "An OU factor adjusted to match the starting yield curve exactly.",
      description: "Hull-White adds a time-dependent adjustment to OU so the model matches the selected starting curve.",
      curves: ["hull-white-nelson-siegel", "hull-white-svensson"]
    },
    "cir-plus-plus": {
      slug: "cir-plus-plus",
      category: "fixed-income",
      name: "CIR++",
      shortName: "CIR++",
      family: "One-factor model fitted to a curve",
      image: "static/cir_plus_plus.png",
      summary: "An exact CIR factor adjusted to match the initial yield curve.",
      description: "CIR++ combines a nonnegative CIR factor with a deterministic curve shift. The total short rate may be negative. Price implementation is available; independent catalogue certification is pending.",
      curves: ["cir-plus-plus-nelson-siegel", "cir-plus-plus-svensson"]
    },
    "g2-plus-plus": {
      slug: "g2-plus-plus",
      category: "fixed-income",
      name: "G2++",
      shortName: "G2++",
      family: "Two-factor model fitted to a curve",
      image: "static/g2_plus_plus.png",
      summary: "Two linked factors adjusted to match the starting yield curve exactly.",
      description: "G2++ adds a time-dependent adjustment to G2 while preserving the relationship between both factors.",
      curves: ["g2-plus-plus-nelson-siegel", "g2-plus-plus-svensson"]
    }
  };

  const curves = {
    "cir-plus-plus-nelson-siegel": {
      slug: "cir-plus-plus-nelson-siegel",
      resourceKey: "nelson-siegel",
      model: "cir-plus-plus",
      name: "Nelson-Siegel",
      image: "static/nelson_siegel.png",
      summary: "A yield curve described by four parameters.",
      description: "Nelson-Siegel describes the starting yield curve that CIR++ matches.",
      products: []
    },
    "cir-plus-plus-svensson": {
      slug: "cir-plus-plus-svensson",
      resourceKey: "svensson",
      model: "cir-plus-plus",
      name: "Svensson",
      image: "static/svensson.png",
      summary: "A six-parameter yield curve with two curvature terms.",
      description: "Svensson describes the starting yield curve that CIR++ matches.",
      products: []
    },
    "hull-white-nelson-siegel": {
      slug: "hull-white-nelson-siegel",
      resourceKey: "nelson-siegel",
      model: "hull-white",
      name: "Nelson-Siegel",
      image: "static/nelson_siegel.png",
      summary: "A yield curve described by four parameters.",
      description: "Nelson-Siegel describes short, medium, and long maturities with a simple formula.",
      products: []
    },
    "g2-plus-plus-nelson-siegel": {
      slug: "g2-plus-plus-nelson-siegel",
      resourceKey: "nelson-siegel",
      model: "g2-plus-plus",
      name: "Nelson-Siegel",
      image: "static/nelson_siegel.png",
      summary: "A yield curve described by four parameters.",
      description: "Nelson-Siegel describes the starting yield curve that G2++ matches.",
      products: []
    },
    "hull-white-svensson": {
      slug: "hull-white-svensson",
      resourceKey: "svensson",
      model: "hull-white",
      name: "Svensson",
      image: "static/svensson.png",
      summary: "A six-parameter yield curve with two curvature terms.",
      description: "Svensson extends Nelson-Siegel with a second medium- and long-term curvature component.",
      products: []
    },
    "g2-plus-plus-svensson": {
      slug: "g2-plus-plus-svensson",
      resourceKey: "svensson",
      model: "g2-plus-plus",
      name: "Svensson",
      image: "static/svensson.png",
      summary: "A six-parameter yield curve with two curvature terms.",
      description: "Svensson describes the starting yield curve that G2++ matches.",
      products: []
    }
  };

  const products = {
    "heston-european-calls": {
      slug: "heston-european-calls", model: "heston", name: "European Calls",
      image: "static/european_call.png", summary: "A call based on the asset price at maturity.",
      description: "The option pays when the final asset price is above its strike.",
      datasets: ["heston-european-calls-01"]
    },
    "heston-european-puts": {
      slug: "heston-european-puts", model: "heston", name: "European Puts",
      image: "static/european_put.png", summary: "A put based on the asset price at maturity.",
      description: "The option pays when the final asset price is below its strike.",
      datasets: ["heston-european-puts-01"]
    },
    "heston-asian-calls": {
      slug: "heston-asian-calls", model: "heston", name: "Asian Calls",
      image: "static/asian_call.png", summary: "A call on the average asset price over time.",
      description: "The payoff uses the average asset price observed from inception through maturity.",
      datasets: ["heston-asian-calls-01"]
    },
    "heston-asian-puts": {
      slug: "heston-asian-puts", model: "heston", name: "Asian Puts",
      image: "static/asian_put.png", summary: "A put on the average asset price over time.",
      description: "The payoff uses the average asset price observed from inception through maturity.",
      datasets: ["heston-asian-puts-01"]
    },
    "heston-digital-calls": {
      slug: "heston-digital-calls", model: "heston", name: "Digital Calls",
      image: "static/digital_call.png", summary: "A fixed cash payment triggered above the strike.",
      description: "The option pays a fixed cash amount when the final asset price exceeds its strike.",
      datasets: ["heston-digital-calls-01"]
    },
    "heston-digital-puts": {
      slug: "heston-digital-puts", model: "heston", name: "Digital Puts",
      image: "static/digital_put.png", summary: "A fixed cash payment triggered below the strike.",
      description: "The option pays a fixed cash amount when the final asset price is below its strike.",
      datasets: ["heston-digital-puts-01"]
    },
    "heston-asset-or-nothing-calls": {
      slug: "heston-asset-or-nothing-calls", model: "heston", name: "Asset-or-Nothing Calls",
      image: "static/asset_or_nothing_call.png", summary: "The final asset value is paid above the strike.",
      description: "The option pays the final asset value when it exceeds the strike.",
      datasets: ["heston-asset-or-nothing-calls-01"]
    },
    "heston-asset-or-nothing-puts": {
      slug: "heston-asset-or-nothing-puts", model: "heston", name: "Asset-or-Nothing Puts",
      image: "static/asset_or_nothing_put.png", summary: "The final asset value is paid below the strike.",
      description: "The option pays the final asset value when it is below the strike.",
      datasets: ["heston-asset-or-nothing-puts-01"]
    },
    "heston-gap-calls": {
      slug: "heston-gap-calls", model: "heston", name: "Gap Calls",
      image: "static/gap_call.png", summary: "A call triggered by one strike and paid from another.",
      description: "The trigger strike activates a call payoff defined by the payoff strike.",
      datasets: ["heston-gap-calls-01"]
    },
    "heston-gap-puts": {
      slug: "heston-gap-puts", model: "heston", name: "Gap Puts",
      image: "static/gap_put.png", summary: "A put triggered by one strike and paid from another.",
      description: "The trigger strike activates a put payoff defined by the payoff strike.",
      datasets: ["heston-gap-puts-01"]
    },
    "heston-straddles": {
      slug: "heston-straddles", model: "heston", name: "Straddles",
      image: "static/straddle.png", summary: "A combined call and put with the same strike and maturity.",
      description: "The payoff is the absolute distance between the final asset price and the strike.",
      datasets: ["heston-straddles-01"]
    },
    "heston-lookback-options": {
      slug: "heston-lookback-options", model: "heston", name: "Lookback Calls",
      image: "static/lookback_option.png", summary: "A call on the highest asset price reached before maturity.",
      description: "The payoff uses the highest asset price observed from inception through maturity.",
      datasets: ["heston-lookback-options-01"]
    },
    "heston-american-puts": {
      slug: "heston-american-puts", model: "heston", name: "American Puts",
      image: "static/american_put.png", summary: "A put that may be exercised before maturity.",
      description: "The price is computed backwards across the available exercise dates.",
      datasets: ["heston-american-puts-01"]
    },
    "heston-geometric-asian-calls": {
      slug: "heston-geometric-asian-calls", model: "heston", name: "Geometric Asian Calls",
      image: "static/geometric_asian_call.png", summary: "A call on the geometric average asset price.",
      description: "The geometric average includes the initial spot and every simulated date through maturity.",
      datasets: ["heston-geometric-asian-calls-01"]
    },
    "heston-geometric-asian-puts": {
      slug: "heston-geometric-asian-puts", model: "heston", name: "Geometric Asian Puts",
      image: "static/geometric_asian_put.png", summary: "A put on the geometric average asset price.",
      description: "The geometric average includes the initial spot and every simulated date through maturity.",
      datasets: ["heston-geometric-asian-puts-01"]
    },
    "heston-forward-start-calls": {
      slug: "heston-forward-start-calls", model: "heston", name: "Forward-Start Calls",
      image: "static/forward_start_call.png", summary: "A call whose strike is fixed from a future spot.",
      description: "At the reset date, the strike becomes a fixed multiple of the asset price.",
      datasets: ["heston-forward-start-calls-01"]
    },
    "heston-forward-start-puts": {
      slug: "heston-forward-start-puts", model: "heston", name: "Forward-Start Puts",
      image: "static/forward_start_put.png", summary: "A put whose strike is fixed from a future spot.",
      description: "At the reset date, the strike becomes a fixed multiple of the asset price.",
      datasets: ["heston-forward-start-puts-01"]
    },
    "heston-up-and-out-calls": {
      slug: "heston-up-and-out-calls", model: "heston", name: "Up-and-Out Calls",
      image: "static/up_and_out_call.png", summary: "A call cancelled when the asset reaches an upper barrier.",
      description: "The upper barrier is monitored at every simulated date with no rebate.",
      datasets: ["heston-up-and-out-calls-01"]
    },
    "heston-down-and-out-puts": {
      slug: "heston-down-and-out-puts", model: "heston", name: "Down-and-Out Puts",
      image: "static/down_and_out_put.png", summary: "A put cancelled when the asset reaches a lower barrier.",
      description: "The lower barrier is monitored at every simulated date with no rebate.",
      datasets: ["heston-down-and-out-puts-01"]
    },
    "heston-up-and-in-calls": {
      slug: "heston-up-and-in-calls", model: "heston", name: "Up-and-In Calls",
      image: "static/up_and_in_call.png", summary: "A call activated when the asset reaches an upper barrier.",
      description: "The upper barrier is monitored at every simulated date with no rebate.",
      datasets: ["heston-up-and-in-calls-01"]
    },
    "heston-down-and-in-puts": {
      slug: "heston-down-and-in-puts", model: "heston", name: "Down-and-In Puts",
      image: "static/down_and_in_put.png", summary: "A put activated when the asset reaches a lower barrier.",
      description: "The lower barrier is monitored at every simulated date with no rebate.",
      datasets: ["heston-down-and-in-puts-01"]
    },
    "heston-up-one-touches": {
      slug: "heston-up-one-touches", model: "heston", name: "Up One-Touches",
      image: "static/up_one_touch.png", summary: "A fixed cash payment activated by an upper barrier.",
      description: "The upper barrier is monitored at every simulated date and cash is paid at maturity.",
      datasets: ["heston-up-one-touches-01"]
    },
    "heston-up-no-touches": {
      slug: "heston-up-no-touches", model: "heston", name: "Up No-Touches",
      image: "static/up_no_touch.png", summary: "A fixed cash payment made if an upper barrier is never reached.",
      description: "The upper barrier is monitored at every simulated date and cash is paid at maturity.",
      datasets: ["heston-up-no-touches-01"]
    },
    "heston-double-knock-out-calls": {
      slug: "heston-double-knock-out-calls", model: "heston", name: "Double-Knock-Out Calls",
      image: "static/double_knock_out_call.png", summary: "A call active only while the asset stays between two barriers.",
      description: "Both barriers are monitored at every simulated date with no rebate.",
      datasets: ["heston-double-knock-out-calls-01"]
    },
    "heston-double-knock-out-puts": {
      slug: "heston-double-knock-out-puts", model: "heston", name: "Double-Knock-Out Puts",
      image: "static/double_knock_out_put.png", summary: "A put active only while the asset stays between two barriers.",
      description: "Both barriers are monitored at every simulated date with no rebate.",
      datasets: ["heston-double-knock-out-puts-01"]
    },
    "heston-american-calls": {
      slug: "heston-american-calls", model: "heston", name: "American Calls",
      image: "static/american_call.png", summary: "A call that may be exercised before maturity.",
      description: "The price is computed backwards across the available exercise dates.",
      datasets: ["heston-american-calls-01"]
    }
  };

  const resource = (title, description, dataUrl, catalogPath) => ({
    title,
    description,
    dataUrl,
    repositoryUrl: repositoryFolder(catalogPath)
  });

  const modelResources = {
    "cir-plus-plus": resource("CIR++ factor parameters", "Mean reversion, long-run factor mean, volatility and nonnegative initial factor; the initial curve is independent.", `${storage}/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json`, "catalog/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01"),
    "black-scholes": resource("Black-Scholes parameters", "Initial spot, constant volatility, risk-free rate, and dividend yield.", `${storage}/model/equity/black_scholes/parameters/black_scholes_01.json`, "catalog/model/equity/black_scholes/parameters/black_scholes_01"),
    heston: resource("Heston parameters", "Heston settings used by the price dataset.", `${storage}/model/equity/heston/parameters/heston_01.json`, "catalog/model/equity/heston/parameters/heston_01"),
    bates: resource("Bates parameters", "Heston variance settings and compound-Poisson lognormal jump parameters.", `${storage}/model/equity/bates/parameters/bates_01.json`, "catalog/model/equity/bates/parameters/bates_01"),
    merton: resource("Merton parameters", "Diffusion volatility and compound-Poisson Gaussian log-jump parameters.", `${storage}/model/equity/merton/parameters/merton_01.json`, "catalog/model/equity/merton/parameters/merton_01"),
    kou: resource("Kou parameters", "Diffusion volatility and asymmetric double-exponential jump parameters.", `${storage}/model/equity/kou/parameters/kou_01.json`, "catalog/model/equity/kou/parameters/kou_01"),
    cev: resource("CEV parameters", "Local-volatility scale, elasticity, rates, and initial spot.", `${storage}/model/equity/cev/parameters/cev_01.json`, "catalog/model/equity/cev/parameters/cev_01"),
    "schobel-zhu": resource("Schöbel-Zhu parameters", "OU volatility, volatility of volatility, correlation, rates, and initial spot.", `${storage}/model/equity/schobel_zhu/parameters/schobel_zhu_01.json`, "catalog/model/equity/schobel_zhu/parameters/schobel_zhu_01"),
    "variance-gamma": resource("Variance-Gamma parameters", "Gamma-clock variance, Brownian volatility, skew, rates, and initial spot.", `${storage}/model/equity/variance_gamma/parameters/variance_gamma_01.json`, "catalog/model/equity/variance_gamma/parameters/variance_gamma_01"),
    "normal-inverse-gaussian": resource("Normal-Inverse-Gaussian parameters", "Tail steepness, asymmetry, scale, rates, and initial spot.", `${storage}/model/equity/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json`, "catalog/model/equity/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01"),
    "ornstein-uhlenbeck": resource("OU parameters", "How quickly the rate moves back toward zero, its volatility, and its starting value.", `${storage}/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01.json`, "catalog/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01"),
    vasicek: resource("Vasicek parameters", "How quickly the rate moves toward its long-run level, its volatility, and its starting value.", `${storage}/model/fixed_income/vasicek/parameters/vasicek_01.json`, "catalog/model/fixed_income/vasicek/parameters/vasicek_01"),
    g2: resource("G2 parameters", "Two factors, how quickly they move back toward zero, their volatilities, relationship, and starting values.", `${storage}/model/fixed_income/g2/parameters/g2_01.json`, "catalog/model/fixed_income/g2/parameters/g2_01"),
    "hull-white": resource("Hull-White parameters", "OU settings used with a separate starting yield curve.", `${storage}/model/fixed_income/hull_white/parameters/hull_white_01.json`, "catalog/model/fixed_income/hull_white/parameters/hull_white_01"),
    "g2-plus-plus": resource("G2++ parameters", "G2 settings used with a separate starting yield curve.", `${storage}/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json`, "catalog/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01")
  };

  const curveResources = {
    "nelson-siegel": resource("Nelson-Siegel curves", "Starting yield curves used by Hull-White and G2++.", `${storage}/curve/nelson_siegel/nelson_siegel_01.json`, "catalog/curve/nelson_siegel/nelson_siegel_01"),
    svensson: resource("Svensson curves", "Starting yield curves with two curvature terms.", `${storage}/curve/svensson/svensson_01.json`, "catalog/curve/svensson/svensson_01")
  };

  const productDefinitions = [
    {
      slug: "caplets", name: "Caplets", image: "static/caplet.png",
      summary: "An option that pays when a future interest rate is above its strike.",
      description: "The price is computed from an equivalent put on a zero-coupon bond.",
      resourcePath: "rate_options", resourceDatasetId: "rate_options_01",
      resourceTitle: "Rate-option parameters", resourceDescription: "Fixing dates, payment dates, year fractions, amounts, and strikes."
    },
    {
      slug: "floorlets", name: "Floorlets", image: "static/floorlet.png",
      summary: "An option that pays when a future interest rate is below its strike.",
      description: "The price is computed from an equivalent call on a zero-coupon bond.",
      resourcePath: "rate_options", resourceDatasetId: "rate_options_01",
      resourceTitle: "Rate-option parameters", resourceDescription: "Fixing dates, payment dates, year fractions, amounts, and strikes."
    },
    {
      slug: "zero-coupon-bond-calls", path: "zero_coupon_bond_calls", datasetId: "zero_coupon_bond_calls_01",
      name: "Zero-Coupon Bond Calls", image: "static/zero_coupon_bond_call.png",
      summary: "A call option on a zero-coupon bond price.",
      description: "The option pays when the bond price at expiry is above its strike.",
      resourcePath: "zero_coupon_bond_options", resourceDatasetId: "zero_coupon_bond_options_01",
      resourceTitle: "Zero-coupon bond option parameters", resourceDescription: "Option expiries, bond maturities, amounts, and strikes."
    },
    {
      slug: "zero-coupon-bond-puts", path: "zero_coupon_bond_puts", datasetId: "zero_coupon_bond_puts_01",
      name: "Zero-Coupon Bond Puts", image: "static/zero_coupon_bond_put.png",
      summary: "A put option on a zero-coupon bond price.",
      description: "The option pays when the bond price at expiry is below its strike.",
      resourcePath: "zero_coupon_bond_options", resourceDatasetId: "zero_coupon_bond_options_01",
      resourceTitle: "Zero-coupon bond option parameters", resourceDescription: "Option expiries, bond maturities, amounts, and strikes."
    }
  ];

  const productResources = {
    "heston-european-calls": resource("European option parameters", "Strikes and maturities.", `${storage}/product/european_options/european_options_01.json`, "catalog/product/equity/european_options/european_options_01"),
    "heston-european-puts": resource("European option parameters", "Strikes and maturities.", `${storage}/product/european_options/european_options_01.json`, "catalog/product/equity/european_options/european_options_01"),
    "heston-asian-calls": resource("Asian option parameters", "Strikes and maturities.", `${storage}/product/asian_options/asian_options_01.json`, "catalog/product/equity/asian_options/asian_options_01"),
    "heston-asian-puts": resource("Asian option parameters", "Strikes and maturities.", `${storage}/product/asian_options/asian_options_01.json`, "catalog/product/equity/asian_options/asian_options_01"),
    "heston-digital-calls": resource("Digital option parameters", "Cash amounts, strikes, and maturities.", `${storage}/product/digital_options/digital_options_01.json`, "catalog/product/equity/digital_options/digital_options_01"),
    "heston-digital-puts": resource("Digital option parameters", "Cash amounts, strikes, and maturities.", `${storage}/product/digital_options/digital_options_01.json`, "catalog/product/equity/digital_options/digital_options_01"),
    "heston-asset-or-nothing-calls": resource("Asset-or-nothing option parameters", "Strikes and maturities.", `${storage}/product/asset_or_nothing_options/asset_or_nothing_options_01.json`, "catalog/product/equity/asset_or_nothing_options/asset_or_nothing_options_01"),
    "heston-asset-or-nothing-puts": resource("Asset-or-nothing option parameters", "Strikes and maturities.", `${storage}/product/asset_or_nothing_options/asset_or_nothing_options_01.json`, "catalog/product/equity/asset_or_nothing_options/asset_or_nothing_options_01"),
    "heston-gap-calls": resource("Gap call parameters", "Trigger strikes, payoff strikes, and maturities.", `${storage}/product/gap_options/gap_call_options_01.json`, "catalog/product/equity/gap_options/gap_call_options_01"),
    "heston-gap-puts": resource("Gap put parameters", "Trigger strikes, payoff strikes, and maturities.", `${storage}/product/gap_options/gap_put_options_01.json`, "catalog/product/equity/gap_options/gap_put_options_01"),
    "heston-straddles": resource("Straddle parameters", "Common call and put strikes and maturities.", `${storage}/product/straddles/straddles_01.json`, "catalog/product/equity/straddles/straddles_01"),
    "heston-lookback-options": resource("Lookback parameters", "Fixed strikes and maturities.", `${storage}/product/lookback_options/lookback_options_01.json`, "catalog/product/equity/lookback_options/lookback_options_01"),
    "heston-american-puts": resource("American option parameters", "Strikes, maturities, and exercise intervals.", `${storage}/product/american_options/american_options_01.json`, "catalog/product/equity/american_options/american_options_01"),
    "heston-geometric-asian-calls": resource("Geometric Asian option parameters", "Strikes and maturities.", `${storage}/product/geometric_asian_options/geometric_asian_options_01.json`, "catalog/product/equity/geometric_asian_options/geometric_asian_options_01"),
    "heston-geometric-asian-puts": resource("Geometric Asian option parameters", "Strikes and maturities.", `${storage}/product/geometric_asian_options/geometric_asian_options_01.json`, "catalog/product/equity/geometric_asian_options/geometric_asian_options_01"),
    "heston-forward-start-calls": resource("Forward-start option parameters", "Moneyness, reset dates, and maturities.", `${storage}/product/forward_start_options/forward_start_options_01.json`, "catalog/product/equity/forward_start_options/forward_start_options_01"),
    "heston-forward-start-puts": resource("Forward-start option parameters", "Moneyness, reset dates, and maturities.", `${storage}/product/forward_start_options/forward_start_options_01.json`, "catalog/product/equity/forward_start_options/forward_start_options_01"),
    "heston-up-and-out-calls": resource("Up-and-out option parameters", "Strikes, upper barriers, and maturities.", `${storage}/product/up_and_out_options/up_and_out_options_01.json`, "catalog/product/equity/up_and_out_options/up_and_out_options_01"),
    "heston-down-and-out-puts": resource("Down-and-out option parameters", "Strikes, lower barriers, and maturities.", `${storage}/product/down_and_out_options/down_and_out_options_01.json`, "catalog/product/equity/down_and_out_options/down_and_out_options_01"),
    "heston-up-and-in-calls": resource("Up-and-in option parameters", "Strikes, upper barriers, and maturities.", `${storage}/product/up_and_in_options/up_and_in_options_01.json`, "catalog/product/equity/up_and_in_options/up_and_in_options_01"),
    "heston-down-and-in-puts": resource("Down-and-in option parameters", "Strikes, lower barriers, and maturities.", `${storage}/product/down_and_in_options/down_and_in_options_01.json`, "catalog/product/equity/down_and_in_options/down_and_in_options_01"),
    "heston-up-one-touches": resource("Up one-touch parameters", "Upper barriers, cash amounts, and maturities.", `${storage}/product/up_one_touches/up_one_touches_01.json`, "catalog/product/equity/up_one_touches/up_one_touches_01"),
    "heston-up-no-touches": resource("Up no-touch parameters", "Upper barriers, cash amounts, and maturities.", `${storage}/product/up_no_touches/up_no_touches_01.json`, "catalog/product/equity/up_no_touches/up_no_touches_01"),
    "heston-double-knock-out-calls": resource("Double-knock-out option parameters", "Strikes, lower and upper barriers, and maturities.", `${storage}/product/double_knock_out_options/double_knock_out_options_01.json`, "catalog/product/equity/double_knock_out_options/double_knock_out_options_01"),
    "heston-double-knock-out-puts": resource("Double-knock-out option parameters", "Strikes, lower and upper barriers, and maturities.", `${storage}/product/double_knock_out_options/double_knock_out_options_01.json`, "catalog/product/equity/double_knock_out_options/double_knock_out_options_01"),
    "heston-american-calls": resource("American option parameters", "Strikes, maturities, and exercise intervals.", `${storage}/product/american_options/american_options_01.json`, "catalog/product/equity/american_options/american_options_01")
  };

  const datasets = {
    "heston-european-calls-01": { title: "Heston European Calls 01", model: "heston", product: "heston-european-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "European call prices under Heston.", priceDataset: resource("Heston European Calls 01", "Complete price dataset.", "https://mlp.lpma.math.upmc.fr/DataCarlo/Assets/Heston/EuropeanCall/heston_01__european_calls_01__01.json", "catalog/model/equity/heston/prices/european_calls/heston_01__european_calls_01__01") },
    "heston-european-puts-01": { title: "Heston European Puts 01", model: "heston", product: "heston-european-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "European put prices under Heston.", priceDataset: resource("Heston European Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/european_puts/heston_01__european_puts_01__01.json`, "catalog/model/equity/heston/prices/european_puts/heston_01__european_puts_01__01") },
    "heston-asian-calls-01": { title: "Heston Asian Calls 01", model: "heston", product: "heston-asian-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Average-price call prices under Heston.", priceDataset: resource("Heston Asian Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/asian_calls/heston_01__asian_calls_01__01.json`, "catalog/model/equity/heston/prices/asian_calls/heston_01__asian_calls_01__01") },
    "heston-asian-puts-01": { title: "Heston Asian Puts 01", model: "heston", product: "heston-asian-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Average-price put prices under Heston.", priceDataset: resource("Heston Asian Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/asian_puts/heston_01__asian_puts_01__01.json`, "catalog/model/equity/heston/prices/asian_puts/heston_01__asian_puts_01__01") },
    "heston-digital-calls-01": { title: "Heston Digital Calls 01", model: "heston", product: "heston-digital-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Cash-or-nothing call prices under Heston.", priceDataset: resource("Heston Digital Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/digital_calls/heston_01__digital_calls_01__01.json`, "catalog/model/equity/heston/prices/digital_calls/heston_01__digital_calls_01__01") },
    "heston-digital-puts-01": { title: "Heston Digital Puts 01", model: "heston", product: "heston-digital-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Cash-or-nothing put prices under Heston.", priceDataset: resource("Heston Digital Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/digital_puts/heston_01__digital_puts_01__01.json`, "catalog/model/equity/heston/prices/digital_puts/heston_01__digital_puts_01__01") },
    "heston-asset-or-nothing-calls-01": { title: "Heston Asset-or-Nothing Calls 01", model: "heston", product: "heston-asset-or-nothing-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Asset-or-nothing call prices under Heston.", priceDataset: resource("Heston Asset-or-Nothing Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01.json`, "catalog/model/equity/heston/prices/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01") },
    "heston-asset-or-nothing-puts-01": { title: "Heston Asset-or-Nothing Puts 01", model: "heston", product: "heston-asset-or-nothing-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Asset-or-nothing put prices under Heston.", priceDataset: resource("Heston Asset-or-Nothing Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/asset_or_nothing_puts/heston_01__asset_or_nothing_puts_01__01.json`, "catalog/model/equity/heston/prices/asset_or_nothing_puts/heston_01__asset_or_nothing_puts_01__01") },
    "heston-gap-calls-01": { title: "Heston Gap Calls 01", model: "heston", product: "heston-gap-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Gap-call prices under Heston.", priceDataset: resource("Heston Gap Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/gap_calls/heston_01__gap_calls_01__01.json`, "catalog/model/equity/heston/prices/gap_calls/heston_01__gap_calls_01__01") },
    "heston-gap-puts-01": { title: "Heston Gap Puts 01", model: "heston", product: "heston-gap-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Gap-put prices under Heston.", priceDataset: resource("Heston Gap Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/gap_puts/heston_01__gap_puts_01__01.json`, "catalog/model/equity/heston/prices/gap_puts/heston_01__gap_puts_01__01") },
    "heston-straddles-01": { title: "Heston Straddles 01", model: "heston", product: "heston-straddles", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "European straddle prices under Heston.", priceDataset: resource("Heston Straddles 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/straddles/heston_01__straddles_01__01.json`, "catalog/model/equity/heston/prices/straddles/heston_01__straddles_01__01") },
    "heston-lookback-options-01": { title: "Heston Lookback Calls 01", model: "heston", product: "heston-lookback-options", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Lookback call prices under Heston.", priceDataset: resource("Heston Lookback Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/lookback_options/heston_01__lookback_options_01__01.json`, "catalog/model/equity/heston/prices/lookback_options/heston_01__lookback_options_01__01") },
    "heston-american-puts-01": { title: "Heston American Puts 01", model: "heston", product: "heston-american-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Longstaff-Schwartz", description: "Early-exercise put prices under Heston.", priceDataset: resource("Heston American Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/american_puts/heston_01__american_puts_01__01.json`, "catalog/model/equity/heston/prices/american_puts/heston_01__american_puts_01__01") },
    "heston-geometric-asian-calls-01": { title: "Heston Geometric Asian Calls 01", model: "heston", product: "heston-geometric-asian-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Geometric-average call prices under Heston.", priceDataset: resource("Heston Geometric Asian Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/geometric_asian_calls/heston_01__geometric_asian_calls_01__01.json`, "catalog/model/equity/heston/prices/geometric_asian_calls/heston_01__geometric_asian_calls_01__01") },
    "heston-geometric-asian-puts-01": { title: "Heston Geometric Asian Puts 01", model: "heston", product: "heston-geometric-asian-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Geometric-average put prices under Heston.", priceDataset: resource("Heston Geometric Asian Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/geometric_asian_puts/heston_01__geometric_asian_puts_01__01.json`, "catalog/model/equity/heston/prices/geometric_asian_puts/heston_01__geometric_asian_puts_01__01") },
    "heston-forward-start-calls-01": { title: "Heston Forward-Start Calls 01", model: "heston", product: "heston-forward-start-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Forward-start call prices under Heston.", priceDataset: resource("Heston Forward-Start Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/forward_start_calls/heston_01__forward_start_calls_01__01.json`, "catalog/model/equity/heston/prices/forward_start_calls/heston_01__forward_start_calls_01__01") },
    "heston-forward-start-puts-01": { title: "Heston Forward-Start Puts 01", model: "heston", product: "heston-forward-start-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Forward-start put prices under Heston.", priceDataset: resource("Heston Forward-Start Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/forward_start_puts/heston_01__forward_start_puts_01__01.json`, "catalog/model/equity/heston/prices/forward_start_puts/heston_01__forward_start_puts_01__01") },
    "heston-up-and-out-calls-01": { title: "Heston Up-and-Out Calls 01", model: "heston", product: "heston-up-and-out-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored up-and-out call prices under Heston.", priceDataset: resource("Heston Up-and-Out Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/up_and_out_calls/heston_01__up_and_out_calls_01__01.json`, "catalog/model/equity/heston/prices/up_and_out_calls/heston_01__up_and_out_calls_01__01") },
    "heston-down-and-out-puts-01": { title: "Heston Down-and-Out Puts 01", model: "heston", product: "heston-down-and-out-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored down-and-out put prices under Heston.", priceDataset: resource("Heston Down-and-Out Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/down_and_out_puts/heston_01__down_and_out_puts_01__01.json`, "catalog/model/equity/heston/prices/down_and_out_puts/heston_01__down_and_out_puts_01__01") },
    "heston-up-and-in-calls-01": { title: "Heston Up-and-In Calls 01", model: "heston", product: "heston-up-and-in-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored up-and-in call prices under Heston.", priceDataset: resource("Heston Up-and-In Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/up_and_in_calls/heston_01__up_and_in_calls_01__01.json`, "catalog/model/equity/heston/prices/up_and_in_calls/heston_01__up_and_in_calls_01__01") },
    "heston-down-and-in-puts-01": { title: "Heston Down-and-In Puts 01", model: "heston", product: "heston-down-and-in-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored down-and-in put prices under Heston.", priceDataset: resource("Heston Down-and-In Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/down_and_in_puts/heston_01__down_and_in_puts_01__01.json`, "catalog/model/equity/heston/prices/down_and_in_puts/heston_01__down_and_in_puts_01__01") },
    "heston-up-one-touches-01": { title: "Heston Up One-Touches 01", model: "heston", product: "heston-up-one-touches", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored maturity-paid one-touch prices under Heston.", priceDataset: resource("Heston Up One-Touches 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/up_one_touches/heston_01__up_one_touches_01__01.json`, "catalog/model/equity/heston/prices/up_one_touches/heston_01__up_one_touches_01__01") },
    "heston-up-no-touches-01": { title: "Heston Up No-Touches 01", model: "heston", product: "heston-up-no-touches", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored maturity-paid no-touch prices under Heston.", priceDataset: resource("Heston Up No-Touches 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/up_no_touches/heston_01__up_no_touches_01__01.json`, "catalog/model/equity/heston/prices/up_no_touches/heston_01__up_no_touches_01__01") },
    "heston-double-knock-out-calls-01": { title: "Heston Double-Knock-Out Calls 01", model: "heston", product: "heston-double-knock-out-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored double-knock-out call prices under Heston.", priceDataset: resource("Heston Double-Knock-Out Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/double_knock_out_calls/heston_01__double_knock_out_calls_01__01.json`, "catalog/model/equity/heston/prices/double_knock_out_calls/heston_01__double_knock_out_calls_01__01") },
    "heston-double-knock-out-puts-01": { title: "Heston Double-Knock-Out Puts 01", model: "heston", product: "heston-double-knock-out-puts", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Monte Carlo", description: "Discretely monitored double-knock-out put prices under Heston.", priceDataset: resource("Heston Double-Knock-Out Puts 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/double_knock_out_puts/heston_01__double_knock_out_puts_01__01.json`, "catalog/model/equity/heston/prices/double_knock_out_puts/heston_01__double_knock_out_puts_01__01") },
    "heston-american-calls-01": { title: "Heston American Calls 01", model: "heston", product: "heston-american-calls", rowCount: "1,000", simulationMethod: "QE-M scheme", pricingMethod: "Longstaff-Schwartz", description: "Early-exercise call prices under Heston.", priceDataset: resource("Heston American Calls 01", "Complete price dataset.", `${storage}/model/equity/heston/prices/american_calls/heston_01__american_calls_01__01.json`, "catalog/model/equity/heston/prices/american_calls/heston_01__american_calls_01__01") }
  };

  // Bates deliberately exposes the same product catalogue as Heston. Clone
  // only the presentation metadata here; model and price resources remain
  // distinct and point to their own reproducible catalog recipes.
  const batesText = (value) => value
    .replaceAll("Heston", "Bates")
    .replaceAll("heston", "bates");

  Object.entries({ ...products }).forEach(([productSlug, product]) => {
    if (product.model !== "heston") return;
    const batesSlug = batesText(productSlug);
    products[batesSlug] = {
      ...product,
      slug: batesSlug,
      model: "bates",
      datasets: product.datasets.map(batesText)
    };
    models.bates.products.push(batesSlug);
    productResources[batesSlug] = productResources[productSlug];
  });

  Object.entries({ ...datasets }).forEach(([datasetSlug, dataset]) => {
    if (dataset.model !== "heston") return;
    const batesSlug = batesText(datasetSlug);
    datasets[batesSlug] = {
      ...dataset,
      title: batesText(dataset.title),
      model: "bates",
      product: batesText(dataset.product),
      simulationMethod: "QE-M with compound-Poisson lognormal jumps",
      description: batesText(dataset.description),
      priceDataset: {
        ...dataset.priceDataset,
        title: batesText(dataset.priceDataset.title),
        dataUrl: batesText(dataset.priceDataset.dataUrl),
        repositoryUrl: batesText(dataset.priceDataset.repositoryUrl)
      }
    };
  });

  // The new V1 equity models share Heston's public non-American payoff
  // catalogue, but every price resource points to its own explicit CUDA
  // implementation and reproducible generator.
  const monteCarloEquityModels = [
    {
      slug: "merton",
      sourceName: "Heston",
      targetName: "Merton",
      modelId: "merton_01",
      simulationMethod: "Exact Gaussian diffusion and compound-Poisson lognormal jumps"
    },
    {
      slug: "kou",
      sourceName: "Heston",
      targetName: "Kou",
      modelId: "kou_01",
      simulationMethod: "Exact Gaussian diffusion and compound-Poisson double-exponential jumps"
    },
    {
      slug: "cev",
      sourceName: "Heston",
      targetName: "CEV",
      modelId: "cev_01",
      simulationMethod: "Absorbed Milstein scheme"
    },
    {
      slug: "schobel-zhu",
      sourceName: "Heston",
      targetName: "Schöbel-Zhu",
      modelId: "schobel_zhu_01",
      simulationMethod: "Exact OU volatility endpoint with log-spot Euler stepping"
    }
  ];

  monteCarloEquityModels.forEach((configuration) => {
    const sourceProducts = Object.entries({ ...products });
    sourceProducts.forEach(([productSlug, product]) => {
      if (product.model !== "heston" || productSlug.includes("-american-")) return;
      const payoffSlug = productSlug.replace("heston-", "");
      const targetSlug = `${configuration.slug}-${payoffSlug}`;
      products[targetSlug] = {
        ...product,
        slug: targetSlug,
        model: configuration.slug,
        datasets: [`${targetSlug}-01`]
      };
      models[configuration.slug].products.push(targetSlug);
      productResources[targetSlug] = productResources[productSlug];
    });

    Object.entries({ ...datasets }).forEach(([datasetSlug, dataset]) => {
      if (dataset.model !== "heston" || datasetSlug.includes("-american-")) return;
      const payoffSlug = dataset.product.replace("heston-", "");
      const targetProductSlug = `${configuration.slug}-${payoffSlug}`;
      const targetDatasetSlug = `${targetProductSlug}-01`;
      const priceFolder = payoffSlug.replaceAll("-", "_");
      const priceId = `${configuration.modelId}__${priceFolder}_01__01`;
      const modelName = configuration.modelId.replace("_01", "");
      datasets[targetDatasetSlug] = {
        ...dataset,
        title: dataset.title.replaceAll(
          configuration.sourceName,
          configuration.targetName
        ),
        model: configuration.slug,
        product: targetProductSlug,
        simulationMethod: configuration.simulationMethod,
        pricingMethod: "Monte Carlo",
        description: dataset.description.replaceAll(
          configuration.sourceName,
          configuration.targetName
        ),
        priceDataset: resource(
          dataset.priceDataset.title.replaceAll(
            configuration.sourceName,
            configuration.targetName
          ),
          "Complete price dataset.",
          `${storage}/model/equity/${modelName}/prices/${priceFolder}/${priceId}.json`,
          `catalog/model/equity/${modelName}/prices/${priceFolder}/${priceId}`
        )
      };
    });
  });

  // Black-Scholes supports the same non-American public payoff catalogue as
  // Heston. Closed-form datasets use one thread per price; path-dependent
  // datasets retain the common one-block-per-price Monte Carlo layout.
  const blackScholesText = (value) => value
    .replaceAll("Heston", "Black-Scholes")
    .replaceAll("heston", "black-scholes");
  const blackScholesClosedFormProducts = new Set([
    "european-calls", "european-puts", "digital-calls", "digital-puts",
    "asset-or-nothing-calls", "asset-or-nothing-puts", "gap-calls",
    "gap-puts", "straddles", "geometric-asian-calls",
    "geometric-asian-puts", "forward-start-calls", "forward-start-puts"
  ]);

  Object.entries({ ...products }).forEach(([productSlug, product]) => {
    if (product.model !== "heston" || productSlug.includes("-american-")) return;
    const blackScholesSlug = blackScholesText(productSlug);
    products[blackScholesSlug] = {
      ...product,
      slug: blackScholesSlug,
      model: "black-scholes",
      datasets: product.datasets.map(blackScholesText)
    };
    models["black-scholes"].products.push(blackScholesSlug);
    productResources[blackScholesSlug] = productResources[productSlug];
  });

  Object.entries({ ...datasets }).forEach(([datasetSlug, dataset]) => {
    if (dataset.model !== "heston" || datasetSlug.includes("-american-")) return;
    const blackScholesSlug = blackScholesText(datasetSlug);
    const payoffSlug = dataset.product.replace("heston-", "");
    const priceFolder = payoffSlug.replaceAll("-", "_");
    const priceId = `black_scholes_01__${priceFolder}_01__01`;
    const priceCatalogPath =
      `catalog/model/equity/black_scholes/prices/${priceFolder}/${priceId}`;
    const closedForm = blackScholesClosedFormProducts.has(payoffSlug);
    datasets[blackScholesSlug] = {
      ...dataset,
      title: blackScholesText(dataset.title),
      model: "black-scholes",
      product: blackScholesText(dataset.product),
      simulationMethod: closedForm
        ? "Black-Scholes closed form"
        : "Exact Gaussian log-price transitions",
      pricingMethod: closedForm ? "Closed-form" : "Monte Carlo",
      description: blackScholesText(dataset.description),
      priceDataset: {
        ...dataset.priceDataset,
        title: blackScholesText(dataset.priceDataset.title),
        dataUrl:
          `${storage}/model/equity/black_scholes/prices/${priceFolder}/${priceId}.json`,
        repositoryUrl: repositoryFolder(priceCatalogPath)
      }
    };
  });

  const fixedIncomeModels = [
    { slug: "ornstein-uhlenbeck", modelId: "ornstein_uhlenbeck_01", pricePath: "fixed_income/ornstein_uhlenbeck" },
    { slug: "vasicek", modelId: "vasicek_01", pricePath: "fixed_income/vasicek" },
    { slug: "g2", modelId: "g2_01", pricePath: "fixed_income/g2" },
    { slug: "hull-white", modelId: "hull_white_01", curve: "hull-white-nelson-siegel", curveId: "nelson_siegel_01", pricePath: "fixed_income/hull_white/nelson_siegel" },
    { slug: "hull-white", modelId: "hull_white_01", curve: "hull-white-svensson", curveId: "svensson_01", pricePath: "fixed_income/hull_white/svensson" },
    { slug: "g2-plus-plus", modelId: "g2_plus_plus_01", curve: "g2-plus-plus-nelson-siegel", curveId: "nelson_siegel_01", pricePath: "fixed_income/g2_plus_plus/nelson_siegel" },
    { slug: "g2-plus-plus", modelId: "g2_plus_plus_01", curve: "g2-plus-plus-svensson", curveId: "svensson_01", pricePath: "fixed_income/g2_plus_plus/svensson" }
  ];

  fixedIncomeModels.forEach((configuration) => {
    const model = models[configuration.slug];
    const productList = configuration.curve
      ? curves[configuration.curve].products
      : model.products;

    productDefinitions.forEach((definition) => {
      const productPath = definition.path || definition.slug;
      const productDatasetId = definition.datasetId || `${productPath}_01`;
      const resourcePath = definition.resourcePath || productPath;
      const resourceDatasetId = definition.resourceDatasetId || productDatasetId;
      const productSlug = configuration.curve
        ? `${configuration.curve}-${definition.slug}`
        : `${configuration.slug}-${definition.slug}`;
      const priceId = [configuration.modelId, configuration.curveId, productDatasetId, "01"]
        .filter(Boolean).join("__");
      const datasetSlug = `${productSlug}-01`;
      const [assetClass, modelName, ...priceSuffix] =
        configuration.pricePath.split("/");
      const modelPricePath = [
        assetClass,
        modelName,
        "prices",
        ...priceSuffix
      ].join("/");
      const priceCatalogPath =
        `catalog/model/${modelPricePath}/${productPath}/${priceId}`;

      products[productSlug] = {
        slug: productSlug,
        model: configuration.slug,
        curve: configuration.curve,
        name: definition.name,
        image: definition.image,
        summary: definition.summary,
        description: definition.description,
        datasets: [datasetSlug]
      };
      productList.push(productSlug);
      productResources[productSlug] = resource(
        definition.resourceTitle,
        definition.resourceDescription,
        `${storage}/product/${resourcePath}/${resourceDatasetId}.json`,
        `catalog/product/fixed_income/${resourcePath}/${resourceDatasetId}`
      );
      datasets[datasetSlug] = {
        title: `${model.shortName} ${definition.name} 01`,
        model: configuration.slug,
        curve: configuration.curve,
        product: productSlug,
        rowCount: "1,000",
        pricingMethod: "Exact",
        description: `${definition.name} prices under ${model.name}.`,
        documentation: documentationSlug,
        priceDataset: resource(
          `${model.shortName} ${definition.name} 01`,
          "Complete price dataset.",
          `${storage}/model/${modelPricePath}/${productPath}/${priceId}.json`,
          priceCatalogPath
        )
      };
    });
  });

  Object.values(datasets).forEach((dataset) => {
    dataset.modelDataset = modelResources[dataset.model];
    dataset.productDataset = productResources[dataset.product];
    if (dataset.curve) {
      dataset.curveDataset = curveResources[curves[dataset.curve].resourceKey];
    }
  });

  window.AI_FACTORY_CATALOG = { categories, models, curves, products, datasets };
})();

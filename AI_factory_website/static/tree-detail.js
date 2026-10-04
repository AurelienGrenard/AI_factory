(function () {
  "use strict";

  const inventory = window.AI_FACTORY_DATASET_INDEX;
  const app = document.getElementById("app");
  const breadcrumbs = document.getElementById("breadcrumbs");
  const labels = {
    curve: "Curves", model: "Models", product: "Products",
    equity: "Equity", fixed_income: "Fixed Income",
    markovian: "Markovian", rough: "Rough", prices: "Prices",
    price_sensitivities: "Price sensitivities", samples: "Samples",
    parameters: "Parameters", cir_plus_plus: "CIR++",
    g2_plus_plus: "G2++", heston_3_2: "Heston 3/2"
  };
  const kindLabels = {
    prices: "Prices", price_sensitivities: "Price sensitivities",
    samples: "Samples", model_parameters: "Model parameters",
    product_parameters: "Product parameters", curve_parameters: "Curve parameters"
  };

  const escapeHtml = (value) => String(value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function label(segment) {
    return labels[segment] || segment.replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function treeHref(segments) {
    return segments.length
      ? `#/catalog-tree/${segments.map(encodeURIComponent).join("/")}`
      : "#/";
  }

  function externalLink(text, url, primary) {
    return `<a class="button${primary ? "" : " secondary"}" href="${escapeHtml(url)}"
      target="_blank" rel="noreferrer">${escapeHtml(text)}</a>`;
  }

  function datasetCard(dataset) {
    return `<a class="dataset-card" href="#/catalog-dataset/${encodeURIComponent(dataset.key)}">
      <div class="card-topline"><span>${escapeHtml(kindLabels[dataset.kind] || dataset.kind)}</span>${dataset.rowCount ? `<span class="tag">${Number(dataset.rowCount).toLocaleString("en-US")} rows</span>` : ""}</div>
      <h3>${escapeHtml(dataset.id)}</h3>
      <p class="card-copy">Open the dataset and its related inputs.</p>
      <span class="card-link">Open</span>
    </a>`;
  }

  function setActiveNav(segments) {
    const active = segments.includes("equity")
      ? "equity" : segments.includes("fixed_income") ? "fixed-income" : "catalog";
    document.querySelectorAll("[data-nav]").forEach((link) => {
      link.classList.toggle("active", link.dataset.nav === active);
    });
  }

  function setDatasetBreadcrumbs(dataset) {
    const segments = dataset.key.split("/");
    const items = [{ label: "Catalog", href: "#/" }];
    segments.forEach((segment, index) => {
      items.push({ label: label(segment), href: treeHref(segments.slice(0, index + 1)) });
    });
    items.push({ label: dataset.id });
    breadcrumbs.innerHTML = items.map((item, index) => {
      const content = item.href
        ? `<a href="${item.href}">${escapeHtml(item.label)}</a>`
        : `<strong>${escapeHtml(item.label)}</strong>`;
      return `${index ? '<span class="breadcrumb-separator">/</span>' : ""}${content}`;
    }).join("");
  }

  function renderLeaf(segments, datasets) {
    const title = label(segments.at(-1));
    app.innerHTML = `<header class="page-header"><div><p class="eyebrow">Catalog tree</p>
      <h1>${escapeHtml(title)}</h1><p class="lead">Choose a dataset to view its downloads and related inputs.</p></div>
      <div class="header-metrics inventory-metric"><div><strong>${datasets.length.toLocaleString("en-US")}</strong><span>Datasets</span></div></div></header>
      <section class="section-band white"><div class="section-heading"><div><p class="eyebrow">Datasets</p><h2>Available datasets</h2></div></div>
      <div class="dataset-grid">${datasets.sort((left, right) => left.id.localeCompare(right.id)).map(datasetCard).join("")}</div></section>`;
  }

  function relatedSection(related) {
    return `<section class="resource-section"><div class="resource-copy">
      <p class="eyebrow">${escapeHtml(label(related.role))} dataset</p>
      <h2>${escapeHtml(related.id)}</h2><p>${escapeHtml(related.catalogPath)}</p></div>
      <div class="resource-actions">${externalLink("Download dataset", related.url, true)}${externalLink("GitHub", related.repositoryUrl, false)}</div></section>`;
  }

  function renderDataset(dataset) {
    const segments = dataset.key.split("/");
    setActiveNav(segments);
    setDatasetBreadcrumbs(dataset);
    const related = dataset.related || [];
    app.innerHTML = `<header class="dataset-header"><div class="dataset-heading">
      <p class="eyebrow">${escapeHtml(kindLabels[dataset.kind] || dataset.kind)}</p>
      <h1>${escapeHtml(dataset.id)}</h1><p class="lead">${escapeHtml(dataset.catalogPath)}</p></div>
      <div class="resource-actions">${externalLink("Download dataset", dataset.url, true)}${externalLink("GitHub", dataset.repositoryUrl, false)}</div></header>
      <section class="summary-strip columns-2"><div><span>Rows</span><strong>${dataset.rowCount ? Number(dataset.rowCount).toLocaleString("en-US") : "Not specified"}</strong></div>
      <div><span>Dataset type</span><strong>${escapeHtml(kindLabels[dataset.kind] || dataset.kind)}</strong></div></section>
      ${related.length ? `<div class="resource-stack">${related.map(relatedSection).join("")}</div>` : ""}`;
  }

  function routeDetail() {
    const path = (location.hash.slice(1) || "/").replace(/\/+$/, "") || "/";
    const parts = path.split("/").filter(Boolean);
    if (parts[0] === "catalog-dataset" && parts[1]) {
      const key = decodeURIComponent(parts[1]);
      const dataset = inventory.datasets.find((entry) => entry.key === key);
      if (dataset) renderDataset(dataset);
      scrollTo(0, 0);
      app.focus({ preventScroll: true });
      return;
    }
    if (parts[0] !== "catalog-tree") return;
    const segments = parts.slice(1).map(decodeURIComponent);
    const prefix = segments.join("/");
    const matches = inventory.datasets.filter((dataset) =>
      dataset.key === prefix || dataset.key.startsWith(`${prefix}/`)
    );
    const direct = matches.filter((dataset) => dataset.key.split("/").length === segments.length + 1);
    const hasFolders = matches.some((dataset) => dataset.key.split("/").length > segments.length + 1);
    if (direct.length && !hasFolders) renderLeaf(segments, direct);
  }

  addEventListener("hashchange", routeDetail);
  routeDetail();
})();

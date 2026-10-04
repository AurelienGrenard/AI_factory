(function () {
  "use strict";

  const inventory = window.AI_FACTORY_DATASET_INDEX;
  const app = document.getElementById("app");
  const breadcrumbs = document.getElementById("breadcrumbs");
  const kindLabels = {
    prices: "Prices",
    price_sensitivities: "Price sensitivities",
    samples: "Samples",
    model_parameters: "Model parameters",
    product_parameters: "Product parameters",
    curve_parameters: "Curve parameters"
  };

  const escapeHtml = (value) => String(value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  function setActiveNav() {
    document.querySelectorAll("[data-nav]").forEach((link) => {
      link.classList.toggle("active", link.dataset.nav === "inventory");
    });
  }

  function setBreadcrumbs(items) {
    breadcrumbs.innerHTML = items.map((item, index) => {
      const content = item.href
        ? `<a href="${item.href}">${escapeHtml(item.label)}</a>`
        : `<strong>${escapeHtml(item.label)}</strong>`;
      return `${index ? '<span class="breadcrumb-separator">/</span>' : ""}${content}`;
    }).join("");
  }

  function externalLink(label, url, primary) {
    return `<a class="button${primary ? "" : " secondary"}" href="${escapeHtml(url)}"
      target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
  }

  function datasetActions(dataset) {
    return `<div class="inventory-actions">
      ${dataset.url ? externalLink("Download dataset", dataset.url, true) : ""}
      ${externalLink("GitHub", dataset.repositoryUrl, false)}
    </div>`;
  }

  function datasetRow(dataset) {
    const descriptors = [
      dataset.assetClass,
      dataset.modelClass,
      dataset.model,
      dataset.family,
      dataset.topic
    ].filter(Boolean).map((value) => escapeHtml(String(value).replaceAll("_", " ")));
    return `<article class="inventory-row">
      <div class="inventory-row-copy">
        <div class="card-topline"><span>${escapeHtml(kindLabels[dataset.kind] || dataset.kind)}</span>${dataset.rowCount ? `<span class="tag">${Number(dataset.rowCount).toLocaleString("en-US")} rows</span>` : ""}</div>
        <h3><a href="#/catalog-datasets/${encodeURIComponent(dataset.key)}">${escapeHtml(dataset.id)}</a></h3>
        <p>${descriptors.join(" · ")}</p>
      </div>
      ${datasetActions(dataset)}
    </article>`;
  }

  function renderInventory(kind) {
    const selectedKind = kind && inventory.counts[kind] ? kind : "all";
    const baseDatasets = selectedKind === "all"
      ? inventory.datasets
      : inventory.datasets.filter((dataset) => dataset.kind === selectedKind);
    const tabs = [
      ["all", "All", inventory.count],
      ...Object.keys(kindLabels)
        .filter((key) => inventory.counts[key])
        .map((key) => [key, kindLabels[key], inventory.counts[key]])
    ];
    setActiveNav();
    setBreadcrumbs([{ label: "All datasets" }]);
    app.innerHTML = `<header class="page-header"><div><p class="eyebrow">Complete catalog</p>
      <h1>All datasets</h1><p class="lead">Every dataset declared in <code>catalog</code>, with direct links to its download and generation code.</p></div>
      <div class="header-metrics inventory-metric"><div><strong>${inventory.count.toLocaleString("en-US")}</strong><span>Datasets</span></div></div></header>
      <nav class="dataset-tabs" aria-label="Dataset types">${tabs.map(([key, label, count]) =>
        `<a class="${selectedKind === key ? "active" : ""}" href="#/inventory${key === "all" ? "" : `/${key}`}">${escapeHtml(label)} <span>${Number(count).toLocaleString("en-US")}</span></a>`
      ).join("")}</nav>
      <section class="inventory-panel">
        <div class="inventory-toolbar"><label for="inventorySearch">Search datasets</label><input id="inventorySearch" type="search" placeholder="CIR++, European swaption, Heston…" autocomplete="off"><span id="inventoryResultCount"></span></div>
        <div class="inventory-list" id="inventoryList"></div>
        <nav class="inventory-pagination" id="inventoryPagination" aria-label="Dataset pages"></nav>
      </section>`;

    const search = document.getElementById("inventorySearch");
    const list = document.getElementById("inventoryList");
    const resultCount = document.getElementById("inventoryResultCount");
    const pagination = document.getElementById("inventoryPagination");
    const pageSize = 100;
    let page = 1;

    function updateList() {
      const normalizedQuery = search.value.toLowerCase().replaceAll("+", " plus ").replaceAll("_", " ").replaceAll("-", " ").trim();
      const terms = normalizedQuery.split(/\s+/).filter(Boolean);
      const filtered = terms.length ? baseDatasets.filter((dataset) => {
        const searchable = [dataset.id, dataset.key, dataset.kind, dataset.model, dataset.family, dataset.topic]
          .filter(Boolean).join(" ").replaceAll("_", " ").toLowerCase();
        return terms.every((term) => searchable.includes(term));
      }) : baseDatasets;
      const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
      page = Math.min(page, totalPages);
      const start = (page - 1) * pageSize;
      list.innerHTML = filtered.slice(start, start + pageSize).map(datasetRow).join("") ||
        `<div class="inventory-empty"><h2>No dataset found</h2><p>Try a model, product, or exact dataset identifier.</p></div>`;
      resultCount.textContent = `${filtered.length.toLocaleString("en-US")} result${filtered.length === 1 ? "" : "s"}`;
      pagination.innerHTML = filtered.length > pageSize
        ? `<button type="button" data-page="previous" ${page === 1 ? "disabled" : ""}>Previous</button><span>Page ${page} of ${totalPages}</span><button type="button" data-page="next" ${page === totalPages ? "disabled" : ""}>Next</button>`
        : "";
    }

    search.addEventListener("input", () => { page = 1; updateList(); });
    pagination.addEventListener("click", (event) => {
      const button = event.target.closest("[data-page]");
      if (!button || button.disabled) return;
      page += button.dataset.page === "next" ? 1 : -1;
      updateList();
      document.querySelector(".inventory-panel").scrollIntoView();
    });
    updateList();
  }

  function renderDataset(key) {
    const dataset = inventory.datasets.find((entry) => entry.key === key);
    if (!dataset) return;
    setActiveNav();
    setBreadcrumbs([{ label: "All datasets", href: "#/inventory" }, { label: dataset.id }]);
    const details = [
      ["Type", kindLabels[dataset.kind] || dataset.kind],
      ["Rows", dataset.rowCount ? Number(dataset.rowCount).toLocaleString("en-US") : "Not specified"],
      ["Catalog path", dataset.catalogPath],
      ["Output path", dataset.outputPath || "Not specified"]
    ];
    app.innerHTML = `<header class="dataset-header"><div class="dataset-heading"><p class="eyebrow">${escapeHtml(kindLabels[dataset.kind] || dataset.kind)}</p><h1>${escapeHtml(dataset.id)}</h1><p class="lead">Dataset declared in the catalog recipe below.</p></div>${datasetActions(dataset)}</header>
      <section class="inventory-detail">${details.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</section>`;
  }

  function routeInventory() {
    const path = (location.hash.slice(1) || "/").replace(/\/+$/, "") || "/";
    const parts = path.split("/").filter(Boolean);
    if (parts[0] === "inventory") {
      renderInventory(parts[1] || "all");
      scrollTo(0, 0);
      app.focus({ preventScroll: true });
    } else if (parts[0] === "catalog-datasets" && parts[1]) {
      renderDataset(decodeURIComponent(parts[1]));
      scrollTo(0, 0);
      app.focus({ preventScroll: true });
    }
  }

  addEventListener("hashchange", routeInventory);
  routeInventory();
})();

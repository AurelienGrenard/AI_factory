(function () {
  "use strict";

  const inventory = window.AI_FACTORY_DATASET_INDEX;
  const app = document.getElementById("app");
  const breadcrumbs = document.getElementById("breadcrumbs");
  const labels = {
    curve: "Curves",
    model: "Models",
    product: "Products",
    equity: "Equity",
    fixed_income: "Fixed Income",
    markovian: "Markovian",
    rough: "Rough",
    prices: "Prices",
    price_sensitivities: "Price sensitivities",
    samples: "Samples",
    parameters: "Parameters",
    cir_plus_plus: "CIR++",
    g2_plus_plus: "G2++",
    heston_3_2: "Heston 3/2"
  };
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

  function label(segment) {
    return labels[segment] || segment
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function treeHref(segments) {
    return segments.length
      ? `#/catalog-tree/${segments.map(encodeURIComponent).join("/")}`
      : "#/";
  }

  function setActiveNav(segments) {
    const active = segments.includes("equity")
      ? "equity"
      : segments.includes("fixed_income") ? "fixed-income" : "catalog";
    document.querySelectorAll("[data-nav]").forEach((link) => {
      link.classList.toggle("active", link.dataset.nav === active);
    });
  }

  function setBreadcrumbs(segments) {
    const items = [{ label: "Catalog", href: segments.length ? "#/" : undefined }];
    segments.forEach((segment, index) => {
      items.push({
        label: label(segment),
        href: index < segments.length - 1 ? treeHref(segments.slice(0, index + 1)) : undefined
      });
    });
    breadcrumbs.innerHTML = items.map((item, index) => {
      const content = item.href
        ? `<a href="${item.href}">${escapeHtml(item.label)}</a>`
        : `<strong>${escapeHtml(item.label)}</strong>`;
      return `${index ? '<span class="breadcrumb-separator">/</span>' : ""}${content}`;
    }).join("");
  }

  function externalLink(text, url, primary) {
    return `<a class="button${primary ? "" : " secondary"}" href="${escapeHtml(url)}"
      target="_blank" rel="noreferrer">${escapeHtml(text)}</a>`;
  }

  function datasetRow(dataset) {
    return `<article class="inventory-row">
      <div class="inventory-row-copy">
        <div class="card-topline"><span>${escapeHtml(kindLabels[dataset.kind] || dataset.kind)}</span>${dataset.rowCount ? `<span class="tag">${Number(dataset.rowCount).toLocaleString("en-US")} rows</span>` : ""}</div>
        <h3>${escapeHtml(dataset.id)}</h3>
        <p>${escapeHtml(dataset.catalogPath)}</p>
      </div>
      <div class="inventory-actions">
        ${externalLink("Download dataset", dataset.url, true)}
        ${externalLink("GitHub", dataset.repositoryUrl, false)}
      </div>
    </article>`;
  }

  function renderTree(segments) {
    const prefix = segments.join("/");
    const matches = inventory.datasets.filter((dataset) =>
      !prefix || dataset.key === prefix || dataset.key.startsWith(`${prefix}/`)
    );
    if (!matches.length) return;

    const folders = new Map();
    const datasets = [];
    matches.forEach((dataset) => {
      const remainder = dataset.key.split("/").slice(segments.length);
      if (remainder.length === 1) {
        datasets.push(dataset);
      } else if (remainder.length > 1) {
        folders.set(remainder[0], (folders.get(remainder[0]) || 0) + 1);
      }
    });

    setActiveNav(segments);
    setBreadcrumbs(segments);
    const folderCards = [...folders.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([folder, count]) => `<a class="product-card tree-folder" href="${treeHref([...segments, folder])}">
        <div class="card-topline"><span>Catalog folder</span><span class="tag">${count.toLocaleString("en-US")} datasets</span></div>
        <h2>${escapeHtml(label(folder))}</h2>
        <span class="card-link">Open</span>
      </a>`).join("");
    const title = segments.length ? label(segments.at(-1)) : "Dataset catalog";

    app.innerHTML = `<header class="page-header"><div><p class="eyebrow">Catalog tree</p>
      <h1>${escapeHtml(title)}</h1><p class="lead">Browse the website exactly like the <code>catalog</code> directory.</p></div>
      <div class="header-metrics inventory-metric"><div><strong>${matches.length.toLocaleString("en-US")}</strong><span>Datasets</span></div></div></header>
      ${folderCards ? `<section class="section-band white"><div class="section-heading"><div><p class="eyebrow">Folders</p><h2>Choose a branch</h2></div></div><div class="product-grid">${folderCards}</div></section>` : ""}
      ${datasets.length ? `<section class="inventory-panel tree-datasets"><div class="inventory-list">${datasets.sort((left, right) => left.id.localeCompare(right.id)).map(datasetRow).join("")}</div></section>` : ""}`;
  }

  function routeTree() {
    const path = (location.hash.slice(1) || "/").replace(/\/+$/, "") || "/";
    const parts = path.split("/").filter(Boolean);
    if (path === "/") {
      renderTree([]);
    } else if (parts[0] === "catalog-tree") {
      renderTree(parts.slice(1).map(decodeURIComponent));
    } else if (parts[0] === "categories" && parts[1] === "equity") {
      renderTree(["model", "equity"]);
    } else if (parts[0] === "categories" && parts[1] === "fixed-income") {
      renderTree(["model", "fixed_income"]);
    } else {
      return;
    }
    scrollTo(0, 0);
    app.focus({ preventScroll: true });
  }

  addEventListener("hashchange", routeTree);
  routeTree();
})();

(function () {
  "use strict";
  const rows = window.AI_FACTORY_DATASET_INDEX.datasets;
  const app = document.getElementById("app");
  const breadcrumbs = document.getElementById("breadcrumbs");
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("sidebarOverlay");
  const menuButton = document.getElementById("menuButton");
  const labels = { samples: "Sample", model_parameters: "Parameters", prices: "Prices" };
  const escapeHtml = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const title = (value) => String(value).replaceAll("_", " ").replaceAll("/", " · ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
  const modelRows = (model) => rows.filter((row) => row.model === model);
  const models = [...new Set(rows.map((row) => row.model))].sort();
  const modelName = (model) => title(model.split("/").at(-1));
  const market = (model) => model.startsWith("equity/") ? "equity" : "fixed_income";
  const marketName = (value) => value === "equity" ? "Equity" : "Fixed Income";
  const urlBase = window.AI_FACTORY_DATASET_BASE_URL || "../datasets";

  function linkFor(row) {
    const relative = row.output.replace(/^datasets\//, "");
    return `${urlBase.replace(/\/$/, "")}/${relative.split("/").map(encodeURIComponent).join("/")}`;
  }
  function setActive(section) {
    document.querySelectorAll("[data-nav]").forEach((link) => link.classList.toggle("active", link.dataset.nav === section));
  }
  function setCrumbs(items) {
    breadcrumbs.innerHTML = items.map((item, index) => `${index ? '<span class="breadcrumb-separator">/</span>' : ""}${item.href
      ? `<a href="${item.href}">${escapeHtml(item.label)}</a>` : `<strong>${escapeHtml(item.label)}</strong>`}`).join("");
  }
  function card(label, description, href, meta) {
    return `<a class="product-card" href="${href}"><div class="card-topline"><span>${escapeHtml(meta)}</span></div>
      <h2>${escapeHtml(label)}</h2><p class="card-copy">${escapeHtml(description)}</p><span class="card-link">Open</span></a>`;
  }
  function layout(heading, lead, body) {
    app.innerHTML = `<header class="page-header"><div><p class="eyebrow">Production catalog</p><h1>${escapeHtml(heading)}</h1>
      <p class="lead">${escapeHtml(lead)}</p></div></header><section class="section-band white">${body}</section>`;
  }
  function notFound() {
    setActive("catalog"); setCrumbs([{ label: "Not found" }]);
    layout("Page not found", "Choose a model from the catalog.", card("Catalog", "Browse available models.", "#/", "Home"));
  }
  function home() {
    setActive("catalog"); setCrumbs([{ label: "Markets" }]);
    layout("Dataset catalog", "Choose a market, then a model.", `<div class="product-grid">${["equity", "fixed_income"].map((value) =>
      card(marketName(value), `${models.filter((model) => market(model) === value).length} models`, `#/markets/${value}`, "Market")
    ).join("")}</div>`);
  }
  function marketPage(value) {
    if (!["equity", "fixed_income"].includes(value)) return notFound();
    setActive(value); setCrumbs([{ label: "Markets", href: "#/" }, { label: marketName(value) }]);
    layout(marketName(value), "Choose a model.", `<div class="product-grid">${models.filter((model) => market(model) === value)
      .map((model) => card(modelName(model), title(model.split("/").slice(0, -1).join("/")), `#/models/${encodeURIComponent(model)}`, "Model"))
      .join("")}</div>`);
  }
  function modelPage(model) {
    if (!models.includes(model)) return notFound();
    setActive(market(model)); setCrumbs([{ label: marketName(market(model)), href: `#/markets/${market(model)}` }, { label: modelName(model) }]);
    const available = modelRows(model);
    const descriptions = {
      samples: "One sample dataset for this model.",
      model_parameters: "One model parameter dataset.",
      prices: "Choose a product to get its aligned price dataset."
    };
    const cards = Object.keys(labels).map((kind) => {
      const matches = available.filter((row) => row.kind === kind);
      if (!matches.length) return `<div class="product-card" aria-disabled="true"><div class="card-topline"><span>Unavailable</span></div>
        <h2>${labels[kind]}</h2><p class="card-copy">No published dataset yet.</p></div>`;
      return card(labels[kind], descriptions[kind], `#/models/${encodeURIComponent(model)}/${kind}`, `${matches.length} dataset${matches.length > 1 ? "s" : ""}`);
    });
    layout(modelName(model), "Choose a dataset type.", `<div class="product-grid">${cards.join("")}</div>`);
  }
  function categoryPage(model, kind) {
    if (!models.includes(model) || !labels[kind]) return notFound();
    const matches = modelRows(model).filter((row) => row.kind === kind);
    if (!matches.length) return notFound();
    if (kind === "samples" || kind === "model_parameters") return detailPage(model, kind, "");
    setActive(market(model));
    setCrumbs([{ label: modelName(model), href: `#/models/${encodeURIComponent(model)}` }, { label: labels[kind] }]);
    layout(labels[kind], "Choose a product. Each product has one dataset.", `<div class="product-grid">${matches
      .sort((left, right) => left.product.localeCompare(right.product))
      .map((row) => card(title(row.product), "Aligned dataset", `#/models/${encodeURIComponent(model)}/${kind}/${encodeURIComponent(row.product)}`, "Product"))
      .join("")}</div>`);
  }
  function detailPage(model, kind, product) {
    const matches = modelRows(model).filter((row) => row.kind === kind &&
      ((kind === "prices") ? row.product === product : true));
    if (matches.length !== 1) return notFound();
    const row = matches[0];
    setActive(market(model));
    const categoryHref = `#/models/${encodeURIComponent(model)}/${kind}`;
    setCrumbs([{ label: modelName(model), href: `#/models/${encodeURIComponent(model)}` },
      { label: labels[kind], href: kind === "prices" ? categoryHref : undefined },
      ...(product ? [{ label: title(product) }] : [])]);
    layout(product ? title(product) : labels[kind], "One production dataset is available here.",
      `<div class="inventory-detail"><div><span>Dataset</span><strong>${escapeHtml(row.id)}</strong></div>
      <div><span>Rows</span><strong>${Number(row.rowCount).toLocaleString("en-US")}</strong></div></div>
      <div class="resource-actions"><a class="button" href="${escapeHtml(linkFor(row))}" download>Download dataset</a></div>`);
  }
  function route() {
    sidebar.classList.remove("open"); overlay.classList.remove("open"); menuButton.setAttribute("aria-expanded", "false");
    const parts = (location.hash.slice(1) || "/").split("/").filter(Boolean);
    if (!parts.length) home();
    else if (parts[0] === "markets" && parts.length === 2) marketPage(parts[1]);
    else if (parts[0] === "models" && parts[1]) {
      const model = decodeURIComponent(parts[1]);
      if (parts.length === 2) modelPage(model);
      else if (parts.length === 3) categoryPage(model, parts[2]);
      else if (parts.length === 4) detailPage(model, parts[2], decodeURIComponent(parts[3]));
      else notFound();
    } else notFound();
    scrollTo(0, 0); app.focus({ preventScroll: true });
  }
  menuButton.addEventListener("click", () => {
    const open = !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", open); overlay.classList.toggle("open", open);
    menuButton.setAttribute("aria-expanded", String(open));
  });
  overlay.addEventListener("click", () => { sidebar.classList.remove("open"); overlay.classList.remove("open"); });
  addEventListener("hashchange", route);
  route();
})();

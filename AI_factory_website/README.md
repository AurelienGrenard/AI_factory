# AI Factory static website

The website reads only `catalog/prod/manifest.json`. Navigation is market →
model → sample or parameters → one dataset, or market → model → prices → product
→ one aligned dataset. Price gradients and all `other` recipes are absent.

Regenerate the index from the project root after changing the prod selection:

```sh
python3 AI_factory_website/build_dataset_index.py
```

The default download base is `../datasets`, relative to the website. A row in
the prod index therefore downloads from `../datasets/prod/...`. Set
`window.AI_FACTORY_DATASET_BASE_URL` before loading `static/catalog.js` if the
server hosts the datasets elsewhere.

See [the production layout guide](../docs/deployment-catalog.md) for copying
`catalog/prod` and `datasets/prod` together.

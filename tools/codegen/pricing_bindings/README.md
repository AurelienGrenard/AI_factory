# Pricing-binding code-generation prototype

This isolated prototype generates repetitive model-product binding files from
an explicit Python manifest. It does not modify `src/`, CMake or the build tree.

```bash
python3 tools/codegen/pricing_bindings/generate.py \
    --output /tmp/ai_factory-pricing-bindings \
    --compare-root .
```

The manifest currently covers Merton and CEV European and Asian options. It
records every schedule family and option-side instantiation explicitly. The
command writes eight files under `/tmp` and verifies that they match the
corresponding hand-written sources byte for byte.

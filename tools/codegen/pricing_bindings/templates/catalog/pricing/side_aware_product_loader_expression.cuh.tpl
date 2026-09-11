// Binds a compile-time option side to a product dataset loader.
[](const auto& path) { return product::$product_loader(path, OptionSide::$side); }

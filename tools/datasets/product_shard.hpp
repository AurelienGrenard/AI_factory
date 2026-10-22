#pragma once

// Reproduce the seeded 100-product subset used by Cartesian MC datasets.
#include "tools/datasets/artifact_io.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ai_factory::workbench::datasets {

template <std::size_t N>
void write_product_shard(
    const std::filesystem::path& source_path,
    const std::filesystem::path& output_path,
    const std::filesystem::path& catalog_path,
    const std::array<std::size_t, N>& source_rows
) {
    const auto started = std::chrono::steady_clock::now();
    auto source = read_json_file(source_path);
    if (!source.is_object() || !source.contains("products")
        || !source["products"].is_array()
        || source["products"].size() != 1000U || N != 100U) {
        throw std::invalid_argument("Expected a 1000-row source and 100 shard indices.");
    }
    nlohmann::ordered_json products = nlohmann::ordered_json::array();
    for (std::size_t row : source_rows) {
        if (row >= source["products"].size()) {
            throw std::out_of_range("Product shard index exceeds source row count.");
        }
        products.push_back(source["products"][row]);
    }
    source["database_id"] = output_path.stem().string();
    source["row_count"] = N;
    source["products"] = std::move(products);
    write_json_file(output_path, source);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    write_generation_receipt(catalog_path, N,
        {{"mode", "deterministic_product_shard"},
         {"source", source_path.generic_string()},
         {"selected_rows", source_rows}},
        elapsed, 0.0);
}

}  // namespace ai_factory::workbench::datasets

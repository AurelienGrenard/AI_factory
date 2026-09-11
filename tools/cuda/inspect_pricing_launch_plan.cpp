// Inspect the same host plan used by recipes, without CUDA or dataset generation.
#include "tools/cuda/pricing_launch_plan.hpp"

#include <charconv>
#include <fstream>
#include <iostream>
#include <string>

namespace tuning = ai_factory::workbench::offline::cuda_tuning;

std::size_t positive_count(std::string_view text) {
    std::size_t value = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0U)
        throw std::invalid_argument("Expected a positive integer count.");
    return value;
}

int main(int argc, char** argv) {
    try {
        const bool price_delta = argc > 1 && std::string_view(argv[argc - 1]) == "--price-delta";
        if (price_delta) --argc;
        if (argc < 3 || argc > 5) {
            std::cerr << "Usage: inspect_pricing_launch_plan MODEL/[CURVE/]PRODUCT PRICES"
                         " [PATHS_PER_PRICE [MAXIMUM_RESIDENT_PRICES]] [--price-delta]\n"
                         "Run from the repository root; no CUDA calls are made.\n";
            return 2;
        }
        std::ifstream stream("cmake/generated/PricingCapabilityManifest.json");
        if (!stream) throw std::runtime_error("Cannot read the generated capability manifest.");
        const auto inventory = nlohmann::json::parse(stream);
        const std::string key = argv[1];
        const auto name = inventory.at("pricing_launch_families").at(key).get<std::string>();
        tuning::PricingFamily family{};
        bool found = false;
        for (int index = 0; index <= static_cast<int>(tuning::PricingFamily::rough_fft); ++index) {
            const auto candidate = static_cast<tuning::PricingFamily>(index);
            if (tuning::pricing_family_name(candidate) == name) {
                family = candidate;
                found = true;
                break;
            }
        }
        if (!found) throw std::invalid_argument("Unknown generated pricing family: " + name);
        const auto first = key.find('/');
        const auto last = key.rfind('/');
        const std::string model = key.substr(0, first);
        const std::string product = key.substr(last + 1U);
        const std::string curve = first == last ? "" : key.substr(first + 1U, last - first - 1U);
        const tuning::PricingIdentity identity{family, model, product, curve};
        const bool analytical = family == tuning::PricingFamily::closed_form
            || family == tuning::PricingFamily::jamshidian;
        if (analytical && argc > 3)
            throw std::invalid_argument("Closed-form pricing has no path count argument.");
        const auto paths = analytical ? 0U : argc > 3 ? positive_count(argv[3]) : tuning::kProductionPathsPerPrice;
        tuning::PricingLaunchLimits limits;
        if (argc > 4) limits.maximum_resident_prices = positive_count(argv[4]);
        if (price_delta && !inventory.at("price_delta_bindings").contains(
                "src/model/equity/markovian/" + model + "/product/" + product + "_price_delta"))
            throw std::invalid_argument("No generated price-delta binding for " + key);
        const auto plan = price_delta
            ? tuning::make_equity_price_delta_launch_plan(identity, positive_count(argv[2]), paths, limits)
            : tuning::make_pricing_launch_plan(identity, positive_count(argv[2]), paths, limits);
        auto metadata = tuning::pricing_launch_metadata(plan);
        metadata["device_resources_checked"] = false;
        metadata["scope"] = "offline proposed launch; native engine guards and qualification still required";
        std::cout << metadata.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Pricing launch plan: " << error.what() << '\n';
        return 1;
    }
}

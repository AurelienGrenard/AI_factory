// Inspect the same host plan used by recipes, without CUDA or dataset generation.
#include "tools/cuda/pricing_launch_plan.hpp"
#include "tools/cuda/price_gradients/launch_plan.hpp"

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
        unsigned int batch_size = tuning::pg::kDefaultSensitivityBatchSize;
        const bool explicit_batch = argc > 2 && std::string_view(argv[argc-2]) == "--sensitivity-batch-size";
        if (explicit_batch) {
            const auto parsed = positive_count(argv[argc-1]);
            if (parsed != 1U && parsed != 2U && parsed != 4U)
                throw std::invalid_argument("Sensitivity batch size must be 1, 2 or 4.");
            batch_size = static_cast<unsigned int>(parsed);
            argc -= 2;
        }
        const bool price_gradients = argc > 2 && std::string_view(argv[argc-2]) == "--price-gradients";
        const std::size_t sensitivity_count = price_gradients
            ? (std::string_view(argv[argc-1]) == "0" ? 0U : positive_count(argv[argc-1])) : 0U;
        if (explicit_batch && !price_gradients) throw std::invalid_argument("Batch size requires --price-gradients K.");
        if (price_gradients) argc -= 2;
        const bool price_delta = argc > 1 && std::string_view(argv[argc - 1]) == "--price-delta";
        if (price_gradients && price_delta) throw std::invalid_argument("Choose one sensitivity interface.");
        if (price_delta) --argc;
        if (argc < 3 || argc > 5) {
            std::cerr << "Usage: inspect_pricing_launch_plan MODEL/[CURVE/]PRODUCT PRICES"
                         " [PATHS_PER_PRICE [MAXIMUM_RESIDENT_PRICES]] [--price-delta | --price-gradients K [--sensitivity-batch-size B]]\n"
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
        if (price_delta) {
            bool available = false;
            for (const auto& binding : inventory.at("price_delta_bindings"))
                available = available || binding.at("identity").get<std::string>() == key;
            if (!available)
                throw std::invalid_argument("No generated price-delta binding for " + key);
        }
        nlohmann::ordered_json metadata;
        if (price_gradients) {
            const auto plan = tuning::make_price_gradient_launch_plan(identity,positive_count(argv[2]),sensitivity_count,paths,limits,batch_size);
            metadata = tuning::price_gradient_launch_metadata(plan,sensitivity_count);
        } else {
            const auto plan = price_delta
                ? tuning::make_equity_price_delta_launch_plan(identity,positive_count(argv[2]),paths,limits)
                : tuning::make_pricing_launch_plan(identity,positive_count(argv[2]),paths,limits);
            metadata = tuning::pricing_launch_metadata(plan);
        }
        metadata["device_resources_checked"] = false;
        metadata["scope"] = "offline proposed launch; native engine guards and qualification still required";
        std::cout << metadata.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Pricing launch plan: " << error.what() << '\n';
        return 1;
    }
}

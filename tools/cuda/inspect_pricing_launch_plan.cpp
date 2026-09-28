// Inspect the same host plan used by recipes, without CUDA or dataset generation.
#include "tools/cuda/pricing_launch_plan.hpp"
#include "tools/cuda/price_gradients/launch_plan.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace tuning = ai_factory::workbench::offline::cuda_tuning;
namespace pg = ai_factory::workbench::price_gradients;

std::size_t positive_count(std::string_view text) {
    std::size_t value = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0U)
        throw std::invalid_argument("Expected a positive integer count.");
    return value;
}

int main(int argc, char** argv) {
    try {
        const bool node_graph = argc > 1
            && std::string_view(argv[argc-1]) == "--node-graph";
        if (node_graph) --argc;
        const bool mixed = argc > 1
            && std::string_view(argv[argc-1]) == "--mixed";
        if (mixed) --argc;
        const bool diagonal = argc > 1
            && std::string_view(argv[argc-1]) == "--diagonal";
        if (diagonal) --argc;
        if (mixed && diagonal)
            throw std::invalid_argument("Choose --diagonal or --mixed.");
        const bool price_gradients = argc > 2 && std::string_view(argv[argc-2]) == "--price-gradients";
        if ((diagonal || mixed) && !price_gradients)
            throw std::invalid_argument("Higher orders require --price-gradients K.");
        if (node_graph && !diagonal)
            throw std::invalid_argument(
                "--node-graph requires --price-gradients K --diagonal."
            );
        const std::size_t sensitivity_count = price_gradients
            ? (std::string_view(argv[argc-1]) == "0" ? 0U : positive_count(argv[argc-1])) : 0U;
        if (price_gradients) argc -= 2;
        const bool price_delta = argc > 1 && std::string_view(argv[argc - 1]) == "--price-delta";
        if (price_gradients && price_delta) throw std::invalid_argument("Choose one sensitivity interface.");
        if (price_delta) --argc;
        if (argc < 3 || argc > 5) {
            std::cerr << "Usage: inspect_pricing_launch_plan MODEL/[CURVE/]PRODUCT PRICES"
                         " [PATHS_PER_PRICE [MAXIMUM_RESIDENT_PRICES]] [--price-delta | --price-gradients K [--diagonal [--node-graph]|--mixed]]\n"
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
            bool device_prepared = false;
            bool gradient_available = false;
            for (auto it = inventory.at("price_gradient_bindings").begin();
                 it != inventory.at("price_gradient_bindings").end(); ++it) {
                if (it.value().at("identity").get<std::string>() == key) {
                    gradient_available = true;
                    device_prepared = it.value().at("preparation_strategy")
                        .get<std::string>().starts_with("device_prepared_");
                    const auto orders = it.value().at("supported_orders")
                        .get<std::vector<std::string>>();
                    if (diagonal && std::find(
                            orders.begin(), orders.end(), "diagonal_second"
                        ) == orders.end())
                        throw std::invalid_argument(
                            "The selected gradient binding does not expose diagonal order."
                        );
                    if (mixed && std::find(
                            orders.begin(), orders.end(), "mixed_second"
                        ) == orders.end())
                        throw std::invalid_argument(
                            "The selected gradient binding does not expose mixed order."
                        );
                }
            }
            if (!gradient_available)
                throw std::invalid_argument(
                    "No generated price-gradient binding for " + key
                );
            if ((diagonal || mixed) && !device_prepared)
                throw std::invalid_argument(
                    "The selected gradient binding has no higher-order dataset plan."
                );
            const auto plan = tuning::make_price_gradient_launch_plan(
                identity,
                positive_count(argv[2]),
                sensitivity_count,
                paths,
                limits
            );
            if (mixed) {
                const auto graph = pg::make_sensitivity_graph_plan(
                    pg::SensitivityRequest::full_hessian(),
                    sensitivity_count
                );
                metadata = tuning::mixed_node_graph_launch_metadata(
                    plan, sensitivity_count, graph
                );
            } else if (node_graph) {
                metadata = tuning::diagonal_node_graph_launch_metadata(
                    plan, sensitivity_count
                );
            } else {
                metadata = device_prepared
                    ? tuning::device_prepared_price_gradient_launch_metadata(plan,sensitivity_count)
                    : tuning::price_gradient_launch_metadata(plan,sensitivity_count);
            }
            if (diagonal) {
                if (!node_graph) {
                    metadata["maximum_live_scenarios"] =
                        sensitivity_count == 0U ? 1U : 4U;
                    metadata["represented_nodes_per_sensitivity"] =
                        sensitivity_count == 0U ? 0U : 4U;
                }
                metadata["requested_orders"] = {"first", "diagonal_second"};
            }
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

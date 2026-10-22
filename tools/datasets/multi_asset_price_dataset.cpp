#include "tools/datasets/multi_asset_price_dataset.hpp"

#include "common/dataset_validation.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

namespace ai_factory::workbench::datasets {
namespace {

void write_dataset(
    const MultiAssetPriceDatasetRecipe& recipe,
    const CompiledRowIndexPlan& row_plan,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    std::size_t paths_per_price,std::uint64_t base_seed,
    const nlohmann::ordered_json& execution,
    double wall_seconds,double kernel_seconds
) {
    const std::size_t result_count=row_plan.result_count;
    if (!recipe.url.empty()) validate_dataset_url(recipe.url);
    const auto asset_count=recipe.model_dataset_paths.size();
    if(asset_count==0U || result_count==0U
        || prices.size()!=result_count
        || standard_errors.size()!=result_count
        || paths_per_price<2U)
        throw std::invalid_argument(
            "Invalid multi-asset price output dimensions.");
    std::map<std::filesystem::path,nlohmann::ordered_json> model_cache;
    std::vector<const nlohmann::ordered_json*> models;
    models.reserve(asset_count);
    for(const auto& path:recipe.model_dataset_paths) {
        auto [entry,inserted]=model_cache.try_emplace(path);
        if(inserted) entry->second=read_json_file(path);
        models.push_back(&entry->second);
    }
    const auto product=read_json_file(recipe.product_dataset_path);
    const auto correlation=read_json_file(recipe.correlation_dataset_path);
    const auto curve=recipe.curve_dataset_path
        ? read_json_file(*recipe.curve_dataset_path)
        : nlohmann::ordered_json{};
    const auto rates=recipe.rate_model_dataset_path
        ? read_json_file(*recipe.rate_model_dataset_path)
        : nlohmann::ordered_json{};
    for(std::size_t asset=0U;asset<asset_count;++asset)
        if(row_plan.selectors[asset].source_count
            !=models[asset]->at("models").size())
            throw std::invalid_argument(
                "Compact model selector disagrees with mother dataset.");
    const auto rate_count=recipe.rate_model_dataset_path ? 1U : 0U;
    const auto product_slot=asset_count+rate_count;
    if((rate_count && row_plan.selectors[asset_count].source_count
                !=rates.at("models").size())
        ||row_plan.selectors[product_slot].source_count
            !=product.at("products").size()
        ||row_plan.selectors[product_slot+1U].source_count
            !=correlation.at("correlations").size()
        ||(recipe.curve_dataset_path
            &&row_plan.selectors[product_slot+2U].source_count
                !=curve.at("curves").size()))
        throw std::invalid_argument(
            "Compact selector disagrees with mother dataset.");
    const auto reference=[](const nlohmann::ordered_json& document) {
        return nlohmann::ordered_json{
            {"id",document.at("database_id")}};
    };
    nlohmann::ordered_json model_references=
        nlohmann::ordered_json::array();
    for(const auto* model:models)
        model_references.push_back(reference(*model));
    nlohmann::ordered_json results=nlohmann::ordered_json::array();
    std::vector<std::uint32_t> model_indices(asset_count);
    const auto correlation_slot=product_slot+1U;
    for(std::size_t i=0U;i<result_count;++i) {
        const auto row_index=static_cast<std::uint32_t>(i);
        const auto product_index=
            row_plan.selectors[product_slot].select(row_index);
        const auto correlation_index=
            row_plan.selectors[correlation_slot].select(row_index);
        const auto curve_index=recipe.curve_dataset_path
            ?row_plan.selectors[correlation_slot+1U].select(row_index):0U;
        const auto rate_index=rate_count
            ?row_plan.selectors[asset_count].select(row_index):0U;
        for(std::size_t asset=0U;asset<asset_count;++asset)
            model_indices[asset]=row_plan.selectors[asset].select(row_index);
        if(product_index>=product.at("products").size()
            ||correlation_index>=correlation.at("correlations").size()
            ||(recipe.rate_model_dataset_path
                &&rate_index>=rates.at("models").size())
            ||(recipe.curve_dataset_path
                &&curve_index>=curve.at("curves").size())
            ||!std::isfinite(prices[i])
            ||!std::isfinite(standard_errors[i])
            ||standard_errors[i]<0.0f)
            throw std::invalid_argument("Invalid multi-asset result row.");
        nlohmann::ordered_json ids=nlohmann::ordered_json::array();
        for(std::size_t asset=0U;asset<asset_count;++asset) {
            if(model_indices[asset]
                >=models[asset]->at("models").size())
                throw std::invalid_argument(
                    "Model index exceeds its mother dataset.");
            ids.push_back(models[asset]->at("models")
                .at(model_indices[asset]).at("id"));
        }
        nlohmann::ordered_json result{
            {"id",format_row_id(i)},
            {"model_ids",ids},
            {"correlation_id",correlation.at("correlations")
                .at(correlation_index).at("id")},
            {"product_id",product.at("products")
                .at(product_index).at("id")},
            {"seed",base_seed+i},
            {"outputs",{
                {"price",prices[i]},
                {"standard_error",standard_errors[i]}}}
        };
        if(recipe.rate_model_dataset_path)
            result["rate_model_id"]=rates.at("models")
                .at(rate_index).at("id");
        if(recipe.curve_dataset_path)
            result["curve_id"]=curve.at("curves")
                .at(curve_index).at("id");
        results.push_back(std::move(result));
    }
    nlohmann::ordered_json document{
        {"database_id",recipe.dataset_path.stem().string()},
        {"row_count",result_count},
        {"time_convention",product.at("time_convention")},
        {"model_datasets",model_references},
        {"product_dataset",reference(product)},
        {"correlation_dataset",reference(correlation)},
        {"price_construction",{{"method","compiled aligned/cartesian plan"},
            {"model_order","model_datasets order"},
            {"selectors",[&] {
                nlohmann::ordered_json descriptors=
                    nlohmann::ordered_json::array();
                for(const auto& selector:row_plan.selectors)
                    descriptors.push_back({
                        {"stride",selector.stride},
                        {"source_count",selector.source_count},
                        {"mode",selector.mode==RowIndexMode::Strided
                            ?"strided":"sampled_curve"}});
                return descriptors;
            }()}}},
        {"timing",{
            {"wall_seconds",wall_seconds},
            {"kernel_seconds",kernel_seconds}}},
        {"results",results}
    };
    if(recipe.curve_dataset_path)
        document["curve_dataset"]=reference(curve);
    if(recipe.rate_model_dataset_path)
        document["rate_model_dataset"]=reference(rates);
    write_json_file(recipe.dataset_path,document);
    validate_price_dataset_file(recipe.dataset_path);
    auto receipt=execution;
    if(!receipt.is_object())
        throw std::invalid_argument(
            "Execution metadata must be an object.");
    receipt["paths_per_price"]=paths_per_price;
    write_generation_receipt(recipe.catalog_path,result_count,receipt,
                             wall_seconds,kernel_seconds);
}

}  // namespace

void write_multi_asset_price_dataset(
    const MultiAssetPriceDatasetRecipe& recipe,
    const CompiledRowIndexPlan& row_plan,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors,
    std::size_t paths_per_price,std::uint64_t base_seed,
    const nlohmann::ordered_json& execution,
    double wall_seconds,double kernel_seconds
) {
    const auto asset_count=recipe.model_dataset_paths.size();
    const auto rate_count=recipe.rate_model_dataset_path ? 1U : 0U;
    const auto curve_count=recipe.curve_dataset_path ? 1U : 0U;
    const auto product_slot=asset_count+rate_count;
    const auto correlation_slot=product_slot+1U;
    if(row_plan.selectors.size()!=asset_count+rate_count+2U+curve_count)
        throw std::invalid_argument("Invalid compact price plan slots.");
    write_dataset(recipe,row_plan,prices,standard_errors,
        paths_per_price,base_seed,execution,wall_seconds,kernel_seconds);
}

}  // namespace ai_factory::workbench::datasets

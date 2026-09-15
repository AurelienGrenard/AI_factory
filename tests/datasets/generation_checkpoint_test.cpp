#include "tools/cuda/generation_checkpoint.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ai_factory::workbench::offline::cuda::GenerationCheckpoint;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(const std::vector<float>& actual,
                   const std::vector<float>& expected) {
    require(actual == expected, "checkpoint restored different values");
}

}  // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path()
        / ("ai_factory_generation_checkpoint_" + std::to_string(getpid()));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    {
        std::ofstream interrupted_manifest(directory / "results.checkpoint.tmp");
        interrupted_manifest << "interrupted manifest";
    }
    const std::string identity(64U, 'a');
    setenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR", directory.c_str(), 1);
    setenv("AI_FACTORY_GENERATION_CHECKPOINT_ID", identity.c_str(), 1);

    const std::vector<float> first_prices{1.0f, 2.0f};
    const std::vector<float> first_errors{0.1f, 0.2f};
    {
        GenerationCheckpoint checkpoint(5U, {"price", "standard_error"});
        require(checkpoint.enabled(), "configured checkpoint is disabled");
        require(checkpoint.completed_prices() == 0U, "new checkpoint is not empty");
        require(!std::filesystem::exists(directory / "results.checkpoint.tmp"),
                "interrupted manifest temporary was not removed");
        checkpoint.commit(0U, 2U, {
            std::span<const float>(first_prices),
            std::span<const float>(first_errors),
        });
    }

    const auto checkpoint_path = directory / "results.checkpoint";
    const auto valid_size = std::filesystem::file_size(checkpoint_path);
    {
        std::ofstream incomplete(checkpoint_path, std::ios::binary | std::ios::app);
        incomplete << "{\"type\":\"batch\",\"offset\":";
    }
    require(std::filesystem::file_size(checkpoint_path) > valid_size,
            "test did not append an interrupted record");

    const std::vector<float> final_prices{3.0f, 4.0f, 5.0f};
    const std::vector<float> final_errors{0.3f, 0.4f, 0.5f};
    {
        GenerationCheckpoint checkpoint(5U, {"price", "standard_error"});
        require(checkpoint.completed_prices() == 2U,
                "valid prefix was not recovered");
        require(checkpoint.resumed_prices() == 2U,
                "resumed price count is wrong");
        require(std::filesystem::file_size(checkpoint_path) == valid_size,
                "incomplete trailing record was not removed");
        checkpoint.commit(2U, 3U, {
            std::span<const float>(final_prices),
            std::span<const float>(final_errors),
        });
        std::vector<float> prices(5U);
        std::vector<float> errors(5U);
        checkpoint.restore_prefix({
            std::span<float>(prices), std::span<float>(errors)
        });
        require_equal(prices, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        require_equal(errors, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
    }

    {
        GenerationCheckpoint checkpoint(5U, {"price", "standard_error"});
        require(checkpoint.completed_prices() == 5U,
                "complete checkpoint was not recovered");
    }

    setenv("AI_FACTORY_GENERATION_CHECKPOINT_ID", std::string(64U, 'b').c_str(), 1);
    bool rejected = false;
    try {
        GenerationCheckpoint checkpoint(5U, {"price", "standard_error"});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "mismatched checkpoint identity was accepted");
    setenv("AI_FACTORY_GENERATION_CHECKPOINT_ID", identity.c_str(), 1);
    {
        GenerationCheckpoint checkpoint(5U, {"price", "standard_error"});
        require(checkpoint.completed_prices() == 5U,
                "failed checkpoint open retained its lock");
    }
    {
        std::fstream corrupt(
            checkpoint_path, std::ios::binary | std::ios::in | std::ios::out
        );
        corrupt.seekg(-2, std::ios::end);
        char value = '\0';
        corrupt.get(value);
        value ^= 1;
        corrupt.seekp(-2, std::ios::end);
        corrupt.put(value);
    }
    rejected = false;
    try {
        GenerationCheckpoint checkpoint(5U, {"price", "standard_error"});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "corrupt checkpoint payload was accepted");

    unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
    unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");
    GenerationCheckpoint disabled(5U, {"price", "standard_error"});
    require(!disabled.enabled(), "unconfigured checkpoint is enabled");
    std::filesystem::remove_all(directory);
    return 0;
}

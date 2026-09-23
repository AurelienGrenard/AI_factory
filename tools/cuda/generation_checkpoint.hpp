// Durable host-side checkpoints for long-running offline price generators.
#pragma once

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ai_factory::workbench::offline::cuda {

class GenerationCheckpoint {
public:
    GenerationCheckpoint(
        std::size_t total_prices,
        std::vector<std::string> channels
    ) : total_prices_(total_prices), channels_(std::move(channels)) {
        const char* configured_directory = std::getenv(
            "AI_FACTORY_GENERATION_CHECKPOINT_DIR"
        );
        const char* configured_identity = std::getenv(
            "AI_FACTORY_GENERATION_CHECKPOINT_ID"
        );
        const bool has_directory = configured_directory != nullptr
            && *configured_directory != '\0';
        const bool has_identity = configured_identity != nullptr
            && *configured_identity != '\0';
        if (!has_directory && !has_identity) return;
        if (!has_directory || !has_identity) {
            throw std::invalid_argument(
                "A generation checkpoint requires both its directory and identity."
            );
        }
        if (total_prices_ == 0U || channels_.empty()
            || std::any_of(channels_.begin(), channels_.end(), [](const auto& name) {
                return name.empty();
            })) {
            throw std::invalid_argument(
                "A generation checkpoint requires prices and named output channels."
            );
        }
        identity_ = configured_identity;
        if (identity_.size() != 64U
            || std::any_of(identity_.begin(), identity_.end(), [](char value) {
                return !((value >= '0' && value <= '9')
                    || (value >= 'a' && value <= 'f'));
            })) {
            throw std::invalid_argument(
                "The generation checkpoint identity must be a lowercase SHA-256."
            );
        }

        directory_ = configured_directory;
        std::filesystem::create_directories(directory_);
        acquire_lock();
        try {
            path_ = directory_ / "results.checkpoint";
            temporary_path_ = directory_ / "results.checkpoint.tmp";
            values_.reserve(channels_.size());
            for (std::size_t index = 0U; index < channels_.size(); ++index) {
                values_.emplace_back(total_prices_);
            }
            if (std::filesystem::exists(path_)) {
                std::filesystem::remove(temporary_path_);
                load();
            } else {
                std::filesystem::remove(temporary_path_);
                initialize();
            }
            resumed_prices_ = completed_prices_;
            enabled_ = true;
        } catch (...) {
            release_lock();
            throw;
        }
    }

    ~GenerationCheckpoint() {
        release_lock();
    }

    GenerationCheckpoint(const GenerationCheckpoint&) = delete;
    GenerationCheckpoint& operator=(const GenerationCheckpoint&) = delete;

    bool enabled() const noexcept { return enabled_; }
    std::size_t completed_prices() const noexcept { return completed_prices_; }
    std::size_t resumed_prices() const noexcept { return resumed_prices_; }

    void commit(
        std::size_t offset,
        std::size_t count,
        const std::vector<std::span<const float>>& channel_values
    ) {
        if (!enabled_) return;
        if (offset != completed_prices_ || count == 0U
            || count > total_prices_ - offset
            || channel_values.size() != channels_.size()
            || std::any_of(channel_values.begin(), channel_values.end(),
                [count](const auto values) { return values.size() != count; })) {
            throw std::invalid_argument(
                "Generation checkpoint batches must form one contiguous prefix."
            );
        }

        std::uint64_t checksum = kFnvOffset;
        for (const auto values : channel_values) {
            checksum = update_checksum(
                checksum,
                reinterpret_cast<const unsigned char*>(values.data()),
                values.size_bytes()
            );
        }
        const std::size_t payload_bytes = checked_payload_bytes(count);
        const nlohmann::ordered_json record{
            {"type", "batch"},
            {"offset", offset},
            {"count", count},
            {"payload_bytes", payload_bytes},
            {"checksum", checksum_text(checksum)},
        };
        const std::string header = record.dump() + "\n";

        int descriptor = open(path_.c_str(), O_WRONLY | O_APPEND | O_CLOEXEC);
        if (descriptor < 0) throw_system_error("open generation checkpoint");
        try {
            write_all(descriptor, header.data(), header.size());
            for (const auto values : channel_values) {
                write_all(descriptor, values.data(), values.size_bytes());
            }
            constexpr char separator = '\n';
            write_all(descriptor, &separator, 1U);
            if (fsync(descriptor) != 0) {
                throw_system_error("flush generation checkpoint");
            }
            if (close(descriptor) != 0) {
                throw_system_error("close generation checkpoint");
            }
            descriptor = -1;
        } catch (...) {
            if (descriptor >= 0) close(descriptor);
            throw;
        }

        for (std::size_t channel = 0U; channel < channels_.size(); ++channel) {
            std::copy(
                channel_values[channel].begin(), channel_values[channel].end(),
                values_[channel].begin() + static_cast<std::ptrdiff_t>(offset)
            );
        }
        completed_prices_ += count;
    }

    void restore_prefix(const std::vector<std::span<float>>& destinations) const {
        if (!enabled_) return;
        if (destinations.size() != channels_.size()
            || std::any_of(destinations.begin(), destinations.end(),
                [this](const auto values) {
                    return values.size() != total_prices_;
                })) {
            throw std::invalid_argument(
                "Generation checkpoint restore destinations have the wrong shape."
            );
        }
        for (std::size_t channel = 0U; channel < channels_.size(); ++channel) {
            std::copy_n(
                values_[channel].begin(), completed_prices_,
                destinations[channel].begin()
            );
        }
    }

private:
    static constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
    static constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    static constexpr int kSchemaVersion = 1;

    void acquire_lock() {
        const auto lock_path = directory_ / "checkpoint.lock";
        lock_fd_ = open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (lock_fd_ < 0) throw_system_error("open generation checkpoint lock");
        if (flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
            close(lock_fd_);
            lock_fd_ = -1;
            throw std::runtime_error(
                "The generation checkpoint is already in use: "
                + directory_.string()
            );
        }
    }

    void release_lock() noexcept {
        if (lock_fd_ < 0) return;
        flock(lock_fd_, LOCK_UN);
        close(lock_fd_);
        lock_fd_ = -1;
    }

    void initialize() {
        const nlohmann::ordered_json manifest{
            {"type", "manifest"},
            {"schema_version", kSchemaVersion},
            {"identity", identity_},
            {"total_prices", total_prices_},
            {"channels", channels_},
        };
        const std::string contents = manifest.dump() + "\n";
        int descriptor = open(
            temporary_path_.c_str(),
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
            0600
        );
        if (descriptor < 0) throw_system_error("create generation checkpoint");
        try {
            write_all(descriptor, contents.data(), contents.size());
            if (fsync(descriptor) != 0) {
                throw_system_error("flush generation checkpoint manifest");
            }
            if (close(descriptor) != 0) {
                throw_system_error("close generation checkpoint manifest");
            }
            descriptor = -1;
            std::filesystem::rename(temporary_path_, path_);
            sync_directory();
        } catch (...) {
            if (descriptor >= 0) close(descriptor);
            throw;
        }
    }

    void load() {
        std::ifstream input(path_, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "Cannot read generation checkpoint: " + path_.string()
            );
        }
        std::string line;
        if (!std::getline(input, line)) {
            throw std::runtime_error("Generation checkpoint manifest is missing.");
        }
        const auto manifest = parse_json(line, "manifest");
        if (manifest.value("type", "") != "manifest"
            || manifest.value("schema_version", 0) != kSchemaVersion
            || manifest.value("identity", "") != identity_
            || manifest.value("total_prices", std::size_t{0}) != total_prices_
            || manifest.value("channels", std::vector<std::string>{}) != channels_) {
            throw std::runtime_error(
                "Generation checkpoint does not match the frozen dataset."
            );
        }

        std::uintmax_t valid_size = static_cast<std::uintmax_t>(input.tellg());
        while (input.peek() != std::char_traits<char>::eof()) {
            if (!std::getline(input, line)) {
                truncate_to(valid_size);
                break;
            }
            if (input.eof()) {
                truncate_to(valid_size);
                break;
            }
            nlohmann::json record;
            try {
                record = nlohmann::json::parse(line);
            } catch (const nlohmann::json::exception&) {
                if (input.peek() == std::char_traits<char>::eof()) {
                    truncate_to(valid_size);
                    break;
                }
                throw std::runtime_error("Generation checkpoint batch header is corrupt.");
            }
            const std::size_t offset = record.value(
                "offset", std::numeric_limits<std::size_t>::max()
            );
            const std::size_t count = record.value("count", std::size_t{0});
            if (record.value("type", "") != "batch"
                || offset != completed_prices_ || count == 0U
                || count > total_prices_ - offset
                || record.value("payload_bytes", std::size_t{0})
                    != checked_payload_bytes(count)) {
                throw std::runtime_error(
                    "Generation checkpoint batches are not a valid contiguous prefix."
                );
            }
            const std::size_t payload_bytes = checked_payload_bytes(count);
            std::vector<unsigned char> payload(payload_bytes);
            input.read(
                reinterpret_cast<char*>(payload.data()),
                static_cast<std::streamsize>(payload.size())
            );
            if (static_cast<std::size_t>(input.gcount()) != payload.size()) {
                truncate_to(valid_size);
                break;
            }
            char separator = '\0';
            if (!input.get(separator)) {
                truncate_to(valid_size);
                break;
            }
            if (separator != '\n') {
                throw std::runtime_error("Generation checkpoint batch separator is corrupt.");
            }
            const auto expected_checksum = record.value("checksum", "");
            if (expected_checksum != checksum_text(update_checksum(
                    kFnvOffset, payload.data(), payload.size()))) {
                throw std::runtime_error("Generation checkpoint batch checksum mismatch.");
            }
            for (std::size_t channel = 0U; channel < channels_.size(); ++channel) {
                std::memcpy(
                    values_[channel].data() + offset,
                    payload.data() + channel * count * sizeof(float),
                    count * sizeof(float)
                );
            }
            completed_prices_ += count;
            valid_size = static_cast<std::uintmax_t>(input.tellg());
        }
    }

    std::size_t checked_payload_bytes(std::size_t count) const {
        if (count > std::numeric_limits<std::size_t>::max()
                / sizeof(float) / channels_.size()) {
            throw std::overflow_error("Generation checkpoint batch is too large.");
        }
        return count * channels_.size() * sizeof(float);
    }

    static nlohmann::json parse_json(
        const std::string& value, const char* description
    ) {
        try {
            return nlohmann::json::parse(value);
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error(
                std::string("Invalid generation checkpoint ") + description
                + ": " + error.what()
            );
        }
    }

    static std::uint64_t update_checksum(
        std::uint64_t checksum,
        const unsigned char* data,
        std::size_t size
    ) noexcept {
        for (std::size_t index = 0U; index < size; ++index) {
            checksum ^= data[index];
            checksum *= kFnvPrime;
        }
        return checksum;
    }

    static std::string checksum_text(std::uint64_t checksum) {
        std::ostringstream output;
        output << std::hex << std::setfill('0') << std::setw(16) << checksum;
        return output.str();
    }

    static void write_all(int descriptor, const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        while (size != 0U) {
            const ssize_t written = write(descriptor, bytes, size);
            if (written < 0) {
                if (errno == EINTR) continue;
                throw_system_error("write generation checkpoint");
            }
            if (written == 0) {
                throw std::runtime_error("Generation checkpoint write made no progress.");
            }
            bytes += written;
            size -= static_cast<std::size_t>(written);
        }
    }

    static void throw_system_error(const char* operation) {
        throw std::system_error(errno, std::generic_category(), operation);
    }

    void truncate_to(std::uintmax_t size) {
        std::filesystem::resize_file(path_, size);
        const int descriptor = open(path_.c_str(), O_WRONLY | O_CLOEXEC);
        if (descriptor < 0) throw_system_error("open truncated generation checkpoint");
        if (fsync(descriptor) != 0) {
            const int saved_errno = errno;
            close(descriptor);
            errno = saved_errno;
            throw_system_error("flush truncated generation checkpoint");
        }
        if (close(descriptor) != 0) {
            throw_system_error("close truncated generation checkpoint");
        }
    }

    void sync_directory() const {
        const int descriptor = open(directory_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (descriptor < 0) throw_system_error("open generation checkpoint directory");
        if (fsync(descriptor) != 0) {
            const int saved_errno = errno;
            close(descriptor);
            errno = saved_errno;
            throw_system_error("flush generation checkpoint directory");
        }
        if (close(descriptor) != 0) {
            throw_system_error("close generation checkpoint directory");
        }
    }

    std::size_t total_prices_ = 0U;
    std::vector<std::string> channels_;
    std::string identity_;
    std::filesystem::path directory_;
    std::filesystem::path path_;
    std::filesystem::path temporary_path_;
    std::vector<std::vector<float>> values_;
    std::size_t completed_prices_ = 0U;
    std::size_t resumed_prices_ = 0U;
    int lock_fd_ = -1;
    bool enabled_ = false;
};

}  // namespace ai_factory::workbench::offline::cuda

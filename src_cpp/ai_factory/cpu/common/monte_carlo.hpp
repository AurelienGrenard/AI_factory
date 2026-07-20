#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ai_factory::simulation {

struct SimulationConfig {
    std::uint64_t seed;
    std::size_t num_paths;
    std::string random_backend;
};

struct TimedTerminalSpots {
    std::vector<double> terminal_spots;
    double simulation_seconds;
};

struct TimedMaxSpots {
    std::vector<double> max_spots;
    double simulation_seconds;
};

}  // namespace ai_factory::simulation

#pragma once

#include <cstdint>

namespace substrate {

struct HiveStorePlasticity {
    std::uint8_t plasticity_delta_q8 = 0u;
    std::uint8_t reserved_q8 = 0u;
    std::uint16_t reserved = 0u;
};

struct HiveStoreEntry {
    std::uint32_t voxel_id = 0u;
    std::uint32_t entry_type = 0u;
    HiveStorePlasticity plasticity{};
};

}  // namespace substrate

// shared_memory_protocol.h — minimal shared-memory protocol types.
// Provides the IPC header used by engine_swarm_heatmap.cu.
#pragma once
#include <cstdint>
#include <cstddef>

namespace substrate {

static constexpr uint64_t SHM_SWARM_HEATMAP_MAGIC = 0x30783148454154ULL;
static constexpr uint32_t SWARM_HEATMAP_IPC_VERSION = 1u;
static constexpr uint32_t SWARM_HEATMAP_IPC_FLAGS_NONE = 0u;

struct SwarmHeatmapIpcHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t concept_count;
    uint32_t payload_bytes;
    uint64_t sequence;
};

} // namespace substrate

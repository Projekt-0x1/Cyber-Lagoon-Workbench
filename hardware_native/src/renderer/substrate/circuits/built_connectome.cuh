#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace substrate::circuits {

static constexpr uint64_t kSubstrateRuntimeMinimumLogicalCells = 80000000000ull;
static constexpr uint64_t kSubstrateRuntimeNeurogenesisLogicalCells = 160000000000ull;

struct BuiltStream {
    std::string id;
    std::string domain;
    std::string owner;
    std::string purpose;
    uint32_t base = 0u;
    uint32_t words = 0u;
    uint32_t init_fill = 0u;
    bool init_fill_set = false;
};

struct BuiltDroneSurface {
    std::string family;
    std::string domain;
    uint32_t base = 0u;
    uint32_t words = 0u;
};

struct BuiltConnectome {
    std::vector<BuiltStream> streams;
    std::vector<BuiltDroneSurface> drone_surfaces;
    uint32_t pool_words = 0u;
    bool exact_memory = true;

    bool apply_init_fills(uint32_t* d_pool, cudaStream_t stream) const;
};

}  // namespace substrate::circuits

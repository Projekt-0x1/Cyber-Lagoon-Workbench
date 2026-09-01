#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "substrate/circuits/built_connectome.cuh"
#include "substrate/circuits/circuit_organoid_cell_dish.cuh"

namespace substrate::circuits {

inline bool BuiltConnectome::apply_init_fills(uint32_t* d_pool, cudaStream_t stream) const {
    if (!d_pool) return false;
    for (const auto& s : streams) {
        if (!s.init_fill_set || s.words == 0u) continue;
        if (s.init_fill == 0u) {
            cudaMemsetAsync(d_pool + s.base, 0, sizeof(uint32_t) * s.words, stream);
        } else if (s.init_fill == 0xFFFFFFFFu) {
            cudaMemsetAsync(d_pool + s.base, 0xFF, sizeof(uint32_t) * s.words, stream);
        } else {
            std::vector<uint32_t> fill(s.words, s.init_fill);
            cudaMemcpyAsync(
                d_pool + s.base,
                fill.data(),
                sizeof(uint32_t) * s.words,
                cudaMemcpyHostToDevice,
                stream);
        }
    }
    return true;
}

inline bool build_connectome(const std::string& raw, BuiltConnectome* out) {
    if (!out) return false;
    if (raw != "organoid_cell_dish") return false;
    *out = BuiltConnectome{};
    circuit_organoid_cell_dish(out);
    return out->pool_words != 0u && !out->streams.empty();
}

}  // namespace substrate::circuits

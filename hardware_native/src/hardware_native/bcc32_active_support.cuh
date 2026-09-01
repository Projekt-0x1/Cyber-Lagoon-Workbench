#pragma once

#include "bcc32_cuda_executor.cuh"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace substrate::bcc32 {

// Fixed-grid graph-safe primitives. The launch capacity is immutable, while
// each kernel reads the active count from device memory. No function here
// reads or writes a host frontier, and unused entries are never executed.
void launch_active_site_graph_safe(
    SiteWord* words, const std::uint64_t* slots,
    const std::uint32_t* device_count, std::uint32_t capacity, bool inverse,
    cudaStream_t stream = nullptr);
void launch_active_edge_graph_safe(
    SiteWord* words, const std::uint64_t* slots,
    const std::uint32_t* device_count, std::uint32_t capacity,
    const DeviceChunkMap& chunks, bool inverse,
    cudaStream_t stream = nullptr);
void launch_active_stream_graph_safe(
    SiteWord* words, const std::uint64_t* slots, std::uint8_t* holes,
    const std::uint32_t* device_count, std::uint32_t capacity,
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream = nullptr);

enum class ActiveSupportPolicy : std::uint8_t {
    dynamic,
    fixed,
};

// Exact sparse scheduling over the direct dense field. The active list is
// derived execution metadata: every persistent material bit remains in the
// executor's ordinary uint32 aperture, and the canonical law is unchanged.
// This first implementation deliberately accepts only CUDA's default stream;
// the caller must not submit another operation to the same executor concurrently.
class CudaBcc32ActiveSupport {
public:
    CudaBcc32ActiveSupport(CudaBcc32Executor& executor,
                           std::span<const DeviceChunkSlot> host_topology,
                           std::span<const std::uint64_t> initial_support,
                           std::span<const std::uint64_t> persistent_anchors = {},
                           ActiveSupportPolicy policy =
                               ActiveSupportPolicy::dynamic);
    CudaBcc32ActiveSupport(const CudaBcc32ActiveSupport&) = delete;
    CudaBcc32ActiveSupport& operator=(const CudaBcc32ActiveSupport&) = delete;
    CudaBcc32ActiveSupport(CudaBcc32ActiveSupport&&) = delete;
    CudaBcc32ActiveSupport& operator=(CudaBcc32ActiveSupport&&) = delete;
    ~CudaBcc32ActiveSupport();

    void include(std::span<const std::uint64_t> slots);
    // Reconcile sparse scheduling metadata with authoritative dense words
    // after an external reciprocal material exchange.  Unlike include(),
    // this is an involution-compatible mid-life operation: touched non-Q
    // sites are enrolled, touched Q sites are retired unless anchored, and
    // prior phase journals remain intact.
    void reconcile_external_contacts(
        std::span<const std::uint64_t> slots,
        cudaStream_t stream = nullptr);
    // Rebinds derived scheduling metadata after an exact external field copy.
    // It never writes material state.
    void reset(std::span<const std::uint64_t> support);
    void apply_superstep(const DeviceChunkMap& chunks,
                         cudaStream_t stream = nullptr);
    void apply_superstep_inverse(const DeviceChunkMap& chunks,
                                 cudaStream_t stream = nullptr);

    [[nodiscard]] const std::vector<std::uint64_t>& active_slots() const {
        return active_;
    }
    [[nodiscard]] const std::uint64_t* device_active_slots(
        cudaStream_t stream = nullptr);
    [[nodiscard]] std::vector<SiteWord> download_active_words(
        cudaStream_t stream = nullptr) const;
    [[nodiscard]] std::size_t reversible_supersteps() const {
        return phase_history_.size();
    }
    [[nodiscard]] bool can_reverse_supersteps(std::size_t count) const {
        return policy_ == ActiveSupportPolicy::fixed ||
               phase_history_.size() >= count;
    }

private:
    void rebind_active(std::span<const std::uint64_t> support);
    struct SupportDelta {
        std::vector<std::uint64_t> added;
        std::vector<std::uint64_t> removed;
    };
    [[nodiscard]] SupportDelta support_delta(
        std::span<const std::uint64_t> before,
        std::span<const std::uint64_t> after) const;
    void reverse_support_delta(std::vector<std::uint64_t>* support,
                               const SupportDelta& delta) const;
    void expand_radius_into(std::vector<std::uint64_t>& expanded,
                            std::span<const std::uint64_t> slots,
                            std::uint32_t radius) const;
    void validate_full_superstep_closure(cudaStream_t stream);
    void refresh(std::span<const std::uint64_t> candidates,
                 cudaStream_t stream);
    void ensure_capacity(std::size_t count);
    void apply_site(bool inverse, cudaStream_t stream);
    void apply_edge(const DeviceChunkMap& chunks, bool inverse,
                    cudaStream_t stream);
    void apply_stream(const DeviceChunkMap& chunks, bool inverse,
                      cudaStream_t stream);

    CudaBcc32Executor& executor_;
    std::vector<DeviceChunkSlot> topology_;
    std::vector<std::uint64_t> anchors_;
    std::vector<std::uint64_t> active_;
    struct PhaseSupportJournal {
        // External contacts may re-enrol a site whose authoritative word has
        // already returned to Q. Forward retires that derived-only root before
        // topology preflight; inverse restores the exact tick-entry scheduler
        // state after all material factors have been undone.
        SupportDelta entry_to_append;
        SupportDelta append_to_edge;
        SupportDelta edge_to_macro;
        SupportDelta macro_to_prediction;
        SupportDelta prediction_to_end;
    };
    // Derived scheduler history only: represented matter remains entirely in
    // the dense aperture. Checkpoint load starts a new reversible boundary.
    std::vector<PhaseSupportJournal> phase_history_;
    mutable std::vector<std::uint8_t> expansion_seen_;
    // Reused BFS, closure, delta, and snapshot scratch.  Capacity persists
    // across supersteps; contents never outlive the call that filled them.
    mutable std::vector<std::uint64_t> expansion_frontier_;
    mutable std::vector<std::uint64_t> expansion_next_;
    mutable std::vector<std::uint64_t> expansion_result_;
    mutable std::vector<std::uint64_t> delta_scratch_a_;
    mutable std::vector<std::uint64_t> delta_scratch_b_;
    mutable std::vector<std::uint64_t> refresh_candidates_;
    mutable std::vector<SiteWord> refresh_samples_;
    std::vector<std::uint64_t> phase_support_;
    std::vector<std::uint64_t> tick_entry_support_;
    ActiveSupportPolicy policy_ = ActiveSupportPolicy::dynamic;
    std::uint64_t* device_slots_ = nullptr;
    SiteWord* device_samples_ = nullptr;
    std::uint8_t* device_holes_ = nullptr;
    std::uint8_t* device_resolution_ = nullptr;
    std::size_t capacity_ = 0u;
};

}  // namespace substrate::bcc32

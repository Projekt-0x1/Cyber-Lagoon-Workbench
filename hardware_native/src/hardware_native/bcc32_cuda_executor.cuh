#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

#include "bcc32_geometry.cuh"
#include "bcc32_membrane.hpp"
#include "bcc32_law_netlist.cuh"
#include "bcc32_types.cuh"

namespace substrate::bcc32 {

constexpr std::uint64_t kCarrierSnapshotBytes = kProductionSites;
constexpr std::uint64_t kProductionExecutorHeadroomBytes =
    128ull * 1024ull * 1024ull;

static_assert(kCarrierSnapshotBytes == 2'500'000'000ull);

// One entry describes one dense 100^3 resident slot. Coordinates are bounded
// aperture-local topology labels, never logical material coordinates.
// Neighbor entries are direct BCC chunk displacements; -1 denotes canonical Q
// outside this page.
// This compact topology is metadata, not material state and is never indexed
// per site.
struct DeviceChunkSlot {
    static constexpr std::int32_t kMissing = -1;

    std::int64_t chunk_x = 0;
    std::int64_t chunk_y = 0;
    std::int64_t chunk_z = 0;
    std::int32_t bcc_neighbors[8]{
        kMissing, kMissing, kMissing, kMissing,
        kMissing, kMissing, kMissing, kMissing,
    };
};

struct DeviceChunkMap {
    const DeviceChunkSlot* slots = nullptr;
    std::uint32_t chunk_count = 0u;
};

// Owns exactly one direct uint32 aperture and one reusable byte-per-site P
// snapshot. Caller-owned chunk metadata supplies open-Z3 residency; no flat
// lattice shape, torus, per-site neighbor table, managed allocation, or host
// mirror exists in this executor. Calls on separate CUDA streams are ordered
// through one executor-local completion event; calls on the same stream retain
// their native asynchronous ordering.
class CudaBcc32ActiveSupport;
namespace developmental_adult { class GrownAdult; }
namespace device_ordinary_f_timeline { class DeviceOrdinaryFTimeline; }

// Fixed-grid graph nodes use this launch shape. The launch width is immutable,
// while every kernel reads the authoritative active count from device memory
// before touching a slot. Unused storage is never represented by a real
// padding site and therefore cannot acquire repeated factor updates.
struct DeviceActiveSupportWindow {
    const std::uint64_t* slots = nullptr;
    const std::uint32_t* count = nullptr;
    // Per-candidate collision result. This is bounded by the active window;
    // it is never a second full-aperture byte array.
    std::uint8_t* resolved = nullptr;
    std::uint32_t capacity = 0u;
};

class CudaBcc32Executor {
public:
    [[nodiscard]] static CudaBcc32Executor production();
    [[nodiscard]] static CudaBcc32Executor testing(std::uint32_t chunk_count);

    CudaBcc32Executor(const CudaBcc32Executor&) = delete;
    CudaBcc32Executor& operator=(const CudaBcc32Executor&) = delete;
    CudaBcc32Executor(CudaBcc32Executor&& other) noexcept;
    CudaBcc32Executor& operator=(CudaBcc32Executor&& other) noexcept;
    ~CudaBcc32Executor();

    [[nodiscard]] std::uint32_t chunk_count() const { return chunk_count_; }
    [[nodiscard]] std::uint64_t site_count() const { return site_count_; }
    [[nodiscard]] std::uint64_t aperture_bytes() const { return aperture_bytes_; }
    [[nodiscard]] std::uint64_t carrier_snapshot_bytes() const {
        return carrier_snapshot_bytes_;
    }
 private:
    [[nodiscard]] SiteWord* mutable_device_words() { return words_; }

    // ⛔ THE MUTABLE ROUTE IS NOT PUBLIC. Only the two classes that legitimately
    // own organism memory may take it: GrownAdult, which performs canonical
    // factor writes and boundary transactions, and CudaBcc32ActiveSupport, which
    // applies K_site over the scheduled support. Everyone else reads through the
    // const overload below.
    //
    // ⚠ This is a friend list, not a guarantee -- a friend can still do anything,
    // and the list is exactly what tools/audit_world_write_authority.sh counts as
    // the owning set. What changes is that a NEW caller cannot take the route by
    // accident; it has to be added here, in the open.
    friend class developmental_adult::GrownAdult;
    friend class ::substrate::bcc32::CudaBcc32ActiveSupport;
    friend class device_ordinary_f_timeline::DeviceOrdinaryFTimeline;

 public:
    [[nodiscard]] const SiteWord* device_words() const { return words_; }

    // ⭐ GENESIS AUTHORITY, WITH A PRECONDITION THAT CAN ACTUALLY REFUSE.
    //
    // §0.12's two mutation classes govern POST-genesis change. Arranging matter
    // before the first tick is neither a factor write nor a boundary
    // transaction, and the search tools that build many candidate worlds in one
    // allocation legitimately need to do it. Before this they took
    // `device_words()` and launched a seeding kernel; when the mutable route
    // became private they stopped compiling, and 28 translation units went red.
    //
    // ⛔ WHAT MAKES IT A CONTROL RATHER THAN A RENAME. `apply_superstep`
    // increments `supersteps_`, and this REFUSES once that is nonzero. So the
    // route is genesis-only by construction, not by convention -- a post-t0
    // caller gets an exception, not a write. The pointer is scoped to the
    // operation and never handed over.
    //
    // ⚠ It is still an authority, not a proof: a callable can copy the pointer
    // out, and requirement 5's declared write set is unbuilt. It converts six
    // anonymous seeding routes into one named one that can say no.
    template <typename Operation>
    void seed_before_first_tick(Operation&& operation) {
        if (supersteps_ != 0u) {
            throw std::logic_error(
                "genesis seeding requested after the first tick");
        }
        // ⛔ THROUGH THE COUNTED ACCESSOR, NOT `words_` DIRECTLY. Touching the
        // member would make this a write route the ratchet cannot see -- a new
        // authority invisible to the instrument that exists to count
        // authorities. It costs nothing and it keeps the number honest.
        std::forward<Operation>(operation)(mutable_device_words());
    }

    [[nodiscard]] std::uint64_t supersteps() const { return supersteps_; }

    // ⛔ THE GUARD ABOVE WAS BLIND ON THE COMMON PATH, AND ITS PROBE WAS TOO.
    //
    // `seed_before_first_tick` refuses once `supersteps_` is nonzero, and
    // `CudaBcc32Executor::apply_superstep` increments it. But the tree advances
    // the world mostly through `CudaBcc32ActiveSupport::apply_superstep`, which
    // does NOT call it -- it calls the apply_active_* family directly. So the
    // counter stayed 0 through any number of ticks and the refusal never fired.
    //
    // ⚠ The probe passed because it used `executor.apply_superstep(chunks)`, the
    // one path that does increment. A control exercised only on the path that
    // works is not evidence about the path that does not.
    //
    // ⇒ the active support notes its own advance here. It is already a friend of
    // this class, so no new route is opened.
    void note_world_advances(std::uint64_t count) {
        if (count > std::numeric_limits<std::uint64_t>::max() - supersteps_)
            throw std::overflow_error("BCC32 world-advance receipt overflow");
        supersteps_ += count;
    }
    void note_world_advance() { note_world_advances(1u); }

    // ⭐ THE SAME TWO NAMED CLASSES, HOSTED WHERE THE OWNER ACTUALLY IS.
    //
    // GrownAdult carries these because it owns an organism. A caller driving the
    // executor directly -- the search tools, and the contracts that build many
    // candidate worlds in one allocation -- has no adult, and had no route at all
    // once the mutable accessor went private.
    //
    // ⛔ NOT A THIRD AND FOURTH CLASS. It is the SAME taxonomy that
    // tools/audit_direct_world_writes.sh showed classifies 72 independent sites
    // with no residual, made reachable without an organism. Inventing executor-
    // specific class names to fit these call sites is what would make the census
    // meaningless.
    //
    // ⚠ Same limits as the adult's: the pointer is scoped to the operation, a
    // callable can still copy it out, and requirement 5's declared write set is
    // unbuilt. These count and name; they do not enforce.
    template <typename Operation>
      // ⭐ THE BROKER FORWARDS THE OPERATION'S RESULT, and that is a measurement
  // result rather than a preference. Nine call sites consume a helper's return
  // value (`auto r = sparse::lesion_transition(words, ...)`), and a broker that
  // returns a receipt cannot wrap them -- the caller would receive the receipt
  // where it expects the helper's value. Twice already a transform produced
  // exactly that and would not have compiled.
  //
  // ⚠ The receipt is not lost, it is moved off the return path: `last_intervene()`
  // exposes it, and no caller consumed the returned receipt before this change
  // (checked, not assumed). Counting and epoch-stamping are unchanged.
  decltype(auto) intervene(InterventionReason reason,
                         Operation&& operation) {
    InterventionReceipt receipt;
    receipt.epoch = supersteps_;
    receipt.intervention = ++interventions_;
    receipt.reason = reason;
    last_intervene_ = receipt;
    return std::forward<Operation>(operation)(mutable_device_words());
  }
    template <typename Operation>
      // ⭐ THE BROKER FORWARDS THE OPERATION'S RESULT, and that is a measurement
  // result rather than a preference. Nine call sites consume a helper's return
  // value (`auto r = sparse::lesion_transition(words, ...)`), and a broker that
  // returns a receipt cannot wrap them -- the caller would receive the receipt
  // where it expects the helper's value. Twice already a transform produced
  // exactly that and would not have compiled.
  //
  // ⚠ The receipt is not lost, it is moved off the return path: `last_resident_stage()`
  // exposes it, and no caller consumed the returned receipt before this change
  // (checked, not assumed). Counting and epoch-stamping are unchanged.
  decltype(auto) resident_stage(ResidentStageReason reason,
                              Operation&& operation) {
    ResidentStageReceipt receipt;
    receipt.epoch = supersteps_;
    receipt.stage = ++resident_stages_;
    receipt.reason = reason;
    last_resident_stage_ = receipt;
    return std::forward<Operation>(operation)(mutable_device_words());
  }
    [[nodiscard]] std::uint64_t interventions() const { return interventions_; }
    [[nodiscard]] std::uint64_t resident_stages() const { return resident_stages_; }


    void initialize_q(cudaStream_t stream = nullptr);
    void upload_words(std::uint64_t offset, std::span<const SiteWord> source,
                      cudaStream_t stream = nullptr);
    void download_words(std::uint64_t offset, std::span<SiteWord> destination,
                        cudaStream_t stream = nullptr) const;

    void apply_superstep(const DeviceChunkMap& chunks,
                         cudaStream_t stream = nullptr);
    void apply_superstep_inverse(const DeviceChunkMap& chunks,
                                 cudaStream_t stream = nullptr);
  void apply_carrier_pair_splitter(const DeviceChunkMap& chunks, bool inverse,
                                   cudaStream_t stream = nullptr);
  void apply_processive_rearm(const DeviceChunkMap& chunks, bool inverse,
                              cudaStream_t stream = nullptr);
  void apply_processive_release(const DeviceChunkMap& chunks, bool inverse,
                                cudaStream_t stream = nullptr);
  void apply_active_processive_release(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream = nullptr);
  void apply_active_processive_release(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      std::uint8_t* resolved,
      cudaStream_t stream = nullptr);
  void apply_carrier_corner(const DeviceChunkMap& chunks, bool inverse,
                            cudaStream_t stream = nullptr);
  void apply_eligibility_residual_junction(
      const DeviceChunkMap& chunks, bool inverse,
      cudaStream_t stream = nullptr);
  void apply_active_eligibility_residual_junction(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream = nullptr);
  void apply_developmental_append(const DeviceChunkMap& chunks, bool inverse,
                                  cudaStream_t stream = nullptr);
  void apply_active_developmental_append(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream = nullptr);
  void apply_developmental_learned_receptor(
      const DeviceChunkMap& chunks, bool inverse,
      cudaStream_t stream = nullptr);
  void apply_active_developmental_learned_receptor(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream = nullptr);
  void apply_developmental_credit_service(
      const DeviceChunkMap& chunks, bool inverse,
      cudaStream_t stream = nullptr);
  void apply_active_developmental_credit_service(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream = nullptr);
  // Sparse scheduler entry point for the four post-edge macro factors. `active_slots`
  // names candidate centers in the same direct field; all neighborhood reads
  // and writes still address the authoritative aperture.
  void apply_active_macro_factors(const DeviceChunkMap& chunks, bool inverse,
                                  const std::uint64_t* active_slots,
                                  std::uint64_t active_count,
                                  cudaStream_t stream = nullptr);
  void apply_active_macro_factors(const DeviceChunkMap& chunks, bool inverse,
                                  const std::uint64_t* active_slots,
                                  std::uint64_t active_count,
                                  std::uint8_t* resolved,
                                  cudaStream_t stream = nullptr);
  void apply_active_prediction_residual_route_toggle(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream = nullptr);
  void apply_active_prediction_residual_route_toggle(
      const DeviceChunkMap& chunks, bool inverse,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      std::uint8_t* resolved,
      cudaStream_t stream = nullptr);
  void apply_prediction_residual_route_toggle(
      const DeviceChunkMap& chunks, bool inverse,
      cudaStream_t stream = nullptr);

  // No-sync launchers used while capturing the ordinary-F device timeline.
  // The grid width is fixed at graph construction, but every factor kernel
  // reads DeviceActiveSupportWindow::count on device.
  void graph_safe_prepare_active_window(
      const std::uint64_t* source_slots,
      const std::uint32_t* source_count,
      std::uint64_t* window_slots,
      std::uint32_t window_capacity,
      cudaStream_t stream = nullptr) const;
  void graph_safe_apply_active_factor(
      const DeviceChunkMap& chunks, LawFactor factor, bool inverse,
      DeviceActiveSupportWindow window,
      cudaStream_t stream = nullptr);
  void graph_safe_apply_active_macro_factors(
      const DeviceChunkMap& chunks, bool inverse,
      DeviceActiveSupportWindow window,
      cudaStream_t stream = nullptr);

  // Applies one globally ordered law factor to immutable page input. The first
    // core_count input slots own output, which is written to the core_count slots
    // beginning at input_count. Input plus output scratch share this aperture;
    // no second field allocation exists. Edge and stream pages require complete
    // chunk closure for every core slot.
    void apply_factor_window(const DeviceChunkMap& chunks,
                             LawFactor factor,
                             bool inverse,
                             std::uint32_t core_count,
                             std::uint32_t input_count,
                             cudaStream_t stream = nullptr);

    void write_word(std::uint64_t slot, SiteWord value,
                    cudaStream_t stream = nullptr);
    [[nodiscard]] SiteWord read_word(std::uint64_t slot,
                                     cudaStream_t stream = nullptr) const;
    [[nodiscard]] bool all_words_equal(SiteWord value,
                                       cudaStream_t stream = nullptr);

    [[nodiscard]] static std::uint64_t checked_site_count(
        std::uint32_t chunk_count);
    [[nodiscard]] static std::uint64_t checked_word_bytes(
        std::uint32_t chunk_count);

private:
    class Submission;

    explicit CudaBcc32Executor(std::uint32_t chunk_count, bool production);

    void validate_superstep_closure(const DeviceChunkMap& chunks,
                                    cudaStream_t stream) const;
    void validate_topology(const DeviceChunkMap& chunks,
                           cudaStream_t stream) const;
    void validate_core_closure(const DeviceChunkMap& chunks,
                               std::uint32_t core_count,
                               cudaStream_t stream) const;
    void apply_k_site(bool inverse, std::uint64_t site_count,
                      cudaStream_t stream);
    void apply_k_edge(const DeviceChunkMap& chunks, bool inverse,
                      std::uint64_t site_count, cudaStream_t stream);
    void apply_k_carrier_pair_splitter(const DeviceChunkMap& chunks, bool inverse,
                                     std::uint64_t site_count, cudaStream_t stream);
  void apply_k_processive_rearm(const DeviceChunkMap& chunks, bool inverse,
                                std::uint64_t site_count, cudaStream_t stream);
  void apply_k_processive_release(const DeviceChunkMap& chunks, bool inverse,
                                  std::uint64_t site_count, cudaStream_t stream);
  void apply_k_carrier_corner(const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
                              cudaStream_t stream);
  void apply_k_eligibility_residual_junction(
      const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream);
  void apply_k_developmental_append(
      const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream);
  void apply_k_developmental_learned_receptor(
      const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream);
  void apply_k_developmental_credit_service(
      const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
      const std::uint64_t* active_slots, std::uint64_t active_count,
      cudaStream_t stream);
  void stream_p(const DeviceChunkMap& chunks, bool inverse,
                std::uint64_t site_count, cudaStream_t stream);
    [[nodiscard]] std::uint8_t* ensure_bounded_resolution(
        std::uint64_t active_count, cudaStream_t stream);
    void release() noexcept;

    std::uint32_t chunk_count_ = 0u;
    std::uint64_t site_count_ = 0ull;
    std::uint64_t aperture_bytes_ = 0ull;
    std::uint64_t carrier_snapshot_bytes_ = 0ull;
    // Counts supersteps so genesis authority above can REFUSE after the first
    // tick. A counter that only rises is what makes that a control.
    std::uint64_t supersteps_ = 0ull;
    std::uint64_t interventions_ = 0ull;
    InterventionReceipt last_intervene_{};
    std::uint64_t resident_stages_ = 0ull;
    ResidentStageReceipt last_resident_stage_{};
    SiteWord* words_ = nullptr;
    std::uint8_t* carrier_snapshot_ = nullptr;
    // Grow-only scratch for the bounded-resolution active factors.  Capacity
    // persists for the executor lifetime; contents never outlive the
    // submission that consumed them.
    std::uint8_t* bounded_resolution_ = nullptr;
    std::size_t bounded_resolution_capacity_ = 0;
    mutable std::mutex submission_mutex_;
    mutable cudaEvent_t completion_event_ = nullptr;
    mutable cudaStream_t last_submission_stream_ = nullptr;
    mutable bool has_submission_ = false;
};

}  // namespace substrate::bcc32

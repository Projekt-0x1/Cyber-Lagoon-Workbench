#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "bcc32_world_store.hpp"
#include "bcc32_processive_credit_return_seed.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {

class ConditionedMatterExecutor {
 public:
  virtual ~ConditionedMatterExecutor() = default;

  virtual bool advance(WorldStore* world, bool inverse,
                       std::uint32_t supersteps,
                       std::uint64_t* page_count,
                       std::string* error) = 0;
  [[nodiscard]] virtual std::uint64_t aperture_bytes() const = 0;
};

// Implemented by the CUDA aperture library. The conditioned owner depends only
// on this physical executor boundary, so the final core never falls back to the
// CPU reference lattice.
[[nodiscard]] std::shared_ptr<ConditionedMatterExecutor>
make_paged_conditioned_matter_executor(
    std::uint32_t aperture_chunks = 64u,
    bool reverse_core_order = false);

// Where route `slot` is placed inside the shared conditioned world.
//
// This is an internal placement detail in the sense that no checkpoint depends on it -- route
// matter is serialised at coordinates RELATIVE to this origin and slots are reassigned on load
// -- but it is emphatically NOT an arbitrary one. The paged executor's resident window is the
// UNION of per-chunk halo dilations (`bcc32_transition.cu:134`), so how route chunks are
// arranged in space sets what it costs to advance many routes in one window. It is exposed so
// that property can be measured instead of assumed; see
// `docs/audits/2026-07-28-route-population-capacity.md`.
[[nodiscard]] Z3Coordinate conditioned_route_origin(std::uint32_t slot);

// How many routes share one 100^3 chunk under that placement.
[[nodiscard]] std::uint32_t conditioned_routes_per_chunk();

// This key names the adult contact route that feeds a physical weight.  It is
// routing metadata, not the weight: deleting every Entry::world deletes all
// learned strength while preserving every key.
struct ConditionedMatterKey {
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;

  friend bool operator==(const ConditionedMatterKey&,
                         const ConditionedMatterKey&) = default;
  friend bool operator<(const ConditionedMatterKey& left,
                        const ConditionedMatterKey& right) {
    if (left.anchor != right.anchor) return left.anchor < right.anchor;
    if (left.previous != right.previous)
      return left.previous < right.previous;
    return left.next < right.next;
  }
};

struct ConditionedMatterCredit {
  ConditionedMatterKey key{};
  std::int32_t polarity = 0;
  std::uint32_t source_event = 0u;
};

// ABI-neutral bridge records for CUDA-owned adult buffers.  The extra `valid`
// word lets the owner consume the fixed event slots without a stream-side
// semantic filter or vector conversion.  These records are deliberately
// unsigned at the route surface: the current owner publishes conductance as
// uint32_t, so this bridge does not claim signed inhibitory publication.
struct ConditionedMatterDeviceCredit {
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;
  std::int32_t polarity = 0;
  std::uint32_t source_event = 0u;
  std::uint32_t valid = 0u;
};

struct ConditionedMatterDeviceKey {
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;
};

// Sparse physical owner for the first canonical-adult composition.  Each
// occupied entry is an actual processive BCC region advanced only by the
// declared canonical law epoch. The host may route an exact contact to an
// entry and page or checkpoint its words; it never stores, increments, or
// restores a scalar
// weight.
class ConditionedLearningMatter {
 public:
  ConditionedLearningMatter() = default;
  ConditionedLearningMatter(
      std::size_t organ_capacity,
      std::shared_ptr<ConditionedMatterExecutor> executor);
  ConditionedLearningMatter(const ConditionedLearningMatter&) = delete;
  ConditionedLearningMatter& operator=(const ConditionedLearningMatter&) =
      delete;
  ConditionedLearningMatter(ConditionedLearningMatter&&) noexcept = default;
  ConditionedLearningMatter& operator=(ConditionedLearningMatter&&) noexcept =
      default;

  [[nodiscard]] std::size_t size() const { return bound_count_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }
  [[nodiscard]] bool empty() const { return bound_count_ == 0u; }
  [[nodiscard]] bool contains(const ConditionedMatterKey& key) const;
  [[nodiscard]] std::size_t remaining_capacity() const {
    return capacity_ - bound_count_;
  }
  [[nodiscard]] std::size_t required_new_entries(
      std::span<const ConditionedMatterCredit> credits) const;

  struct ConsumeReceipt {
    std::uint32_t requested = 0u;
    std::uint32_t admitted = 0u;
    std::uint32_t abstained = 0u;
    std::uint32_t executor_waves = 0u;
    std::uint32_t peak_wave_routes = 0u;
    std::uint32_t fixed_returns = 0u;
    std::uint32_t conserved_contacts = 0u;
  };

  [[nodiscard]] ConsumeReceipt consume_with_receipt(
      std::span<const ConditionedMatterCredit> credits);
  // Consume a device-resident fixed event batch.  The owner remains the sole
  // authority: device records are copied into the owner's transaction only
  // inside this boundary, then committed by the existing paged executor.
  [[nodiscard]] ConsumeReceipt consume_device_batch(
      const ConditionedMatterDeviceCredit* device_credits,
      std::uint32_t count);
  // Host-record ingestion remains an owner-internal transaction primitive and
  // an explicit diagnostic API. The adult production callback supplies device
  // records directly through consume_device_batch.
  [[nodiscard]] ConsumeReceipt consume_event_batch(
      std::span<const ConditionedMatterDeviceCredit> events);
  void consume(std::span<const ConditionedMatterCredit> credits) {
    (void)consume_with_receipt(credits);
  }
  void consume(const ConditionedMatterCredit& credit);

  // Returns one unit for every processive cell that physically conducts a
  // three-hole query.  The query is compute/uncompute on a fork; no marker
  // census or cached level participates in the result.
  [[nodiscard]] std::uint32_t conductance(
      const ConditionedMatterKey& key) const;
  [[nodiscard]] std::vector<std::uint32_t> conductances(
      std::span<const ConditionedMatterKey> keys) const;
  // Quiesced migration/checkpoint observer. Production never uses this host
  // inventory; it exists only to translate pre-device-owner v6 checkpoints.
  [[nodiscard]] std::vector<ConditionedMatterKey> inventory_keys() const;
  // Read exact inventory keys and publish the owner's conductance directly to
  // a device surface.  The caller never receives route keys or host vectors.
  void publish_conductance_device(
      const ConditionedMatterDeviceKey* device_keys,
      std::uint32_t count,
      std::uint32_t* device_conductance) const;

  // Causal intervention used by contracts.  The removed material remains in
  // the entry's stage-addressed escrow and is serialized with the world.
  void lesion_stage(const ConditionedMatterKey& key, std::uint32_t stage);
  void restore_lesioned_stage(
      const ConditionedMatterKey& key, std::uint32_t stage);

  // Stage-addressed record of removed material.  Slot s holds exactly the
  // quanta taken from stage s; a slot still at its vacuum means that stage was
  // never lesioned.  Nothing is overwritten, so damage is distinguishable from
  // ignorance across repeated lesions.
  [[nodiscard]] std::array<SiteWord, kProcessiveCreditReturnStageCount>
      lesion_escrow(const ConditionedMatterKey& key) const;

  // delta_n_q over the route's own world plus tape_delta over every escrow
  // slot.  Invariant under lesion_stage: removal moves material, never
  // destroys it.
  [[nodiscard]] DeltaNQ route_conserved_delta(
      const ConditionedMatterKey& key) const;

  void save(std::ostream& output) const;
  static ConditionedLearningMatter load(
      std::istream& input,
      std::shared_ptr<ConditionedMatterExecutor> executor);

  [[nodiscard]] std::uint64_t physical_hash() const;
  [[nodiscard]] bool uses_physical_executor() const {
    return executor_ != nullptr;
  }
  [[nodiscard]] std::uint64_t executor_aperture_bytes() const {
    return executor_ == nullptr ? 0u : executor_->aperture_bytes();
  }
  [[nodiscard]] std::uint64_t resident_sites() const {
    return world_.support().non_quiescent_sites;
  }
  [[nodiscard]] std::uint64_t resident_chunks() const {
    return world_.support().materialized_chunks;
  }
  [[nodiscard]] std::uint64_t completed_supersteps() const {
    std::uint64_t total = 0u;
    for (const Entry& entry : entries_)
      total += entry.completed_supersteps;
    return total;
  }

 private:
  struct Entry {
    ConditionedMatterKey key{};
    std::array<SiteWord, kProcessiveCreditReturnStageCount> input_tapes{};
    std::array<SiteWord, kProcessiveCreditReturnStageCount> output_tapes{};
    // Per-stage escrow (see the public `lesion_escrow` accessor). The default
    // here only guards a default-constructed Entry that never goes through
    // `seeded_entry` or `load`; every real entry overwrites this with the
    // seeded vacuum at birth. `{}` alone would zero-init, which is a real
    // (non-vacuum) word, so every slot is filled explicitly instead.
    std::array<SiteWord, kProcessiveCreditReturnStageCount> lesion_escrow =
        [] {
          std::array<SiteWord, kProcessiveCreditReturnStageCount> vacuum{};
          vacuum.fill(kQuiescentWord);
          return vacuum;
        }();
    std::uint64_t completed_supersteps = 0u;
    std::uint32_t slot = 0u;
    bool bound = false;
  };

  [[nodiscard]] std::vector<Entry>::iterator find(
      const ConditionedMatterKey& key);
  [[nodiscard]] std::vector<Entry>::const_iterator find(
      const ConditionedMatterKey& key) const;
  [[nodiscard]] Entry seeded_entry(const ConditionedMatterKey& key,
                                   std::uint32_t slot) const;
  [[nodiscard]] WorldStore seeded_route_world() const;
  [[nodiscard]] WorldStore read_route_world(const Entry& entry) const;
  void write_route_world(const Entry& entry, const WorldStore& route);
  void advance_world(WorldStore* world, bool inverse,
                     std::uint32_t supersteps,
                     std::uint64_t* page_count) const;

  std::vector<Entry> entries_;
  // The bank is declared at birth as this many seed-backed virtual organ
  // pages. Only contacted pages are resident in entries_; materializing one
  // is a pager operation over already-declared matter, not adult-time growth.
  std::size_t capacity_ = 0u;
  std::size_t bound_count_ = 0u;
  std::uint32_t next_slot_ = 0u;
  WorldStore world_;
  std::shared_ptr<ConditionedMatterExecutor> executor_;
};

}  // namespace substrate::bcc32

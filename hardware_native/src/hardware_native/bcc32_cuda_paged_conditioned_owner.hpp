#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>

namespace substrate::bcc32 {
struct ConditionedMatterDeviceCredit;
struct ConditionedMatterDeviceKey;
class ConditionedLearningMatter;
}  // namespace substrate::bcc32

namespace bcc32::paged_conditioned_owner {

constexpr std::int32_t kConductanceCeiling = 8;
constexpr std::uint32_t kFactorRegionCount = 4u;
constexpr std::uint32_t kFactorHorizon = 4u;
constexpr std::uint32_t kFactorWordsPerRegion = 12288u;
constexpr std::uint32_t kResidentFactorWords =
    kFactorRegionCount * kFactorWordsPerRegion;
constexpr std::uint32_t kInvalidFactorSlot = 0xffffffffu;

enum class OperationCode : std::uint32_t {
  kNotRun = 0u,
  kOk = 1u,
  kInvalidInput = 2u,
  kBusy = 3u,
  kRejected = 4u,
  kStale = 5u,
};

struct ResidentFactorRing {
  std::uint32_t* lane[kFactorHorizon]{};
};

struct ResidentFactorBinding {
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;
  std::uint32_t region = 0u;
  std::uint32_t factor_index = 0xffffffffu;
  std::uint32_t slot = kInvalidFactorSlot;
  std::uint64_t lesion_generation = 0u;
  std::uint32_t bound = 0u;
  std::uint32_t reserved = 0u;

  friend bool operator==(const ResidentFactorBinding&,
                         const ResidentFactorBinding&) = default;
};

struct ResidentFactorClock {
  std::uint64_t contact = 0u;

  friend bool operator==(const ResidentFactorClock&,
                         const ResidentFactorClock&) = default;
};

struct ResidentFactorStateView {
  ResidentFactorRing factors{};
  std::uint32_t* eligibility_supply = nullptr;
  ResidentFactorBinding* bindings = nullptr;
  std::uint32_t binding_count = 0u;
  ResidentFactorClock* clock = nullptr;
  std::uint32_t* positive_regions = nullptr;
  std::uint32_t* negative_regions = nullptr;
  std::uint32_t* matched_regions = nullptr;
  std::uint32_t* residual_escrow = nullptr;
};

struct ResidentPredictionWitness {
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;
  std::uint32_t region = 0u;
  std::uint32_t factor_index = 0xffffffffu;
  std::uint32_t source_event = 0u;
  std::uint32_t valid = 0u;
};

// A transient device-side prediction witness. The producer names the route it
// actually selected. factor_index is reserved ABI space and must be invalid:
// the owner derives the endpoint from resident route matter.
struct ResidentFactorParticipation {
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;
  std::uint32_t region = 0u;
  std::uint32_t factor_index = 0xffffffffu;
  std::uint32_t* eligibility_source = nullptr;
  std::uint32_t valid = 0u;
};

struct WorldResidualSource {
  const std::uint32_t* world = nullptr;
  std::uint64_t world_words = 0u;
  std::uint64_t positive_endpoint = 0u;
  std::uint64_t negative_endpoint = 0u;
  std::uint32_t vacancy = 0u;
  std::uint32_t* positive_supply = nullptr;
  std::uint32_t* negative_supply = nullptr;
  std::uint32_t region = 0u;
};

struct ConsumeReceipt {
  std::uint32_t requested = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t inserted = 0u;
  std::uint32_t rejected = 0u;
  std::uint32_t failure_index = 0xffffffffu;
};

struct PhysicalMeasure {
  std::uint64_t hash = 0u;
  std::uint64_t matter_bits = 0u;
  std::uint64_t free_bits = 0u;
  std::uint64_t route_bits = 0u;
  std::uint64_t lesion_bits = 0u;
  std::uint64_t factor_hash = 0u;
  std::uint64_t factor_bits = 0u;
  std::uint32_t occupied_routes = 0u;
  std::uint32_t committed_transactions = 0u;
  std::uint32_t factor_lane_words = 0u;
  std::uint32_t factor_bindings = 0u;
};

struct RouteLesionReceipt {
  std::uint32_t code = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t slot = kInvalidFactorSlot;
  std::uint32_t anchor = 0u;
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;
  std::uint32_t region = 0u;
  std::uint32_t positive_word = 0u;
  std::uint32_t negative_word = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t restore_epoch = 0u;
};

struct FactorCreditReceipt {
  std::uint32_t code = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t matched_regions = 0u;
  std::uint32_t consumed_regions = 0u;
  std::uint32_t unbound = 0u;
};

struct FactorParticipationReceipt {
  std::uint32_t code = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t factor_index = 0xffffffffu;
  std::uint32_t slot = kInvalidFactorSlot;
};

struct PredictionWitnessReceipt {
  std::uint32_t requested = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t rejected = 0u;
};

struct DelayedFactorCreditReceipt {
  std::uint32_t code = 0u;
  std::uint32_t requested = 0u;
  std::uint32_t admitted = 0u;
  std::uint32_t matched = 0u;
  std::uint32_t errors = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t expired = 0u;
};

class PagedConditionedOwner {
 public:
  PagedConditionedOwner();
  explicit PagedConditionedOwner(std::uint32_t capacity);
  ~PagedConditionedOwner();

  PagedConditionedOwner(const PagedConditionedOwner&) = delete;
  PagedConditionedOwner& operator=(const PagedConditionedOwner&) = delete;
  PagedConditionedOwner(PagedConditionedOwner&&) noexcept;
  PagedConditionedOwner& operator=(PagedConditionedOwner&&) noexcept;

  void reset(std::uint32_t capacity);
  [[nodiscard]] std::uint32_t capacity() const;
  [[nodiscard]] std::uint32_t size() const;
  [[nodiscard]] std::uint32_t remaining_capacity() const;

  ConsumeReceipt consume_device_batch(
      const substrate::bcc32::ConditionedMatterDeviceCredit* device_events, std::uint32_t count);
  void publish_conductance_device(const substrate::bcc32::ConditionedMatterDeviceKey* keys,
                                  std::uint32_t count, std::uint32_t* output) const;
  RouteLesionReceipt lesion_route(std::uint32_t anchor,
                                  std::uint32_t previous,
                                  std::uint32_t next,
                                  std::uint32_t region = 0u);
  bool restore_route_lesion(const RouteLesionReceipt& receipt);
  void reset_resident_factor_state(std::uint32_t lane_words,
                                   std::uint32_t binding_count);
  [[nodiscard]] ResidentFactorStateView resident_factor_state_device() const;
  FactorParticipationReceipt capture_resident_factor_participation(
      const ResidentFactorParticipation* device_participation);
  PredictionWitnessReceipt capture_resident_prediction_batch(
      const ResidentPredictionWitness* device_witnesses,
      std::uint32_t count);
  DelayedFactorCreditReceipt apply_resident_conditioned_factor_credit(
      const substrate::bcc32::ConditionedMatterDeviceCredit* device_events,
      std::uint32_t count);
  [[nodiscard]] bool inverse_resident_conditioned_factor_credit();
  [[nodiscard]] bool inverse_resident_factor_participation(
      const ResidentFactorParticipation* device_participation);
  void resolve_resident_factor_bindings();
  FactorCreditReceipt apply_resident_factor_credit();
  FactorCreditReceipt apply_resident_world_factor_credit(
      const WorldResidualSource& source);
  FactorCreditReceipt apply_resident_atomic_world_factor_credit(
      const ResidentFactorParticipation* device_participation,
      const WorldResidualSource& source);
  [[nodiscard]] bool inverse_resident_factor_credit();
  [[nodiscard]] bool inverse_resident_world_factor_credit(
      const WorldResidualSource& source);
  [[nodiscard]] bool inverse_resident_atomic_world_factor_credit(
      const ResidentFactorParticipation* device_participation,
      const WorldResidualSource& source);
  void resolve_factor_bindings_device(ResidentFactorBinding* bindings,
                                      std::uint32_t binding_count) const;
  FactorCreditReceipt apply_factor_credit(const ResidentFactorRing& factors,
                                          const ResidentFactorBinding* bindings,
                                          std::uint32_t binding_count, ResidentFactorClock* clock,
                                          std::uint32_t* positive_regions,
                                          std::uint32_t* negative_regions,
                                          std::uint32_t* matched_regions,
                                          std::uint32_t* residual_escrow,
                                          const WorldResidualSource* world = nullptr);
  [[nodiscard]] bool inverse_factor_credit(const ResidentFactorRing& factors,
                                           const ResidentFactorBinding* bindings,
                                           std::uint32_t binding_count, ResidentFactorClock* clock,
                                           std::uint32_t* positive_regions,
                                           std::uint32_t* negative_regions,
                                           std::uint32_t* matched_regions,
                                           std::uint32_t* residual_escrow,
                                           const WorldResidualSource* world = nullptr);

  [[nodiscard]] PhysicalMeasure physical_measure() const;
  [[nodiscard]] std::uint64_t physical_hash() const;
  void save(std::ostream& output) const;

  static PagedConditionedOwner load(std::istream& input);
  static PagedConditionedOwner migrate_legacy(
      const substrate::bcc32::ConditionedLearningMatter& legacy);

 private:
  struct Impl;
  explicit PagedConditionedOwner(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace bcc32::paged_conditioned_owner

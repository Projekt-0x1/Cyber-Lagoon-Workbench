#pragma once

#if defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)
#include "hardware_native/bcc32_cuda_adult_complete_checkpoint_v2.cuh"
#else
#include "hardware_native/bcc32_cuda_adult_v1.cuh"
#include "hardware_native/bcc32_cuda_adult_complete_checkpoint_v2.cuh"
#endif
#include "hardware_native/bcc32_cuda_appraisal_v1.cuh"
#include "hardware_native/bcc32_cuda_resident_discourse_plan.cuh"
#include "hardware_native/bcc32_cuda_resident_plan_eligibility.cuh"
#include "hardware_native/bcc32_cuda_resident_population_surface.cuh"
#include "hardware_native/bcc32_cuda_resident_question_ticket.cuh"
#include "hardware_native/bcc32_cuda_resident_proposition_chain.cuh"
#include "hardware_native/bcc32_conditioned_learning_matter.hpp"
#include "hardware_native/bcc32_cuda_paged_conditioned_owner.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bcc32_cuda_adult_stream_v1 {

namespace adult = bcc32_cuda_adult_v1;
namespace complete_checkpoint = bcc32_cuda_adult_complete_checkpoint_v2;
namespace appraisal = bcc32_cuda_appraisal_v1;
namespace discourse_plan = bcc32_cuda_resident_discourse_plan;
namespace question_goal = bcc32_cuda_resident_question_goal;
namespace question_ticket = bcc32_cuda_resident_question_ticket;
namespace plan_eligibility = substrate::bcc32::resident_plan_eligibility;
namespace population_surface = bcc32_cuda_resident_population_surface;
namespace proposition_chain = bcc32_cuda_resident_proposition_chain;
namespace surface_organ = substrate::bcc32::resident_surface_organ_v2;

constexpr std::uint32_t kStreamVersion = 7u;
constexpr std::uint32_t kDefaultChunkCapacity = 4096u;
constexpr std::uint32_t kDefaultEmissionCapacity = 64u;
constexpr std::uint32_t kPinnedEmissionPublicationLimit =
    kDefaultChunkCapacity;
// Sparse route identities are materialized only on contact. This declared
// extent therefore lifts the old 65,536-route ceiling without preallocating
// route worlds. The conditioned owner packs 110 disjoint route supports per
// 4 MB shared chunk, so this full declared extent is under 20 GiB of direct
// words before container overhead and remains inside the disclosed envelope.
constexpr std::uint32_t kDefaultConditionedOrganCapacity = 524288u;
constexpr std::uint64_t kDriveMagic = 0x3156524453434342ull;

#include "hardware_native/bcc32_cuda_adult_stream_host_transport.inl"

struct StreamConfig {
  std::uint32_t chunk_capacity = kDefaultChunkCapacity;
  std::uint32_t emission_capacity = kDefaultEmissionCapacity;
  std::uint32_t emission_threshold = 32u;
  std::uint32_t surprise_threshold = 0xffffffffu;
  std::uint32_t autonomous_quiet_ticks = 2u;
  std::uint32_t contact_refractory_ticks = 1u;
  std::uint32_t emission_refractory_ticks = 1u;
  std::uint32_t emission_base_cost = 8u;
  std::uint32_t basal_energy_per_tick = 1u;
  std::uint32_t reafference_enabled = 1u;
  std::uint32_t appraisal_holdout_bytes = appraisal::kOperationalHoldoutBytes;
  // Seed-backed organ pages are declared at stream birth. Neutral pages cost
  // only a hash-backed pager declaration; contacted pages become resident.
  std::uint32_t conditioned_organ_capacity =
      kDefaultConditionedOrganCapacity;
  // The stream owns the learned population/tissue route used by query
  // settlement. Disabling it is reserved for focused transport contracts.
  std::uint32_t resident_language_enabled = 1u;
  // The statistical byte generator is diagnostic-only once a learned query
  // has been recognized; it can never answer a failed resident settlement.
  std::uint32_t legacy_generator_enabled = 1u;
};

// Every field in this block is ordinary mutable device state and is checkpointed verbatim.
struct DriveState {
  std::uint64_t magic;
  std::uint64_t ticks;
  std::uint64_t contact_events;
  std::uint64_t external_chunks;
  std::uint64_t external_bytes;
  std::uint64_t reafferent_chunks;
  std::uint64_t reafferent_bytes;
  std::uint64_t activity_total;
  std::uint64_t surprise_total;
  std::uint64_t plasticity_events;
  std::uint64_t revision_events;
  std::uint64_t improvement_events;
  std::uint64_t appraisal_events;
  std::uint64_t appraisal_commits;
  std::uint64_t appraisal_rejections;
  std::uint64_t appraisal_blocked_commits;
  std::uint64_t chronological_bytes;
  std::uint64_t predictive_loss_total;
  std::uint64_t emissions;
  std::uint64_t autonomous_emissions;
  std::uint64_t contact_emissions;
  std::uint64_t emitted_bytes;
  std::uint64_t energy_initial;
  std::uint64_t energy;
  std::uint64_t energy_gained;
  std::uint64_t energy_spent;
  std::uint64_t last_emission_tick;
  std::uint32_t recent_activity;
  std::uint32_t recent_surprise;
  std::uint32_t previous_surprise;
  std::uint32_t last_contact_hash;
  std::uint32_t last_output_hash;
  std::uint32_t quiet_ticks;
  std::uint32_t refractory_ticks;
  std::uint32_t emission_threshold;
  std::uint32_t surprise_threshold;
  std::uint32_t autonomous_quiet_ticks;
  std::uint32_t contact_refractory_ticks;
  std::uint32_t emission_refractory_ticks;
  std::uint32_t emission_base_cost;
  std::uint32_t basal_energy_per_tick;
  std::uint32_t emission_capacity;
  std::uint32_t emit_pending;
  std::uint32_t pending_is_autonomous;
  std::uint32_t drive_enabled;
  std::uint32_t plasticity_enabled;
  std::uint32_t reafference_enabled;
  std::uint32_t drive_lesioned;
  std::uint32_t plasticity_lesioned;
  std::uint32_t energy_ledger_ok;
  std::uint32_t recent_predictive_loss;
  std::int32_t learning_parameters[appraisal::kGenomeParameters];
  std::uint32_t byte_exposure[256];
};

struct ContactSummary {
  unsigned long long activity;
  std::uint32_t unseen;
  std::uint32_t hash;
};

// This is an observer receipt for the device-resident query path. It names no
// vocabulary or answer: it only records whether learned cue matter reached a
// uniquely settled ordered binding and whether that binding supplied bytes.
struct QueryAnswerReceipt {
  std::uint32_t attempted = 0u;
  // Issued by the live Goal/Plan pair; it carries no vocabulary or answer.
  question_ticket::ResidentQuestionActionTicket question_ticket{};
  // A residently learned interrogative onset was present in the pre-contact
  // cue. This records admission only; it names no lexical item or answer.
  std::uint32_t learned_question_form = 0u;
  // Observer-only evidence for the earliest exact cue unit.  This exposes the
  // learned admission signal without exporting any lexical content.
  std::uint32_t question_onset_evidence = 0u;
  // Retained for stream-report ABI compatibility.  Response admission is not
  // a question/answer rule: a resident contact population must settle one
  // unambiguous learned action population before bytes may be emitted.
  std::uint32_t learned_relation_gap = 0u;
  std::uint32_t exact_cue_units = 0u;
  std::uint32_t claimed_ordered_bindings = 0u;
  std::uint32_t exact_topic_matches = 0u;
  std::uint32_t qualified_best_count = 0u;
  std::uint32_t qualified_witnessed_bindings = 0u;
  std::uint32_t witnessed_role_cell_overlap_bindings = 0u;
  std::uint32_t max_witnessed_role_cell_overlap = 0u;
  std::uint32_t selected_binding_index = 0xffffffffu;
  std::uint32_t selected_present_roles = 0u;
  std::uint32_t episode_spine_steps = 0u;
  std::uint32_t episode_spine_terminal = 0xffffffffu;
  std::uint32_t episode_spine_ambiguous = 0u;
  std::uint32_t selected_unresolved_roles = 0u;
  // Copied at :1086-1087 from the settlement result. `selected_present_roles`
  // above SUMS POPULATION COUNTS despite naming roles, and it is only the
  // tiebreak key; episode coverage is what actually decides the winner.
  std::uint32_t selected_episode_coverage = 0u;
  std::uint32_t max_role_coverage_any_episode = 0u;
  std::uint32_t join_gate_qualified_count = 0u;
  std::uint32_t join_gate_entered = 0u;
  std::uint32_t ambiguous = 0u;
  std::uint32_t staged = 0u;
  std::uint32_t anchor_count = 0u;
  std::uint32_t serialized_units = 0u;
  std::uint32_t emission_authorized = 0u;
  std::uint32_t emitted_bytes = 0u;
  // Direct ordered-tissue execution receipt. The selector consumes the
  // pre-contact exact cue on device; these fields only attribute the resulting
  // candidate or its mandatory silence and never choose an answer.
  std::uint32_t ordered_relation_route_authorized = 0u;
  std::uint32_t ordered_relation_topology_matches = 0u;
  std::uint32_t ordered_relation_qualified_candidates = 0u;
  std::uint32_t ordered_relation_withdrawn_candidates = 0u;
  std::uint32_t ordered_relation_invalid_sources = 0u;
  std::uint32_t ordered_relation_conflict = 0u;
  std::uint32_t ordered_relation_ready = 0u;
  std::uint32_t ordered_relation_clarification_ready = 0u;
  std::uint32_t ordered_relation_source_current = 0u;
  std::uint32_t ordered_relation_stale_source_rejected = 0u;
  std::uint32_t ordered_relation_source_binding = 0xffffffffu;
  std::uint32_t ordered_relation_composed_source_binding = 0xffffffffu;
  std::uint32_t ordered_relation_terminal_source_binding = 0xffffffffu;
  std::uint32_t ordered_relation_output_units = 0u;
  std::uint32_t ordered_relation_composition_depth = 0u;
  std::uint32_t candidate_producer = 0u;
  std::uint64_t ordered_relation_tissue_revision = 0u;
  std::uint64_t ordered_relation_source_revision = 0u;
  std::uint64_t ordered_relation_composed_source_revision = 0u;
  std::uint64_t ordered_relation_terminal_source_revision = 0u;
  std::uint64_t ordered_relation_ticketed_return_revision = 0u;
  std::uint32_t ordered_relation_ticketed_return_source_mask = 0u;
  std::uint64_t ordered_relation_source_positive_mass = 0u;
  std::uint64_t ordered_relation_source_counterevidence = 0u;
  // Surface-organ discriminators. SurfaceOrganResult already computes all of
  // these with real writers; the receipt carried only eight of its sixteen
  // fields, so "ready=0" could not be split into WHY. Added together with the
  // copies that populate them in adopt_stream_construction_surface_kernel.
  std::uint32_t surface_closure_supported = 0u;
  std::uint32_t surface_anchors_preserved = 0u;
  std::uint32_t surface_capacity_exceeded = 0u;
  std::uint32_t surface_output_units = 0u;
  std::uint32_t surface_connectors = 0u;
  std::uint32_t surface_selected_permutation = 0u;
  // The stream's OWN realizer reports through construction_ready, which is a
  // verdict: complete = expected_steps != 0 && failed == 0 &&
  // realized_steps == expected_steps && byte_count != 0. When ready=0 these
  // four say WHICH conjunct broke. All four have writers (3/5/2/2 sites);
  // added together with the copies below.
  std::uint32_t tx_expected_steps = 0u;
  std::uint32_t tx_failed = 0u;
  std::uint32_t tx_realized_steps = 0u;
  std::uint32_t tx_byte_count = 0u;
  std::uint32_t ngram_fallback = 0u;
  std::uint32_t completion_cells = 0u;
  // Observer-only construction receipt. These are counts produced by the
  // resident witness competition; they do not admit, rank, or name content.
  std::uint32_t construction_count = 0u;
  std::uint32_t construction_supported = 0u;
  std::uint32_t construction_shape_matched = 0u;
  std::uint32_t construction_mapping_matched = 0u;
  std::uint32_t construction_tied = 0u;
  std::uint32_t construction_ready = 0u;
  std::uint32_t construction_grammar_supported = 0u;
  std::uint32_t construction_output_bytes = 0u;
  // Observer-only: exact-event trajectory reconstruction before the surface
  // organ sees a plan. These values never select content or relax abstention.
  std::uint32_t surface_trajectory_slots = 0u;
  std::uint32_t surface_trajectory_grounded = 0u;
  std::uint32_t surface_trajectory_ambiguous = 0u;
  std::uint32_t trajectory_closed = 0u;
  std::uint32_t trajectory_observed = 0u;
  std::uint32_t trajectory_pending = 0u;
  // Observer-only receipt for a resident-only quiet step.  This is set only
  // after existing resident route matter was advanced; it is never an output
  // authorization and carries no content, source, or semantic label.
  std::uint32_t endogenous_resident_steps = 0u;
  // Observer-only relation-graph receipt. These counts expose which learned
  // graph stage abstained; they do not influence retrieval or emission.
  std::uint32_t relation_candidate_count = 0u;
  std::uint32_t relation_plan_units = 0u;
  std::uint32_t relation_plan_clauses = 0u;
  std::uint32_t relation_topic_fallback = 0u;
  std::uint32_t relation_gap_field = 0u;
  std::uint32_t relation_surface_events = 0u;
  std::uint32_t relation_surface_witnesses = 0u;
  std::uint32_t relation_surface_missing_events = 0u;
  std::uint32_t relation_surface_missing_constructions = 0u;
  std::uint32_t relation_surface_anchor_frames = 0u;
};

// Observer-only receipt for the query Plan's population grounding seam. It
// records bounded plan/result state across the generic materializer, the
// question-goal overwrite, and final authorization without exposing a cell,
// unit, byte, word, or semantic label.
struct PlanAnchorGroundingObserverReceipt {
  std::uint32_t pre_valid = 0u;
  std::uint32_t pre_status = 0u;
  std::uint32_t pre_step_count = 0u;
  std::uint32_t pre_anchor_count = 0u;
  std::uint32_t pre_opaque_steps = 0u;
  std::uint32_t pre_ordered_steps = 0u;
  std::uint32_t pre_other_steps = 0u;
  std::uint32_t pre_zero_population_steps = 0u;
  std::uint32_t pre_population_references = 0u;
  std::uint32_t pre_question_goal_dependency = 0u;
  std::uint32_t pre_plan_revision = 0u;
  std::uint32_t materializer_ready = 0u;
  std::uint32_t materializer_grounded_steps = 0u;
  std::uint32_t materializer_anchor_count = 0u;
  std::uint32_t materializer_ambiguous_step = 0xffffffffu;
  std::uint32_t materializer_ungrounded_step = 0xffffffffu;
  std::uint32_t materializer_weakest_overlap = 0u;
  std::uint32_t materializer_plan_revision = 0u;
  std::uint32_t post_status = 0u;
  std::uint32_t post_anchor_count = 0u;
  std::uint32_t post_plan_revision = 0u;
  std::uint32_t final_attempted = 0u;
  std::uint32_t final_staged = 0u;
  std::uint32_t final_anchor_count = 0u;
  std::uint32_t final_plan_valid = 0u;
  std::uint32_t final_plan_status = 0u;
};

#include "hardware_native/bcc32_cuda_ordered_relation_public_output.inl"

// Device-local transaction state for one dependency-linked surface span.  It
// contains only bounded plan positions and byte accounting; construction
// choice and anchor identities remain in the resident plan and witnesses.
struct SurfaceSpanTransaction {
  std::uint32_t first_step = 0u;
  std::uint32_t expected_steps = 0u;
  std::uint32_t realized_steps = 0u;
  std::uint32_t failed = 0u;
  std::uint32_t byte_count = 0u;
  std::uint32_t anchor_count = 0u;
  std::uint32_t construction_count = 0u;
  std::uint32_t construction_supported = 0u;
  std::uint32_t construction_shape_matched = 0u;
  std::uint32_t construction_mapping_matched = 0u;
  std::uint32_t construction_tied = 0u;
  std::uint32_t rejected_steps = 0u;
  std::uint32_t last_rejection = 0u;
  std::uint32_t closure_supported = 0u;
  std::uint32_t emitted_evidence_count = 0u;
  std::uint64_t
      emitted_evidence[adult::construction::kRelationSurfaceEvidenceCap]{};
  std::uint32_t emitted_unit_count = 0u;
  std::uint32_t emitted_units[surface_organ::kSurfaceOrganMaxOutputUnits]{};
};

// One pending physical contact->action trace.  It has no language-act,
// entity, role, correctness, or reward field.  The trace opens only after a
// resident plan has crossed the motor boundary and closes only on a later
// external contact; self-reafference is deliberately not a consequence.
struct PendingActionTrajectory {
  std::uint8_t source_surface[kDefaultChunkCapacity]{};
  std::uint32_t source_byte_count = 0u;
  std::uint32_t source_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
  std::uint32_t source_count = 0u;
  std::uint8_t action_surface[kDefaultEmissionCapacity]{};
  std::uint32_t action_byte_count = 0u;
  std::uint32_t action_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
  std::uint32_t action_count = 0u;
  std::uint32_t active = 0u;
  std::uint32_t source_revision = 0u;
  std::uint32_t action_revision = 0u;
  std::uint64_t observed_transitions = 0u;
  std::uint64_t closed_transitions = 0u;
  std::uint64_t superseded_sources = 0u;
  // This state survives per-contact receipt reset and is consumed by the next
  // external contact that either matches or invalidates the live ticket.
  question_ticket::ResidentQuestionActionTicket question_ticket{};
  std::uint32_t question_return_accepted = 0u;
  std::uint32_t question_return_rejected = 0u;
};

constexpr std::uint32_t kActionTransitionCapacity = 256u;
constexpr std::uint32_t kMinimumActionTransitionSupport = 3u;

// A bounded resident trace of what actually followed an emitted action.  All
// fields are opaque cell populations; the three positions mean temporal
// order only (earlier contact, performed action, later contact).  They never
// encode a question, answer, entity, grammatical role, or host judgement.
struct ActionTransitionEvidence {
  std::uint8_t source_surface[kDefaultChunkCapacity]{};
  std::uint32_t source_byte_count = 0u;
  std::uint32_t source_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
  std::uint32_t source_count = 0u;
  std::uint8_t action_surface[kDefaultEmissionCapacity]{};
  std::uint32_t action_byte_count = 0u;
  std::uint32_t action_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
  std::uint32_t action_count = 0u;
  std::uint8_t later_surface[kDefaultChunkCapacity]{};
  std::uint32_t later_byte_count = 0u;
  std::uint32_t later_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
  std::uint32_t later_count = 0u;
  std::uint64_t observational_support = 0u;
  std::uint64_t revision = 0u;
  std::uint32_t claimed = 0u;
};

struct ActionTransitionScalars {
  std::uint64_t revision = 0u;
  std::uint64_t observed = 0u;
  std::uint64_t capacity_rejections = 0u;
  std::uint32_t occupied = 0u;
};

struct StreamState {
  adult::AdultState adult;
  bcc32::paged_conditioned_owner::PagedConditionedOwner
      conditioned_device_owner;
  // Legacy checkpoint migration source only. Production credit and
  // conductance are owned exclusively by conditioned_device_owner.
  substrate::bcc32::ConditionedLearningMatter conditioned_learning_matter;
  adult::DeviceArray<DriveState> drive;
  adult::DeviceArray<appraisal::ResidentAppraisal> appraisal;
  adult::DeviceArray<appraisal::AppraisalWorkspace> appraisal_workspace;
  adult::DeviceArray<ContactSummary> contact_summary;
  adult::DeviceArray<std::uint8_t> ingress;
  adult::DeviceArray<std::uint8_t> candidate;
  adult::DeviceArray<std::uint8_t> egress;
  adult::DeviceArray<std::uint32_t> generated_count;
  adult::DeviceArray<std::uint32_t> transport_counts;
  adult::DeviceArray<std::uint32_t> candidate_rng;
  adult::DeviceArray<discourse_plan::ResidentDiscoursePlanState> query_plan;
  adult::DeviceArray<proposition_chain::OrderedSettlementResult>
      query_settlement;
  adult::DeviceArray<QueryAnswerReceipt> query_answer_receipt;
  adult::DeviceArray<std::uint32_t> ordered_relation_output_units;
  adult::DeviceArray<std::uint32_t> ordered_relation_output_count;
  adult::DeviceArray<adult::ordered_relation::OrderedRelationExecutionReceipt>
      ordered_relation_execution_receipt;
  adult::DeviceArray<population_surface::GroundingResult> query_grounding;
  adult::DeviceArray<PlanAnchorGroundingObserverReceipt>
      query_plan_grounding_observer;
  // Transient cue-to-relation support and its deterministic winner.  This is
  // transport workspace, never persisted semantic state.
  adult::DeviceArray<std::uint64_t> query_topic_support;
  adult::DeviceArray<std::uint64_t> query_topic_key;
  // Frozen pre-contact projections. Each retained relation event owns one
  // opaque construction selection and one ordered anchor frame.
  adult::DeviceArray<std::uint32_t> query_surface_selection;
  adult::DeviceArray<std::uint32_t> query_surface_anchors;
  adult::DeviceArray<std::uint32_t> query_surface_anchor_counts;
  // Exact retained event revision that authorizes the current relation
  // surface. This is transient routing matter, not persisted meaning.
  adult::DeviceArray<std::uint64_t> query_relation_evidence_revision;
  adult::DeviceArray<std::uint32_t> query_relation_evidence_count;
  adult::DeviceArray<adult::construction::WitnessedRelationPlanReceipt>
      query_relation_plan_receipt;
  adult::DeviceArray<SurfaceSpanTransaction> query_surface_transaction;
  adult::DeviceArray<PendingActionTrajectory> pending_action_trajectory;
  adult::DeviceArray<ActionTransitionEvidence> action_transitions;
  adult::DeviceArray<ActionTransitionScalars> action_transition_scalars;
  std::uint32_t chunk_capacity = 0u;
  std::uint32_t emission_capacity = 0u;
  std::uint32_t appraisal_holdout_bytes = 0u;
  std::uint64_t chronological_bytes = 0u;
  bool plasticity_disabled = false;
  bool legacy_generator_enabled = true;
  // Host-only transport mode. Duplex framing closes a contact only on an
  // explicit zero-length frame; the resident adult still owns all content
  // and learned-closure decisions in the default mode.
  bool explicit_streaming_flush = false;
  // Host transaction guard only. One completed external contact advances the
  // resident eligibility clock exactly once, even when it yields no credit.
  bool conditioned_credit_advanced_this_contact = false;
  // ⭐ SKIPS ARE COUNTED, NEVER SILENT. The delayed-factor step may legitimately
  // have nothing to do -- an owner whose routes exist but whose factor state is
  // not configured yet. A silent skip would hide a GENUINE misconfiguration
  // behind the same path, so every skip is counted with the precondition mask
  // that caused it and "skipped" is a measured state rather than an absence.
  std::uint32_t delayed_factor_credit_skipped = 0u;
  std::uint32_t delayed_factor_credit_skip_mask = 0u;
  // Host-only transaction handle for a fragmented contact that may exhaust the
  // declared physical organ bank. It is never part of organism state.
  std::string contact_rollback_checkpoint;
};

inline void save_checkpoint(const StreamState& state, const std::string& path);
inline StreamState load_checkpoint(const std::string& path);

inline std::string contact_rollback_path(const StreamState& state) {
  static std::atomic<std::uint64_t> next{0u};
  std::ostringstream path;
  path << "/tmp/bcc32-contact-rollback-"
       << reinterpret_cast<std::uintptr_t>(&state) << '-'
       << next.fetch_add(1u, std::memory_order_relaxed) << ".chk";
  return path.str();
}

inline void publish_conditioned_conductance(StreamState& state) {
  const std::uint32_t count =
      state.adult.online_conditioned_transition_count;
  adult::DeviceArray<std::uint32_t> published(count == 0u ? 1u : count);
  if (count == 0u || state.conditioned_device_owner.capacity() == 0u) {
    adult::cuda_require(cudaMemset(published.get(), 0, published.bytes()),
                        "clear empty physical conductance surface");
    state.adult.online_conditioned_transition_conductance =
        std::move(published);
    return;
  }
  static_assert(sizeof(adult::ConditionedTransitionKey) ==
                    sizeof(substrate::bcc32::ConditionedMatterDeviceKey));
  static_assert(offsetof(adult::ConditionedTransitionKey, anchor) == 0u);
  static_assert(offsetof(adult::ConditionedTransitionKey, previous) ==
                sizeof(std::uint32_t));
  static_assert(offsetof(adult::ConditionedTransitionKey, next) ==
                2u * sizeof(std::uint32_t));
  state.conditioned_device_owner.publish_conductance_device(
      reinterpret_cast<const substrate::bcc32::ConditionedMatterDeviceKey*>(
          state.adult.online_conditioned_transitions.get()),
      count, published.get());
  state.adult.online_conditioned_transition_conductance =
      std::move(published);
}

inline bool conditioned_credit_owner_active(const StreamState& state) {
  return state.conditioned_device_owner.capacity() != 0u &&
         !state.plasticity_disabled;
}

// ⭐ THE CONTACT BOUNDARY, NAMED. Eligibility may advance at most once per
// contact, and `consume_conditioned_credit_chunk` enforces that with a flag. The
// only thing that cleared it lived INSIDE `contact_host_bytes`, the full
// host-bytes ingress path, so a caller driving credit directly had no way to
// begin a second contact at all.
//
// ⛔ THIS IS EXTRACTION, NOT NEW MECHANISM, and that was measured before it was
// written: the ingress prologue's ONLY assignment to `state` is this flag, and no
// allocation, kernel launch or clock advance precedes the ingest call. So
// "begin a contact" was already separable from "ingest host bytes"; it simply
// had no name.
//
// ⚠ IT DOES NOT WEAKEN THE GUARD. The once-per-contact throw is untouched; this
// only gives the boundary an explicit caller. Two advances without an
// intervening `begin_contact` still throw.
inline void begin_contact(StreamState& state) {
  state.conditioned_credit_advanced_this_contact = false;
}

inline void consume_conditioned_credit_chunk(
    void* context, const adult::ConditionedCreditEvent* device_events,
    std::uint32_t event_count) {
  if (context == nullptr)
    throw std::runtime_error("null conditioned credit consumer context");
  if (event_count != 0u && device_events == nullptr)
    throw std::runtime_error("null conditioned device credit batch");
  StreamState& state = *static_cast<StreamState*>(context);
  // ⭐ AN EMPTY BATCH ADVANCED NOTHING, SO IT MUST NOT MARK THE CONTACT.
  // The once-per-contact guard below exists because eligibility may advance at
  // most once per contact. A zero-event callback applies no credit, so counting
  // it as the contact's advance makes the next real batch throw -- which is
  // exactly what happened when the owner-side empty-batch no-op was added
  // without this. Returning here keeps both properties: no state change, and the
  // contact's one advance still available.
  if (event_count == 0u) return;
  if (state.conditioned_credit_advanced_this_contact)
    throw std::runtime_error(
        "conditioned eligibility advanced more than once for one contact");
  static_assert(sizeof(adult::ConditionedCreditEvent) ==
                    sizeof(substrate::bcc32::ConditionedMatterDeviceCredit));
  static_assert(offsetof(adult::ConditionedCreditEvent, key) == 0u);
  static_assert(offsetof(adult::ConditionedCreditEvent, polarity) ==
                3u * sizeof(std::uint32_t));
  static_assert(offsetof(adult::ConditionedCreditEvent, source_event) ==
                4u * sizeof(std::uint32_t));
  static_assert(offsetof(adult::ConditionedCreditEvent, valid) ==
                5u * sizeof(std::uint32_t));
  if (event_count != 0u) {
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_status =
        cudaPointerGetAttributes(&attributes, device_events);
    if (attribute_status != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice) {
      (void)cudaGetLastError();
      throw std::runtime_error(
          "conditioned credit callback did not receive device memory");
    }
  }
  // ⭐⭐ THE ROUTE-CREDIT STEP THIS CALLBACK NEVER HAD.
  //
  // `consume_conditioned_credit_chunk` ran ONLY the delayed-factor step, so
  // conditioned credit was never applied to routes at all:
  // `PagedConditionedOwner::consume_device_batch` -- the route path, taking this
  // exact batch type -- had ZERO call sites in this header.
  //
  // ⇒ that is why bcc32_cuda_adult_stream_v7_restart_contract measured
  // `occupied_routes == 0` and then died in the factor step with
  // `precondition_mask=14`: the factor step legitimately has no bindings yet,
  // because the contract configures factor state only AFTER a route exists.
  // The contract's ordering was right and the callback was missing a step.
  //
  // ⚠ ORDER MATTERS AND IS NOT ARBITRARY. Routes materialize first; delayed
  // factor credit acts on bindings that name routes. Running the factor step
  // first is what made an unconfigured owner fatal to ordinary credit.
  const auto consume_receipt =
      state.conditioned_device_owner.consume_device_batch(
          reinterpret_cast<
              const substrate::bcc32::ConditionedMatterDeviceCredit*>(
              device_events),
          event_count);
  if (consume_receipt.rejected != 0u)
    throw std::runtime_error(
        "conditioned route credit rejected: requested=" +
        std::to_string(consume_receipt.requested) +
        " admitted=" + std::to_string(consume_receipt.admitted) +
        " inserted=" + std::to_string(consume_receipt.inserted) +
        " rejected=" + std::to_string(consume_receipt.rejected) +
        " failure_index=" + std::to_string(consume_receipt.failure_index));

  const auto factor_receipt =
      state.conditioned_device_owner.apply_resident_conditioned_factor_credit(
          reinterpret_cast<
              const substrate::bcc32::ConditionedMatterDeviceCredit*>(
              device_events),
          event_count);
  // ⚠ THIS SKIP IS ONLY CORRECT BECAUSE THE ROUTE STEP ABOVE EXISTS. Applied on
  // its own it made the whole callback a no-op and the owner never changed --
  // measured, and reverted, before the route step was found missing. With routes
  // materialized here, an unconfigured factor state genuinely has nothing to
  // credit.
  //
  // ⚠ NARROW: only mask bits 2|4|8 -- no bindings, unsized lane, no supply --
  // mean "nothing here to credit". Bit 1, a null batch carrying events, stays
  // fatal: that is a caller error, not an unconfigured owner.
  constexpr std::uint32_t kNoResidentFactorState = 2u | 4u | 8u;
  const bool nothing_to_factor_credit =
      factor_receipt.code ==
          static_cast<std::uint32_t>(
              bcc32::paged_conditioned_owner::OperationCode::kInvalidInput) &&
      (factor_receipt.errors & 1u) == 0u &&
      (factor_receipt.errors & kNoResidentFactorState) != 0u;
  if (nothing_to_factor_credit) {
    ++state.delayed_factor_credit_skipped;
    state.delayed_factor_credit_skip_mask = factor_receipt.errors;
  } else if (factor_receipt.code != static_cast<std::uint32_t>(
          bcc32::paged_conditioned_owner::OperationCode::kOk))
    // ⚠ NAME THE CODE. This threw a bare sentence, so a rejection said only
    // THAT the owner refused and never WHICH refusal it was -- and the target
    // could not compile at all, so nobody had read the message either way.
    throw std::runtime_error(
        "resident delayed factor credit rejected: code=" +
        std::to_string(factor_receipt.code) +
        " events=" + std::to_string(event_count) +
        " precondition_mask=" + std::to_string(factor_receipt.errors) +
        " (1=null batch 2=no bindings 4=lane unsized 8=no eligibility supply)");
  state.conditioned_credit_advanced_this_contact = true;
}

inline void consume_conditioned_prediction_witness_chunk(
    void* context,
    const adult::ConditionedPredictionWitness* device_witnesses,
    std::uint32_t witness_count) {
  if (context == nullptr)
    throw std::runtime_error("null prediction-witness consumer context");
  if (witness_count != 0u && device_witnesses == nullptr)
    throw std::runtime_error("null prediction-witness batch");
  if (witness_count == 0u) return;
  StreamState& state = *static_cast<StreamState*>(context);
  using OwnerWitness =
      bcc32::paged_conditioned_owner::ResidentPredictionWitness;
  static_assert(sizeof(adult::ConditionedPredictionWitness) ==
                sizeof(OwnerWitness));
  static_assert(offsetof(adult::ConditionedPredictionWitness, key) == 0u);
  static_assert(offsetof(adult::ConditionedPredictionWitness, region) ==
                3u * sizeof(std::uint32_t));
  static_assert(offsetof(adult::ConditionedPredictionWitness, factor_index) ==
                4u * sizeof(std::uint32_t));
  static_assert(offsetof(adult::ConditionedPredictionWitness, source_event) ==
                5u * sizeof(std::uint32_t));
  static_assert(offsetof(adult::ConditionedPredictionWitness, valid) ==
                6u * sizeof(std::uint32_t));
  static_assert(adult::kConditionedPredictionFactorWords ==
                bcc32::paged_conditioned_owner::kFactorWordsPerRegion);
  cudaPointerAttributes attributes{};
  const cudaError_t attribute_status =
      cudaPointerGetAttributes(&attributes, device_witnesses);
  if (attribute_status != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice) {
    (void)cudaGetLastError();
    throw std::runtime_error(
        "prediction witness callback did not receive device memory");
  }
  const auto receipt =
      state.conditioned_device_owner.capture_resident_prediction_batch(
          reinterpret_cast<const OwnerWitness*>(device_witnesses),
          witness_count);
  if (receipt.rejected != 0u)
    throw std::runtime_error("resident prediction witness rejected");
}

// ⛔ THIS GUARD IS LOAD-BEARING FOR THE ONLY CHECKPOINT-ONLY TRANSLATION UNIT.
//
// `bcc32_cuda_adult_stream_v7_restart_contract.cu` is the single file in the
// repository that defines `BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY`. In that
// mode the stream header deliberately does NOT include
// `bcc32_cuda_adult_v1.cuh` (see the include selector at the top of this
// file): only the adult STATE surface arrives, via
// `bcc32_cuda_adult_complete_checkpoint_v2.cuh` ->
// `bcc32_cuda_adult_state.cuh`. Everything between here and the matching
// `#endif` is the adult's executable language/drive surface, and it depends on
// symbols that live behind the executable adult header
// (`adult::TrainReport`, `adult::mix32`, ...). It also defines
// `initialize_query_answer_state_kernel`, whose checkpoint-only counterpart is
// supplied by `bcc32_cuda_adult_stream_checkpoint_tail.inl`.
//
// This exact `#if !defined(...)` / `#endif` pair existed until commit
// 2370976158 ("refactor: extract adult stream checkpoint tail", 2026-08-13),
// which moved the tail of this header into an `.inl` and dropped the opening
// and closing lines of the region that stayed behind. Because every other
// includer of this header compiles the `#else` arm of the include selector,
// the resulting breakage was invisible to all ~20 of them and only the one
// checkpoint-only contract went RED.
#if !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)

inline void publish_conditioned_conductance_chunk(
    void* context, adult::AdultState& adult_state) {
  if (context == nullptr)
    throw std::runtime_error("null conditioned publisher context");
  StreamState& state = *static_cast<StreamState*>(context);
  if (&state.adult != &adult_state)
    throw std::runtime_error("conditioned publisher adult mismatch");
  publish_conditioned_conductance(state);
}

struct TickResult {
  std::vector<std::uint8_t> bytes;
  bool input_free = false;
};

struct StreamReport {
  DriveState drive{};
  appraisal::ResidentAppraisal appraisal{};
  adult::TrainReport adult{};
  QueryAnswerReceipt query{};
  std::size_t query_topic_support_slots = 0u;
  std::size_t stream_resident_bytes = 0u;
};

__device__ __forceinline__ std::uint32_t hash_raw_bytes(const std::uint8_t* bytes,
                                                        std::uint32_t count) {
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash == 0u ? 1u : hash;
}

__global__ void initialize_drive_kernel(DriveState* drive, StreamConfig config) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  drive->magic = kDriveMagic;
  drive->previous_surprise = 0xffffffffu;
  drive->emission_threshold = config.emission_threshold;
  drive->surprise_threshold = config.surprise_threshold;
  drive->autonomous_quiet_ticks = config.autonomous_quiet_ticks;
  drive->contact_refractory_ticks = config.contact_refractory_ticks;
  drive->emission_refractory_ticks = config.emission_refractory_ticks;
  drive->emission_base_cost = config.emission_base_cost;
  drive->basal_energy_per_tick = config.basal_energy_per_tick;
  drive->emission_capacity = config.emission_capacity;
  drive->drive_enabled = 1u;
  drive->plasticity_enabled = 1u;
  drive->reafference_enabled = config.reafference_enabled != 0u;
  drive->energy_ledger_ok = 1u;
}

__global__ void summarize_contact_parallel_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, DriveState* drive,
    ContactSummary* summary) {
  std::uint64_t local_activity = 0u;
  std::uint32_t local_unseen = 0u;
  std::uint32_t local_hash = 0u;
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < byte_count; i += stride) {
    const std::uint8_t byte = bytes[i];
    const std::uint8_t previous = i == 0u ? bytes[0] : bytes[i - 1u];
    local_activity += 1u + __popc(static_cast<std::uint32_t>(byte ^ previous));
    local_unseen += atomicAdd(drive->byte_exposure + byte, 1u) == 0u;
    local_hash ^= adult::mix32(static_cast<std::uint32_t>(byte) ^
                               (i * 0x9e3779b9u));
  }

  __shared__ unsigned long long activity[adult::kBlock];
  __shared__ std::uint32_t unseen[adult::kBlock];
  __shared__ std::uint32_t hashes[adult::kBlock];
  activity[threadIdx.x] = local_activity;
  unseen[threadIdx.x] = local_unseen;
  hashes[threadIdx.x] = local_hash;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset) {
      activity[threadIdx.x] += activity[threadIdx.x + offset];
      unseen[threadIdx.x] += unseen[threadIdx.x + offset];
      hashes[threadIdx.x] ^= hashes[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    atomicAdd(&summary->activity, activity[0]);
    atomicAdd(&summary->unseen, unseen[0]);
    atomicXor(&summary->hash, hashes[0]);
  }
}

__global__ void sync_appraisal_parameters_kernel(
    const appraisal::ResidentAppraisal* resident, DriveState* drive) {
  if (blockIdx.x == 0u && threadIdx.x < appraisal::kGenomeParameters) {
    drive->learning_parameters[threadIdx.x] =
        resident->adult.parameter[threadIdx.x];
  }
}

__global__ void finalize_contact_summary_kernel(
    std::uint32_t byte_count, std::uint32_t reafferent,
    std::uint32_t plasticity_applied, const ContactSummary* summary,
    const appraisal::ResidentAppraisal* resident, DriveState* drive) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const appraisal::AppraisalReceipt& receipt = resident->receipt;
  const std::uint32_t hash = summary->hash == 0u ? 1u : summary->hash;
  const std::uint64_t predictive_loss = receipt.after_loss;
  const std::uint32_t predictive_surprise = receipt.heldout_count == 0u
      ? 0u
      : static_cast<std::uint32_t>(min(
            predictive_loss / receipt.heldout_count,
            static_cast<std::uint64_t>(0xffffffffu)));
  const std::uint32_t surprise = summary->unseen +
      (drive->last_contact_hash == 0u
           ? 32u
           : __popc(hash ^ drive->last_contact_hash)) +
      predictive_surprise;

  ++drive->contact_events;
  if (reafferent != 0u) {
    ++drive->reafferent_chunks;
    drive->reafferent_bytes += byte_count;
  } else {
    ++drive->external_chunks;
    drive->external_bytes += byte_count;
  }
  drive->activity_total += summary->activity;
  drive->surprise_total += surprise;
  drive->predictive_loss_total += predictive_loss;
  drive->recent_activity = static_cast<std::uint32_t>(
      min(summary->activity, static_cast<unsigned long long>(0xffffffffu)));
  drive->recent_surprise = surprise;
  drive->recent_predictive_loss = predictive_surprise;
  drive->plasticity_events += plasticity_applied != 0u;
  ++drive->appraisal_events;
  drive->appraisal_commits += receipt.accepted;
  drive->appraisal_rejections += receipt.accepted == 0u;
  drive->appraisal_blocked_commits += receipt.lesion_blocked_improvement;
  drive->improvement_events += receipt.accepted;
  if (drive->last_contact_hash != 0u && drive->last_contact_hash != hash)
    ++drive->revision_events;
  drive->previous_surprise = surprise;
  drive->last_contact_hash = hash;
  drive->chronological_bytes += byte_count;

  const std::uint64_t gained =
      static_cast<std::uint64_t>(byte_count) + summary->activity + surprise;
  drive->energy += gained;
  drive->energy_gained += gained;
  drive->energy_ledger_ok =
      drive->energy_initial + drive->energy_gained ==
      drive->energy + drive->energy_spent;
}

__global__ void drive_tick_kernel(
    DriveState* drive, std::uint32_t inbound_count,
    const discourse_plan::ResidentDiscoursePlanState* query_plan = nullptr,
    QueryAnswerReceipt* query_receipt = nullptr) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  ++drive->ticks;
  if (query_receipt != nullptr)
    query_receipt->endogenous_resident_steps = 0u;
  const std::uint64_t basal = drive->basal_energy_per_tick;
  drive->energy += basal;
  drive->energy_gained += basal;

  const std::uint32_t prior_refractory = drive->refractory_ticks;
  if (inbound_count != 0u) {
    drive->quiet_ticks = 0u;
    if (drive->refractory_ticks != 0u) --drive->refractory_ticks;
  } else {
    if (drive->quiet_ticks != 0xffffffffu) ++drive->quiet_ticks;
    if (drive->refractory_ticks != 0u) --drive->refractory_ticks;
  }

  const bool proposition_plan_ready =
      query_plan != nullptr && discourse_plan::valid(*query_plan) &&
      query_plan->status == discourse_plan::PlanStatus::committed &&
      query_plan->anchor_reference_count != 0u;
  const bool relation_plan_ready =
      query_receipt != nullptr && query_receipt->attempted != 0u &&
      (query_receipt->relation_plan_units != 0u ||
       query_receipt->ordered_relation_ready != 0u ||
       query_receipt->ordered_relation_clarification_ready != 0u);
  const bool query_plan_ready = proposition_plan_ready || relation_plan_ready;
  const bool autonomous_ready = inbound_count == 0u &&
      drive->quiet_ticks >= drive->autonomous_quiet_ticks;
  const bool contact_ready = inbound_count != 0u &&
      (query_plan_ready || drive->recent_surprise >= drive->surprise_threshold);
  // Refractory suppresses autonomous continuation of the previous action.
  // A newly committed plan on an external contact is a distinct causal event;
  // carrying the old action's refractory state into it creates alternating
  // answered and muted contacts even after resident matter has produced a
  // grounded candidate.
  const bool fresh_contact_plan = inbound_count != 0u && query_plan_ready;
  const bool ready = drive->drive_enabled != 0u &&
      (prior_refractory == 0u || fresh_contact_plan) &&
      drive->energy >= drive->emission_threshold &&
      (autonomous_ready || contact_ready);
  drive->emit_pending = ready ? 1u : 0u;
  drive->pending_is_autonomous = ready && inbound_count == 0u ? 1u : 0u;
  if (ready && inbound_count != 0u)
    drive->refractory_ticks = drive->contact_refractory_ticks;
  drive->recent_activity >>= 1u;
  drive->recent_surprise >>= 1u;
  drive->energy_ledger_ok =
      drive->energy_initial + drive->energy_gained == drive->energy + drive->energy_spent;
}

__global__ void initialize_query_answer_state_kernel(
    discourse_plan::ResidentDiscoursePlanState* plan,
    proposition_chain::OrderedSettlementResult* settlement,
    QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (plan != nullptr) *plan = discourse_plan::ResidentDiscoursePlanState{};
  if (settlement != nullptr)
    *settlement = proposition_chain::OrderedSettlementResult{};
  if (receipt != nullptr) *receipt = QueryAnswerReceipt{};
}

// Compress a complete contact or committed action into a bounded sparse
// population using existing learned unit populations.  The sampling is purely
// temporal and address-based; it never names a word, role, language act, or
// outcome.  The same population geometry is used for source, action, and
// later contact, so no slot has authored semantic authority.
__device__ inline std::uint32_t collect_sequence_population(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    population_surface::UnitPopulationView units, std::uint32_t* output) {
  constexpr std::uint32_t kCapacity =
      adult::proposition_tissue::kMaximumPopulationCells;
  if (sequence == nullptr || output == nullptr || sequence_count == 0u ||
      units.cells == nullptr || units.population_width == 0u ||
      units.unit_count == 0u) {
    return 0u;
  }
  const std::uint32_t sample_count =
      sequence_count < (kCapacity / 2u) ? sequence_count : (kCapacity / 2u);
  const std::uint32_t cells_per_unit = units.population_width < 2u
      ? units.population_width : 2u;
  std::uint32_t result_count = 0u;
  for (std::uint32_t sample = 0u; sample < sample_count; ++sample) {
    const std::uint32_t offset = sample_count == 1u
        ? 0u
        : sample * (sequence_count - 1u) / (sample_count - 1u);
    const std::uint32_t unit = sequence[offset];
    if (unit < units.unit_begin || unit - units.unit_begin >= units.unit_count)
      continue;
    const std::uint32_t* cells =
        units.cells + static_cast<std::size_t>(unit) * units.population_width;
    for (std::uint32_t slot = 0u; slot < cells_per_unit && result_count < kCapacity;
         ++slot) {
      bool present = false;
      for (std::uint32_t existing = 0u; existing < result_count; ++existing)
        present = present || output[existing] == cells[slot];
      if (!present) output[result_count++] = cells[slot];
    }
  }
  return result_count;
}

__device__ inline bool same_population_cells(const std::uint32_t* left,
                                             std::uint32_t left_count,
                                             const std::uint32_t* right,
                                             std::uint32_t right_count) {
  if (left == nullptr || right == nullptr || left_count != right_count) return false;
  for (std::uint32_t index = 0u; index < left_count; ++index)
    if (left[index] != right[index]) return false;
  return true;
}

__device__ inline bool same_surface_bytes(const std::uint8_t* left,
                                          std::uint32_t left_count,
                                          const std::uint8_t* right,
                                          std::uint32_t right_count) {
  if (left == nullptr || right == nullptr || left_count != right_count) return false;
  for (std::uint32_t index = 0u; index < left_count; ++index)
    if (left[index] != right[index]) return false;
  return true;
}

__device__ inline bool same_action_transition(
    const ActionTransitionEvidence& evidence,
    const PendingActionTrajectory& trace, const std::uint8_t* later_surface,
    std::uint32_t later_byte_count) {
  return evidence.claimed != 0u &&
      same_surface_bytes(evidence.source_surface, evidence.source_byte_count,
                         trace.source_surface, trace.source_byte_count) &&
      same_surface_bytes(evidence.action_surface, evidence.action_byte_count,
                         trace.action_surface, trace.action_byte_count) &&
      same_surface_bytes(evidence.later_surface, evidence.later_byte_count,
                         later_surface, later_byte_count);
}

__global__ void settle_pending_action_trajectory_kernel(
    adult::proposition_tissue::TissueView tissue,
    population_surface::UnitPopulationView units, const std::uint32_t* sequence,
    const std::uint32_t* sequence_count, const std::uint8_t* later_surface,
    std::uint32_t later_byte_count, PendingActionTrajectory* trace,
    const question_goal::ResidentQuestionGoalState* question,
    const discourse_plan::ResidentDiscoursePlanState* question_plan,
    ActionTransitionEvidence* transitions, std::uint32_t transition_capacity,
    ActionTransitionScalars* transition_scalars) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || trace == nullptr ||
      trace->active == 0u || sequence_count == nullptr)
    return;
  if (trace->question_ticket.active != 0u) {
    const bool have_live_goal = question != nullptr && question_plan != nullptr;
    const bool accepted = have_live_goal &&
        question_ticket::consume_return(
            &trace->question_ticket, *question, *question_plan,
            adult::kDistributedMotorPopulation, trace->source_surface,
            trace->source_byte_count, later_surface, later_byte_count);
    if (accepted) {
      if (trace->question_return_accepted != 0xffffffffu)
        ++trace->question_return_accepted;
    } else if (!have_live_goal) {
      trace->question_ticket.active = 0u;
      if (trace->question_ticket.rejected_returns != 0xffffffffu)
        ++trace->question_ticket.rejected_returns;
      if (trace->question_return_rejected != 0xffffffffu)
        ++trace->question_return_rejected;
    }
  }
  std::uint32_t later_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
  const std::uint32_t later_count = collect_sequence_population(
      sequence, sequence_count[0], units, later_cells);
  const auto source = adult::proposition_tissue::SparsePopulationView{
      trace->source_cells, trace->source_count};
  const auto action = adult::proposition_tissue::SparsePopulationView{
      trace->action_cells, trace->action_count};
  const auto later = adult::proposition_tissue::SparsePopulationView{
      later_cells, later_count};
  // A later external contact is temporal evidence, not reward. Only a real
  // embodied polarity/outcome path may write intervention or counterevidence.
  bool observed = adult::proposition_tissue::assimilate_experience(
      tissue, source, action, later, 0u, 0, 1u);
  if (transitions != nullptr && transition_scalars != nullptr &&
      later_surface != nullptr && later_byte_count != 0u &&
      later_byte_count <= kDefaultChunkCapacity) {
    ActionTransitionEvidence* slot = nullptr;
    for (std::uint32_t index = 0u; index < transition_capacity; ++index) {
      ActionTransitionEvidence& candidate = transitions[index];
      if (same_action_transition(candidate, *trace, later_surface,
                                 later_byte_count)) {
        slot = &candidate;
        break;
      }
      if (slot == nullptr && candidate.claimed == 0u) slot = &candidate;
    }
    if (slot == nullptr) {
      ++transition_scalars->capacity_rejections;
    } else {
      if (slot->claimed == 0u) {
        for (std::uint32_t index = 0u; index < trace->source_byte_count; ++index)
          slot->source_surface[index] = trace->source_surface[index];
        for (std::uint32_t index = 0u; index < trace->source_count; ++index)
          slot->source_cells[index] = trace->source_cells[index];
        for (std::uint32_t index = 0u; index < trace->action_byte_count; ++index)
          slot->action_surface[index] = trace->action_surface[index];
        for (std::uint32_t index = 0u; index < trace->action_count; ++index)
          slot->action_cells[index] = trace->action_cells[index];
        for (std::uint32_t index = 0u; index < later_byte_count; ++index)
          slot->later_surface[index] = later_surface[index];
        for (std::uint32_t index = 0u; index < later_count; ++index)
          slot->later_cells[index] = later_cells[index];
        slot->source_byte_count = trace->source_byte_count;
        slot->source_count = trace->source_count;
        slot->action_byte_count = trace->action_byte_count;
        slot->action_count = trace->action_count;
        slot->later_byte_count = later_byte_count;
        slot->later_count = later_count;
        slot->claimed = 1u;
        ++transition_scalars->occupied;
      }
      ++slot->observational_support;
      slot->revision = ++transition_scalars->revision;
      ++transition_scalars->observed;
      observed = true;
    }
  }
  if (observed) ++trace->observed_transitions;
  ++trace->closed_transitions;
  trace->active = 0u;
  trace->action_byte_count = 0u;
  trace->action_count = 0u;
}

__global__ void capture_contact_source_kernel(
    population_surface::UnitPopulationView units, const std::uint32_t* sequence,
    const std::uint32_t* sequence_count, const std::uint8_t* source_surface,
    std::uint32_t source_byte_count, PendingActionTrajectory* trace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || trace == nullptr ||
      sequence_count == nullptr || source_surface == nullptr ||
      source_byte_count == 0u || source_byte_count > kDefaultChunkCapacity)
    return;
  std::uint32_t source_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
  const std::uint32_t source_count = collect_sequence_population(
      sequence, sequence_count[0], units, source_cells);
  if (trace->active != 0u) ++trace->superseded_sources;
  for (std::uint32_t index = 0u; index < source_byte_count; ++index)
    trace->source_surface[index] = source_surface[index];
  trace->source_byte_count = source_byte_count;
  for (std::uint32_t index = 0u; index < source_count; ++index)
    trace->source_cells[index] = source_cells[index];
  trace->source_count = source_count;
  trace->action_byte_count = 0u;
  trace->action_count = 0u;
  trace->active = 0u;
  ++trace->source_revision;
}

__global__ void stage_contact_response_plan_kernel(
    adult::proposition_tissue::TissueView tissue,
    adult::proposition_tissue::CompletionWorkspaceView workspace,
    population_surface::UnitPopulationView units, const std::uint32_t* sequence,
    const std::uint32_t* sequence_count, const std::uint32_t* exact_cue_units,
    const std::uint32_t* cue_scores, const std::uint32_t* cue_orders,
    const std::uint32_t* onset_mass,
    const std::uint32_t* role_canon,
    const std::uint32_t* closed_class_mask,
    const std::uint32_t* construction_witnesses,
    std::uint32_t construction_witness_capacity,
    const std::uint8_t* source_surface,
    std::uint32_t source_byte_count,
    adult::proposition_tissue::CompletionResult* completion,
    discourse_plan::ResidentDiscoursePlanState* plan,
    proposition_chain::OrderedSettlementResult* settlement,
    PendingActionTrajectory* trace, const ActionTransitionEvidence* transitions,
    std::uint32_t transition_capacity,
    const ActionTransitionScalars* transition_scalars, QueryAnswerReceipt* receipt,
    // Env-gated ablation of the bounded two-binding join inside the ordered
    // settlement. Device code cannot read getenv, so the host reads
    // BCC32_ORDERED_JOIN_LESION once and passes the decision in. Defaulted
    // false == join enabled == today's behaviour for every other caller.
    bool ordered_join_lesioned = false) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || plan == nullptr ||
      settlement == nullptr || receipt == nullptr || completion == nullptr) {
    return;
  }
  *settlement = proposition_chain::OrderedSettlementResult{};
  *receipt = QueryAnswerReceipt{};
  discourse_plan::clear(plan);
  (void)tissue;
  receipt->trajectory_observed = trace == nullptr
      ? 0u : static_cast<std::uint32_t>(trace->observed_transitions);
  receipt->trajectory_closed = trace == nullptr
      ? 0u : static_cast<std::uint32_t>(trace->closed_transitions);
  receipt->trajectory_pending = trace != nullptr && trace->active != 0u ? 1u : 0u;
  if (sequence_count == nullptr)
    return;

  receipt->exact_cue_units = sequence_count[0];
  *completion = adult::proposition_tissue::CompletionResult{};
  // A contact may settle resident matter without being an attempt to elicit a
  // response. The only admission signal here is a form the organism learned
  // from earlier interrogative contacts: an exact learned opener at the
  // START of this cue. Treating any later matching unit (including learned
  // whitespace/fragments) as an opener makes ordinary factual contacts look
  // interrogative and prevents them from becoming relation evidence. There
  // is no byte test, word list, or host classifier, so this remains learned
  // and can admit a previously learned form without a final '?'.
  bool learned_question_form = false;
  if (exact_cue_units != nullptr && onset_mass != nullptr && cue_orders != nullptr) {
    for (std::uint32_t offset = 0u; offset < units.unit_count; ++offset) {
      const std::uint32_t unit = units.unit_begin + offset;
      // Modality belongs to the exact witnessed opener. A role analogue may
      // transfer slot behavior later, but must not turn a statement into a
      // request before that statement can become evidence.
      const std::uint32_t onset_evidence = onset_mass[unit];
      // relation_cue_orders is intentionally one-based: zero means no
      // retained cue order and the first retained cue unit is one.
      if (exact_cue_units[unit] != 0u && cue_orders[unit] == 1u)
        receipt->question_onset_evidence = onset_evidence;
      if (exact_cue_units[unit] != 0u && cue_orders[unit] == 1u &&
          onset_evidence >= adult::construction::kQonsetTopicFloor) {
        learned_question_form = true;
        break;
      }
    }
  }
  receipt->learned_question_form = learned_question_form ? 1u : 0u;
  if (!learned_question_form)
    return;
  // Prefer a qualified ordered binding because it retains each learned role
  // identity through grounding.  The generic population path below remains
  // available for resident matter that has not formed an ordered binding.
  const std::uint32_t surface_revision = tissue.scalars == nullptr
      ? 0u
      : static_cast<std::uint32_t>(tissue.scalars->revision);
  const proposition_chain::ExactUnitCueView exact_cue{
      exact_cue_units, units.unit_begin, units.unit_count, onset_mass, cue_orders,
      cue_scores, closed_class_mask};
  proposition_chain::OrderedSettlementResult ordered{};
  const bool ordered_staged = proposition_chain::stage_plan_from_ordered_exact_units(
      tissue, exact_cue,
      // The contact is an exact learned cue, so its first admissible route is
      // an episodic binding.  Broader discourse qualification remains the
      // generic route below and still requires independent contexts.
      adult::proposition_tissue::OrderedBindingQualification::exact_episode,
      units, plan, &ordered, surface_revision, nullptr, construction_witnesses,
      construction_witness_capacity, ordered_join_lesioned);
  receipt->claimed_ordered_bindings = ordered.claimed_bindings;
  receipt->exact_topic_matches = ordered.exact_topic_matches;
  receipt->qualified_best_count = ordered.qualified_bindings;
  receipt->qualified_witnessed_bindings = ordered.qualified_witnessed_bindings;
  receipt->witnessed_role_cell_overlap_bindings =
      ordered.witnessed_role_cell_overlap_bindings;
  receipt->max_witnessed_role_cell_overlap =
      ordered.max_witnessed_role_cell_overlap;
  receipt->selected_binding_index = ordered.selected_binding_index;
  receipt->selected_present_roles = ordered.selected_role_coverage;
  receipt->selected_episode_coverage = ordered.selected_episode_coverage;
  receipt->max_role_coverage_any_episode = ordered.max_role_coverage_any_episode;
  receipt->join_gate_qualified_count = ordered.join_gate_qualified_count;
  receipt->join_gate_entered = ordered.join_gate_entered;
  receipt->episode_spine_steps = ordered.episode_spine_steps;
  receipt->episode_spine_terminal = ordered.episode_spine_terminal;
  receipt->episode_spine_ambiguous = ordered.episode_spine_ambiguous;
  if (ordered_staged)
    *settlement = ordered;
  proposition_chain::SettlementResult generic{};
  if (!ordered_staged) {
    // A new contact is not an address into a replay table. Its exact learned
    // populations remain the cue for the recurrent population settlement.
    proposition_chain::stage_plan_from_units(
        tissue, units, sequence, sequence_count[0],
        {workspace.cell_scores, workspace.cell_capacity}, plan, &generic,
        surface_revision, adult::proposition_tissue::CompletionPolicy::discourse,
        0u);
    settlement->attempted = generic.attempted;
    settlement->staged = generic.staged;
    settlement->step_count = generic.step_count;
    settlement->capacity_stopped = generic.capacity_stopped;
    settlement->tissue_revision = generic.tissue_revision;
  }
  receipt->attempted = settlement->attempted;
  receipt->staged = settlement->staged;
  receipt->completion_cells = ordered_staged ? settlement->exact_topic_matches
                                             : generic.cue_cell_count;
  if (!ordered_staged)
    receipt->qualified_best_count = generic.staged;
  // `uncertain_mass` is a residual completion metric, not a tied-winner
  // receipt. Do not relabel residual mass as semantic ambiguity.
  receipt->ambiguous = 0u;
  completion->ready = settlement->staged;
  completion->qualified_synapses = ordered_staged
      ? settlement->exact_topic_matches
      : static_cast<std::uint32_t>(generic.strongest_score);
  completion->strongest_score = generic.strongest_score;
  completion->tissue_revision = settlement->tissue_revision;
  (void)source_surface;
  (void)source_byte_count;
  (void)trace;
  (void)transitions;
  (void)transition_capacity;
  (void)transition_scalars;
}

__global__ void finalize_contact_response_grounding_kernel(
    discourse_plan::ResidentDiscoursePlanState* plan,
    question_goal::ResidentQuestionGoalState* goal,
    const population_surface::GroundingResult* grounding,
    PendingActionTrajectory* trace, QueryAnswerReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr) return;
  receipt->trajectory_pending = trace != nullptr && trace->active != 0u ? 1u : 0u;
  receipt->staged = 0u;
  receipt->question_ticket = question_ticket::ResidentQuestionActionTicket{};
  receipt->anchor_count = 0u;
  // GroundingResult remains the materializer receipt for ordinary relation
  // answers.  A question-bound Plan has stronger canonical authority: its live
  // Goal/Plan pair decides admission, so forging or clearing the observer
  // projection cannot create or suppress a public question action.
  const bool question_bound =
      plan != nullptr && plan->question_goal_dependency != 0u;
  const bool authorized = question_bound
      ? goal != nullptr &&
            question_goal::materialized_question_plan_is_authoritative(*goal,
                *plan, adult::kDistributedMotorPopulation)
      : plan != nullptr && grounding != nullptr && grounding->ready != 0u &&
            discourse_plan::valid(*plan) &&
            plan->status == discourse_plan::PlanStatus::committed &&
            plan->anchor_reference_count != 0u;
  if (!authorized) {
    if (plan != nullptr) {
      if (question_bound && goal != nullptr) {
        discourse_plan::invalidate(plan, goal->revision);
        question_goal::normalize_plan_goal_pair(goal, plan);
      } else {
        discourse_plan::clear(plan);
      }
    }
    return;
  }
  receipt->attempted = 1u;
  receipt->staged = 1u;
  receipt->anchor_count = plan->anchor_reference_count;
  if (question_bound && goal != nullptr)
    question_ticket::issue(&receipt->question_ticket, *goal, *plan);
}

__global__ void capture_emitted_action_kernel(
    const std::uint32_t* transport_counts, const std::uint8_t* action_surface,
    population_surface::UnitPopulationView units, const std::uint32_t* sequence,
    const std::uint32_t* sequence_count, PendingActionTrajectory* trace,
    QueryAnswerReceipt* receipt,
    const question_goal::ResidentQuestionGoalState* question,
    const discourse_plan::ResidentDiscoursePlanState* question_plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || transport_counts == nullptr ||
      sequence_count == nullptr || trace == nullptr)
    return;
  if (transport_counts[0] == 0u || transport_counts[0] > kDefaultEmissionCapacity ||
      action_surface == nullptr || trace->source_byte_count == 0u)
    return;
  // A committed resident action keeps the population that selected the motor
  // plan. A legacy output may have no population organ at all (the generic
  // motor/reafference contract deliberately disables resident language). In
  // that case the performed raw surface is still real action evidence, but it
  // must remain surface-only: never fabricate a proposition cell from a unit
  // index or a host-selected sentinel.
  if (receipt == nullptr || receipt->attempted == 0u || trace->action_count == 0u) {
    std::uint32_t action_cells[adult::proposition_tissue::kMaximumPopulationCells]{};
    const std::uint32_t action_count = collect_sequence_population(
        sequence, sequence_count[0], units, action_cells);
    if (action_count == 0u &&
        ((receipt != nullptr && receipt->attempted != 0u) ||
         units.cells != nullptr))
      return;
    for (std::uint32_t index = 0u; index < action_count; ++index)
      trace->action_cells[index] = action_cells[index];
    trace->action_count = action_count;
  }
  for (std::uint32_t index = 0u; index < transport_counts[0]; ++index)
    trace->action_surface[index] = action_surface[index];
  trace->action_byte_count = transport_counts[0];
  trace->active = 1u;
  ++trace->action_revision;
  if (receipt != nullptr && receipt->question_ticket.issued != 0u &&
      question != nullptr && question_plan != nullptr) {
    trace->question_ticket = receipt->question_ticket;
    if (!question_ticket::arm(
            &trace->question_ticket, *question, *question_plan,
            adult::kDistributedMotorPopulation, action_surface,
            transport_counts[0])) {
      trace->question_ticket = question_ticket::ResidentQuestionActionTicket{};
      receipt->question_ticket = question_ticket::ResidentQuestionActionTicket{};
    }
  }
  if (receipt != nullptr) receipt->trajectory_pending = 1u;
}

// The raw return is admitted by the ticket before assimilation.  Only the
// accepted ticket may discharge the resident Goal; generic later contacts may
// still remain ordinary temporal action evidence.
__global__ void discharge_ticketed_question_goal_kernel(
    adult::proposition_tissue::TissueView tissue,
    question_goal::ResidentQuestionGoalState* question,
    PendingActionTrajectory* trace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || question == nullptr ||
      trace == nullptr || trace->question_return_accepted == 0u)
    return;
  (void)question_goal::discharge_question_goal_from_tissue(tissue, question);
  trace->question_return_accepted = 0u;
}

// Diagnosed 2026-08-14 (0X1-159): this kernel structurally never reads a
// plan. Every reachable path below already has generated_count[0] == 0u (the
// prior guard returns otherwise), so a construction-readiness branch gated on
// generated_count[0] != 0u can never fire -- it was dead code, not a
// conditional fallback. The `plan`/`construction_result`/`unit_*`/
// `candidate*` parameters this kernel used to take were only ever void-cast;
// real surface realization for a committed learned action already happened
// earlier in the tick via stage_stream_surface_plan_kernel /
// append_stream_surface_step_kernel (see finalize_stream_surface_span_kernel
// just above this call). This step is the always-silent no-op that resets
// the receipt when that earlier path produced nothing. Anchors are evidence
// for construction matching, not an alternate text generator: a failed
// construction remains silent until resident learning supplies a lawful
// surface witness. The two-binding join upstream in
// bcc32_cuda_resident_proposition_chain.cuh is NOT retired by this change --
// its discourse-plan output is a real input to stage_stream_surface_plan_kernel
// (selection[1] indexes plan->steps there), it remains fully instrumented
// (join_gate_qualified_count/join_gate_entered/episode_spine_*) and
// lesion-gated (BCC32_ORDERED_JOIN_LESION); only the redundant plumbing that
// carried its output into this kernel, which never consulted it, is removed.
__global__ void realize_query_answer_plan_kernel(
    QueryAnswerReceipt* receipt, std::uint32_t* generated_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr ||
      generated_count == nullptr) {
    return;
  }
  if (receipt->attempted == 0u) {
    receipt->ngram_fallback = 1u;
    return;
  }
  if (generated_count[0] != 0u) return;
  *generated_count = 0u;
  receipt->ngram_fallback = 0u;
  receipt->serialized_units = 0u;
}

#endif  // !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)

#include "bcc32_cuda_adult_stream_surface_realization.inl"

#include "hardware_native/bcc32_cuda_adult_stream_endogenous_step.inl"
inline void allocate_transport(StreamState& state,
                               std::uint32_t chunk_capacity,
                               std::uint32_t emission_capacity,
                               std::uint32_t appraisal_holdout_bytes) {
  if (chunk_capacity == 0u || emission_capacity == 0u ||
      appraisal_holdout_bytes == 0u) {
    throw std::runtime_error("stream capacities must be nonzero");
  }
  state.chunk_capacity = chunk_capacity;
  state.emission_capacity = emission_capacity;
  state.appraisal_holdout_bytes = appraisal_holdout_bytes;
  state.drive.allocate(1u);
  state.appraisal.allocate(1u);
  state.appraisal_workspace.allocate(1u);
  state.contact_summary.allocate(1u);
  state.ingress.allocate(chunk_capacity);
  state.candidate.allocate(emission_capacity);
  state.egress.allocate(emission_capacity);
  state.generated_count.allocate(1u);
  state.transport_counts.allocate(2u);
  state.candidate_rng.allocate(1u);
  state.query_plan.allocate(1u);
  state.query_settlement.allocate(1u);
  state.query_answer_receipt.allocate(1u);
  state.ordered_relation_output_units.allocate(
      adult::ordered_relation::kMaximumExecutionOutputUnits);
  state.ordered_relation_output_count.allocate(1u);
  state.ordered_relation_execution_receipt.allocate(1u);
  state.query_grounding.allocate(1u);
  state.query_plan_grounding_observer.allocate(1u);
  // Online contacts can materialize new units after stream startup. This
  // device tournament is indexed by unit identity, so its extent must follow
  // the adult's fixed capacity rather than the startup population.
  state.query_topic_support.allocate(state.adult.unit_capacity);
  state.query_topic_key.allocate(1u);
  state.query_surface_selection.allocate(
      adult::construction::kRelationSurfaceEvidenceCap * 4u);
  state.query_surface_anchors.allocate(
      adult::construction::kRelationSurfaceEvidenceCap *
      adult::construction::kConstructionMaxSlots);
  state.query_surface_anchor_counts.allocate(
      adult::construction::kRelationSurfaceEvidenceCap);
  state.query_relation_evidence_revision.allocate(
      adult::construction::kRelationSurfaceEvidenceCap);
  state.query_relation_evidence_count.allocate(1u);
  state.query_relation_plan_receipt.allocate(1u);
  state.query_surface_transaction.allocate(1u);
  state.pending_action_trajectory.allocate(1u);
  state.action_transitions.allocate(kActionTransitionCapacity);
  state.action_transition_scalars.allocate(1u);
  adult::cuda_require(cudaMemset(state.query_grounding.get(), 0,
                                 state.query_grounding.bytes()),
                      "clear allocated resident contact-response grounding");
  adult::cuda_require(cudaMemset(state.query_plan_grounding_observer.get(), 0,
                                 state.query_plan_grounding_observer.bytes()),
                      "clear allocated resident plan grounding observer");
  adult::cuda_require(cudaMemset(state.ordered_relation_output_units.get(), 0,
                                 state.ordered_relation_output_units.bytes()),
                      "clear allocated ordered relation output units");
  adult::cuda_require(cudaMemset(state.ordered_relation_output_count.get(), 0,
                                 state.ordered_relation_output_count.bytes()),
                      "clear allocated ordered relation output count");
  adult::cuda_require(
      cudaMemset(state.ordered_relation_execution_receipt.get(), 0,
                 state.ordered_relation_execution_receipt.bytes()),
      "clear allocated ordered relation execution receipt");
  adult::cuda_require(cudaMemset(state.query_surface_selection.get(), 0xff,
                                 state.query_surface_selection.bytes()),
                      "clear allocated resident surface selection");
  adult::cuda_require(cudaMemset(state.query_surface_anchors.get(), 0xff,
                                 state.query_surface_anchors.bytes()),
                      "clear allocated resident surface anchors");
  adult::cuda_require(cudaMemset(state.query_surface_anchor_counts.get(), 0,
                                 state.query_surface_anchor_counts.bytes()),
                      "clear allocated resident surface anchor extents");
  adult::cuda_require(cudaMemset(state.query_relation_evidence_revision.get(), 0,
                                 state.query_relation_evidence_revision.bytes()),
                      "clear allocated relation surface evidence");
  adult::cuda_require(cudaMemset(state.query_relation_evidence_count.get(), 0,
                                 state.query_relation_evidence_count.bytes()),
                      "clear allocated relation surface extent");
  adult::cuda_require(cudaMemset(state.query_relation_plan_receipt.get(), 0,
                                 state.query_relation_plan_receipt.bytes()),
                      "clear allocated witnessed relation plan receipt");
  adult::cuda_require(cudaMemset(state.query_surface_transaction.get(), 0,
                                 state.query_surface_transaction.bytes()),
                      "clear allocated resident surface transaction");
  adult::cuda_require(cudaMemset(state.pending_action_trajectory.get(), 0,
                                 state.pending_action_trajectory.bytes()),
                      "clear allocated pending resident action trajectory");
  adult::cuda_require(cudaMemset(state.action_transitions.get(), 0,
                                 state.action_transitions.bytes()),
                      "clear allocated resident action transitions");
  adult::cuda_require(cudaMemset(state.action_transition_scalars.get(), 0,
                                 state.action_transition_scalars.bytes()),
                      "clear allocated resident action transition scalars");
}

#include "bcc32_cuda_adult_stream_tick_execution.inl"

#include "bcc32_cuda_adult_stream_checkpoint_tail.inl"

#pragma once

// Process/storage membrane for the current RWR0 resident-rewrite production
// law epoch. Older scalar, ordinary-F, and recurrent implementations remain
// focused donor controls only and are absent from canonical organismd linkage.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "bcc32_provenance.hpp"

namespace substrate::bcc32::persistent_kernel {

using BoundaryWord = std::uint32_t;

// Host-selected observer fidelity. This value is never resident matter and is
// never visible to resident selection, learning, rewriting, or publication.
// RUNG 1 implements only off and l0. Higher levels are reserved so the public
// configuration surface does not need to change when later rungs land.
enum class BitBusCircuitLevel : std::uint32_t {
  off = 0u,
  l0 = 1u,
  l1 = 2u,
  l2 = 3u,
  l3 = 4u,
  l4 = 5u,
};

// One observer-only physical boundary transfer.
//
// No resident word, semantic category, region name, token, answer, score, or
// learned feature appears here. The before/after values are compact
// fingerprints of already-public commitment state, not substrate dumps.
struct BitBusCircuitEvent {
  std::uint64_t epoch = 0u;
  std::uint32_t source_physical_locus = 0xffffffffu;
  std::uint32_t destination_physical_locus = 0xffffffffu;
  std::uint64_t native_state_before = 0u;
  std::uint64_t native_state_after = 0u;
  std::uint32_t transfer_kind = 0u;
  std::uint32_t fanout_count = 0u;
  std::uint64_t source_revision = 0u;
  std::uint32_t morphology_changed = 0u;
  std::uint32_t reserved = 0u;

  friend bool operator==(const BitBusCircuitEvent&,
                         const BitBusCircuitEvent&) = default;
};

struct BitBusCircuitSnapshot {
  std::vector<BitBusCircuitEvent> events;
  std::uint64_t next_sequence = 0u;
  std::uint64_t overwritten = 0u;
};

// Configure process-wide CUDA device-runtime resources required by the
// resident rewrite adult. This must run before earlier CUDA work can make
// device-runtime limits immutable; repeated calls are process-wide no-ops.
void bootstrap_resident_device_runtime();

inline constexpr std::uint32_t kAdaptiveInitialStructuralMatterQ8PerCell = 64u;
inline constexpr std::uint32_t kAdaptiveEdgeStructuralDoseQ8 = 16u;
inline constexpr std::uint32_t kAdaptiveLocalProposalRadius = 8u;

// A physical world event, not a semantic writer.  The controller can expose
// a bounded local injury or coupling block, but cannot name an owner, route,
// operand, answer, operator, or desired behavior.
struct RawPhysicalIntervention {
  std::uint32_t center = 0u;
  std::uint32_t radius = 0u;
  std::uint32_t matter_q8 = 0u;
  std::uint32_t coupling_q8 = 0u;
  std::uint32_t duration_epochs = 0u;
  // Optional raw packet transport channels.  These are physical flux,
  // polarity, and displacement, never a field/region/operator selector.
  std::int32_t flux_q8 = 0;
  std::int32_t polarity_q8 = 0;
  std::int32_t displacement = 0;
  std::uint32_t transport_q8 = 0u;
  // Additional unlabelled physical channels.  Diffusion is a local spread
  // coefficient, receptor gain is a local tissue coupling, and repair is a
  // bounded resource pulse.  None names a source, route, relation, or goal.
  std::uint32_t diffusion_q8 = 0u;
  std::uint32_t receptor_gain_q8 = 256u;
  std::uint32_t repair_q8 = 0u;
};

// An opaque, device-issued capability for one raw consequence of one committed
// public action.  It names no route, source, target, or interpretation.  A
// body may echo it with raw contact exactly once; it cannot mint one that the
// resident writer has not first issued.  This is an action-to-return routing
// receipt, not authentication that the returned bytes came from a physical
// world rather than a dishonest body adapter.
struct ActionReturnTicket {
  // Rotated for each resident boot in this host process.  It prevents a ticket
  // from an otherwise identical live sibling adult from being accepted here;
  // process-restart and checkpoint-restore separation remain unproven.
  std::uint64_t issuer_instance = 0u;
  std::uint64_t action_sequence = 0u;
  std::uint64_t nonce = 0u;

  friend bool operator==(const ActionReturnTicket&, const ActionReturnTicket&) = default;
};

// A body attachment is outside the cognitive authority boundary: it selects a
// fixed device-local transition, never a word, action, route, or resident
// write. The simulated device body is a bounded closed-loop control, not a
// connection to the separately declared host route or an external
// physical-source claim.
enum class DeviceBodyAttachment : std::uint32_t {
  detached = 0u,
  simulated_device = 1u,
};

struct TickReceipt {
  std::uint64_t tick = 0u;
  std::uint64_t phase = 0u;
  std::uint64_t contact_sequence = 0u;
  ContentAddress sealed_execution{};
  ContentAddress law{};
  ContentAddress image{};
  ContentAddress input{};
  ContentAddress output{};
  ContentAddress predecessor{};
  ContentAddress commitment{};
  // The numeric projection remains device-owned and observer-only; a zero
  // valid bit is explicit source-only/lesion/interaction-off abstention.
  std::int32_t joint_output = 0;
  std::int32_t joint_residual = 0;
  std::uint32_t joint_output_valid = 0u;
  std::uint64_t joint_route = 0u;
  // Device-authored field ecology metrics.  These are a bounded tuple of
  // route-local observations, not a semantic field id or an abstraction
  // scalar.  The resident field itself never crosses the passive boundary.
  std::int32_t field_response_l1 = 0;
  std::int32_t field_residual_l1 = 0;
  std::int32_t field_peak_response = 0;
  std::int32_t field_peak_residual = 0;
  std::uint32_t field_active_routes = 0u;
  std::uint32_t field_contact_routes = 0u;
  std::uint32_t field_responsive_routes = 0u;
  std::uint64_t field_owner_mix = 0u;
  std::int32_t developmental_pressure_l1 = 0;
  std::int32_t functional_horizon = 0;
  std::int32_t functional_integration = 0;
  std::int32_t functional_delay = 0;
  std::int32_t functional_spatial_tv_l1 = 0;
  ContentAddress structure{};
  ContentAddress intervention{};
  std::uint64_t intervention_sequence = 0u;
  std::uint32_t structural_focus_cell = 0u;
  std::uint32_t removed_matter_q8_sum = 0u;
  std::uint32_t cut_coupling_q8_sum = 0u;
  std::uint32_t structural_support_q8 = 0u;
  std::int32_t effective_field_response_l1 = 0;
  std::uint32_t intervention_active = 0u;
  std::uint32_t field_source_count = 0u;
  std::uint32_t field_supported_source_count = 0u;
  std::uint32_t field_receptor_count = 0u;
  std::int32_t field_packet_flux_l1 = 0;
  std::int32_t field_polarity_l1 = 0;
  std::uint64_t field_return_packets_q8 = 0u;
  std::uint32_t field_near_profile = 0u;
  std::uint32_t field_middle_profile = 0u;
  std::uint32_t field_far_profile = 0u;
  std::uint64_t field_anchor_mix = 0u;
  std::uint32_t field_anchor_moment = 0u;
  std::uint32_t field_profile_tv_l1 = 0u;
  std::uint64_t field_relation_mix = 0u;
  std::uint32_t field_relation_l1 = 0u;
  std::int32_t field_relation_sum = 0;
  std::int32_t field_polarity_sum = 0;
  std::uint32_t field_receptor_gain_q8 = 256u;
  std::uint32_t field_diffusion_l1 = 0u;
  std::uint32_t field_damage_q8 = 0u;
  std::uint32_t field_recruited_sources = 0u;
  std::uint32_t field_repaired_sources = 0u;
  std::uint32_t field_growth_front_count = 0u;
  // Observer-only summaries of the first two resident adjacency components,
  // ordered by their physical minimum anchor. These are not host region IDs.
  std::uint32_t field_component_count = 0u;
  std::int32_t field_component0_response_l1 = 0;
  std::int32_t field_component1_response_l1 = 0;
  std::int32_t field_component0_residual_l1 = 0;
  std::int32_t field_component1_residual_l1 = 0;
  std::uint32_t field_component0_support_q8 = 0u;
  std::uint32_t field_component1_support_q8 = 0u;
  std::uint32_t field_withdrawn_sources = 0u;
  // Device-owned physical expression state at the learned joint locus.
  std::uint32_t active_joint_locus = 0u;
  std::uint32_t tissue_matter_q8 = 256u;
  std::uint32_t tissue_coupling_q8 = 256u;
  // Observer-only projection of per-source temporal composition. The
  // resident braid itself never crosses the passive boundary.
  std::int32_t field_packet_braid_sum = 0;
  std::uint32_t field_packet_braid_l1 = 0u;
  std::uint32_t field_packet_braid_sources = 0u;
  // Coupled adaptive-ecology projection. These fields summarize resident
  // fast/slow credit, connectivity, resource, turnover, repair, and replay;
  // they do not expose a host-selected mode or semantic state vector.
  std::uint32_t adaptive_context_l1 = 0u;
  std::uint32_t adaptive_context_anchor = 0u;
  std::uint32_t adaptive_context_matter_q8 = 0u;
  std::uint32_t adaptive_plasticity_q8 = 0u;
  std::uint32_t adaptive_consolidation_q8 = 0u;
  std::uint32_t adaptive_growth_q8 = 0u;
  std::uint32_t adaptive_turnover_q8 = 0u;
  std::uint32_t adaptive_repair_q8 = 0u;
  std::uint32_t adaptive_replay_q8 = 0u;
  std::uint32_t adaptive_live_edges = 0u;
  std::uint32_t adaptive_consolidated_routes = 0u;
  std::uint64_t adaptive_structural_revision = 0u;
  std::uint64_t adaptive_recruited_edges = 0u;
  std::uint64_t adaptive_pruned_edges = 0u;
  std::uint64_t adaptive_repaired_cells = 0u;
  std::uint64_t adaptive_free_structural_matter_q8 = 0u;
  std::uint64_t adaptive_edge_structural_matter_q8 = 0u;
  std::uint64_t adaptive_cumulative_structural_debit_q8 = 0u;
  std::uint64_t adaptive_cumulative_structural_refund_q8 = 0u;
  std::uint64_t adaptive_local_proposal_activity = 0u;

  // Repeatedly credited resident joints may develop two expression endpoints.
  // These are passive geometry/gain observations, never host route selection.
  std::uint32_t alternate_expression_endpoint = 0u;
  std::uint32_t joint_expression_endpoints = 0u;
  std::uint32_t primary_expression_gain_q8 = 0u;
  std::uint32_t alternate_expression_gain_q8 = 0u;

  // The symmetric packet-history product is independent of the existing
  // antisymmetric braid. Both are local resident state.
  std::int32_t field_packet_alignment_sum = 0;
  std::uint32_t field_packet_alignment_l1 = 0u;
  std::uint32_t field_packet_alignment_sources = 0u;
  std::uint32_t field_mature_packet_sources = 0u;

  // Effective, tissue-gated causal projections. Signed sums retain direction;
  // L1 values support matched lesion comparisons.
  std::int32_t field_effective_alignment_sum = 0;
  std::int32_t field_effective_braid_sum = 0;
  std::uint32_t field_effective_alignment_l1 = 0u;
  std::uint32_t field_effective_braid_l1 = 0u;
  std::uint32_t field_alignment_locus = 0u;
  std::uint32_t field_braid_locus = 0u;

  // Generic founder-signed projection of the resident body into the recurrent
  // language child. No semantic label or host-computed feature crosses this
  // boundary.
  std::int32_t recurrent_context_l1_q16 = 0;
  std::uint64_t recurrent_context_mix = 0u;
  std::uint64_t recurrent_context_revision = 0u;

  // RWR0 production-law epoch. The complete resident Record population is
  // private; these are passive causal/accounting receipts and raw motor matter.
  ContentAddress rewrite_world{};
  std::uint64_t rewrite_revision = 0u;
  std::uint64_t rewrite_admitted_events = 0u;
  // Observer receipt for the parallel live-frontier pass. It reports how
  // much currently active trajectory matter participated in this epoch; it
  // is not a semantic population count or an output selector.
  std::uint32_t rewrite_live_frontier_records = 0u;
  std::uint64_t rewrite_live_frontier_digest = 0u;
  std::uint32_t rewrite_fault = 0u;
  std::uint32_t rewrite_owned_clock = 0u;
  std::uint32_t rewrite_descriptions = 0u;
  std::uint32_t rewrite_mature_descriptions = 0u;
  std::uint32_t rewrite_partial_matches = 0u;
  std::uint32_t rewrite_direct_fires = 0u;
  std::uint32_t rewrite_staged_fires = 0u;
  // How staged hypotheses END, which the fire counters above cannot say. A
  // partial has two independent deaths -- aging out under kPartialLifetime, or
  // the unconditional retirement in the staged sweep -- and without separating
  // them "no construction formed" cannot be distinguished from "the hypothesis
  // expired first". The two want opposite repairs.
  std::uint32_t rewrite_partials_aged_out = 0u;
  std::uint32_t rewrite_partials_retired_matched = 0u;
  std::uint32_t rewrite_partials_retired_unmatched = 0u;
  std::uint32_t rewrite_conflict_abstentions = 0u;
  std::uint32_t rewrite_constructor_rewrites = 0u;
  std::uint32_t rewrite_motor_value = 0u;
  std::uint32_t rewrite_motor_valid = 0u;
  // Passive cumulative count of generic resident prelinguistic motor
  // exploration actions originated by the generic resident fallback.
  std::uint64_t rewrite_motor_babble_actions = 0u;
  std::uint32_t rewrite_active_locus = 0xffffffffu;
  std::uint32_t rewrite_constructor_locus = 0xffffffffu;
  std::uint32_t rewrite_removed_matter_q8 = 0u;
  std::uint32_t rewrite_program_rules = 0u;
  std::uint32_t rewrite_mature_program_rules = 0u;
  std::uint32_t rewrite_trajectory_records = 0u;
  std::uint32_t rewrite_retained_exemplars = 0u;
  std::uint32_t rewrite_program_generated_events = 0u;
  std::uint32_t rewrite_program_conflict_abstentions = 0u;
  std::uint32_t rewrite_rejected_unbound_variables = 0u;
  std::uint32_t rewrite_completed_inductions = 0u;
  std::uint32_t rewrite_span_program_rules = 0u;
  std::uint32_t rewrite_mature_span_program_rules = 0u;
  std::uint32_t rewrite_span_generated_events = 0u;
  std::uint32_t rewrite_span_conflict_abstentions = 0u;
  std::uint32_t rewrite_span_rejected_unbound_variables = 0u;
  std::uint32_t rewrite_span_ambiguous_abstentions = 0u;
  std::uint32_t rewrite_span_completed_inductions = 0u;
  std::uint32_t rewrite_causal_relation_generated_events = 0u;
  std::uint32_t rewrite_causal_relation_probe_steps = 0u;
  std::uint32_t rewrite_causal_relation_participating_records = 0u;
  std::uint32_t rewrite_causal_relation_candidate_query_observed = 0u;
  std::uint32_t rewrite_causal_relation_candidate_query_owner = 0xffffffffu;
  std::uint32_t rewrite_causal_relation_candidate_query_revision = 0u;
  std::uint32_t rewrite_causal_relation_candidate_query_extent = 0u;
  std::uint32_t rewrite_causal_relation_candidate_readiness_stage = 0u;
  std::uint32_t rewrite_causal_relation_candidate_source_provenance_failure = 0u;
  std::uint32_t rewrite_causal_relation_candidate_source_provenance_owner = 0xffffffffu;
  std::uint32_t rewrite_causal_relation_candidate_applicable = 0u;
  std::uint32_t rewrite_causal_relation_candidate_ready = 0u;
  std::uint32_t rewrite_causal_relation_candidate_ambiguous = 0u;
  std::uint32_t rewrite_causal_relation_candidate_query_linked = 0u;
  std::uint32_t rewrite_causal_relation_candidate_trajectory_owner = 0xffffffffu;
  std::uint32_t rewrite_causal_relation_candidate_trajectory_revision = 0u;
  std::uint32_t rewrite_causal_relation_candidate_relation = 0u;
  std::uint32_t rewrite_causal_relation_candidate_extent = 0u;
  // Passive live-population receipt, distinct from the candidate-specific
  // participating_records field above.
  std::uint32_t rewrite_causal_relation_live_records = 0u;
  std::uint32_t rewrite_causal_relation_source_witness_records = 0u;
  std::uint32_t rewrite_causal_relation_source_witness_leaves = 0u;
  std::uint32_t rewrite_causal_relation_independent_sources = 0u;
  std::uint32_t rewrite_causal_relation_source_contributions = 0u;
  std::uint32_t rewrite_causal_relation_max_source_contribution = 0u;
  std::uint32_t rewrite_causal_relation_contribution_concentration_q16 = 0u;
  std::uint32_t rewrite_causal_relation_singleton_supported_steps = 0u;
  std::uint32_t rewrite_causal_relation_minimum_probe_support = 0u;
  std::uint64_t rewrite_causal_relation_component_digest = 0u;
  std::uint64_t rewrite_causal_relation_component_revision_digest = 0u;
  std::uint64_t rewrite_causal_relation_external_provenance_digest = 0u;
  std::uint32_t rewrite_causal_relation_external_leaves = 0u;
  // Exact permit carried by the distributed resident emission that reached
  // the common public writer. These are passive falsifiers of the public
  // boundary, not a selector or semantic answer representation.
  std::uint32_t rewrite_public_emission_receipt_valid = 0u;
  std::uint32_t rewrite_public_emission_owner = 0xffffffffu;
  std::uint32_t rewrite_public_emission_participant_records = 0u;
  std::uint32_t rewrite_public_emission_external_leaves = 0u;
  std::uint32_t rewrite_public_emission_independent_sources = 0u;
  std::uint32_t rewrite_public_emission_source_contributions = 0u;
  std::uint64_t rewrite_public_emission_topology_digest = 0u;
  std::uint64_t rewrite_public_emission_revision_digest = 0u;
  std::uint64_t rewrite_public_emission_provenance_digest = 0u;
  std::uint64_t rewrite_public_emission_participation_digest = 0u;
  std::uint64_t rewrite_public_emission_epoch = 0u;
  // Passive close scheduling telemetry. These fields expose progress for
  // diagnosis only; no host path may select, advance, or publish resident
  // work from them.
  std::uint32_t rewrite_close_work_pending = 0u;
  std::uint32_t rewrite_close_work_phase = 0u;
  // Passive RWR24 phase receipt. These counters summarize resident Record
  // matter only; they cannot select, advance, or publish an inquiry.
  std::uint32_t rewrite_open_inquiries = 0u;
  std::uint32_t rewrite_open_inquiry_constructors = 0u;
  std::uint32_t rewrite_open_inquiry_captured = 0u;
  std::uint32_t rewrite_open_inquiry_bound = 0u;
  std::uint32_t rewrite_open_inquiry_settled = 0u;
  std::uint32_t rewrite_open_inquiry_resumed = 0u;
  std::uint32_t rewrite_open_inquiry_prefix_current = 0u;
  std::uint32_t rewrite_open_inquiry_prefix_yielded = 0u;
  std::uint32_t rewrite_open_inquiry_prefix_flags = 0u;
  // RWR24 open-inquiry scheduling diagnostics (0X1-163/0X1-206). Cumulative
  // per-branch event counters for open_from_yielded_version_space, copied
  // verbatim from live resident state every call -- not a pool scan, since
  // a decline leaves no Record behind to scan for. Two real-time-stamped
  // samples can be diffed to see which branch (if any) fired between them,
  // distinguishing "scheduler never reached this call" from "reached it and
  // declined at precondition X". Passive; nothing reads these to select,
  // gate, or advance construction.
  std::uint32_t rewrite_open_inquiry_construction_attempts = 0u;
  std::uint32_t rewrite_open_inquiry_decline_active_inquiry = 0u;
  std::uint32_t rewrite_open_inquiry_decline_no_suspended_trajectory = 0u;
  std::uint32_t rewrite_open_inquiry_decline_not_wholly_external = 0u;
  std::uint32_t rewrite_open_inquiry_decline_not_yielded = 0u;
  std::uint32_t rewrite_open_inquiry_decline_already_open = 0u;
  std::uint32_t rewrite_open_inquiry_decline_fork_failed = 0u;
  std::uint32_t rewrite_open_inquiry_decline_multi_constructor = 0u;
  std::uint32_t rewrite_open_inquiry_decline_free_records = 0u;
  std::uint32_t rewrite_open_inquiry_decline_owner_failed = 0u;
  std::uint32_t rewrite_open_inquiry_construction_admitted = 0u;
  // RWR24 reply-continuation scheduling diagnostics (0X1-163/0X1-206,
  // continuing the diagnostics above): once open_from_yielded_version_space
  // has admitted a first OpenInquiry, wait_for_generated_word(a) is the next
  // organism-visible stall. These cumulative counters, copied verbatim from
  // live resident state every call, cover the five ordered stages between
  // that admission and the first reply-continuation word actually reaching
  // egress: capturing the teacher's surface, binding a fresh external reply,
  // settling it, reactivating the suspended trajectory, and emitting the
  // reply-continuation word itself. Passive; nothing reads these to select,
  // gate, or advance any of these functions.
  std::uint32_t rewrite_oi_capture_surface_attempts = 0u;
  // capture_teacher_surface_before_end's own combined-guard decline reasons,
  // one counter per precondition, same technique as the RWR24 open-inquiry
  // construction decline set above.
  // kInvalid on either side of the combined guard above conflates "no
  // matching candidate" with "more than one -- ambiguous"; these four
  // counters separate them, per the source-side comment.
  std::uint32_t rewrite_oi_capture_decline_no_active_inquiry = 0u;
  std::uint32_t rewrite_oi_capture_decline_ambiguous_inquiry = 0u;
  std::uint32_t rewrite_oi_capture_decline_no_current_trajectory = 0u;
  std::uint32_t rewrite_oi_capture_decline_ambiguous_trajectory = 0u;
  std::uint32_t rewrite_oi_capture_decline_already_progressed = 0u;
  std::uint32_t rewrite_oi_capture_decline_reserved_pending = 0u;
  std::uint32_t rewrite_oi_capture_decline_surface_owner_is_suspended = 0u;
  std::uint32_t rewrite_oi_capture_decline_not_wholly_external = 0u;
  std::uint32_t rewrite_oi_capture_decline_surface_zero_length = 0u;
  std::uint32_t rewrite_oi_capture_decline_surface_too_long = 0u;
  std::uint32_t rewrite_oi_capture_decline_insufficient_free_records = 0u;
  std::uint32_t rewrite_oi_capture_surface_admitted = 0u;
  std::uint32_t rewrite_oi_bind_reply_attempts = 0u;
  std::uint32_t rewrite_oi_bind_reply_admitted = 0u;
  std::uint32_t rewrite_oi_settle_reply_attempts = 0u;
  // settle_bound_reply's own combined-guard decline reasons (0X1-163/0X1-206,
  // this session), same one-counter-per-precondition technique as the
  // rewrite_oi_capture_decline_* set above. The first four mirror capture's
  // own no-candidate-vs-ambiguous reclassification over the same two helper
  // functions; the rest decompose settle's three further combined guards
  // (inquiry/reply shape, suspended-trajectory shape, post-loop
  // alternative/witness/emission consensus) plus its mid-loop
  // malformed-witness guard, in source clause order.
  std::uint32_t rewrite_oi_settle_decline_no_active_inquiry = 0u;
  std::uint32_t rewrite_oi_settle_decline_ambiguous_inquiry = 0u;
  std::uint32_t rewrite_oi_settle_decline_no_current_trajectory = 0u;
  std::uint32_t rewrite_oi_settle_decline_ambiguous_trajectory = 0u;
  std::uint32_t rewrite_oi_settle_decline_not_reply_bound = 0u;
  std::uint32_t rewrite_oi_settle_decline_already_settled = 0u;
  std::uint32_t rewrite_oi_settle_decline_not_awaiting_reply = 0u;
  std::uint32_t rewrite_oi_settle_decline_selected_owner_zero = 0u;
  std::uint32_t rewrite_oi_settle_decline_selected_owner_invalid = 0u;
  std::uint32_t rewrite_oi_settle_decline_selected_revision_zero = 0u;
  std::uint32_t rewrite_oi_settle_decline_selected_revision_invalid = 0u;
  std::uint32_t rewrite_oi_settle_decline_reply_matter_zero = 0u;
  std::uint32_t rewrite_oi_settle_decline_reply_not_trajectory_form = 0u;
  std::uint32_t rewrite_oi_settle_decline_reply_owner_zero = 0u;
  std::uint32_t rewrite_oi_settle_decline_reply_owner_invalid = 0u;
  std::uint32_t rewrite_oi_settle_decline_reply_length_zero = 0u;
  std::uint32_t rewrite_oi_settle_decline_reply_not_current = 0u;
  std::uint32_t rewrite_oi_settle_decline_no_suspended_trajectory = 0u;
  std::uint32_t rewrite_oi_settle_decline_suspended_lane3_zero = 0u;
  std::uint32_t rewrite_oi_settle_decline_suspended_owner_mismatch = 0u;
  std::uint32_t rewrite_oi_settle_decline_suspended_not_open_inquiry = 0u;
  std::uint32_t rewrite_oi_settle_decline_witness_owner_mismatch = 0u;
  std::uint32_t rewrite_oi_settle_decline_witness_selected_owner_mismatch =
      0u;
  std::uint32_t
      rewrite_oi_settle_decline_witness_selected_revision_mismatch = 0u;
  std::uint32_t rewrite_oi_settle_decline_witness_reply_revision_mismatch =
      0u;
  std::uint32_t rewrite_oi_settle_decline_witness_reply_length_mismatch =
      0u;
  std::uint32_t rewrite_oi_settle_decline_witness_reply_tail_mismatch = 0u;
  std::uint32_t rewrite_oi_settle_decline_witness_not_external = 0u;
  std::uint32_t rewrite_oi_settle_decline_witness_revision_not_one = 0u;
  std::uint32_t rewrite_oi_settle_decline_alternative_count = 0u;
  std::uint32_t rewrite_oi_settle_decline_selected_binding_count = 0u;
  std::uint32_t rewrite_oi_settle_decline_selected_consequence_invalid = 0u;
  std::uint32_t rewrite_oi_settle_decline_witness_count = 0u;
  std::uint32_t rewrite_oi_settle_decline_witness_consequence_mismatch = 0u;
  std::uint32_t rewrite_oi_settle_decline_emission_count = 0u;
  std::uint32_t rewrite_oi_settle_decline_selected_emission_count = 0u;
  std::uint32_t rewrite_oi_settle_reply_admitted = 0u;
  std::uint32_t rewrite_oi_reactivate_attempts = 0u;
  std::uint32_t rewrite_oi_reactivate_admitted = 0u;
  std::uint32_t rewrite_oi_reply_continuation_attempts = 0u;
  std::uint32_t rewrite_oi_reply_continuation_admitted = 0u;
  // advance_surface_once diagnostics (0X1-163/0X1-206, sibling session): the
  // surface-word emission path that runs after an already-admitted inquiry's
  // Emission record exists. Same passive cumulative-counter contract as the
  // blocks above.
  std::uint32_t rewrite_open_inquiry_surface_attempts = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_ambiguous_emission = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_no_emission = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_no_inquiry_header = 0u;
  std::uint32_t rewrite_open_inquiry_surface_reply_dispatch = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_bad_kind = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_constructor_stale = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_suspended_stale = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_snapshot_mismatch = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_exhausted = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_term_lookup_failed = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_alternative_path_failed = 0u;
  std::uint32_t rewrite_open_inquiry_surface_decline_append_failed = 0u;
  std::uint32_t rewrite_open_inquiry_surface_word_emitted = 0u;
  // settle_constructor_from_complete_episodes diagnostics (0X1-163, this
  // session). See the source-side comment on the matching field block in
  // bcc32_resident_open_inquiry_diagnostic_counters.inl for the full
  // rationale; these are the same fields, copied verbatim into the receipt
  // every call, same passive-instrumentation contract as every block above.
  std::uint32_t rewrite_oi_ctor_gate_eligible = 0u;
  std::uint32_t rewrite_oi_ctor_gate_resume_observed = 0u;
  std::uint32_t rewrite_oi_ctor_settle_attempts = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_already_authoritative = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_episode_overflow = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_insufficient_episodes = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_conflicting_template = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_no_template = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_insufficient_free_records = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_owner_failed = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_header_alloc_failed = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_term_alloc_failed = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_witness_alloc_failed = 0u;
  std::uint32_t rewrite_oi_ctor_settle_decline_final_check_failed = 0u;
  std::uint32_t rewrite_oi_ctor_settle_admitted = 0u;
  std::uint32_t rewrite_oi_ctor_settle_last_episode_count = 0u;
  // episode_complete_valid diagnostics (0X1-163, this session). See the
  // source-side comment on the matching field block in
  // bcc32_resident_open_inquiry_diagnostic_counters.inl for the full
  // rationale and the cumulative-across-call-sites caveat.
  std::uint32_t rewrite_oi_episode_complete_attempts = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_not_open_inquiry_form = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_flags_incomplete = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_lane2_invalid = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_lane3_zero = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_lane4_not_two = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_lane5_invalid = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_lane6_invalid = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_reserved1_not_invalid =
      0u;
  std::uint32_t rewrite_oi_episode_complete_decline_reserved0_zero = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_reserved0_too_long = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_surface_count_mismatch =
      0u;
  std::uint32_t
      rewrite_oi_episode_complete_decline_surface_source_owner_invalid = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_reply_source_owner_invalid =
      0u;
  std::uint32_t
      rewrite_oi_episode_complete_decline_surface_word_lookup_failed = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_alternative_not_grounded =
      0u;
  std::uint32_t
      rewrite_oi_episode_complete_decline_alternative_consensus_failed = 0u;
  std::uint32_t
      rewrite_oi_episode_complete_decline_alternatives_lookup_failed = 0u;
  std::uint32_t rewrite_oi_episode_complete_decline_reply_witness_malformed =
      0u;
  std::uint32_t rewrite_oi_episode_complete_decline_resume_witness_malformed =
      0u;
  std::uint32_t rewrite_oi_episode_complete_decline_witness_count_mismatch =
      0u;
  std::uint32_t rewrite_oi_episode_complete_valid_count = 0u;
  // inquiry_reply_source_owner diagnostics (0X1-163, this session). See the
  // source-side comment on the matching field block in
  // bcc32_resident_open_inquiry_diagnostic_counters.inl.
  std::uint32_t rewrite_oi_reply_source_owner_attempts = 0u;
  std::uint32_t rewrite_oi_reply_source_owner_decline_ambiguous = 0u;
  std::uint32_t rewrite_oi_reply_source_owner_decline_witness_owner_bad = 0u;
  std::uint32_t rewrite_oi_reply_source_owner_decline_witness_lane6_zero =
      0u;
  std::uint32_t
      rewrite_oi_reply_source_owner_decline_witness_not_external = 0u;
  std::uint32_t rewrite_oi_reply_source_owner_decline_reply_terms_invalid =
      0u;
  std::uint32_t rewrite_oi_reply_source_owner_decline_no_witness_found = 0u;
  std::uint32_t rewrite_oi_reply_source_owner_found = 0u;
  // reply_terms_valid diagnostics (0X1-163, this session). See the
  // source-side comment on the matching field block in
  // bcc32_resident_open_inquiry_diagnostic_counters.inl.
  std::uint32_t rewrite_oi_reply_terms_attempts = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_witness_guard = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_term_lookup_failed = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_term_lane4_owner_mismatch =
      0u;
  std::uint32_t
      rewrite_oi_reply_terms_decline_term_lane5_revision_mismatch = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_term_lane6_length_mismatch =
      0u;
  std::uint32_t rewrite_oi_reply_terms_decline_term_not_external = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_term_revision_not_one = 0u;
  std::uint32_t rewrite_oi_reply_terms_last_bad_term_revision = 0u;
  std::uint32_t rewrite_oi_reply_terms_last_bad_term_slot = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_term_count_mismatch = 0u;
  std::uint32_t rewrite_oi_reply_terms_fastpath_matched = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_program_ambiguous = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_program_not_found = 0u;
  std::uint32_t rewrite_oi_reply_terms_decline_program_exact_mismatch = 0u;
  std::uint32_t rewrite_oi_reply_terms_program_exact_matched = 0u;
  std::uint32_t rewrite_causal_germline_episodes = 0u;
  std::uint32_t rewrite_causal_germline_constructors = 0u;
  std::uint32_t rewrite_causal_germline_applications = 0u;
  std::uint32_t rewrite_causal_germline_reconstructions = 0u;
  // Counterevidence is a live count; suppression fields are cumulative events.
  std::uint32_t rewrite_causal_germline_counterevidence = 0u;
  std::uint32_t rewrite_causal_germline_product_suppressions = 0u;
  std::uint32_t rewrite_causal_germline_constructor_suppressions = 0u;
  std::uint32_t rewrite_causal_germline_conflict_abstentions = 0u;
  std::uint32_t rewrite_causal_germline_constructor_locus = 0xffffffffu;
  std::uint32_t rewrite_causal_germline_product_locus = 0xffffffffu;
  std::uint64_t rewrite_organization_digest = 0u;

  // Constitutional ordinary-F ownership remains transitional for actions:
  // f_world is published as a physical observation, while the legacy scalar
  // action writer stays explicit until raw-boundary-to-F causality closes.
  ContentAddress genesis_manifest{};
  ContentAddress f_world{};
  std::uint64_t completed_f_ticks = 0u;
  std::uint64_t f_generation = 0u;
  std::uint32_t f_fault = 0u;
  // Device-published ordinary-F support population at the last publication.
  // This is an observer-only capacity diagnostic; it is not a host-selected
  // route or an authority signal.
  std::uint32_t f_active_count = 0u;
  std::uint32_t f_owned_clock = 0u;
  // Observer-only CUDA graph chronology for the resident ordinary-F leaf.
  // These fields never authorize output or participate in resident choice.
  std::uint32_t f_continuation_phase = 0u;
  std::uint32_t f_continuation_status = 0u;
  std::uint32_t legacy_action_authority = 0u;
  // Passive raw outward matter after the committed ordinary-F tick. These
  // are physical rails, not decoded actions or host-selected targets.
  std::uint32_t f_motor_zero = 0u;
  std::uint32_t f_motor_one = 0u;

  // Resident-owned action-to-return accounting.  The digest commits the raw
  // returned packet before it enters rewrite learning; it is not a host label
  // or a claim that a physical exterior was authenticated.
  std::uint64_t action_return_issued = 0u;
  std::uint64_t action_return_accepted = 0u;
  std::uint64_t action_return_rejected = 0u;
  std::uint64_t action_return_action_sequence = 0u;
  std::uint64_t action_return_pending_action_sequence = 0u;
  std::uint64_t action_return_contact_sequence = 0u;
  // The exact raw-word count committed by the latest completed action return.
  // It is zero until a terminal return chunk has been accepted.
  std::uint64_t action_return_contact_words = 0u;
  // Passive progress for a ticket whose transport spans ingress slots or
  // continuation pages. These fields are transport receipts only; they are
  // never a semantic context window or permission to truncate contact.
  std::uint64_t action_return_stream_words = 0u;
  std::uint64_t action_return_stream_next_chunk = 1u;
  std::uint32_t action_return_stream_active = 0u;
  std::uint32_t action_return_pending = 0u;
  // Set only in the device epoch that consumes a valid resident ticket and
  // opaque return bytes.  It is a transport event, not proof of an external
  // body consequence or source identity.
  std::uint32_t action_return_ticketed_external_return = 0u;
  // Observer-only accounting for constraint matter rederived inside the
  // already accepted action-return staging transaction. These measurements
  // do not participate in ticket validation or public-output selection.
  std::uint64_t action_return_constraint_reafferent_attempted = 0u;
  std::uint64_t action_return_constraint_reafferent_accepted = 0u;
  std::uint64_t action_return_constraint_reafferent_rejected = 0u;
  std::uint64_t action_return_constraint_countered_records = 0u;
  std::uint64_t action_return_constraint_admitted_records = 0u;
  std::uint64_t action_return_constraint_resident_revision = 0u;
  // Latest post-contact component observation. Unique/cut-closed resident
  // structure decides readiness; these scalars are passive copies only.
  std::uint32_t action_return_constraint_component_ready = 0u;
  std::uint32_t action_return_constraint_component_ambiguous = 0u;
  std::uint32_t action_return_constraint_component_records = 0u;
  std::uint32_t action_return_constraint_component_sources = 0u;
  std::uint32_t action_return_constraint_rederived_event = 0u;
  // Passive accounting for distributed constraint-participation formation
  // staged at an ordinary raw-contact physical END. Never gates ingress,
  // chooses content, or authorizes an outward action.
  std::uint64_t rewrite_participation_end_attempted = 0u;
  std::uint64_t rewrite_participation_end_admitted = 0u;
  std::uint64_t rewrite_participation_end_rejected = 0u;
  // Passive latest-ingress-END lifecycle receipt. These expose only counts of
  // already-resident physical participation records at close boundaries; they
  // neither identify a relation nor authorize generation.
  std::uint32_t rewrite_participation_end_materialized_records = 0u;
  std::uint32_t rewrite_participation_end_precommit_records = 0u;
  std::uint32_t rewrite_participation_end_committed_records = 0u;
  // UNIMPLEMENTED PLACEHOLDERS -- these three rails are permanently zero. They
  // were introduced already-hardcoded by 4b1678ccb1, whose own receipt declared
  // `no_physical_body_source_reafference_proof`, and no writer has ever computed
  // them since; the only assignment in the runtime is the literal `0u` in
  // bcc32_resident_rewrite_runtime_egress.inl.
  //
  // Consequence for test authors: an `== 0u` assertion on any of these CANNOT
  // FAIL and is not a control. As of 2026-08-16 there are 26 such conjuncts
  // across 5 production contracts (mixed_provenance, egress_history,
  // ticketed_discourse_revision, motor_babble_reafference, and the organismd
  // dump), several carrying messages that read as real guards -- e.g.
  // "simulated body return falsely raised external proof rails". They are
  // pre-registered guards awaiting an implementation, not evidence that
  // physical-consequence, body-reafference, or source-identity provenance is
  // currently verified anywhere. Do not cite them as coverage, and do not add
  // more of them; when a real writer lands, these assertions become live and
  // this comment must go.
  std::uint32_t action_return_physical_consequence_proven = 0u;
  std::uint32_t action_return_body_reafference_proven = 0u;
  std::uint32_t action_return_source_identity_proven = 0u;
  // Device-body route fields prove only that a device-resident simulated body
  // produced this raw return after a resident action. They are deliberately
  // separate from the three physical proof rails above.
  std::uint32_t action_return_device_body_enabled = 0u;
  std::uint32_t action_return_device_body_closed_loop = 0u;
  std::uint64_t action_return_device_body_producer_instance = 0u;
  std::uint64_t action_return_device_body_source_epoch = 0u;
  std::uint64_t action_return_device_body_route_sequence = 0u;
  // Device-local simulated-body evidence. The state is advanced from the
  // resident action before its raw sensor return is published; it is not a
  // physical-source or external-body proof rail.
  std::uint64_t action_return_device_body_state = 0u;
  std::uint64_t action_return_device_body_transition_count = 0u;
  std::uint32_t action_return_device_body_consequence_word = 0u;
  std::uint32_t action_return_world_cell_slot = 0xffffffffu;
  std::uint32_t action_return_world_claim_slot = 0xffffffffu;
  std::uint32_t action_return_world_write_count = 0u;
  ContentAddress action_return_contact{};

  // Predictive-shadow receipt counters (0X1-267). Observer instrumentation
  // only, copied from the canonical `predictive_shadow` field's own
  // PredictiveShadowReceipt -- never resident selection authority. Exposed
  // so a host-side falsifier can assert these are unchanged across a
  // rejected/faulted action-return transaction and moved by an accepted
  // one, proving the staged-settlement fix (requirement 4) from outside the
  // process rather than only from code structure.
  std::uint32_t predictive_shadow_external_matches = 0u;
  std::uint32_t predictive_shadow_external_violations = 0u;
  std::uint32_t predictive_shadow_omissions = 0u;
  std::uint32_t predictive_shadow_relations_formed = 0u;
  // Raw per-observe_contact counter, unconditionally incremented for every
  // external contact regardless of match/violation/omission outcome or
  // relation-capacity abstention. Unlike predictive_shadow_omissions (which
  // advance_matter's unconditional per-epoch sweep can also move,
  // independent of any action-return activity), this field is only touched
  // inside observe_contact, making it the most reliable falsifier signal for
  // a rejected mid-stream action-return chunk that already ran
  // apply_action_return_chunk before the reject fired.
  std::uint32_t predictive_shadow_external_contacts = 0u;

  // 0X1-267 requirement 5, step 1 (Linear 0X1-267 comment c98064fd). Purely
  // diagnostic, non-gating measurement of whether this epoch's already-
  // computed predictive-shadow route projection physically intersects the
  // resident morphology the relation-candidate reader
  // (run_autonomous_generation_device) already judged eligible this epoch.
  // Copied unchanged from DeviceState's predictive_shadow_route_probe_*
  // scalars, which are themselves computed strictly after that reader's
  // dispatch cascade completes each epoch. Never read by
  // run_autonomous_generation_device's dispatch cascade or by any other
  // selection/route-choice path in the resident runtime -- promoting this
  // probe into real gating authority is a separate, explicitly deferred
  // follow-up.
  std::uint32_t predictive_shadow_route_probe_eligible_morphology = 0u;
  std::uint32_t predictive_shadow_route_probe_intersection = 0u;
  std::uint32_t predictive_shadow_route_probe_intersection_popcount = 0u;

  // Versioned, fail-closed projection of the resident logical Record
  // organization. This is derived by the device writer from the canonical
  // world and is deliberately independent of host packet/contact counters.
  // It is not a process checkpoint; 0X1-178 owns full restart state.
  std::uint32_t rewrite_world_lineage_version = 0u;
  std::uint32_t rewrite_world_lineage_valid = 0u;
  std::uint64_t rewrite_world_lineage_revision = 0u;
  std::uint64_t rewrite_world_lineage_organization_digest = 0u;
  std::uint64_t rewrite_world_lineage_admitted_events = 0u;

  // Identity of the unique live resident OpenInquiry, when its exact
  // suspended fork can still be rederived from authoritative resident matter.
  // owner is the resident-created inquiry owner; identity is the existing
  // unresolved-fork identity; generation is the live inquiry Record revision.
  std::uint32_t rewrite_open_inquiry_identity_version = 0u;
  std::uint32_t rewrite_open_inquiry_identity_valid = 0u;
  std::uint32_t rewrite_open_inquiry_owner = 0xffffffffu;
  std::uint32_t rewrite_open_inquiry_identity = 0u;
  std::uint32_t rewrite_open_inquiry_generation = 0u;

  // Resident version-space counts explain why an open inquiry was or was not
  // available without exposing a host-selected program or semantic label.
  std::uint32_t rewrite_version_space_factors = 0u;
  std::uint32_t rewrite_version_space_alternatives = 0u;
  std::uint32_t rewrite_mature_version_space_alternatives = 0u;
  std::uint32_t rewrite_version_space_witnesses = 0u;
  std::uint32_t rewrite_version_space_conflict_abstentions = 0u;
  friend bool operator==(const TickReceipt&, const TickReceipt&) = default;
};

struct PassiveSnapshot {
  std::vector<BoundaryWord> actions;
  // Device-generated raw motor form. The host receives bytes and length only;
  // it cannot select a trajectory, cursor, token, or internal population.
  std::vector<std::uint8_t> language_bytes;
  TickReceipt receipt{};
  std::uint64_t energy = 0u;
  // Host graph launches. Exactly one is allowed; epochs are device-owned graph
  // tail continuations and are represented by receipt.tick/device_epochs.
  std::uint64_t host_bootstrap_launches = 0u;
  std::uint64_t device_epochs = 0u;
  std::uint32_t continuation_fault = 0u;
  std::uint64_t language_generation = 0u;
  std::uint64_t language_contacts = 0u;
  std::uint64_t language_recruited_cells = 0u;
  std::uint64_t language_reused_cells = 0u;
  std::uint64_t language_strengthened_edges = 0u;
  // GitHub #1351. The hand-coded recurrent-language/GRU child that used to
  // publish this field was deleted from bcc32_persistent_kernel.cu after
  // #1208 rung 6 translated its surviving invariants onto the Direct lane.
  // Nothing writes it any more, and that is the point: it is the tripwire
  // bcc32_cuda_resident_rewrite_production_contract reads to assert the
  // canonical runtime carries no legacy cognitive child. Reintroducing a
  // publisher for it turns that contract RED, which is the only reason the
  // field survives its producer.
  std::uint64_t recurrent_language_tape_bytes = 0u;
  // Passive copy of the complete retained raw-word event ring. The metadata
  // describes the same publication and is never an acknowledgement cursor.
  std::vector<std::uint8_t> egress_history_bytes;
  std::uint64_t egress_history_next_sequence = 0u;
  std::uint64_t egress_history_oldest_sequence = 0u;
  std::uint64_t egress_history_overwrite_count = 0u;
  std::uint32_t egress_history_fault = 0u;
  // Passive copy of the sole currently returnable action.  Reading it does not
  // acknowledge, consume, or otherwise alter resident state.
  ActionReturnTicket action_return_ticket{};
};

// Only raw contact, passive boundary reads, and lifecycle shutdown are exposed.
// In particular there is no write_cell, set_weight, set_route, set_memory, or
// semantic selector.  State and morphology remain private to the writer.
class PersistentKernel final {
 public:
  // A raw contact is streamed as fixed-size physical packets when it exceeds
  // this aperture.  The aperture is transport-only: it never authorizes
  // semantic chunking, truncation, or a separate training route.
  static constexpr std::size_t kMaximumRawContactWords = 256u;

  // Canonical organismd construction: the disclosed hash-grown resident world
  // owns the one clock. No founder, cell-count, or language feature selector
  // is accepted.
  PersistentKernel();

  // Observer configuration is fixed at boot and never enters resident matter.
  // Allowing it to change mid-life would make instrumentation topology
  // mutable during an adult's causal history, weakening both the zero-cost
  // and no-resident-effect proofs below. RUNG 1 accepts only off and l0.
  explicit PersistentKernel(BitBusCircuitLevel bitbus_circuit_level);

  // Explicit developmental-demo construction. The ordinary-F timeline is
  // still resident-owned and device-advanced; the flag only selects whether
  // this bounded execution assay attaches that already-disclosed adult
  // aperture. Canonical organismd construction above remains unchanged.
  PersistentKernel(BitBusCircuitLevel bitbus_circuit_level,
                   bool enable_ordinary_f);

  // Legacy focused-contract constructor. It is defined only by the donor
  // aperture and is absent from the canonical organismd runtime library.
  explicit PersistentKernel(std::uint64_t founder,
                            std::size_t cell_count = 64u);
  ~PersistentKernel();

  PersistentKernel(const PersistentKernel&) = delete;
  PersistentKernel& operator=(const PersistentKernel&) = delete;
  PersistentKernel(PersistentKernel&&) = delete;
  PersistentKernel& operator=(PersistentKernel&&) = delete;

  // The only inbound mutation: unlabelled boundary words.  Oversized contacts
  // are emitted in consecutive fixed-size raw packets without dropping or
  // semantically dividing their words. The caller never receives a state view.
  void present_raw(std::span<const BoundaryWord> contact);

  // Return opaque raw contact for a device-issued action ticket.  The ticket
  // is checked and consumed only by the resident writer; this call does not
  // report acceptance because host-side enqueueing is not causal evidence.
  void present_action_return(ActionReturnTicket ticket,
                             std::span<const BoundaryWord> contact);

  // Submit one ordered part of an opaque raw return.  Chunk ordinals begin at
  // one for each ticket and are checked by the resident writer.  Only the
  // terminal chunk consumes the ticket; nonterminal chunks remain buffered
  // transport and cannot alter resident learning or report acceptance.
  void present_action_return_chunk(ActionReturnTicket ticket,
                                   std::uint64_t chunk_sequence,
                                   std::span<const BoundaryWord> contact,
                                   bool final_chunk);

  // Attach a declared simulated body whose producer and raw action-to-sensor
  // transition run inside the existing device graph. Once attached, the host
  // return API is closed so it cannot become a second return writer.
  void attach_device_body(DeviceBodyAttachment attachment);

  // Configure only the opaque device-local actuator-to-sensor body mapping.
  // This changes the body consequence for later resident actions; it never
  // selects an action, writes a target, or raises a physical proof rail.
  void configure_device_body_actuator_permutation(std::uint32_t permutation);

  // Deliver a raw physical boundary event.  This is intentionally separate
  // from sensory ingress and contains no semantic selector.
  void present_physical(const RawPhysicalIntervention& event);

  // Request lifecycle shutdown. This does not write or alter resident
  // cognitive state; it only joins the device-owned graph tail execution.
  void shutdown() noexcept;

  // Passive snapshots are copied from the latest device publication.  They are
  // not mutable aliases to internal state and never schedule device work.
  [[nodiscard]] PassiveSnapshot read_snapshot() const;
  [[nodiscard]] std::vector<BoundaryWord> read_actions() const;
  [[nodiscard]] std::vector<std::uint8_t> read_language_bytes() const;
  [[nodiscard]] TickReceipt read_receipt() const;

  // Passive copy of the observer ring. Reading this does not acknowledge,
  // consume, repair, score, select, or mutate resident state.
  [[nodiscard]] BitBusCircuitSnapshot read_bitbus_circuit() const;

  [[nodiscard]] std::size_t cell_count() const noexcept { return cell_count_; }

 private:
  PersistentKernel(std::uint64_t founder, std::size_t cell_count,
                   bool f_owned_clock);

  void* device_state_ = nullptr;
  void* ingress_ = nullptr;
  void* action_return_ = nullptr;
  void* device_body_ = nullptr;
  void* physical_ = nullptr;
  void* egress_ = nullptr;
  void* lifecycle_ = nullptr;
  void* bitbus_circuit_ = nullptr;
  void* resident_census_ = nullptr;
  BitBusCircuitLevel bitbus_circuit_level_ = BitBusCircuitLevel::off;
  std::uint64_t founder_ = 0u;
  ContentAddress sealed_execution_{};
  ContentAddress law_{};
  ContentAddress image_{};
  ContentAddress genesis_manifest_{};
  std::size_t cell_count_ = 0u;
  void* private_stream_ = nullptr;
  void* graph_ = nullptr;
  void* graph_exec_ = nullptr;
  void* cleanup_graph_ = nullptr;
  void* cleanup_graph_exec_ = nullptr;
  void* post_graph_ = nullptr;
  void* post_graph_exec_ = nullptr;
  void* grown_adult_ = nullptr;
  void* ordinary_f_timeline_ = nullptr;
  std::uint64_t next_contact_sequence_ = 1u;
  std::uint64_t next_action_return_sequence_ = 1u;
  std::uint64_t next_intervention_sequence_ = 1u;
  bool shutdown_requested_ = false;
  bool shutdown_complete_ = false;
  bool f_owned_clock_ = false;
  mutable std::vector<std::uint8_t> egress_history_cache_{};
  mutable std::uint64_t egress_history_cache_next_sequence_ = 0u;
  mutable std::uint64_t egress_history_cache_oldest_sequence_ = 0u;
  mutable std::uint64_t egress_history_cache_overwrite_count_ = 0u;
  mutable std::uint32_t egress_history_cache_fault_ = 0u;
};

}  // namespace substrate::bcc32::persistent_kernel

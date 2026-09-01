#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MISMATCH_OMISSION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MISMATCH_OMISSION_CUH

inline constexpr std::uint32_t kResidentMismatchExpectationCapacity =
    kResidentActualFrontierCapacity;
inline constexpr std::uint32_t kResidentMismatchReceiptCapacity =
    2u * kResidentMismatchExpectationCapacity;

enum class ResidentMismatchExpectationState : std::uint32_t {
  free = 0u, pending = 1u, matched = 2u, omitted = 3u,
};
enum class ResidentMismatchKind : std::uint32_t {
  unexpected_presence = 1u, omitted_expected_consequence = 2u,
};

struct alignas(8) ResidentMismatchExpectation {
  std::uint64_t identity, parent_occurrence_identity;
  std::uint64_t parent_participation_identity, logical_recipe_id;
  std::uint64_t revision_identity, current_revision_identity;
  std::uint64_t source_identity, route_incarnation;
  std::uint64_t binding_signature, matched_actual_occurrence_identity;
  std::uint32_t source_incarnation, context_signature, route_index;
  std::uint32_t recipe_cell, derivation_index;
  std::uint32_t generation_tick, horizon_tick, prediction_count;
  std::int32_t predicted_outputs_q16[kResidentPredictionHorizonCount];
  ResidentMismatchExpectationState state;
};
static_assert(std::is_standard_layout_v<ResidentMismatchExpectation> &&
              std::is_trivial_v<ResidentMismatchExpectation> &&
              std::has_unique_object_representations_v<ResidentMismatchExpectation>);

struct alignas(8) ResidentMismatchCreditReceipt {
  std::uint64_t identity, expectation_identity, actual_occurrence_identity;
  std::uint64_t target_occurrence_identity, target_participation_identity;
  std::uint64_t target_logical_recipe_id, target_revision_identity;
  std::uint64_t target_source_identity, target_route_incarnation;
  std::uint64_t committed_revision_identity;
  std::uint32_t target_source_incarnation, target_context_signature;
  std::uint32_t target_route_index, resident_tick, horizon_tick;
  std::uint32_t target_recipe_cell, target_derivation_index;
  std::int32_t causal_credit_delta_q16;
  ResidentMismatchKind kind;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<ResidentMismatchCreditReceipt> &&
              std::is_trivial_v<ResidentMismatchCreditReceipt> &&
              std::has_unique_object_representations_v<ResidentMismatchCreditReceipt>);

struct alignas(8) ResidentMismatchOmissionFrontier {
  ResidentMismatchExpectation expectations[kResidentMismatchExpectationCapacity];
  ResidentMismatchCreditReceipt receipts[kResidentMismatchReceiptCapacity];
  std::uint32_t pending_count, receipt_count, matches, unexpected_presences;
  std::uint32_t omissions, refusals, total_work_units, committed_receipt_count;
};
static_assert(std::is_standard_layout_v<ResidentMismatchOmissionFrontier> &&
              std::is_trivial_v<ResidentMismatchOmissionFrontier> &&
              std::has_unique_object_representations_v<ResidentMismatchOmissionFrontier>);

DIRECT_ADULT_HD inline std::uint64_t resident_mismatch_fold(
    std::uint64_t value, std::uint64_t word) {
  value ^= word + 0x9e3779b97f4a7c15ull + (value << 6u) + (value >> 2u);
  return value == 0u ? 1u : value;
}

DIRECT_ADULT_HD inline std::uint64_t resident_occurrence_binding_signature(
    const ResidentRecipeOccurrence& occurrence) {
  if (occurrence.binding_count == 0u ||
      occurrence.binding_count > direct_network::kResidentDerivationWidth)
    return 0u;
  std::uint64_t identity = 0x6d69736d61746368ull;
  for (std::uint32_t i = 0u; i < occurrence.binding_count; ++i) {
    if (occurrence.bindings[i].formal_port_index != i ||
        occurrence.bindings[i].variable_identity == 0u)
      return 0u;
    identity = resident_mismatch_fold(identity,
                                      occurrence.bindings[i].variable_identity);
  }
  return resident_mismatch_fold(identity, occurrence.binding_count);
}

DIRECT_ADULT_HD inline bool resident_mismatch_actual_is_authorized(
    const ResidentActualFrontierEntry& actual, std::uint32_t resident_tick) {
  const auto& occurrence = actual.occurrence;
  return actual.state == ResidentActualFrontierState::live &&
         occurrence.state == kResidentRecipeOccurrenceLive &&
         occurrence.lineage_kind == ResidentOccurrenceLineageKind::actual &&
         occurrence.authority == DirectParticipationAuthority::independent_external &&
         occurrence.occurrence_identity != 0u &&
         occurrence.participation_identity != 0u &&
         occurrence.logical_recipe_id != 0u && occurrence.revision_identity != 0u &&
         occurrence.source_identity != 0u && occurrence.source_incarnation != 0u &&
         occurrence.route_incarnation != 0u &&
         actual.route_incarnation == occurrence.route_incarnation &&
         actual.work.route_count != 0u &&
         actual.work.route_indices[0] != direct_network::kInvalidIndex &&
         occurrence.timestamp == resident_tick &&
         resident_occurrence_binding_signature(occurrence) != 0u;
}

DIRECT_ADULT_HD inline std::int32_t resident_mismatch_nonzero_delta(
    std::int64_t residual) {
  if (residual > INT32_MAX) return INT32_MAX;
  if (residual < INT32_MIN) return INT32_MIN;
  if (residual == 0) return 1;
  return static_cast<std::int32_t>(residual);
}

// static: keeps this out of cross-TU nvlink merging; under -rdc=true its
// emitted outlines carry TU-local $N ordinals that collide across objects
// ("Size doesn't match"). Same discipline as the bcc32 HD headers.
[[maybe_unused]] static DIRECT_ADULT_HD inline __noinline__ bool
reclaim_quiescent_resident_mismatch_frontier(
    ResidentMismatchOmissionFrontier* frontier) {
  if (frontier == nullptr || frontier->pending_count != 0u ||
      frontier->receipt_count == 0u ||
      frontier->committed_receipt_count != frontier->receipt_count)
    return false;
  for (std::uint32_t i = 0u; i < kResidentMismatchExpectationCapacity; ++i)
    frontier->expectations[i] = {};
  for (std::uint32_t i = 0u; i < kResidentMismatchReceiptCapacity; ++i)
    frontier->receipts[i] = {};
  frontier->receipt_count = 0u;
  frontier->committed_receipt_count = 0u;
  return true;
}

// Import one complete resident-generated horizon bank. The bank is only a
// pending expectation; it owns no participation or credit authority.
DIRECT_ADULT_HD inline bool refresh_resident_mismatch_expectations(
    const DirectBrain& brain, const ResidentActualFrontier& actual_frontier,
    const ResidentMultiHorizonPredictionFrontier& predictions,
    ResidentMismatchOmissionFrontier* frontier) {
  if (frontier == nullptr || brain.development == nullptr ||
      brain.route_incarnations == nullptr ||
      frontier->pending_count > kResidentMismatchExpectationCapacity ||
      frontier->receipt_count > kResidentMismatchReceiptCapacity ||
      frontier->committed_receipt_count > frontier->receipt_count) {
    if (frontier != nullptr) ++frontier->refusals;
    return false;
  }
  reclaim_quiescent_resident_mismatch_frontier(frontier);
  ResidentMismatchOmissionFrontier candidate = *frontier;
  for (std::uint32_t slot = 0u; slot < kResidentMultiHorizonCapacity; ++slot) {
    if (predictions.entries[slot].state != ResidentMultiHorizonState::live)
      continue;
    const auto& first = predictions.entries[slot];
    const auto parent_slot = first.shadow.parent_frontier_slot;
    if (parent_slot >= kResidentActualFrontierCapacity) {
      ++frontier->refusals; return false;
    }
    const auto& parent = actual_frontier.entries[parent_slot];
    const auto& occurrence = parent.occurrence;
    if (brain.postbirth_constructor == nullptr ||
        brain.postbirth_derivations == nullptr ||
        parent.derivation_index >= brain.postbirth_constructor->derivation_count ||
        !resident_mismatch_actual_is_authorized(parent, occurrence.timestamp) ||
        first.shadow.parent_occurrence_identity != occurrence.occurrence_identity ||
        first.shadow.parent_revision_identity != occurrence.revision_identity ||
        first.shadow.parent_context_signature != occurrence.context_signature ||
        parent.work.route_count == 0u) {
      ++frontier->refusals; return false;
    }
    const auto& derivation = brain.postbirth_derivations[parent.derivation_index];
    if (derivation.recipe_cell >= brain.development->recipe_cell_count ||
        derivation.route_index >= brain.route_capacity ||
        derivation.logical_recipe_id != occurrence.logical_recipe_id ||
        derivation.revision_identity != occurrence.revision_identity ||
        brain.route_incarnations[derivation.route_index] !=
            occurrence.route_incarnation) {
      ++frontier->refusals; return false;
    }
    bool duplicate = false;
    for (std::uint32_t i = 0u; i < kResidentMismatchExpectationCapacity; ++i)
      duplicate |= candidate.expectations[i].state !=
                       ResidentMismatchExpectationState::free &&
                   candidate.expectations[i].parent_occurrence_identity ==
                       occurrence.occurrence_identity;
    if (duplicate) continue;
    std::int32_t outputs[kResidentPredictionHorizonCount]{};
    std::uint32_t horizons[kResidentPredictionHorizonCount]{};
    std::uint32_t seen[kResidentPredictionHorizonCount]{};
    std::uint32_t count = 0u;
    for (std::uint32_t prediction_slot = 0u;
         prediction_slot < kResidentMultiHorizonCapacity; ++prediction_slot) {
      const auto& entry = predictions.entries[prediction_slot];
      if (entry.state != ResidentMultiHorizonState::live ||
          entry.shadow.parent_occurrence_identity != occurrence.occurrence_identity)
        continue;
      const std::uint32_t ordinal = entry.horizon_ordinal;
      if (ordinal >= kResidentPredictionHorizonCount || seen[ordinal] != 0u ||
          entry.shadow.state != ResidentSuccessorShadowState::live ||
          entry.shadow.occurrence.lineage_kind !=
              ResidentOccurrenceLineageKind::endogenous ||
          entry.shadow.occurrence.authority != DirectParticipationAuthority::none ||
          entry.shadow.occurrence.eligibility_q16 != 0 ||
          entry.shadow.parent_revision_identity != occurrence.revision_identity ||
          entry.shadow.parent_context_signature != occurrence.context_signature) {
        ++frontier->refusals; return false;
      }
      outputs[ordinal] = entry.shadow.projected_state_q16;
      horizons[ordinal] = entry.shadow.horizon_tick;
      seen[ordinal] = 1u;
      ++count;
    }
    if (count != kResidentPredictionHorizonCount ||
        horizons[0] <= first.shadow.generation_tick ||
        horizons[1] <= horizons[0] || horizons[2] <= horizons[1]) {
      ++frontier->refusals; return false;
    }
    std::uint32_t free_slot = direct_network::kInvalidIndex;
    for (std::uint32_t i = 0u; i < kResidentMismatchExpectationCapacity; ++i)
      if (free_slot == direct_network::kInvalidIndex &&
          candidate.expectations[i].state !=
              ResidentMismatchExpectationState::pending)
        free_slot = i;
    if (free_slot == direct_network::kInvalidIndex) {
      ++frontier->refusals; return false;
    }
    ResidentMismatchExpectation expected{};
    expected.parent_occurrence_identity = occurrence.occurrence_identity;
    expected.parent_participation_identity = occurrence.participation_identity;
    expected.logical_recipe_id = occurrence.logical_recipe_id;
    expected.revision_identity = occurrence.revision_identity;
    expected.current_revision_identity = occurrence.revision_identity;
    expected.source_identity = occurrence.source_identity;
    expected.source_incarnation = occurrence.source_incarnation;
    expected.context_signature = occurrence.context_signature;
    expected.route_index = derivation.route_index;
    expected.route_incarnation = occurrence.route_incarnation;
    expected.recipe_cell = derivation.recipe_cell;
    expected.derivation_index = parent.derivation_index;
    expected.binding_signature = resident_occurrence_binding_signature(occurrence);
    expected.generation_tick = first.shadow.generation_tick;
    expected.horizon_tick = horizons[kResidentPredictionHorizonCount - 1u];
    expected.prediction_count = count;
    expected.state = ResidentMismatchExpectationState::pending;
    for (std::uint32_t i = 0u; i < count; ++i)
      expected.predicted_outputs_q16[i] = outputs[i];
    expected.identity = resident_mismatch_fold(
        resident_mismatch_fold(0x6578706563746174ull,
                               expected.parent_occurrence_identity),
        expected.horizon_tick);
    candidate.expectations[free_slot] = expected;
    ++candidate.pending_count;
    candidate.total_work_units += count;
  }
  *frontier = candidate;
  return true;
}

DIRECT_ADULT_HD inline bool reconcile_resident_actual_mismatch(
    const DirectBrain& brain, const ResidentActualFrontierEntry& actual,
    std::uint32_t resident_tick,
    ResidentMismatchOmissionFrontier* frontier) {
  if (frontier == nullptr ||
      frontier->pending_count > kResidentMismatchExpectationCapacity ||
      frontier->receipt_count > kResidentMismatchReceiptCapacity ||
      frontier->committed_receipt_count > frontier->receipt_count ||
      !resident_mismatch_actual_is_authorized(actual, resident_tick)) {
    if (frontier != nullptr) ++frontier->refusals;
    return false;
  }
  const auto& occurrence = actual.occurrence;
  if (brain.development == nullptr || brain.postbirth_constructor == nullptr ||
      brain.route_incarnations == nullptr ||
      brain.postbirth_derivations == nullptr ||
      actual.derivation_index >= brain.postbirth_constructor->derivation_count) {
    ++frontier->refusals; return false;
  }
  const auto& derivation = brain.postbirth_derivations[actual.derivation_index];
  if (derivation.recipe_cell >= brain.development->recipe_cell_count ||
      derivation.route_index >= brain.route_capacity ||
      derivation.logical_recipe_id != occurrence.logical_recipe_id ||
      derivation.revision_identity != occurrence.revision_identity ||
      brain.route_incarnations[derivation.route_index] !=
          occurrence.route_incarnation) {
    ++frontier->refusals; return false;
  }
  reclaim_quiescent_resident_mismatch_frontier(frontier);
  for (std::uint32_t i = 0u; i < kResidentMismatchExpectationCapacity; ++i)
    if (frontier->expectations[i].matched_actual_occurrence_identity ==
        occurrence.occurrence_identity) {
      ++frontier->refusals; return false;
    }
  for (std::uint32_t i = 0u; i < frontier->receipt_count; ++i)
    if (frontier->receipts[i].actual_occurrence_identity ==
        occurrence.occurrence_identity) {
      ++frontier->refusals; return false;
    }
  const std::uint64_t binding = resident_occurrence_binding_signature(occurrence);
  for (std::uint32_t i = 0u; i < kResidentMismatchExpectationCapacity; ++i) {
    auto& expected = frontier->expectations[i];
    if (expected.state != ResidentMismatchExpectationState::pending ||
        resident_tick < expected.generation_tick || resident_tick > expected.horizon_tick ||
        expected.logical_recipe_id != occurrence.logical_recipe_id ||
        expected.current_revision_identity != occurrence.revision_identity ||
        expected.source_identity != occurrence.source_identity ||
        expected.source_incarnation != occurrence.source_incarnation ||
        expected.context_signature != occurrence.context_signature ||
        expected.binding_signature != binding)
      continue;
    for (std::uint32_t prediction = 0u;
         prediction < expected.prediction_count; ++prediction)
      if (expected.predicted_outputs_q16[prediction] == actual.output_q16) {
        expected.state = ResidentMismatchExpectationState::matched;
        expected.matched_actual_occurrence_identity = occurrence.occurrence_identity;
        --frontier->pending_count; ++frontier->matches;
        return true;
      }
  }
  if (frontier->receipt_count >= kResidentMismatchReceiptCapacity) {
    ++frontier->refusals; return false;
  }
  ResidentMismatchCreditReceipt receipt{};
  receipt.actual_occurrence_identity = occurrence.occurrence_identity;
  receipt.target_occurrence_identity = occurrence.occurrence_identity;
  receipt.target_participation_identity = occurrence.participation_identity;
  receipt.target_logical_recipe_id = occurrence.logical_recipe_id;
  receipt.target_revision_identity = occurrence.revision_identity;
  receipt.target_source_identity = occurrence.source_identity;
  receipt.target_source_incarnation = occurrence.source_incarnation;
  receipt.target_context_signature = occurrence.context_signature;
  receipt.target_route_index = derivation.route_index;
  receipt.target_route_incarnation = occurrence.route_incarnation;
  receipt.target_recipe_cell = derivation.recipe_cell;
  receipt.target_derivation_index = actual.derivation_index;
  receipt.resident_tick = resident_tick;
  receipt.horizon_tick = occurrence.expiry_tick;
  receipt.causal_credit_delta_q16 = resident_mismatch_nonzero_delta(actual.output_q16);
  receipt.kind = ResidentMismatchKind::unexpected_presence;
  receipt.identity = resident_mismatch_fold(
      resident_mismatch_fold(0x756e657870656374ull,
                             occurrence.occurrence_identity), resident_tick);
  frontier->receipts[frontier->receipt_count++] = receipt;
  ++frontier->unexpected_presences;
  return true;
}

// Omission is not processor silence. A later authenticated independent-external
// contact from the same physical source/context must prove that the lived world
// advanced beyond the expectation horizon. Endogenous activity, descendants,
// stale context and resident ticks alone cannot mint absence evidence.
DIRECT_ADULT_HD inline const ResidentActualFrontierEntry*
resident_authenticated_omission_closure(
    const ResidentMismatchExpectation& expected,
    const ResidentActualFrontier* actual_frontier,
    std::uint32_t resident_tick) {
  if (actual_frontier == nullptr ||
      actual_frontier->live_count > kResidentActualFrontierCapacity)
    return nullptr;
  const ResidentActualFrontierEntry* found = nullptr;
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
    const auto& entry = actual_frontier->entries[i];
    const auto& occurrence = entry.occurrence;
    if (entry.state != ResidentActualFrontierState::live ||
        entry.contact_identity == 0u || entry.ingress_sequence == 0u ||
        occurrence.lineage_kind != ResidentOccurrenceLineageKind::actual ||
        occurrence.authority != DirectParticipationAuthority::independent_external ||
        occurrence.occurrence_identity == 0u ||
        occurrence.occurrence_identity == expected.parent_occurrence_identity ||
        occurrence.source_identity != expected.source_identity ||
        occurrence.context_signature != expected.context_signature ||
        occurrence.timestamp <= expected.horizon_tick ||
        occurrence.timestamp > resident_tick)
      continue;
    if (found != nullptr) return nullptr;
    found = &entry;
  }
  return found;
}

DIRECT_ADULT_HD inline std::uint32_t advance_resident_mismatch_omissions(
    std::uint32_t resident_tick, const ResidentActualFrontier* actual_frontier,
    ResidentMismatchOmissionFrontier* frontier) {
  if (frontier == nullptr) return 0u;
  if (frontier->pending_count > kResidentMismatchExpectationCapacity ||
      frontier->receipt_count > kResidentMismatchReceiptCapacity ||
      frontier->committed_receipt_count > frontier->receipt_count) {
    ++frontier->refusals;
    return 0u;
  }
  std::uint32_t emitted = 0u;
  for (std::uint32_t i = 0u; i < kResidentMismatchExpectationCapacity; ++i) {
    auto& expected = frontier->expectations[i];
    if (expected.state != ResidentMismatchExpectationState::pending ||
        resident_tick <= expected.horizon_tick)
      continue;
    const auto* closure = resident_authenticated_omission_closure(
        expected, actual_frontier, resident_tick);
    if (closure == nullptr) continue;
    if (frontier->receipt_count >= kResidentMismatchReceiptCapacity) {
      ++frontier->refusals; continue;
    }
    ResidentMismatchCreditReceipt receipt{};
    receipt.expectation_identity = expected.identity;
    receipt.actual_occurrence_identity = closure->occurrence.occurrence_identity;
    receipt.target_occurrence_identity = expected.parent_occurrence_identity;
    receipt.target_participation_identity = expected.parent_participation_identity;
    receipt.target_logical_recipe_id = expected.logical_recipe_id;
    receipt.target_revision_identity = expected.revision_identity;
    receipt.target_source_identity = expected.source_identity;
    receipt.target_source_incarnation = expected.source_incarnation;
    receipt.target_context_signature = expected.context_signature;
    receipt.target_route_index = expected.route_index;
    receipt.target_route_incarnation = expected.route_incarnation;
    receipt.target_recipe_cell = expected.recipe_cell;
    receipt.target_derivation_index = expected.derivation_index;
    receipt.resident_tick = resident_tick;
    receipt.horizon_tick = expected.horizon_tick;
    receipt.causal_credit_delta_q16 = resident_mismatch_nonzero_delta(
        -static_cast<std::int64_t>(expected.predicted_outputs_q16[
            expected.prediction_count - 1u]));
    receipt.kind = ResidentMismatchKind::omitted_expected_consequence;
    receipt.identity = resident_mismatch_fold(
        resident_mismatch_fold(0x6f6d697373696f6eull, expected.identity),
        closure->occurrence.occurrence_identity);
    frontier->receipts[frontier->receipt_count++] = receipt;
    expected.state = ResidentMismatchExpectationState::omitted;
    --frontier->pending_count; ++frontier->omissions; ++emitted;
  }
  return emitted;
}

DIRECT_ADULT_HD inline bool admit_and_reconcile_resident_contact(
    const DirectBrain& brain, const ActivityEvent& event,
    ResidentContactEpochReceipt* credential, std::uint64_t ingress_sequence,
    const NodeCausalParticipation* active_participation,
    std::uint32_t current_tick, std::uint32_t horizon_ticks,
    ResidentActualFrontier* actual_frontier,
    ResidentMismatchOmissionFrontier* mismatch_frontier,
    DirectActualFrontierCausalCounters* counters = nullptr) {
  if (!admit_resident_actual_frontier_contact(
          brain, event, credential, ingress_sequence, active_participation,
          current_tick, horizon_ticks, actual_frontier, counters))
    return false;
  if (mismatch_frontier != nullptr)
    for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
      const auto& actual = actual_frontier->entries[i];
      if (actual.state == ResidentActualFrontierState::live &&
          actual.contact_identity == credential->identity &&
          actual.ingress_sequence == ingress_sequence) {
        reconcile_resident_actual_mismatch(brain, actual, current_tick,
                                           mismatch_frontier);
        break;
      }
    }
  return true;
}

static __global__ void refresh_resident_predictions_and_expectations_kernel(
    DirectBrain brain, const ResidentActualFrontier* actual_frontier,
    std::uint32_t current_tick,
    ResidentMultiHorizonPredictionFrontier* predictions,
    ResidentMismatchOmissionFrontier* mismatch) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && actual_frontier != nullptr) {
    refresh_resident_causal_credit_predictions(
        brain, *actual_frontier, current_tick, predictions);
    if (predictions != nullptr)
      refresh_resident_mismatch_expectations(brain, *actual_frontier, *predictions,
                                             mismatch);
  }
}

#endif

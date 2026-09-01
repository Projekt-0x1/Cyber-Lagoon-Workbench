#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_construction_composer.cuh"
#include "bcc32_cuda_resident_population_surface.cuh"
#include "bcc32_cuda_resident_proposition_tissue.cuh"

namespace bcc32_cuda_resident_ordered_relation_assimilation {

namespace construction = substrate::bcc32::resident_construction;
namespace population_surface = bcc32_cuda_resident_population_surface;
namespace tissue = bcc32_cuda_resident_proposition_tissue;

constexpr std::uint32_t kMaximumEventsPerPosition =
    construction::kRelationTripleContentArgumentSpan;
constexpr std::uint32_t kEventTileWidth = 32u;
constexpr std::uint32_t kMinimumRecurrentObservations = 3u;
constexpr std::uint32_t kMaximumExecutionOutputUnits = 5u;
// Bound on the compact per-binding candidate lists execute_ordered_relation_
// from_exact_cue_kernel precomputes before its three/two/one-edge exhaustive
// scans (0X1-213). A dense, real resident store made those scans O(capacity^3)
// on a single CUDA thread; see the kernel body for the full explanation.
constexpr std::uint32_t kMaximumOrderedRelationCandidates = 4096u;

// The relation-triple store is a compact aggregate recurrence cache. It keeps
// an exact A/K/B key and total occurrence count, but deliberately does not
// claim episode provenance. Ordered tissue retains every complete physical
// episode; recurrence only changes how later contacts consolidate that record.
struct RelationObservationView {
  const construction::RelationTriple* triples = nullptr;
  const std::uint32_t* counts = nullptr;
  std::uint32_t minimum_recurrence = kMinimumRecurrentObservations;
};

// Which of build_event_populations' return-false routes fired. The aggregate
// invalid/incomplete_population counters cannot tell them apart: they are
// incremented together in one branch, so a whole corpus can be discarded
// without naming what actually failed. Diagnostic only -- no authorization
// predicate reads these, and the aggregates keep their exact prior meaning.
enum class PopulationFailure : std::uint32_t {
  none = 0u,
  missing_view = 1u,       // null sequence or null populations
  primary_unit = 2u,       // agent, connective, connective2 or value append
  context_unit = 3u,       // the one selected context unit's append
  context_distinct = 4u,   // valid_distinct_population over the context
};

struct AssimilationReceipt {
  std::uint64_t contact_revision = 0u;
  std::uint64_t events = 0u;
  std::uint64_t accepted = 0u;
  std::uint64_t replays = 0u;
  std::uint64_t invalid = 0u;
  std::uint64_t incomplete_population = 0u;
  std::uint64_t whole_contact_abstained = 0u;
  std::uint64_t event_overflow = 0u;
  std::uint64_t capacity_overflow = 0u;
  std::uint64_t insufficient_mass = 0u;
  std::uint64_t below_recurrence = 0u;
  std::uint64_t recurrent = 0u;
  std::uint64_t context_replays = 0u;
  std::uint64_t generalized = 0u;
  // Four-way split of the incomplete_population route. These sum to
  // incomplete_population exactly while :561's default-invalid route stays
  // silent; invalid > incomplete_population means :561 fired and the sum
  // no longer accounts for invalid.
  std::uint64_t population_missing_view = 0u;
  std::uint64_t population_primary_unit = 0u;
  std::uint64_t population_context_unit = 0u;
  std::uint64_t population_context_distinct = 0u;
};

// A held-out cue is ordinary resident unit matter in its physical order:
// [earlier role, connective, optional second connective].  Execution scans the
// resident ordered tissue itself and publishes the later-role unit only when
// exactly one intact discourse-qualified binding has that complete topology.
// No caller-supplied candidate, expected unit, binding index, or qualification
// mode participates in selection.
struct OrderedRelationExecutionReceipt {
  std::uint64_t tissue_revision = 0u;
  std::uint64_t source_evidence_revision = 0u;
  std::uint64_t composed_source_evidence_revision = 0u;
  std::uint64_t terminal_source_evidence_revision = 0u;
  // Provenance only: an admitted resident question return may name the exact
  // evidence revision it created. It never gates or ranks relation output.
  std::uint64_t ticketed_return_evidence_revision = 0u;
  std::uint64_t source_positive_mass = 0u;
  std::uint64_t source_counterevidence = 0u;
  std::uint32_t cue_units[3]{};
  std::uint32_t cue_unit_count = 0u;
  std::uint32_t topology_matches = 0u;
  std::uint32_t qualified_candidates = 0u;
  std::uint32_t withdrawn_candidates = 0u;
  std::uint32_t invalid_sources = 0u;
  std::uint32_t conflict = 0u;
  std::uint32_t capacity_overflow = 0u;
  std::uint32_t source_binding_index = 0xffffffffu;
  std::uint32_t composed_source_binding_index = 0xffffffffu;
  std::uint32_t terminal_source_binding_index = 0xffffffffu;
  std::uint32_t source_context_count = 0u;
  std::uint32_t output_unit_count = 0u;
  std::uint32_t composition_depth = 0u;
  std::uint32_t ticketed_return_source_mask = 0u;
  std::uint32_t clarification_opener = 0xffffffffu;
  std::uint32_t alternative_source_count = 0u;
  std::uint32_t alternative_source_binding_indices[kMaximumExecutionOutputUnits]{};
  std::uint32_t clarification_ready = 0u;
  std::uint32_t route_authorized = 0u;
  std::uint32_t public_source_validated = 0u;
  std::uint32_t ready = 0u;
  std::uint32_t abstained = 1u;
};

__device__ inline bool append_complete_unit_population(
    population_surface::UnitPopulationView units, std::uint32_t cell_capacity,
    std::uint32_t unit, std::uint32_t* cells, std::uint32_t* cell_count);

__device__ inline bool ordered_relation_exact_cue_role(
    const tissue::OrderedRoleBindingEvidence& binding, std::uint32_t role,
    const std::uint32_t* cue_exact, const std::uint32_t* cue_orders,
    std::uint32_t cue_unit_begin, std::uint32_t cue_unit_count,
    population_surface::UnitPopulationView units,
    std::uint32_t cell_capacity, std::uint32_t* first_order,
    std::uint32_t* last_order) {
  if (binding.claimed == 0u || role >= 2u || cue_exact == nullptr ||
      cue_orders == nullptr || first_order == nullptr || last_order == nullptr ||
      binding.role_unit_counts[role] == 0u ||
      binding.role_unit_counts[role] > tissue::kMaximumOrderedRoleUnits)
    return false;
  std::uint32_t cells[tissue::kMaximumPopulationCells]{};
  std::uint32_t cell_count = 0u;
  std::uint32_t previous_order = 0u;
  for (std::uint32_t index = 0u;
       index < binding.role_unit_counts[role]; ++index) {
    const std::uint32_t unit = binding.role_units[role][index];
    if (unit < cue_unit_begin || unit - cue_unit_begin >= cue_unit_count ||
        cue_exact[unit] == 0u || cue_orders[unit] == 0u ||
        (previous_order != 0u && cue_orders[unit] <= previous_order) ||
        !append_complete_unit_population(units, cell_capacity, unit, cells,
                                         &cell_count))
      return false;
    if (index == 0u)
      *first_order = cue_orders[unit];
    previous_order = cue_orders[unit];
  }
  *last_order = previous_order;
  return tissue::exact_population_equals(binding.role_cells[role],
                                         binding.role_counts[role],
                                         {cells, cell_count});
}

__device__ inline bool ordered_relation_patient_is_absent(
    const tissue::OrderedRoleBindingEvidence& binding,
    const std::uint32_t* cue_exact, std::uint32_t cue_unit_begin,
    std::uint32_t cue_unit_count) {
  for (std::uint32_t index = 0u; index < binding.role_unit_counts[2]; ++index) {
    const std::uint32_t unit = binding.role_units[2][index];
    if (unit >= cue_unit_begin && unit - cue_unit_begin < cue_unit_count &&
        cue_exact[unit] != 0u)
      return false;
  }
  return binding.role_unit_counts[2] != 0u;
}

__device__ inline bool ordered_relation_same_patient_agent(
    const tissue::OrderedRoleBindingEvidence& earlier,
    const tissue::OrderedRoleBindingEvidence& later) {
  if (earlier.role_unit_counts[2] != later.role_unit_counts[0])
    return false;
  for (std::uint32_t index = 0u; index < earlier.role_unit_counts[2]; ++index)
    if (earlier.role_units[2][index] != later.role_units[0][index])
      return false;
  return tissue::exact_population_equals(
      earlier.role_cells[2], earlier.role_counts[2],
      {later.role_cells[0], later.role_counts[0]});
}

__device__ inline bool ordered_relation_source_revision_is_current(
    tissue::TissueView resident_tissue, std::uint32_t binding_index,
    std::uint64_t evidence_revision) {
  return resident_tissue.ordered_bindings != nullptr &&
      binding_index < resident_tissue.ordered_binding_capacity &&
      evidence_revision != 0u &&
      resident_tissue.ordered_bindings[binding_index].claimed != 0u &&
      resident_tissue.ordered_bindings[binding_index].last_evidence_revision ==
          evidence_revision;
}

// A producer receipt is a transient authorization, never checkpointed
// authority.  Restart may restore the exact retained bindings into a new tissue
// revision; the old receipt must then be rejected and the path rederived.  A
// current answer additionally proves every physical edge still has the exact
// binding identity and evidence revision that produced it.
__device__ inline bool ordered_relation_execution_receipt_is_current(
    tissue::TissueView resident_tissue,
    const OrderedRelationExecutionReceipt& receipt) {
  if (resident_tissue.scalars == nullptr || receipt.tissue_revision == 0u ||
      resident_tissue.scalars->revision != receipt.tissue_revision)
    return false;
  if (receipt.ready == 0u)
    return receipt.clarification_ready != 0u && receipt.conflict != 0u;
  if (receipt.conflict != 0u || receipt.composition_depth == 0u ||
      receipt.composition_depth > 3u ||
      !ordered_relation_source_revision_is_current(
          resident_tissue, receipt.source_binding_index,
          receipt.source_evidence_revision))
    return false;
  if (receipt.composition_depth >= 2u &&
      !ordered_relation_source_revision_is_current(
          resident_tissue, receipt.composed_source_binding_index,
          receipt.composed_source_evidence_revision))
    return false;
  if (receipt.composition_depth == 3u &&
      !ordered_relation_source_revision_is_current(
          resident_tissue, receipt.terminal_source_binding_index,
          receipt.terminal_source_evidence_revision))
    return false;
  return (receipt.composition_depth >= 2u ||
          receipt.composed_source_binding_index == 0xffffffffu) &&
      (receipt.composition_depth == 3u ||
       receipt.terminal_source_binding_index == 0xffffffffu);
}

__device__ inline bool append_ordered_relation_alternatives(
    const tissue::OrderedRoleBindingEvidence& binding,
    std::uint32_t* output_units, std::uint32_t output_capacity,
    std::uint32_t* output_count) {
  for (std::uint32_t role_unit = 0u;
       role_unit < binding.role_unit_counts[2]; ++role_unit) {
    const std::uint32_t unit = binding.role_units[2][role_unit];
    bool present = false;
    for (std::uint32_t prior = 0u; prior < *output_count; ++prior)
      present = present || output_units[prior] == unit;
    if (present)
      continue;
    if (*output_count >= output_capacity)
      return false;
    std::uint32_t position = *output_count;
    while (position != 0u && output_units[position - 1u] > unit) {
      output_units[position] = output_units[position - 1u];
      --position;
    }
    output_units[position] = unit;
    ++*output_count;
  }
  return true;
}

// Join keys are only a conservative index. The exact population comparison
// remains in ordered_relation_same_patient_agent below, so hash collisions
// cannot authorize a route. Stable ordering within each equal-hash range
// preserves the original binding-index traversal and therefore preserves the
// exact ambiguity/conflict result while replacing the cubic candidate scan with
// indexed joins.
__device__ inline std::uint32_t ordered_relation_role_population_hash(
    const tissue::OrderedRoleBindingEvidence& binding, std::uint32_t role) {
  if (role >= tissue::kOrderedBindingRoleCount)
    return 0u;
  const tissue::SparsePopulationView population{
      binding.role_cells[role], binding.role_counts[role]};
  return tissue::ordered_population_hash(population);
}

__device__ inline std::uint32_t ordered_relation_join_lower_bound(
    const tissue::OrderedRoleBindingEvidence* bindings,
    const std::uint32_t* predicate_candidates,
    const std::uint32_t* sorted_candidate_slots, std::uint32_t count,
    std::uint32_t target_hash) {
  std::uint32_t first = 0u;
  std::uint32_t last = count;
  while (first < last) {
    const std::uint32_t middle = first + (last - first) / 2u;
    const std::uint32_t slot = sorted_candidate_slots[middle];
    const std::uint32_t candidate_index = predicate_candidates[slot];
    const std::uint32_t candidate_hash =
        ordered_relation_role_population_hash(bindings[candidate_index], 0u);
    if (candidate_hash < target_hash)
      first = middle + 1u;
    else
      last = middle;
  }
  return first;
}

__device__ inline std::uint32_t ordered_relation_join_upper_bound(
    const tissue::OrderedRoleBindingEvidence* bindings,
    const std::uint32_t* predicate_candidates,
    const std::uint32_t* sorted_candidate_slots, std::uint32_t count,
    std::uint32_t target_hash) {
  std::uint32_t first = 0u;
  std::uint32_t last = count;
  while (first < last) {
    const std::uint32_t middle = first + (last - first) / 2u;
    const std::uint32_t slot = sorted_candidate_slots[middle];
    const std::uint32_t candidate_index = predicate_candidates[slot];
    const std::uint32_t candidate_hash =
        ordered_relation_role_population_hash(bindings[candidate_index], 0u);
    if (candidate_hash <= target_hash)
      first = middle + 1u;
    else
      last = middle;
  }
  return first;
}

// Canonical adult adapter: select from the pre-contact exact-cue projection
// rather than a host-authored compact cue. Extra residently observed units may
// surround the ordered A/K cue, but the complete A then K topology must be
// present and the learned B must be absent. Exact resident question-onset
// evidence gates execution; no observer receipt, host route, candidate, or
// expected output participates.
__global__ void execute_ordered_relation_from_exact_cue_kernel(
    tissue::TissueView resident_tissue,
    population_surface::UnitPopulationView units,
    const std::uint32_t* cue_exact, const std::uint32_t* cue_orders,
    std::uint32_t cue_unit_begin, std::uint32_t cue_unit_count,
    const std::uint64_t* question_onset_evidence,
    const std::uint64_t* ticketed_return_evidence_revision,
    std::uint32_t* output_units,
    std::uint32_t output_capacity, std::uint32_t* output_count,
    OrderedRelationExecutionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || output_count == nullptr ||
      receipt == nullptr)
    return;
  *receipt = OrderedRelationExecutionReceipt{};
  *output_count = 0u;
  if (resident_tissue.scalars != nullptr)
    receipt->tissue_revision = resident_tissue.scalars->revision;
  receipt->ticketed_return_evidence_revision =
      ticketed_return_evidence_revision == nullptr
          ? 0u
          : ticketed_return_evidence_revision[0];
  if (cue_exact != nullptr && question_onset_evidence != nullptr)
    for (std::uint32_t offset = 0u; offset < cue_unit_count; ++offset) {
      const std::uint32_t unit = cue_unit_begin + offset;
      if (cue_exact[unit] != 0u && question_onset_evidence[unit] != 0u) {
        receipt->route_authorized = 1u;
        break;
      }
    }
  if (receipt->route_authorized == 0u ||
      resident_tissue.ordered_bindings == nullptr || cue_exact == nullptr ||
      cue_orders == nullptr || output_units == nullptr || cue_unit_count == 0u)
    return;

  std::uint32_t selected = 0xffffffffu;
  std::uint32_t selected_composed = 0xffffffffu;
  std::uint32_t selected_terminal = 0xffffffffu;
  std::uint32_t alternatives[kMaximumExecutionOutputUnits]{};
  std::uint32_t alternative_count = 0u;

  // Every scan below re-tests the SAME pairing-independent predicates
  // (ordered_relation_exact_cue_role for role 0 and role 1,
  // ordered_relation_patient_is_absent) against the full
  // ordered_binding_capacity at every nesting level. That made the
  // exhaustive ambiguity search O(capacity^3) in the worst case: a binding
  // store dense with real, cue-relevant matter -- exactly what accumulates
  // from real repeated contact -- turns this single CUDA thread into a
  // multi-billion-iteration scan (0X1-213: bcc32_cuda_adult_stream_contract's
  // birth corpus alone made this hang past a two-minute bound). Precompute
  // each binding's pairing-independent candidacy ONCE, in a single
  // O(capacity) pass -- these helper calls are pure and side-effect free, so
  // filtering first and re-running the identical calls only for surviving
  // candidates is exactly equivalent to the original scan-with-continue, not
  // an approximation -- then let every loop below iterate the resulting
  // compact candidate list instead of the raw capacity. Fixed-size, bounded:
  // if the resident store is dense enough to overflow the cap, this abstains
  // and reports capacity_overflow honestly rather than silently truncating
  // the ambiguity search, the same idiom this kernel already uses for the
  // alternatives list below.
  std::uint32_t predicate_candidates[kMaximumOrderedRelationCandidates];
  std::uint32_t predicate_candidate_count = 0u;
  std::uint32_t agent_candidates[kMaximumOrderedRelationCandidates];
  std::uint32_t agent_candidate_count = 0u;
  bool candidate_overflow = false;
  for (std::uint32_t precompute_index = 0u;
       precompute_index < resident_tissue.ordered_binding_capacity;
       ++precompute_index) {
    const tissue::OrderedRoleBindingEvidence& candidate =
        resident_tissue.ordered_bindings[precompute_index];
    std::uint32_t scratch_first = 0u;
    std::uint32_t scratch_last = 0u;
    if (!ordered_relation_exact_cue_role(
            candidate, 1u, cue_exact, cue_orders, cue_unit_begin,
            cue_unit_count, units, resident_tissue.cell_capacity,
            &scratch_first, &scratch_last) ||
        !ordered_relation_patient_is_absent(candidate, cue_exact,
                                            cue_unit_begin, cue_unit_count))
      continue;
    if (predicate_candidate_count >= kMaximumOrderedRelationCandidates) {
      candidate_overflow = true;
      break;
    }
    predicate_candidates[predicate_candidate_count++] = precompute_index;
    if (ordered_relation_exact_cue_role(
            candidate, 0u, cue_exact, cue_orders, cue_unit_begin,
            cue_unit_count, units, resident_tissue.cell_capacity,
            &scratch_first, &scratch_last)) {
      if (agent_candidate_count >= kMaximumOrderedRelationCandidates) {
        candidate_overflow = true;
        break;
      }
      agent_candidates[agent_candidate_count++] = precompute_index;
    }
  }
  if (candidate_overflow) {
    receipt->capacity_overflow = 1u;
    return;
  }

  // Build a stable order over predicate candidates by the population used as
  // their agent role. A first edge's patient and the next edge's agent can
  // then meet through one bounded hash range instead of rescanning every
  // predicate candidate. The exact join predicate below still checks all
  // retained cells and units.
  std::uint32_t predicate_join_slots[kMaximumOrderedRelationCandidates];
  for (std::uint32_t slot = 0u; slot < predicate_candidate_count; ++slot)
    predicate_join_slots[slot] = slot;
  for (std::uint32_t slot = 1u; slot < predicate_candidate_count; ++slot) {
    const std::uint32_t displaced_slot = predicate_join_slots[slot];
    const std::uint32_t displaced_index =
        predicate_candidates[displaced_slot];
    const std::uint32_t displaced_hash =
        ordered_relation_role_population_hash(
            resident_tissue.ordered_bindings[displaced_index], 0u);
    std::uint32_t insertion = slot;
    while (insertion != 0u) {
      const std::uint32_t previous_slot = predicate_join_slots[insertion - 1u];
      const std::uint32_t previous_index =
          predicate_candidates[previous_slot];
      const std::uint32_t previous_hash =
          ordered_relation_role_population_hash(
              resident_tissue.ordered_bindings[previous_index], 0u);
      if (previous_hash <= displaced_hash)
        break;
      predicate_join_slots[insertion] = previous_slot;
      --insertion;
    }
    predicate_join_slots[insertion] = displaced_slot;
  }

  // Prefer a complete three-edge resident path when the cue supplies three
  // ordered connectives. Both joins require exact retained unit identity and
  // complete population equality; a digest, overlap, or caller intermediate
  // cannot bridge either step. Finding the topology but losing qualification
  // at any link blocks every shorter fallback.
  for (std::uint32_t first_slot = 0u; first_slot < agent_candidate_count;
       ++first_slot) {
    const std::uint32_t first_index = agent_candidates[first_slot];
    const tissue::OrderedRoleBindingEvidence& first =
        resident_tissue.ordered_bindings[first_index];
    std::uint32_t agent_first = 0u;
    std::uint32_t agent_last = 0u;
    std::uint32_t first_predicate_first = 0u;
    std::uint32_t first_predicate_last = 0u;
    if (!ordered_relation_exact_cue_role(
            first, 0u, cue_exact, cue_orders, cue_unit_begin, cue_unit_count,
            units, resident_tissue.cell_capacity, &agent_first, &agent_last) ||
        !ordered_relation_exact_cue_role(
            first, 1u, cue_exact, cue_orders, cue_unit_begin, cue_unit_count,
            units, resident_tissue.cell_capacity, &first_predicate_first,
            &first_predicate_last) ||
        agent_last >= first_predicate_first ||
        !ordered_relation_patient_is_absent(first, cue_exact, cue_unit_begin,
                                            cue_unit_count))
      continue;
    const std::uint32_t second_begin = ordered_relation_join_lower_bound(
        resident_tissue.ordered_bindings, predicate_candidates,
        predicate_join_slots, predicate_candidate_count,
        ordered_relation_role_population_hash(first, 2u));
    const std::uint32_t second_end = ordered_relation_join_upper_bound(
        resident_tissue.ordered_bindings, predicate_candidates,
        predicate_join_slots, predicate_candidate_count,
        ordered_relation_role_population_hash(first, 2u));
    for (std::uint32_t second_slot = second_begin; second_slot < second_end;
         ++second_slot) {
      const std::uint32_t second_index =
          predicate_candidates[predicate_join_slots[second_slot]];
      const tissue::OrderedRoleBindingEvidence& second =
          resident_tissue.ordered_bindings[second_index];
      std::uint32_t second_predicate_first = 0u;
      std::uint32_t second_predicate_last = 0u;
      if (!ordered_relation_same_patient_agent(first, second) ||
          !ordered_relation_exact_cue_role(
              second, 1u, cue_exact, cue_orders, cue_unit_begin,
              cue_unit_count, units, resident_tissue.cell_capacity,
              &second_predicate_first, &second_predicate_last) ||
          first_predicate_last >= second_predicate_first ||
          !ordered_relation_patient_is_absent(second, cue_exact,
                                              cue_unit_begin, cue_unit_count))
        continue;
      const std::uint32_t third_begin = ordered_relation_join_lower_bound(
          resident_tissue.ordered_bindings, predicate_candidates,
          predicate_join_slots, predicate_candidate_count,
          ordered_relation_role_population_hash(second, 2u));
      const std::uint32_t third_end = ordered_relation_join_upper_bound(
          resident_tissue.ordered_bindings, predicate_candidates,
          predicate_join_slots, predicate_candidate_count,
          ordered_relation_role_population_hash(second, 2u));
      for (std::uint32_t third_slot = third_begin; third_slot < third_end;
           ++third_slot) {
        const std::uint32_t third_index =
            predicate_candidates[predicate_join_slots[third_slot]];
        const tissue::OrderedRoleBindingEvidence& third =
            resident_tissue.ordered_bindings[third_index];
        std::uint32_t third_predicate_first = 0u;
        std::uint32_t third_predicate_last = 0u;
        if (!ordered_relation_same_patient_agent(second, third) ||
            !ordered_relation_exact_cue_role(
                third, 1u, cue_exact, cue_orders, cue_unit_begin,
                cue_unit_count, units, resident_tissue.cell_capacity,
                &third_predicate_first, &third_predicate_last) ||
            second_predicate_last >= third_predicate_first ||
            !ordered_relation_patient_is_absent(
                third, cue_exact, cue_unit_begin, cue_unit_count))
          continue;
        ++receipt->topology_matches;
        const bool first_qualified = tissue::ordered_binding_qualified(
            first, tissue::OrderedBindingQualification::discourse,
            resident_tissue.cell_capacity);
        const bool second_qualified = tissue::ordered_binding_qualified(
            second, tissue::OrderedBindingQualification::discourse,
            resident_tissue.cell_capacity);
        const bool third_qualified = tissue::ordered_binding_qualified(
            third, tissue::OrderedBindingQualification::discourse,
            resident_tissue.cell_capacity);
        if (!first_qualified || !second_qualified || !third_qualified) {
          ++receipt->withdrawn_candidates;
          continue;
        }
        if (third.role_unit_counts[2] == 0u ||
            third.role_unit_counts[2] > tissue::kMaximumOrderedRoleUnits) {
          ++receipt->invalid_sources;
          continue;
        }
        ++receipt->qualified_candidates;
        selected = first_index;
        selected_composed = second_index;
        selected_terminal = third_index;
        if (!append_ordered_relation_alternatives(
                third, alternatives, kMaximumExecutionOutputUnits,
                &alternative_count) ||
            receipt->alternative_source_count >= kMaximumExecutionOutputUnits)
          receipt->capacity_overflow = 1u;
        else
          receipt->alternative_source_binding_indices[
              receipt->alternative_source_count++] = third_index;
      }
    }
  }

  // Prefer a complete two-edge resident path when the cue supplies two
  // ordered connectives: A/K1 plus K2 can traverse A/K1/B and B/K2/C without
  // exposing B or C through a caller DTO.
  if (receipt->topology_matches == 0u) {
    for (std::uint32_t index_slot = 0u; index_slot < agent_candidate_count;
         ++index_slot) {
    const std::uint32_t index = agent_candidates[index_slot];
    const tissue::OrderedRoleBindingEvidence& first =
        resident_tissue.ordered_bindings[index];
    std::uint32_t agent_first = 0u;
    std::uint32_t agent_last = 0u;
    std::uint32_t predicate_first = 0u;
    std::uint32_t predicate_last = 0u;
    if (!ordered_relation_exact_cue_role(
            first, 0u, cue_exact, cue_orders, cue_unit_begin, cue_unit_count,
            units, resident_tissue.cell_capacity, &agent_first, &agent_last) ||
        !ordered_relation_exact_cue_role(
            first, 1u, cue_exact, cue_orders, cue_unit_begin, cue_unit_count,
            units, resident_tissue.cell_capacity, &predicate_first,
            &predicate_last) ||
        agent_last >= predicate_first ||
        !ordered_relation_patient_is_absent(first, cue_exact, cue_unit_begin,
                                            cue_unit_count))
      continue;
    const std::uint32_t next_begin = ordered_relation_join_lower_bound(
        resident_tissue.ordered_bindings, predicate_candidates,
        predicate_join_slots, predicate_candidate_count,
        ordered_relation_role_population_hash(first, 2u));
    const std::uint32_t next_end = ordered_relation_join_upper_bound(
        resident_tissue.ordered_bindings, predicate_candidates,
        predicate_join_slots, predicate_candidate_count,
        ordered_relation_role_population_hash(first, 2u));
    for (std::uint32_t next_slot = next_begin; next_slot < next_end;
         ++next_slot) {
      const std::uint32_t next =
          predicate_candidates[predicate_join_slots[next_slot]];
      const tissue::OrderedRoleBindingEvidence& second =
          resident_tissue.ordered_bindings[next];
      std::uint32_t second_predicate_first = 0u;
      std::uint32_t second_predicate_last = 0u;
      if (!ordered_relation_same_patient_agent(first, second) ||
          !ordered_relation_exact_cue_role(
              second, 1u, cue_exact, cue_orders, cue_unit_begin,
              cue_unit_count, units, resident_tissue.cell_capacity,
              &second_predicate_first, &second_predicate_last) ||
          predicate_last >= second_predicate_first ||
          !ordered_relation_patient_is_absent(second, cue_exact,
                                              cue_unit_begin, cue_unit_count))
        continue;
      ++receipt->topology_matches;
      const bool first_qualified = tissue::ordered_binding_qualified(
          first, tissue::OrderedBindingQualification::discourse,
          resident_tissue.cell_capacity);
      const bool second_qualified = tissue::ordered_binding_qualified(
          second, tissue::OrderedBindingQualification::discourse,
          resident_tissue.cell_capacity);
      if (!first_qualified || !second_qualified) {
        ++receipt->withdrawn_candidates;
        continue;
      }
      if (second.role_unit_counts[2] == 0u ||
          second.role_unit_counts[2] > tissue::kMaximumOrderedRoleUnits) {
        ++receipt->invalid_sources;
        continue;
      }
      ++receipt->qualified_candidates;
      selected = index;
      selected_composed = next;
      if (!append_ordered_relation_alternatives(
              second, alternatives, kMaximumExecutionOutputUnits,
              &alternative_count) ||
          receipt->alternative_source_count >= kMaximumExecutionOutputUnits)
        receipt->capacity_overflow = 1u;
      else
        receipt->alternative_source_binding_indices[
            receipt->alternative_source_count++] = next;
    }
    }
  }

  // If no two-edge topology exists, execute the original one-edge route.
  if (receipt->topology_matches == 0u) {
    for (std::uint32_t index_slot = 0u; index_slot < agent_candidate_count;
         ++index_slot) {
      const std::uint32_t index = agent_candidates[index_slot];
      const tissue::OrderedRoleBindingEvidence& binding =
          resident_tissue.ordered_bindings[index];
      std::uint32_t agent_first = 0u;
      std::uint32_t agent_last = 0u;
      std::uint32_t predicate_first = 0u;
      std::uint32_t predicate_last = 0u;
      if (!ordered_relation_exact_cue_role(
              binding, 0u, cue_exact, cue_orders, cue_unit_begin,
              cue_unit_count, units, resident_tissue.cell_capacity,
              &agent_first, &agent_last) ||
          !ordered_relation_exact_cue_role(
              binding, 1u, cue_exact, cue_orders, cue_unit_begin,
              cue_unit_count, units, resident_tissue.cell_capacity,
              &predicate_first, &predicate_last) ||
          agent_last >= predicate_first ||
          !ordered_relation_patient_is_absent(binding, cue_exact,
                                              cue_unit_begin, cue_unit_count))
        continue;
      ++receipt->topology_matches;
      if (!tissue::ordered_binding_qualified(
              binding, tissue::OrderedBindingQualification::discourse,
              resident_tissue.cell_capacity)) {
        ++receipt->withdrawn_candidates;
        continue;
      }
      if (binding.role_unit_counts[2] == 0u ||
          binding.role_unit_counts[2] > tissue::kMaximumOrderedRoleUnits) {
        ++receipt->invalid_sources;
        continue;
      }
      ++receipt->qualified_candidates;
      selected = index;
      if (!append_ordered_relation_alternatives(
              binding, alternatives, kMaximumExecutionOutputUnits,
              &alternative_count) ||
          receipt->alternative_source_count >= kMaximumExecutionOutputUnits)
        receipt->capacity_overflow = 1u;
      else
        receipt->alternative_source_binding_indices[
            receipt->alternative_source_count++] = index;
    }
  }
  if (receipt->qualified_candidates != 1u) {
    receipt->conflict = receipt->qualified_candidates > 1u ? 1u : 0u;
    if (receipt->conflict != 0u && receipt->capacity_overflow == 0u &&
        alternative_count >= 2u) {
      // Prefix alternatives with the earliest exact learned question opener.
      // This unit is selected by resident onset evidence and cue chronology,
      // never supplied as a clarification string by the caller.
      std::uint32_t opener = 0xffffffffu;
      std::uint32_t opener_order = 0xffffffffu;
      if (question_onset_evidence != nullptr)
        for (std::uint32_t offset = 0u; offset < cue_unit_count; ++offset) {
          const std::uint32_t unit = cue_unit_begin + offset;
          if (cue_exact[unit] != 0u && cue_orders[unit] != 0u &&
              question_onset_evidence[unit] != 0u &&
              cue_orders[unit] < opener_order) {
            opener = unit;
            opener_order = cue_orders[unit];
          }
        }
      const std::uint32_t needed = alternative_count +
          (opener == 0xffffffffu ? 0u : 1u);
      if (needed <= output_capacity) {
        std::uint32_t written = 0u;
        if (opener != 0xffffffffu) {
          output_units[written++] = opener;
          receipt->clarification_opener = opener;
        }
        for (std::uint32_t index = 0u; index < alternative_count; ++index)
          output_units[written++] = alternatives[index];
        *output_count = written;
        receipt->output_unit_count = written;
        receipt->clarification_ready = 1u;
        receipt->abstained = 0u;
      } else {
        receipt->capacity_overflow = 1u;
      }
    }
    return;
  }
  const tissue::OrderedRoleBindingEvidence& source =
      resident_tissue.ordered_bindings[selected];
  const tissue::OrderedRoleBindingEvidence& output_source =
      resident_tissue.ordered_bindings[
          selected_terminal != 0xffffffffu
              ? selected_terminal
              : (selected_composed == 0xffffffffu ? selected
                                                  : selected_composed)];
  if (output_source.role_unit_counts[2] > output_capacity) {
    receipt->capacity_overflow = 1u;
    return;
  }
  for (std::uint32_t index = 0u;
       index < output_source.role_unit_counts[2]; ++index)
    output_units[index] = output_source.role_units[2][index];
  *output_count = output_source.role_unit_counts[2];
  receipt->source_binding_index = selected;
  receipt->composed_source_binding_index = selected_composed;
  receipt->terminal_source_binding_index = selected_terminal;
  receipt->source_evidence_revision = source.last_evidence_revision;
  receipt->composed_source_evidence_revision =
      selected_composed == 0xffffffffu
          ? 0u
          : resident_tissue.ordered_bindings[selected_composed]
                .last_evidence_revision;
  receipt->terminal_source_evidence_revision =
      selected_terminal == 0xffffffffu
          ? 0u
          : output_source.last_evidence_revision;
  const std::uint64_t ticket_revision =
      receipt->ticketed_return_evidence_revision;
  if (ticket_revision != 0u) {
    if (receipt->source_evidence_revision == ticket_revision)
      receipt->ticketed_return_source_mask |= 1u;
    if (receipt->composed_source_evidence_revision == ticket_revision)
      receipt->ticketed_return_source_mask |= 2u;
    if (receipt->terminal_source_evidence_revision == ticket_revision)
      receipt->ticketed_return_source_mask |= 4u;
  }
  receipt->source_positive_mass =
      tissue::ordered_binding_positive_mass(source) +
      (selected_composed == 0xffffffffu
           ? 0u
           : tissue::ordered_binding_positive_mass(
                 resident_tissue.ordered_bindings[selected_composed])) +
      (selected_terminal == 0xffffffffu
           ? 0u
           : tissue::ordered_binding_positive_mass(output_source));
  receipt->source_counterevidence = source.counterevidence +
      (selected_composed == 0xffffffffu ? 0u
          : resident_tissue.ordered_bindings[selected_composed]
                .counterevidence) +
      (selected_terminal == 0xffffffffu ? 0u
                                        : output_source.counterevidence);
  receipt->source_context_count = source.qualifying_context_count;
  receipt->output_unit_count = output_source.role_unit_counts[2];
  receipt->composition_depth =
      selected_terminal != 0xffffffffu
          ? 3u
          : (selected_composed == 0xffffffffu ? 1u : 2u);
  receipt->ready = 1u;
  receipt->abstained = 0u;
}

__device__ inline bool binding_revision_is_current_contact(
    const tissue::OrderedRoleBindingEvidence& binding,
    std::uint64_t current_contact_revision) {
  return current_contact_revision != 0u &&
         binding.last_evidence_revision != 0u &&
         (binding.last_evidence_revision >> 32u) == current_contact_revision;
}

// Invalidate only bindings revised by this contact. This is a bounded resident
// maintenance pass; it does not inspect construction aggregates or surface
// slots, and therefore cannot invent a replacement witness.
__global__ void invalidate_current_ordered_construction_links_kernel(
    const tissue::OrderedRoleBindingEvidence* bindings,
    std::uint32_t binding_capacity,
    const std::uint64_t* current_contact_revision, std::uint32_t* links) {
  const std::uint32_t binding_index =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (bindings == nullptr || current_contact_revision == nullptr ||
      links == nullptr || binding_index >= binding_capacity)
    return;
  if (binding_revision_is_current_contact(bindings[binding_index],
                                          current_contact_revision[0]))
    links[binding_index] = construction::kNoConstruction;
}

// A six-counter AND across a whole contact's per-event outcomes used to gate
// this contact's ENTIRE shadow-tissue commit: one malformed event among
// thousands of correctly-assimilated ones zeroed every binding in the same
// contact (0X1-155). Each event's own validity, capacity, and mass are
// already enforced independently inside assimilate_ordered_binding_detailed
// and build_event_populations, and each failure there withholds only that
// one event's own binding write -- it never touches any other event's
// already-correct write in the same shadow tissue. A per-event outcome tally
// therefore no longer has anything left to veto at the whole-contact grain;
// this predicate is kept only as the named checkpoint two-phase
// (build-then-apply) callers still invoke.
[[nodiscard]] __device__ inline bool assimilation_authorized(
    const AssimilationReceipt*) {
  return true;
}

__device__ inline std::uint32_t observation_count(
    RelationObservationView observations,
    const construction::RelationTripleEvent& event) {
  if (observations.triples == nullptr || observations.counts == nullptr ||
      observations.minimum_recurrence <= 1u)
    return observations.minimum_recurrence;
  return construction::relation_triple_lookup(
      observations.triples, observations.counts, event.triple.subject,
      event.triple.connective, event.triple.connective2, event.triple.value);
}

__device__ inline bool append_complete_unit_population(
    population_surface::UnitPopulationView units, std::uint32_t cell_capacity,
    std::uint32_t unit, std::uint32_t* cells, std::uint32_t* cell_count) {
  if (units.cells == nullptr || units.contact_mass == nullptr || cells == nullptr ||
      cell_count == nullptr || unit < units.unit_begin ||
      unit - units.unit_begin >= units.unit_count || units.population_width == 0u ||
      units.population_width > tissue::kMaximumPopulationCells ||
      units.contact_mass[unit] == 0u ||
      (units.population_count != nullptr &&
       units.population_count[unit] != units.population_width))
    return false;

  std::uint32_t novel[tissue::kMaximumPopulationCells]{};
  std::uint32_t novel_count = 0u;
  const std::uint32_t* source =
      units.cells + static_cast<std::size_t>(unit) * units.population_width;
  for (std::uint32_t slot = 0u; slot < units.population_width; ++slot) {
    const std::uint32_t cell = source[slot];
    if (cell == 0xffffffffu || cell >= cell_capacity)
      return false;
    bool present = false;
    for (std::uint32_t prior = 0u; prior < *cell_count; ++prior)
      present = present || cells[prior] == cell;
    for (std::uint32_t prior = 0u; prior < novel_count; ++prior)
      if (novel[prior] == cell)
        return false;
    if (!present)
      novel[novel_count++] = cell;
  }
  if (novel_count == 0u ||
      *cell_count + novel_count > tissue::kMaximumPopulationCells)
    return false;
  for (std::uint32_t index = 0u; index < novel_count; ++index)
    cells[(*cell_count)++] = novel[index];
  return true;
}

__device__ inline bool ordered_relation_cue_role_matches(
    const tissue::OrderedRoleBindingEvidence& binding, std::uint32_t role,
    tissue::SparsePopulationView cue_population, const std::uint32_t* cue_units,
    std::uint32_t cue_unit_count) {
  if (binding.claimed == 0u || role >= 2u || cue_units == nullptr ||
      cue_unit_count == 0u ||
      cue_unit_count > tissue::kMaximumOrderedRoleUnits ||
      binding.role_unit_counts[role] != cue_unit_count ||
      !tissue::exact_population_equals(binding.role_cells[role],
                                       binding.role_counts[role],
                                       cue_population))
    return false;
  for (std::uint32_t index = 0u; index < cue_unit_count; ++index)
    if (binding.role_units[role][index] != cue_units[index])
      return false;
  return true;
}

// Execute one learned ordered relation from a raw held-out cue.  The output
// buffer is only a transport surface: the selected unit and its provenance are
// copied from the unique authoritative resident binding after the complete
// scan.  Conflict, lesion/withdrawal, malformed topology, and output pressure
// all leave output_count at zero, so stale bytes never become an action.
__global__ void execute_ordered_relation_kernel(
    tissue::TissueView resident_tissue, const std::uint32_t* cue_sequence,
    std::uint32_t cue_count, population_surface::UnitPopulationView units,
    std::uint32_t* output_units, std::uint32_t output_capacity,
    std::uint32_t* output_count, OrderedRelationExecutionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr ||
      output_count == nullptr)
    return;
  *receipt = OrderedRelationExecutionReceipt{};
  *output_count = 0u;
  if (resident_tissue.scalars != nullptr)
    receipt->tissue_revision = resident_tissue.scalars->revision;
  if (resident_tissue.ordered_bindings == nullptr || cue_sequence == nullptr ||
      output_units == nullptr || cue_count < 2u || cue_count > 3u)
    return;
  receipt->cue_unit_count = cue_count;
  for (std::uint32_t index = 0u; index < cue_count; ++index)
    receipt->cue_units[index] = cue_sequence[index];

  std::uint32_t agent_cells[tissue::kMaximumPopulationCells]{};
  std::uint32_t predicate_cells[tissue::kMaximumPopulationCells]{};
  std::uint32_t agent_count = 0u;
  std::uint32_t predicate_count = 0u;
  if (!append_complete_unit_population(
          units, resident_tissue.cell_capacity, cue_sequence[0], agent_cells,
          &agent_count))
    return;
  for (std::uint32_t index = 1u; index < cue_count; ++index)
    if (!append_complete_unit_population(
            units, resident_tissue.cell_capacity, cue_sequence[index],
            predicate_cells, &predicate_count))
      return;

  const tissue::SparsePopulationView agent{agent_cells, agent_count};
  const tissue::SparsePopulationView predicate{predicate_cells,
                                                predicate_count};
  std::uint32_t selected = 0xffffffffu;
  for (std::uint32_t index = 0u;
       index < resident_tissue.ordered_binding_capacity; ++index) {
    const tissue::OrderedRoleBindingEvidence& binding =
        resident_tissue.ordered_bindings[index];
    const bool topology_match =
        ordered_relation_cue_role_matches(binding, 0u, agent, cue_sequence, 1u) &&
        ordered_relation_cue_role_matches(binding, 1u, predicate,
                                          cue_sequence + 1u, cue_count - 1u);
    if (!topology_match)
      continue;
    ++receipt->topology_matches;
    if (!tissue::ordered_binding_qualified(
            binding, tissue::OrderedBindingQualification::discourse,
            resident_tissue.cell_capacity)) {
      ++receipt->withdrawn_candidates;
      continue;
    }
    if (binding.role_unit_counts[2] == 0u ||
        binding.role_unit_counts[2] > tissue::kMaximumOrderedRoleUnits) {
      ++receipt->invalid_sources;
      continue;
    }
    ++receipt->qualified_candidates;
    selected = index;
  }
  if (receipt->qualified_candidates != 1u) {
    receipt->conflict = receipt->qualified_candidates > 1u ? 1u : 0u;
    return;
  }

  const tissue::OrderedRoleBindingEvidence& source =
      resident_tissue.ordered_bindings[selected];
  if (source.role_unit_counts[2] > output_capacity) {
    receipt->capacity_overflow = 1u;
    return;
  }
  for (std::uint32_t index = 0u; index < source.role_unit_counts[2]; ++index)
    output_units[index] = source.role_units[2][index];
  *output_count = source.role_unit_counts[2];
  receipt->source_binding_index = selected;
  receipt->source_evidence_revision = source.last_evidence_revision;
  receipt->source_positive_mass =
      tissue::ordered_binding_positive_mass(source);
  receipt->source_counterevidence = source.counterevidence;
  receipt->source_context_count = source.qualifying_context_count;
  receipt->output_unit_count = source.role_unit_counts[2];
  receipt->ready = 1u;
  receipt->abstained = 0u;
}

struct EventPopulations {
  std::uint32_t agent[tissue::kMaximumPopulationCells]{};
  std::uint32_t predicate[tissue::kMaximumPopulationCells]{};
  std::uint32_t patient[tissue::kMaximumPopulationCells]{};
  std::uint32_t context[tissue::kMaximumPopulationCells]{};
  std::uint32_t agent_count = 0u;
  std::uint32_t predicate_count = 0u;
  std::uint32_t patient_count = 0u;
  std::uint32_t context_count = 0u;
};

__device__ inline tissue::OrderedRoleUnitIdentity event_role_identity(
    const construction::RelationTripleEvent& event) {
  tissue::OrderedRoleUnitIdentity identity{};
  identity.units[0][0] = event.triple.subject;
  identity.counts[0] = 1u;
  identity.units[1][0] = event.triple.connective;
  identity.counts[1] = 1u;
  if (event.triple.connective2 != construction::kNoTripleUnit) {
    identity.units[1][1] = event.triple.connective2;
    identity.counts[1] = 2u;
  }
  identity.units[2][0] = event.triple.value;
  identity.counts[2] = 1u;
  return identity;
}

__device__ inline tissue::OrderedRoleUnitIdentity event_role_identity(
    const construction::WitnessedRelationEvent& event) {
  tissue::OrderedRoleUnitIdentity identity{};
  identity.units[0][0] = event.triple.subject;
  identity.counts[0] = 1u;
  identity.units[1][0] = event.triple.connective;
  identity.counts[1] = 1u;
  if (event.triple.connective2 != construction::kNoTripleUnit) {
    identity.units[1][1] = event.triple.connective2;
    identity.counts[1] = 2u;
  }
  identity.units[2][0] = event.triple.value;
  identity.counts[2] = 1u;
  return identity;
}

__device__ inline bool build_event_populations(
    tissue::TissueView resident_tissue,
    const construction::RelationTripleEvent& event,
    const std::uint32_t* sequence,
    population_surface::UnitPopulationView units,
    EventPopulations* populations,
    PopulationFailure* failure = nullptr) {
  if (failure != nullptr)
    *failure = PopulationFailure::none;
  if (sequence == nullptr || populations == nullptr) {
    if (failure != nullptr)
      *failure = PopulationFailure::missing_view;
    return false;
  }
  *populations = EventPopulations{};
  if (!append_complete_unit_population(
          units, resident_tissue.cell_capacity, event.triple.subject,
          populations->agent, &populations->agent_count) ||
      !append_complete_unit_population(
          units, resident_tissue.cell_capacity, event.triple.connective,
          populations->predicate, &populations->predicate_count) ||
      (event.triple.connective2 != construction::kNoTripleUnit &&
       !append_complete_unit_population(
           units, resident_tissue.cell_capacity, event.triple.connective2,
           populations->predicate, &populations->predicate_count)) ||
      !append_complete_unit_population(
          units, resident_tissue.cell_capacity, event.triple.value,
          populations->patient, &populations->patient_count)) {
    if (failure != nullptr)
      *failure = PopulationFailure::primary_unit;
    return false;
  }

  bool context_unit_found = false;
  for (std::uint32_t position = event.segment_begin;
       position < event.segment_end; ++position) {
    if (position == event.subject_position || position == event.value_position ||
        (position >= event.connective_begin && position < event.connective_end))
      continue;
    const std::uint32_t unit = sequence[position];
    if (unit == event.triple.subject || unit == event.triple.connective ||
        unit == event.triple.connective2 || unit == event.triple.value)
      continue;
    context_unit_found = true;
    if (!append_complete_unit_population(
            units, resident_tissue.cell_capacity, unit, populations->context,
            &populations->context_count)) {
      if (failure != nullptr)
        *failure = PopulationFailure::context_unit;
      return false;
    }
    break;
  }
  if (!context_unit_found) {
    // An exact complete A population is a lawful one-episode fallback. It does
    // not manufacture context diversity across repeated A/K/B observations.
    for (std::uint32_t index = 0u; index < populations->agent_count; ++index)
      populations->context[populations->context_count++] =
          populations->agent[index];
  }
  const bool context_distinct = tissue::valid_distinct_population(
      {populations->context, populations->context_count},
      resident_tissue.cell_capacity);
  if (!context_distinct && failure != nullptr)
    *failure = PopulationFailure::context_distinct;
  return context_distinct;
}

__device__ inline bool build_witnessed_event_populations(
    tissue::TissueView resident_tissue,
    const construction::WitnessedRelationEvent& event,
    population_surface::UnitPopulationView units,
    EventPopulations* populations) {
  if (populations == nullptr)
    return false;
  *populations = EventPopulations{};
  return append_complete_unit_population(
             units, resident_tissue.cell_capacity, event.triple.subject,
             populations->agent, &populations->agent_count) &&
         append_complete_unit_population(
             units, resident_tissue.cell_capacity, event.triple.connective,
             populations->predicate, &populations->predicate_count) &&
         (event.triple.connective2 == construction::kNoTripleUnit ||
          append_complete_unit_population(
              units, resident_tissue.cell_capacity, event.triple.connective2,
              populations->predicate, &populations->predicate_count)) &&
         append_complete_unit_population(
             units, resident_tissue.cell_capacity, event.triple.value,
             populations->patient, &populations->patient_count);
}

// Exact event provenance is the authoritative relation-to-surface bridge. One
// thread consumes one retained event, reconstructs only that event's three
// resident populations, and uses the existing identity-indexed tissue lookup.
// Aggregate construction slots are intentionally absent from this kernel: an
// overflowed or irrelevant aggregate cannot veto a complete event-local proof.
__global__ void bridge_exact_event_constructions_kernel(
    tissue::TissueView resident_tissue,
    const construction::WitnessedRelationEvent* witnessed_events,
    const std::uint32_t* witnessed_event_cursor,
    const std::uint32_t* witnessed_event_constructions,
    const std::uint32_t* construction_count,
    std::uint32_t construction_capacity,
    population_surface::UnitPopulationView units, std::uint32_t* links) {
  const std::uint32_t event_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (witnessed_events == nullptr || witnessed_event_cursor == nullptr ||
      witnessed_event_constructions == nullptr || construction_count == nullptr ||
      links == nullptr)
    return;
  const std::uint32_t extent =
      min(witnessed_event_cursor[0], construction::kWitnessedRelationEventCap);
  if (event_index >= extent)
    return;
  const construction::WitnessedRelationEvent event = witnessed_events[event_index];
  const std::uint32_t construction_index =
      witnessed_event_constructions[event_index];
  if (event.live == 0u || event.evidence_revision == 0u ||
      construction_index == construction::kNoConstruction ||
      construction_index == construction::kAmbiguousConstruction ||
      construction_index >= construction_capacity ||
      construction_index >= construction_count[0])
    return;

  EventPopulations populations{};
  if (!build_witnessed_event_populations(resident_tissue, event, units,
                                         &populations))
    return;
  tissue::OrderedRoleBindingEvidence* binding = tissue::find_ordered_binding(
      resident_tissue, {populations.agent, populations.agent_count},
      {populations.predicate, populations.predicate_count},
      {populations.patient, populations.patient_count}, nullptr,
      event_role_identity(event));
  if (binding == nullptr ||
      binding->last_evidence_revision != event.evidence_revision)
    return;
  links[static_cast<std::size_t>(binding - resident_tissue.ordered_bindings)] =
      construction_index;
}

__device__ inline tissue::OrderedBindingAssimilationStatus assimilate_event(
    tissue::TissueView resident_tissue,
    const construction::RelationTripleEvent& event,
    const std::uint32_t* sequence,
    population_surface::UnitPopulationView units,
    std::uint64_t evidence_revision, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present) {
  EventPopulations populations{};
  if (!build_event_populations(resident_tissue, event, sequence, units,
                               &populations))
    return tissue::OrderedBindingAssimilationStatus::invalid;

  const auto result = tissue::assimilate_ordered_binding_detailed(
      resident_tissue, {populations.agent, populations.agent_count},
      {populations.predicate, populations.predicate_count},
      {populations.patient, populations.patient_count},
      {populations.context, populations.context_count}, evidence_revision,
      efferent_contact, efferent_polarity, outcome_present,
      event_role_identity(event));
  return result.status;
}

__device__ inline tissue::OrderedBindingAssimilationStatus
assimilate_recurrent_event(
    tissue::TissueView resident_tissue,
    const construction::RelationTripleEvent& event,
    const std::uint32_t* sequence,
    population_surface::UnitPopulationView units,
    std::uint64_t evidence_revision, std::uint32_t efferent_contact,
    std::int32_t efferent_polarity, std::uint32_t outcome_present,
    std::uint32_t* context_replay, std::uint32_t* generalized) {
  EventPopulations populations{};
  if (!build_event_populations(resident_tissue, event, sequence, units,
                               &populations))
    return tissue::OrderedBindingAssimilationStatus::invalid;

  tissue::OrderedRoleBindingEvidence* binding = tissue::find_ordered_binding(
      resident_tissue, {populations.agent, populations.agent_count},
      {populations.predicate, populations.predicate_count},
      {populations.patient, populations.patient_count}, nullptr,
      event_role_identity(event));
  const tissue::SparsePopulationView context{populations.context,
                                              populations.context_count};
  // A non-efferent (passive) contact never carries bodily-consequence
  // evidence regardless of whether an outcome bit happens to be set --
  // assimilate_ordered_binding_detailed() already ignores outcome_present
  // for !intervention ("the outcome bit is deliberately ignored" for
  // passive observation; see bcc32_cuda_resident_proposition_tissue_tail.inl).
  // This fast-path pruning check previously required outcome_present == 0u
  // too, so a passive contact whose outcome bit happened to be set (0X1-349:
  // "non-efferent outcome escaped passive replay pruning") fell through to a
  // full assimilate call and mutated resident mass/revision on a redundant
  // re-observation instead of being pruned as a replay.
  const bool redundant_passive_observation = efferent_contact == 0u;
  if (redundant_passive_observation && binding != nullptr &&
      evidence_revision > binding->last_evidence_revision &&
      (binding->qualifying_context_count >= tissue::kMinimumIndependentContexts ||
       tissue::ordered_binding_context_slot(*binding, context) >= 0)) {
    if (context_replay != nullptr)
      *context_replay = 1u;
    return tissue::OrderedBindingAssimilationStatus::replay;
  }
  const std::uint32_t prior_contexts =
      binding == nullptr ? 0u : binding->qualifying_context_count;
  const auto result = tissue::assimilate_ordered_binding_detailed(
      resident_tissue, {populations.agent, populations.agent_count},
      {populations.predicate, populations.predicate_count},
      {populations.patient, populations.patient_count}, context,
      evidence_revision, efferent_contact, efferent_polarity, outcome_present,
      event_role_identity(event));
  if (generalized != nullptr &&
      result.status == tissue::OrderedBindingAssimilationStatus::accepted &&
      prior_contexts < tissue::kMinimumIndependentContexts) {
    tissue::OrderedRoleBindingEvidence* retained = tissue::find_ordered_binding(
        resident_tissue, {populations.agent, populations.agent_count},
        {populations.predicate, populations.predicate_count},
        {populations.patient, populations.patient_count}, nullptr,
        event_role_identity(event));
    if (retained != nullptr &&
        retained->qualifying_context_count >= tissue::kMinimumIndependentContexts)
      *generalized = 1u;
  }
  return result.status;
}

__global__ void advance_contact_revision_kernel(std::uint64_t* revision) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && revision != nullptr)
    ++*revision;
}

// Extraction is tiled across one warp; mutation remains physically ordered by
// stream position and event ordinal. This avoids a bounded atomic event buffer,
// nondeterministic subsets, and host-side content decisions.
//
// A single pass builds and assimilates every event: each event's own
// validity, recurrence classification, and (if valid) assimilation outcome
// are decided independently of every other event in the same contact. A
// malformed role, missing formation mass, an over-full binding table, or an
// extractor overflow only ever withholds THAT event's own binding write --
// never another event's already-correct write in the same contact. Prior to
// this, a two-pass preflight-then-mutate design abstained the entire
// contact's mutation loop whenever any single event failed preflight, which
// discarded every correctly-assimilated binding alongside the bad one; see
// 0X1-155.
__global__ void assimilate_relation_events_kernel(
    tissue::TissueView resident_tissue, const std::uint32_t* sequence,
    std::uint32_t sequence_count, const std::uint32_t* segment_ids,
    const std::uint32_t* closed_class_mask,
    population_surface::UnitPopulationView units,
    const std::uint64_t* contact_revision,
    std::uint32_t efferent_contact, std::int32_t efferent_polarity,
    std::uint32_t outcome_present,
    const AssimilationReceipt* authorization, AssimilationReceipt* receipt,
    RelationObservationView observations = {},
    const std::uint32_t* contact_event_count = nullptr) {
  if (blockIdx.x != 0u || blockDim.x != kEventTileWidth)
    return;
  __shared__ construction::RelationTripleEvent
      events[kEventTileWidth][kMaximumEventsPerPosition];
  __shared__ std::uint32_t event_counts[kEventTileWidth];
  __shared__ std::uint64_t revision;

  if (threadIdx.x == 0u) {
    if (receipt != nullptr)
      *receipt = AssimilationReceipt{};
    revision = contact_revision == nullptr ? 0u : *contact_revision;
    if (receipt != nullptr)
      receipt->contact_revision = revision;
  }
  __syncthreads();
  // The parallel relation extractor already inspected the complete contact.
  // When it found no event, rescanning millions of positions twice in this
  // deliberately ordered one-warp mutation kernel cannot change resident
  // state. Keep the zero-event decision on device and abstain immediately.
  if (contact_event_count != nullptr && *contact_event_count == 0u)
    return;
  // A per-event outcome tally on a prior receipt is not a whole-contact
  // veto (see assimilation_authorized): the check is retained only as the
  // named checkpoint a two-phase build-then-apply caller still invokes.
  if (!assimilation_authorized(authorization))
    return;
  if (sequence == nullptr || segment_ids == nullptr ||
      closed_class_mask == nullptr || units.cells == nullptr ||
      units.contact_mass == nullptr || revision == 0u ||
      units.population_width == 0u)
    return;

  for (std::uint32_t tile = 0u; tile < sequence_count;
       tile += kEventTileWidth) {
    const std::uint32_t position = tile + threadIdx.x;
    std::uint32_t count = 0u;
    if (position < sequence_count) {
      for (std::uint32_t ordinal = 0u;
           ordinal < kMaximumEventsPerPosition; ++ordinal) {
        construction::RelationTripleEvent event{};
        if (!construction::extract_relation_triple_event(
                sequence, sequence_count, segment_ids, closed_class_mask,
                position, ordinal, &event))
          break;
        events[threadIdx.x][count++] = event;
      }
    }
    event_counts[threadIdx.x] = count;
    if (position < sequence_count && count == kMaximumEventsPerPosition) {
      construction::RelationTripleEvent overflow{};
      if (construction::extract_relation_triple_event(
              sequence, sequence_count, segment_ids, closed_class_mask,
              position, kMaximumEventsPerPosition, &overflow) &&
          receipt != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(&receipt->event_overflow),
                  1ull);
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
      for (std::uint32_t lane = 0u; lane < kEventTileWidth; ++lane) {
        for (std::uint32_t ordinal = 0u; ordinal < event_counts[lane];
             ++ordinal) {
          const auto& event = events[lane][ordinal];
          if (receipt != nullptr)
            ++receipt->events;
          const std::uint32_t recurrence = observation_count(observations, event);
          if (receipt != nullptr) {
            if (efferent_contact == 0u &&
                recurrence < observations.minimum_recurrence)
              ++receipt->below_recurrence;
            else
              ++receipt->recurrent;
          }
          EventPopulations populations{};
          PopulationFailure population_failure = PopulationFailure::none;
          if (!build_event_populations(resident_tissue, event, sequence, units,
                                       &populations, &population_failure)) {
            if (receipt != nullptr) {
              ++receipt->invalid;
              ++receipt->incomplete_population;
              receipt->whole_contact_abstained = 1u;
              // Same branch, same event: the split is a partition of
              // incomplete_population, never an additional rejection.
              switch (population_failure) {
                case PopulationFailure::missing_view:
                  ++receipt->population_missing_view;
                  break;
                case PopulationFailure::primary_unit:
                  ++receipt->population_primary_unit;
                  break;
                case PopulationFailure::context_unit:
                  ++receipt->population_context_unit;
                  break;
                case PopulationFailure::context_distinct:
                  ++receipt->population_context_distinct;
                  break;
                case PopulationFailure::none:
                  break;
              }
            }
            if (resident_tissue.ordered_binding_admission != nullptr)
              ++resident_tissue.ordered_binding_admission->invalid_attempts;
            // No device print here: device printf writes to stdout, which under
            // --duplex-stdio is the motor frame channel -- it corrupts the frame
            // stream and hangs the gate. Observables leave as receipt fields.
            // This event's own binding write is withheld; every other event
            // in this contact is still attempted below.
            continue;
          }
          const std::uint64_t event_revision =
              construction::relation_event_evidence_revision(
                  revision, event.subject_position, ordinal);
          std::uint32_t context_replay = 0u;
          std::uint32_t generalized = 0u;
          const bool consolidation_enabled =
              observations.triples != nullptr && observations.counts != nullptr;
          const bool recurrent_event =
              efferent_contact != 0u ||
              recurrence >= observations.minimum_recurrence;
          const auto status = consolidation_enabled && recurrent_event
                                  ? assimilate_recurrent_event(
                                        resident_tissue, event, sequence, units,
                                        event_revision, efferent_contact,
                                        efferent_polarity, outcome_present,
                                        &context_replay, &generalized)
                                  : assimilate_event(
                                        resident_tissue, event, sequence, units,
                                        event_revision, efferent_contact,
                                        efferent_polarity, outcome_present);
          if (receipt == nullptr)
            continue;
          receipt->context_replays += context_replay;
          receipt->generalized += generalized;
          if (status == tissue::OrderedBindingAssimilationStatus::accepted)
            ++receipt->accepted;
          else if (status == tissue::OrderedBindingAssimilationStatus::replay)
            ++receipt->replays;
          else if (status ==
                   tissue::OrderedBindingAssimilationStatus::capacity_overflow) {
            ++receipt->capacity_overflow;
            receipt->whole_contact_abstained = 1u;
          } else if (status ==
                     tissue::OrderedBindingAssimilationStatus::insufficient_mass) {
            ++receipt->insufficient_mass;
            receipt->whole_contact_abstained = 1u;
          } else {
            ++receipt->invalid;
            receipt->whole_contact_abstained = 1u;
          }
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u && receipt != nullptr && receipt->event_overflow != 0u)
    receipt->whole_contact_abstained = 1u;
}

// The kernel above already produces the exact committed state in a shadow
// tissue, one binding at a time: every index the shadow-build kernel touched
// holds either an unchanged copy of the resident binding (never reached, or
// reached and rejected on its own terms) or a correctly-updated one (that
// event's own assimilation succeeded). Publish the whole table in parallel;
// no further per-event gate is needed here because each shadow index is
// already self-consistent. Only a genuinely unsafe transaction -- no
// authorization receipt at all, or a structural view mismatch (null
// pointers, mismatched capacity) -- withholds the publish; a per-event
// outcome tally in the receipt (invalid/capacity_overflow/insufficient_mass/
// event_overflow/whole_contact_abstained) never does (0X1-155). The host
// launches this transaction but never reads relation content or chooses
// whether it commits.
__global__ void commit_assimilation_shadow_kernel(
    tissue::TissueView resident_tissue, tissue::TissueView shadow_tissue,
    const AssimilationReceipt* authorization, AssimilationReceipt* receipt) {
  const bool views_valid =
      resident_tissue.ordered_bindings != nullptr &&
      shadow_tissue.ordered_bindings != nullptr &&
      resident_tissue.scalars != nullptr && shadow_tissue.scalars != nullptr &&
      resident_tissue.ordered_binding_admission != nullptr &&
      shadow_tissue.ordered_binding_admission != nullptr &&
      resident_tissue.ordered_binding_capacity ==
          shadow_tissue.ordered_binding_capacity;
  const bool authorized = authorization != nullptr &&
                          assimilation_authorized(authorization) && views_valid;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0u) {
    if (receipt != nullptr) {
      *receipt = authorization == nullptr ? AssimilationReceipt{}
                                          : *authorization;
      if (!authorized && (authorization == nullptr || !views_valid))
        receipt->whole_contact_abstained = 1u;
    }
  }
  if (!authorized)
    return;

  if (index < resident_tissue.ordered_binding_capacity)
    resident_tissue.ordered_bindings[index] =
        shadow_tissue.ordered_bindings[index];
  if (index == 0u) {
    *resident_tissue.scalars = *shadow_tissue.scalars;
    *resident_tissue.ordered_binding_admission =
        *shadow_tissue.ordered_binding_admission;
  }
}

}  // namespace bcc32_cuda_resident_ordered_relation_assimilation

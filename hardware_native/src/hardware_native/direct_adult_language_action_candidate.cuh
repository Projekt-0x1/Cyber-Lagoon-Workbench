#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ACTION_CANDIDATE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ACTION_CANDIDATE_CUH
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_action_control_runtime.cuh"
#include "hardware_native/direct_language_plan_causal_lowering.cuh"
namespace substrate::direct_adult_core {
#if defined(__CUDACC__)
#define DIRECT_LANGUAGE_ACTION_HD __host__ __device__
#else
#define DIRECT_LANGUAGE_ACTION_HD
#endif
DIRECT_LANGUAGE_ACTION_HD inline const direct_network::DirectExactHistoryRecord*
latest_verified_sensory_contact(const direct_network::DirectExactHistoryRecord* records,
                                std::uint32_t count) {
  using namespace direct_network;
  if (records == nullptr) return nullptr;
  const DirectExactHistoryRecord* best = nullptr;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& row = records[i];
    if (row.kind != DirectExactHistoryKind::sensory_contact ||
        (row.flags & kDirectHistoryVerifiedObservation) == 0u || row.identity == 0u)
      continue;
    if (best == nullptr || row.sequence > best->sequence) best = &row;
  }
  return best;
}
DIRECT_LANGUAGE_ACTION_HD inline const ResidentRecipeOccurrence*
current_actual_parent_for_sensory(const ResidentRecipeOccurrence* occurrences,
                                  std::uint32_t occurrence_count,
                                  const direct_network::DirectExactHistoryRecord& sensory,
                                  std::uint32_t current_tick) {
  if (occurrences == nullptr) return nullptr;
  const ResidentRecipeOccurrence* match = nullptr;
  for (std::uint32_t i = 0u; i < occurrence_count; ++i) {
    const auto& occ = occurrences[i];
    if (occ.state != kResidentRecipeOccurrenceLive ||
        occ.lineage_kind != ResidentOccurrenceLineageKind::actual ||
        occ.authority != DirectParticipationAuthority::independent_external ||
        occ.participation_identity != sensory.identity ||
        occ.context_signature != sensory.context || occ.expiry_tick < current_tick)
      continue;
    if (match != nullptr) return nullptr;
    match = &occ;
  }
  return match;
}

DIRECT_LANGUAGE_ACTION_HD inline bool current_participation_eligibility_tail(
    const NodeCausalParticipation* active, std::uint32_t node_count,
    std::uint32_t node, std::uint64_t ticket, std::uint32_t current_tick,
    std::uint64_t* tail, std::uint32_t* expiry_tick) {
  if (active == nullptr || tail == nullptr || expiry_tick == nullptr ||
      node >= node_count || ticket == 0u) return false;
  const NodeCausalParticipation* base = active + node * kNodeParticipationAperture;
  std::uint64_t found_tail = 0u; std::uint32_t found_expiry = 0u;
  for (std::uint32_t i = 0u; i < kNodeParticipationAperture; ++i) {
    const auto& slot = base[i];
    if (slot.ticket_id != ticket || slot.expiry_tick < current_tick ||
        slot.current_drive == 0u || slot.authority == DirectParticipationAuthority::none)
      continue;
    const std::uint64_t candidate_tail = participation_eligibility_tail(slot);
    if (candidate_tail == 0u) continue;
    if (found_tail != 0u && found_tail != candidate_tail) return false;
    found_tail = candidate_tail;
    found_expiry = found_expiry == 0u || slot.expiry_tick < found_expiry
        ? slot.expiry_tick : found_expiry;
  }
  if (found_tail == 0u || found_expiry == 0u) return false;
  *tail = found_tail; *expiry_tick = found_expiry; return true;
}
DIRECT_LANGUAGE_ACTION_HD inline direct_causal_program::ProgramBankEntry*
admit_language_plan_as_action_candidate(
    const direct_network::DirectLanguageMotorPlan& plan,
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t record_count,
    const ResidentRecipeOccurrence* current_occurrences, std::uint32_t occurrence_count,
    const NodeCausalParticipation* active_participation, std::uint32_t node_count,
    std::uint32_t current_tick, DirectAdultActionControlRuntimeBlock* control,
    std::uint64_t* evicted_identity = nullptr) {
  if (control == nullptr || plan.admitted == 0u) return nullptr;
  const auto* sensory = latest_verified_sensory_contact(records, record_count);
  if (sensory == nullptr || sensory->source >= node_count) return nullptr;
  const auto* parent = current_actual_parent_for_sensory(
      current_occurrences, occurrence_count, *sensory, current_tick);
  if (parent == nullptr) return nullptr;
  std::uint64_t tail = 0u; std::uint32_t participation_expiry = 0u;
  if (!current_participation_eligibility_tail(active_participation, node_count,
          sensory->source, sensory->identity, current_tick, &tail, &participation_expiry))
    return nullptr;
  const std::uint32_t expiry = parent->expiry_tick < participation_expiry
      ? parent->expiry_tick : participation_expiry;
  auto program = direct_causal_program::lower_language_plan_to_causal_program(
      plan, parent->participation_identity, tail, expiry);
  if (program.identity == 0u) return nullptr;
  return direct_causal_program::admit_program(&control->programs, program, evicted_identity);
}

#undef DIRECT_LANGUAGE_ACTION_HD
}  // namespace substrate::direct_adult_core
#endif

#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_PARTICIPATION_BRIDGE_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_PARTICIPATION_BRIDGE_CUH

#include <cstdint>
#include "hardware_native/direct_causal_program.cuh"

namespace substrate::direct_causal_program {

#if defined(__CUDACC__)
#define DIRECT_CAUSAL_PROGRAM_BRIDGE_HD __host__ __device__
#else
#define DIRECT_CAUSAL_PROGRAM_BRIDGE_HD
#endif

template <typename ParticipationT>
DIRECT_CAUSAL_PROGRAM_BRIDGE_HD inline bool prepare_program_initiation_from_current_participation(
    const ParticipationT* slots, std::uint32_t slot_count,
    std::uint64_t source_eligibility_tail, std::uint32_t current_tick, Program* program) {
  if (slots == nullptr || program == nullptr || program->identity == 0u ||
      slot_count == 0u || source_eligibility_tail == 0u)
    return false;
  const ParticipationT* match = nullptr;
  for (std::uint32_t i = 0u; i < slot_count; ++i) {
    const auto& slot = slots[i];
    if (slot.ticket_id == 0u || slot.expiry_tick < current_tick ||
        slot.current_drive == 0u || static_cast<std::uint32_t>(slot.authority) == 0u)
      continue;
    if (match != nullptr && match->ticket_id != slot.ticket_id)
      return false;
    match = &slot;
  }
  if (match == nullptr) return false;
  program->initiation_participation_identity = match->ticket_id;
  program->initiation_parent_eligibility_ref = source_eligibility_tail;
  program->initiation_expiry_tick = match->expiry_tick;
  return true;
}

#undef DIRECT_CAUSAL_PROGRAM_BRIDGE_HD

}  // namespace substrate::direct_causal_program

#endif

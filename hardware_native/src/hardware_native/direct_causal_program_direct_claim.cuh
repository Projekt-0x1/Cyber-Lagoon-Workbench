#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_DIRECT_CLAIM_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_DIRECT_CLAIM_CUH

#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_causal_program_executor.cuh"

namespace substrate::direct_causal_program {

#if defined(__CUDACC__)
#define DIRECT_CAUSAL_DIRECT_CLAIM_HD __host__ __device__
#else
#define DIRECT_CAUSAL_DIRECT_CLAIM_HD
#endif

struct PreparedDirectDescendantClaim {
  std::uint64_t ticket_id;
  std::uint64_t program_identity;
  std::uint32_t target_node;
  std::uint32_t step_index;
  std::uint32_t expiry_tick;
  std::uint64_t parent_eligibility_ref;
  std::uint32_t authority_incarnation;
  std::uint32_t claim_incarnation;
  substrate::direct_adult_core::DirectParticipationAuthority causal_authority;
  substrate::direct_adult_core::DirectParticipationAuthority occurrence_authority;
  std::uint32_t valid;
};
static_assert(std::is_standard_layout_v<PreparedDirectDescendantClaim> &&
              std::is_trivial_v<PreparedDirectDescendantClaim>);

template <typename ParticipationT>
DIRECT_CAUSAL_DIRECT_CLAIM_HD inline bool prepare_direct_descendant_claim(
    const Program& program, const DueProgramCandidate& candidate,
    const ParticipationT& source, std::uint32_t current_tick,
    PreparedDirectDescendantClaim* out) {
  using substrate::direct_adult_core::DirectParticipationAuthority;
  if (out == nullptr) return false;
  *out = PreparedDirectDescendantClaim{};
  if (candidate.admitted == 0u || candidate.program_identity != program.identity ||
      candidate.participation_identity != program.initiation_participation_identity ||
      source.ticket_id != program.initiation_participation_identity ||
      program.initiation_parent_eligibility_ref == 0u ||
      source.ticket_id == 0u || source.current_drive == 0u ||
      source.expiry_tick < current_tick || program.initiation_expiry_tick < current_tick ||
      source.authority_incarnation == 0u || source.claim_incarnation == 0u ||
      source.authority != DirectParticipationAuthority::independent_external)
    return false;
  out->ticket_id = source.ticket_id;
  out->program_identity = program.identity;
  out->target_node = candidate.step.node;
  out->step_index = candidate.step_index;
  out->expiry_tick = source.expiry_tick < program.initiation_expiry_tick
                         ? source.expiry_tick
                         : program.initiation_expiry_tick;
  out->parent_eligibility_ref = program.initiation_parent_eligibility_ref;
  out->authority_incarnation = source.authority_incarnation;
  out->claim_incarnation = source.claim_incarnation;
  out->causal_authority = source.authority;
  out->occurrence_authority = DirectParticipationAuthority::resident_external_descendant;
  out->valid = 1u;
  return true;
}

#undef DIRECT_CAUSAL_DIRECT_CLAIM_HD

}  // namespace substrate::direct_causal_program

#endif

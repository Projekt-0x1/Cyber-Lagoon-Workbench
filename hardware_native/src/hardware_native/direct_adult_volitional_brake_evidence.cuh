#ifndef HARDWARE_NATIVE_DIRECT_ADULT_VOLITIONAL_BRAKE_EVIDENCE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_VOLITIONAL_BRAKE_EVIDENCE_CUH
#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_adult_volitional_veto.cuh"
namespace substrate::direct_adult_core {
enum class ResidentBrakeEvidenceStatus : std::uint32_t { none=0u, unique=1u, ambiguous=2u, malformed=3u };
#if defined(__CUDACC__)
#define DIRECT_BRAKE_EVIDENCE_HD __host__ __device__
#else
#define DIRECT_BRAKE_EVIDENCE_HD
#endif
DIRECT_BRAKE_EVIDENCE_HD inline std::uint32_t volitional_mismatch_magnitude_q16(std::int32_t delta) {
  std::uint64_t magnitude=delta<0?static_cast<std::uint64_t>(-(static_cast<std::int64_t>(delta))):static_cast<std::uint64_t>(delta);
  return magnitude>direct_network::kVolitionalQ16One?direct_network::kVolitionalQ16One:static_cast<std::uint32_t>(magnitude);
}
template <typename Frontier>
DIRECT_BRAKE_EVIDENCE_HD inline ResidentBrakeEvidenceStatus resident_mismatch_brake_evidence_status(
    const Frontier* frontier,std::uint64_t parent_occurrence_identity,
    std::uint64_t participation_identity,std::uint32_t context_signature,
    std::uint32_t current_tick,direct_network::ResidentVolitionalBrakeEvidence* out) {
  if(out==nullptr||frontier==nullptr||parent_occurrence_identity==0u||participation_identity==0u)return ResidentBrakeEvidenceStatus::malformed;
  constexpr std::uint32_t capacity=sizeof(frontier->receipts)/sizeof(frontier->receipts[0]);
  if(frontier->committed_receipt_count>frontier->receipt_count||frontier->receipt_count>capacity)return ResidentBrakeEvidenceStatus::malformed;
  using Receipt=std::remove_reference_t<decltype(frontier->receipts[0])>;
  const Receipt* match=nullptr;
  for(std::uint32_t i=0;i<frontier->committed_receipt_count;++i){const auto& r=frontier->receipts[i];
    if(r.identity==0u||r.committed_revision_identity==0u||r.target_occurrence_identity!=parent_occurrence_identity||
       r.target_participation_identity!=participation_identity||r.target_context_signature!=context_signature||r.resident_tick>current_tick)continue;
    if(match==nullptr||r.resident_tick>match->resident_tick)match=&r;
    else if(r.resident_tick==match->resident_tick&&r.identity!=match->identity)return ResidentBrakeEvidenceStatus::ambiguous;
  }
  if(match==nullptr)return ResidentBrakeEvidenceStatus::none;
  direct_network::ResidentVolitionalBrakeEvidence e{};e.evidence_identity=match->identity;
  e.parent_occurrence_identity=match->target_occurrence_identity;e.participation_identity=match->target_participation_identity;
  e.context_signature=match->target_context_signature;e.resident_tick=match->resident_tick;
  e.mismatch_magnitude_q16=volitional_mismatch_magnitude_q16(match->causal_credit_delta_q16);
  e.mismatch_kind=static_cast<std::uint32_t>(match->kind);*out=e;return ResidentBrakeEvidenceStatus::unique;
}

template <typename Frontier>
DIRECT_BRAKE_EVIDENCE_HD inline bool resident_mismatch_brake_evidence(
    const Frontier* frontier,std::uint64_t parent_occurrence_identity,
    std::uint64_t participation_identity,std::uint32_t context_signature,
    std::uint32_t current_tick,direct_network::ResidentVolitionalBrakeEvidence* out) {
  return resident_mismatch_brake_evidence_status(frontier,parent_occurrence_identity,participation_identity,context_signature,current_tick,out)==ResidentBrakeEvidenceStatus::unique;
}
#undef DIRECT_BRAKE_EVIDENCE_HD
}
#endif

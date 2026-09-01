#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_VOLITIONAL_BRIDGE_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_VOLITIONAL_BRIDGE_CUH
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_volitional_veto.cuh"
#include "hardware_native/direct_causal_program_competition.cuh"
#include "hardware_native/direct_exact_history.cuh"
namespace substrate::direct_adult_core {
#if defined(__CUDACC__)
#define DIRECT_PROGRAM_VOLITION_HD __host__ __device__
#else
#define DIRECT_PROGRAM_VOLITION_HD
#endif
DIRECT_PROGRAM_VOLITION_HD inline const direct_causal_program::ProgramBankEntry*
competition_leader_entry(const direct_causal_program::ProgramBank& bank,
                         const direct_causal_program::ProgramCompetitionReceipt& competition) {
  if (competition.decision != direct_causal_program::ProgramCompetitionDecision::unique_leader ||
      competition.leader_identity == 0u) return nullptr;
  for (std::uint32_t i=0;i<direct_causal_program::kProgramBankCapacity;++i)
    if (bank.entries[i].occupied!=0u && bank.entries[i].program.identity==competition.leader_identity)
      return &bank.entries[i];
  return nullptr;
}
DIRECT_PROGRAM_VOLITION_HD inline const ResidentRecipeOccurrence*
current_root_for_program(const direct_causal_program::Program& program,
                         const ResidentRecipeOccurrence* occurrences,
                         std::uint32_t count,std::uint32_t current_tick) {
  if (occurrences==nullptr || program.initiation_participation_identity==0u) return nullptr;
  const ResidentRecipeOccurrence* match=nullptr;
  for(std::uint32_t i=0;i<count;++i){const auto& o=occurrences[i];
    if(o.state!=kResidentRecipeOccurrenceLive || o.lineage_kind!=ResidentOccurrenceLineageKind::actual ||
       o.authority!=DirectParticipationAuthority::independent_external ||
       o.participation_identity!=program.initiation_participation_identity ||
       o.occurrence_identity==0u || o.revision_identity==0u || o.expiry_tick<current_tick) continue;
    if(match!=nullptr) return nullptr;
    match=&o;
  }
  return match;
}
DIRECT_PROGRAM_VOLITION_HD inline bool competition_leader_to_prospective(
    const direct_causal_program::ProgramBank& bank,
    const direct_causal_program::ProgramCompetitionReceipt& competition,
    const ResidentRecipeOccurrence* occurrences,std::uint32_t occurrence_count,
    std::uint64_t prediction_identity,std::uint32_t current_tick,
    direct_network::ResidentProspectiveTrajectory* out) {
  using namespace direct_network;
  if(out==nullptr || prediction_identity==0u || current_tick==0u)
    return false;
  const auto* entry=competition_leader_entry(bank,competition); if(entry==nullptr)return false;
  const auto* root=current_root_for_program(entry->program,occurrences,occurrence_count,current_tick);
  if(root==nullptr || !direct_causal_program::initiation_current(entry->program,root->participation_identity,current_tick))return false;
  std::uint64_t trajectory=direct_network::exact_history_fold_word(entry->program.identity,prediction_identity);
  trajectory=direct_network::exact_history_fold_word(trajectory,root->occurrence_identity);
  trajectory=direct_network::exact_history_fold_word(trajectory,root->context_signature);
  if(trajectory==0u)trajectory=entry->program.identity;
  ResidentProspectiveTrajectory p{};p.trajectory_identity=trajectory;p.prediction_identity=prediction_identity;
  p.parent_occurrence_identity=root->occurrence_identity;p.parent_revision_identity=root->revision_identity;
  p.participation_identity=root->participation_identity;p.context_signature=root->context_signature;
  p.generation_tick=current_tick;*out=p;return true;
}
#undef DIRECT_PROGRAM_VOLITION_HD
}
#endif

#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_COMMIT_PHASE_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_COMMIT_PHASE_CUH
#include "hardware_native/direct_adult_action_control_runtime.cuh"
#include "hardware_native/direct_adult_volitional_brake_evidence.cuh"
#include "hardware_native/direct_causal_program_competition.cuh"
#include "hardware_native/direct_causal_program_volitional_bridge.cuh"
namespace substrate::direct_adult_core {
#if defined(__CUDACC__)
#define DIRECT_PROGRAM_COMMIT_HD __host__ __device__
#else
#define DIRECT_PROGRAM_COMMIT_HD
#endif
struct DirectProgramCommitPhaseReceipt {std::uint32_t actionable_roots;std::uint32_t leaders;std::uint64_t leader_program_identity;std::uint64_t prediction_identity;direct_network::ResidentVolitionalDecision decision;};
static_assert(std::is_trivial_v<DirectProgramCommitPhaseReceipt> && std::is_standard_layout_v<DirectProgramCommitPhaseReceipt>);
DIRECT_PROGRAM_COMMIT_HD inline bool action_commitment_empty(const direct_network::ResidentPersistentCommitment& c){return c.generation==0u&&c.trajectory_identity==0u&&c.prediction_identity==0u&&c.parent_occurrence_identity==0u&&c.participation_identity==0u;}
template <typename MismatchFrontier>
DIRECT_PROGRAM_COMMIT_HD inline bool resident_program_competition_commit_phase(
    DirectAdultActionControlRuntimeBlock* control,
    const ResidentRecipeOccurrence* occurrences,std::uint32_t occurrence_count,
    const MismatchFrontier* mismatch,std::uint32_t current_tick,
    DirectProgramCommitPhaseReceipt* out) {
  if(control==nullptr||out==nullptr||occurrences==nullptr||occurrence_count==0u||current_tick==0u)return false;
  DirectProgramCommitPhaseReceipt receipt{};
  if(!action_commitment_empty(control->commitment)){receipt.decision=direct_network::ResidentVolitionalDecision::refused;*out=receipt;return true;}
  direct_causal_program::ProgramCompetitionReceipt selected{};
  const ResidentRecipeOccurrence* selected_root=nullptr;
  const direct_causal_program::ProgramBankEntry* selected_entry=nullptr;
  for(std::uint32_t i=0;i<occurrence_count;++i){const auto& root=occurrences[i];
    if(root.state!=kResidentRecipeOccurrenceLive||root.lineage_kind!=ResidentOccurrenceLineageKind::actual||
       root.authority!=DirectParticipationAuthority::independent_external||root.occurrence_identity==0u||
       root.revision_identity==0u||root.participation_identity==0u||root.context_signature==0u||root.expiry_tick<current_tick)continue;
    direct_causal_program::CurrentState state{};
    const auto competition=direct_causal_program::arbitrate_program_bank(&control->programs,root.context_signature,root.participation_identity,current_tick,state);
    if(competition.decision!=direct_causal_program::ProgramCompetitionDecision::unique_leader)continue;
    ++receipt.actionable_roots;++receipt.leaders;
    if(selected_root!=nullptr){receipt.decision=direct_network::ResidentVolitionalDecision::refused;*out=receipt;return true;}
    selected=competition;selected_root=&root;selected_entry=competition_leader_entry(control->programs,competition);
  }
  if(selected_root==nullptr||selected_entry==nullptr){receipt.decision=direct_network::ResidentVolitionalDecision::refused;*out=receipt;return true;}
  const std::uint64_t prediction=direct_causal_program::program_prediction_identity(*selected_entry,selected_root->context_signature);
  if(prediction==0u){receipt.decision=direct_network::ResidentVolitionalDecision::refused;*out=receipt;return true;}
  direct_network::ResidentProspectiveTrajectory prospective{};
  if(!competition_leader_to_prospective(control->programs,selected,occurrences,occurrence_count,prediction,current_tick,&prospective)){
    receipt.decision=direct_network::ResidentVolitionalDecision::refused;*out=receipt;return true;
  }
  direct_network::ResidentVolitionalBrakeEvidence brake{};
  const auto brake_status=mismatch==nullptr?ResidentBrakeEvidenceStatus::none:
      resident_mismatch_brake_evidence_status(mismatch,selected_root->occurrence_identity,
          selected_root->participation_identity,selected_root->context_signature,current_tick,&brake);
  if(brake_status==ResidentBrakeEvidenceStatus::ambiguous||brake_status==ResidentBrakeEvidenceStatus::malformed){
    receipt.decision=direct_network::ResidentVolitionalDecision::refused;*out=receipt;return true;
  }
  direct_network::ResidentVolitionalVetoReceipt gate{};
  const auto* brake_ptr=brake_status==ResidentBrakeEvidenceStatus::unique?&brake:nullptr;
  if(!direct_network::resident_volitional_precommit_gate(
      prospective,prediction,selected_root->occurrence_identity,selected_root->revision_identity,
      selected_root->participation_identity,selected_root->context_signature,brake_ptr,
      direct_network::kVolitionalQ16One/2u,current_tick,&control->commitment,&gate))return false;
  receipt.leader_program_identity=selected.leader_identity;receipt.prediction_identity=prediction;
  receipt.decision=gate.decision;*out=receipt;return true;
}
#undef DIRECT_PROGRAM_COMMIT_HD
}
#endif

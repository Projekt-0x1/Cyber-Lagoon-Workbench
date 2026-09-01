#include <cstdio>
#include <vector>
#include "hardware_native/direct_adult_source_epistemics.cuh"
using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;
int main(){
  constexpr std::uint64_t claim_ticket=42, assertion_id=100, action_ticket=200;
  constexpr std::uint32_t root_channel=7, motor_word=123;
  DirectExactHistoryRecord records[3]{};
  records[0].sequence=1; records[0].identity=claim_ticket; records[0].resident_tick=1; records[0].event_tick=1;
  records[0].kind=DirectExactHistoryKind::sensory_contact; records[0].source=0; records[0].subject=root_channel;
  records[0].value=9; records[0].context=13; records[0].flags=kDirectHistoryVerifiedObservation;
  stage_source_assertion_history_record(&records[1],assertion_id,claim_ticket,2,2,500,600,0x200000001ull,0x400000003ull,true); records[1].sequence=2;
  records[2]=records[1]; records[2].sequence=3; records[2].identity=101; records[2].parent_identity=assertion_id;
  records[2].resident_tick=3; records[2].event_tick=3; records[2].flags=static_cast<std::uint32_t>(DirectSourceAssertionIntegrity::withdrawn);
  ResidentPostbirthConstructorState state{};
  DirectCausalWorldModel model{}; model.model_identity=0xabc; model.relation_count=1; model.relations[0].action_value=motor_word;
  model.relations[0].root_channel=root_channel; model.relations[0].observations=kCausalModelMinimumSupport;
  model.relations[0].current_outcome_value=88; model.relations[0].current_outcome_tick=4;
  std::vector<AsynchronousTicket> tickets(kMaxAsynchronousTickets); std::vector<DirectActionOccurrence> actions(kMaxAsynchronousTickets);
  constexpr std::uint32_t cap=2; std::vector<DirectActionParticipationLink> links(kMaxAsynchronousTickets*cap);
  const std::uint32_t slot=static_cast<std::uint32_t>(action_ticket & (kMaxAsynchronousTickets-1u));
  tickets[slot].ticket_id=action_ticket; tickets[slot].settled=1; tickets[slot].motor_word=motor_word; tickets[slot].emission_tick=4;
  actions[slot].action_ticket_id=action_ticket; actions[slot].state=kActionOccurrenceSettled; actions[slot].participant_offset=0; actions[slot].participant_count=1; actions[slot].emission_tick=4;
  links[0].participant_ticket_id=claim_ticket;
  DirectSourceWithdrawnRelationReceipt receipt{};
  const bool ok=reconstruct_source_withdrawn_relation(records,3,state,model,tickets.data(),actions.data(),links.data(),cap,assertion_id,&receipt);
  const bool receipt_ok=ok && receipt.source_identity==500 && receipt.assertion_contact_identity==assertion_id && receipt.action_ticket_id==action_ticket && receipt.causal_model_identity==model.model_identity && receipt.action_value==motor_word && receipt.root_channel==root_channel && receipt.current_outcome_value==88 && receipt.observations==kCausalModelMinimumSupport;
  DirectCausalWorldModel lesion=model; lesion.relation_count=0; DirectSourceWithdrawnRelationReceipt gone{};
  const bool lesion_refuse=!reconstruct_source_withdrawn_relation(records,3,state,lesion,tickets.data(),actions.data(),links.data(),cap,assertion_id,&gone);
  std::printf("FOUNDRY_SOURCE_WITHDRAWN_JOIN_HOST %s join=%u lesion_refuse=%u model=%llu action=%llu\n",receipt_ok&&lesion_refuse?"GREEN":"RED",receipt_ok,lesion_refuse,(unsigned long long)receipt.causal_model_identity,(unsigned long long)receipt.action_ticket_id);
  return receipt_ok&&lesion_refuse?0:1;
}

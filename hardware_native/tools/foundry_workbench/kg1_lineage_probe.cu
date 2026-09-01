#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "direct_canonical_developmental_language_fixture.cuh"
#include "hardware_native/direct_adult_source_epistemics.cuh"

using namespace substrate::direct_test;
using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

namespace {
struct ProductionContact { ResidentContactEpochReceipt receipt; ActivityEvent event; };
ProductionContact inject_receipt(Subject* subject,const DirectBoundaryPort& port,std::uint64_t ticket,std::uint32_t word){
  ActivityEvent event{}; event.ticket_id=ticket;event.node=port.node;event.channel=port.channel;event.word=word;event.origin=CausalOrigin::external_contact;event.context=port.physical_route;event.timestamp=subject->runtime->current_tick;
  const auto ports=read_many(subject->brain.boundary_ports,subject->brain.boundary_port_count);std::uint32_t matched=substrate::direct_adult_core::kInvalidIndex;
  for(std::uint32_t i=0;i<ports.size();++i)if(ports[i].node==event.node&&ports[i].channel==event.channel&&ports[i].physical_route==event.context&&(ports[i].role_mask&static_cast<std::uint32_t>(BoundaryRole::sensor))){if(matched!=substrate::direct_adult_core::kInvalidIndex)throw std::runtime_error("ambiguous port");matched=i;}
  require(matched!=substrate::direct_adult_core::kInvalidIndex,"no port");auto receipt=make_resident_contact_epoch_receipt(*subject->runtime->brain,ports[matched],matched,event,static_cast<std::uint64_t>(subject->runtime->host_ingress_write_tail)+1u);require(inject_actual_sensory_contact(subject->runtime,event,receipt),"inject");return {receipt,event};
}
DirectExactHistoryHotPage read_history(const Subject& s){DirectExactHistoryHotPage p{};cuda_require(cudaMemcpy(&p,&s.brain.development->exact_history,sizeof(p),cudaMemcpyDeviceToHost),"read history");return p;}
void write_history(Subject* s,const DirectExactHistoryHotPage& p){cuda_require(cudaMemcpy(&s->brain.development->exact_history,&p,sizeof(p),cudaMemcpyHostToDevice),"write history");}
}
int main(){
  int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
  try{
    Subject adult=birth_subject();constexpr std::uint64_t ticket=0xabc991ull;constexpr std::uint32_t word=0x00018000u;
    auto pc=inject_receipt(&adult,adult.sensor,ticket,word);MotorEvent motor=wait_motor(&adult,ticket,true);
    auto page=read_history(adult);DirectSourceAssertionClaim claim{};require(commit_source_assertion_to_history(&page,pc.receipt,ticket,adult.runtime->current_tick,&claim),"commit assertion");write_history(&adult,page);
    const Word returned=word<=UINT32_MAX-static_cast<Word>(substrate::direct_adult_core::kQ16One/16)?word+static_cast<Word>(substrate::direct_adult_core::kQ16One/16):word-static_cast<Word>(substrate::direct_adult_core::kQ16One/16);
    require(inject_raw_reafferent_contact(adult.runtime,motor.ticket_id,returned),"return");
    const ResidentRawContactKey key=resident_raw_contact_key(pc.event.node,pc.event.channel,pc.event.word,pc.event.context);
    const std::uint32_t derivation_index=require_experience_incidence(adult,key);const auto deriv=read_one(adult.brain.postbirth_derivations+derivation_index);const auto recipe=read_one(adult.brain.recipe_cells+deriv.recipe_cell);
    page=read_history(adult);const bool ticket_ancestry=exact_history_descends_from(page.records,page.committed_slots,deriv.witness_identity,ticket);const bool assertion_ancestry=exact_history_descends_from(page.records,page.committed_slots,deriv.witness_identity,claim.contact_identity);
    std::printf("KG1_LINEAGE ticket=%llu claim_contact=%llu witness=%llu ticket_ancestry=%u assertion_ancestry=%u authority=%u revision=%llu\n",(unsigned long long)ticket,(unsigned long long)claim.contact_identity,(unsigned long long)deriv.witness_identity,ticket_ancestry?1u:0u,assertion_ancestry?1u:0u,(unsigned)resident_recipe_current_revision_authority(recipe),(unsigned long long)recipe.revision_identity);
    destroy(&adult);return ticket_ancestry?0:1;
  }catch(const std::exception& e){std::fprintf(stderr,"KG1_LINEAGE RED %s\n",e.what());return 1;}
}

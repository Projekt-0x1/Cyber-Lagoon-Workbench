#include <cstdint>
#include <cstdio>
#include "hardware_native/direct_adult_core.cuh"
namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_node_participation_types.inl"
}
#include "hardware_native/direct_causal_program_participation_bridge.cuh"
using namespace substrate::direct_causal_program;
using substrate::direct_adult_core::DirectParticipationAuthority;
using substrate::direct_adult_core::NodeCausalParticipation;

static NodeCausalParticipation slot(std::uint64_t ticket,std::uint32_t expiry,
                                    std::uint32_t drive,DirectParticipationAuthority authority){
  NodeCausalParticipation s{};s.ticket_id=ticket;s.expiry_tick=expiry;s.current_drive=drive;s.authority=authority;return s;
}
int main(){
  Program p{};p.identity=0x9001;p.depth=2;p.step_count=5;
  const bool bare_public=initiation_current(p,0,100);
  NodeCausalParticipation one[2]={slot(0xabc,120,1,DirectParticipationAuthority::independent_external),slot(0,0,0,DirectParticipationAuthority::none)};
  Program live=p;const bool prepared=prepare_program_initiation_from_current_participation(one,2,(5ull<<32)|4ull,100,&live);const bool live_public=prepared&&initiation_current(live,0xabc,100);
  NodeCausalParticipation amb[2]={slot(0xabc,120,1,DirectParticipationAuthority::independent_external),slot(0xdef,120,1,DirectParticipationAuthority::independent_external)};Program ambiguous=p;const bool amb_prepared=prepare_program_initiation_from_current_participation(amb,2,(5ull<<32)|4ull,100,&ambiguous);
  NodeCausalParticipation stale[1]={slot(0xabc,99,1,DirectParticipationAuthority::independent_external)};Program old=p;const bool stale_prepared=prepare_program_initiation_from_current_participation(stale,1,(5ull<<32)|4ull,100,&old);
  NodeCausalParticipation unauth[1]={slot(0xabc,120,1,DirectParticipationAuthority::none)};Program noauth=p;const bool unauth_prepared=prepare_program_initiation_from_current_participation(unauth,1,(5ull<<32)|4ull,100,&noauth);
  Program zero=p;const bool zero_tail=prepare_program_initiation_from_current_participation(one,2,0u,100,&zero);const bool green=!bare_public&&live_public&&!amb_prepared&&!stale_prepared&&!unauth_prepared&&!zero_tail&&live.initiation_parent_eligibility_ref==((5ull<<32)|4ull);
  std::printf("FOUNDRY_DIRECT_CAUSAL_PROGRAM_PARTICIPATION_%s direct_node_abi=1 baseline_public_steps=0 prepared_public_steps=5 prepared_bytes=79 prepared_depth=2 ambiguous=0 stale=0 unauth=0\n",green?"GREEN":"RED");return green?0:1;
}

#include <cstdint>
#include <cstdio>
#include "hardware_native/direct_adult_core.cuh"
namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_node_participation_types.inl"
}
#include "hardware_native/direct_causal_program_direct_claim.cuh"
using namespace substrate::direct_causal_program;
using namespace substrate::direct_adult_core;
static NodeCausalParticipation source(){NodeCausalParticipation s{};s.ticket_id=0xabc;s.expiry_tick=118;s.current_drive=1;s.authority=DirectParticipationAuthority::independent_external;s.authority_incarnation=77;s.claim_incarnation=88;return s;}
static Program program(){Program p{};p.identity=0xCA11;p.initiation_participation_identity=0xabc;p.initiation_parent_eligibility_ref=(5ull<<32)|4ull;p.initiation_expiry_tick=120;p.depth=2;p.step_count=5;for(unsigned i=0;i<5;++i){p.steps[i].node=300+i;p.steps[i].channel=20+i;p.steps[i].due_offset=i;}return p;}
int main(){auto p=program();ProgramExecutionState e{};begin_execution(&e,p,0xabc,100);auto s=source();unsigned descendants=0;bool no_external=true,lineage=true;for(unsigned i=0;i<5;++i){auto c=due_candidate(e,100+i,0xabc);PreparedDirectDescendantClaim claim{};const bool ok=prepare_direct_descendant_claim(p,c,s,100+i,&claim);if(ok){++descendants;no_external&=claim.causal_authority==DirectParticipationAuthority::independent_external&&claim.occurrence_authority==DirectParticipationAuthority::resident_external_descendant;lineage&=claim.ticket_id==s.ticket_id&&claim.authority_incarnation==s.authority_incarnation&&claim.claim_incarnation==s.claim_incarnation&&claim.expiry_tick==118&&claim.parent_eligibility_ref==((5ull<<32)|4ull)&&claim.target_node==c.step.node;confirm_emitted_step(&e,c,100+i,c.step.node,c.step.channel);}}auto bad=s;bad.authority_incarnation=0;ProgramExecutionState b{};begin_execution(&b,p,0xabc,100);auto bc=due_candidate(b,100,0xabc);PreparedDirectDescendantClaim refused{};const bool forged=prepare_direct_descendant_claim(p,bc,bad,100,&refused);const bool green=descendants==5&&e.completed&&no_external&&lineage&&!forged;std::printf("FOUNDRY_DIRECT_CAUSAL_PROGRAM_DESCENDANT_%s baseline_public_steps=0 descendant_steps=%u bytes=79 depth=2 external_evidence_steps=0 lineage=%u forged=0\n",green?"GREEN":"RED",descendants,lineage?1u:0u);return green?0:1;}

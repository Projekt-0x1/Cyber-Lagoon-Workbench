#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "hardware_native/direct_adult_core.cuh"
namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_node_participation_types.inl"
}
#include "hardware_native/direct_causal_program_direct_claim.cuh"
using namespace substrate::direct_causal_program;
using namespace substrate::direct_adult_core;
struct Receipt{std::uint32_t green,descendants,completed,no_external,lineage,forged;};
__device__ NodeCausalParticipation source(){NodeCausalParticipation s{};s.ticket_id=0xabc;s.expiry_tick=118;s.current_drive=1;s.authority=DirectParticipationAuthority::independent_external;s.authority_incarnation=77;s.claim_incarnation=88;return s;}
__device__ Program program(){Program p{};p.identity=0xCA11;p.initiation_participation_identity=0xabc;p.initiation_parent_eligibility_ref=(5ull<<32)|4ull;p.initiation_expiry_tick=120;p.depth=2;p.step_count=5;for(std::uint32_t i=0;i<5;++i){p.steps[i].node=300+i;p.steps[i].channel=20+i;p.steps[i].due_offset=i;}return p;}
__global__ void run(Receipt* r){if(blockIdx.x||threadIdx.x)return;auto p=program();auto s=source();ProgramExecutionState e{};begin_execution(&e,p,0xabc,100);r->no_external=1;r->lineage=1;for(std::uint32_t i=0;i<5;++i){auto c=due_candidate(e,100+i,0xabc);PreparedDirectDescendantClaim claim{};if(prepare_direct_descendant_claim(p,c,s,100+i,&claim)){++r->descendants;r->no_external&=claim.causal_authority==DirectParticipationAuthority::independent_external&&claim.occurrence_authority==DirectParticipationAuthority::resident_external_descendant;r->lineage&=claim.ticket_id==s.ticket_id&&claim.authority_incarnation==s.authority_incarnation&&claim.claim_incarnation==s.claim_incarnation&&claim.expiry_tick==118&&claim.parent_eligibility_ref==((5ull<<32)|4ull)&&claim.target_node==c.step.node;confirm_emitted_step(&e,c,100+i,c.step.node,c.step.channel);}}r->completed=e.completed;auto bad=s;bad.authority_incarnation=0;ProgramExecutionState b{};begin_execution(&b,p,0xabc,100);auto bc=due_candidate(b,100,0xabc);PreparedDirectDescendantClaim claim{};r->forged=prepare_direct_descendant_claim(p,bc,bad,100,&claim);r->green=r->descendants==5&&r->completed&&r->no_external&&r->lineage&&!r->forged;}
int main(){Receipt *d=nullptr,h{};if(cudaMalloc(&d,sizeof(Receipt))!=cudaSuccess)return 2;cudaMemset(d,0,sizeof(Receipt));run<<<1,1>>>(d);if(cudaDeviceSynchronize()!=cudaSuccess)return 3;cudaMemcpy(&h,d,sizeof(h),cudaMemcpyDeviceToHost);cudaFree(d);std::printf("FOUNDRY_DIRECT_CAUSAL_PROGRAM_DESCENDANT_CUDA_%s baseline_public_steps=0 descendant_steps=%u bytes=79 depth=2 completed=%u external_evidence_steps=0 lineage=%u forged=0\n",h.green?"GREEN":"RED",h.descendants,h.completed,h.lineage);return h.green?0:1;}

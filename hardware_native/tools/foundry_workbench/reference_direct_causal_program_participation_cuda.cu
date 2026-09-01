#include <cuda_runtime.h>
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
struct Receipt{std::uint32_t green,live,ambiguous,stale,unauth;};
__device__ NodeCausalParticipation slot(std::uint64_t ticket,std::uint32_t expiry,std::uint32_t drive,DirectParticipationAuthority authority){NodeCausalParticipation s{};s.ticket_id=ticket;s.expiry_tick=expiry;s.current_drive=drive;s.authority=authority;return s;}
__global__ void run(Receipt* r){if(blockIdx.x||threadIdx.x)return;Program p{};p.identity=0x9001;p.depth=2;p.step_count=5;NodeCausalParticipation one[2]={slot(0xabc,120,1,DirectParticipationAuthority::independent_external),slot(0,0,0,DirectParticipationAuthority::none)};Program live=p;r->live=prepare_program_initiation_from_current_participation(one,2,(5ull<<32)|4ull,100,&live)&&initiation_current(live,0xabc,100);NodeCausalParticipation amb[2]={slot(0xabc,120,1,DirectParticipationAuthority::independent_external),slot(0xdef,120,1,DirectParticipationAuthority::independent_external)};Program a=p;r->ambiguous=prepare_program_initiation_from_current_participation(amb,2,(5ull<<32)|4ull,100,&a);NodeCausalParticipation oldslot[1]={slot(0xabc,99,1,DirectParticipationAuthority::independent_external)};Program old=p;r->stale=prepare_program_initiation_from_current_participation(oldslot,1,(5ull<<32)|4ull,100,&old);NodeCausalParticipation none[1]={slot(0xabc,120,1,DirectParticipationAuthority::none)};Program no=p;r->unauth=prepare_program_initiation_from_current_participation(none,1,(5ull<<32)|4ull,100,&no);Program zero=p;const bool zero_tail=prepare_program_initiation_from_current_participation(one,2,0u,100,&zero);r->green=r->live&&!r->ambiguous&&!r->stale&&!r->unauth&&!zero_tail&&live.initiation_parent_eligibility_ref==((5ull<<32)|4ull);}
int main(){Receipt *d=nullptr,h{};if(cudaMalloc(&d,sizeof(Receipt))!=cudaSuccess)return 2;cudaMemset(d,0,sizeof(Receipt));run<<<1,1>>>(d);if(cudaDeviceSynchronize()!=cudaSuccess)return 3;cudaMemcpy(&h,d,sizeof(h),cudaMemcpyDeviceToHost);cudaFree(d);std::printf("FOUNDRY_DIRECT_CAUSAL_PROGRAM_PARTICIPATION_CUDA_%s direct_node_abi=1 baseline_public_steps=0 prepared_public_steps=5 prepared_bytes=79 prepared_depth=2 live=%u ambiguous=%u stale=%u unauth=%u\n",h.green?"GREEN":"RED",h.live,h.ambiguous,h.stale,h.unauth);return h.green?0:1;}

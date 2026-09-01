#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_relational_sequence_bridge.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

__global__ void bridge_eval_kernel(const DirectRelSeqRecipe* recipes, unsigned rn,
                                   const DirectRelSeqOccurrence* occ, unsigned on,
                                   unsigned long long root, DirectRelSeqOutput* out,
                                   int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  *ok = direct_relseq_evaluate_occurrence(recipes, rn, occ, on, root, out) ? 1 : 0;
}
static void ck(cudaError_t e, const char* where) {
  if (e != cudaSuccess) { std::fprintf(stderr,"CUDA %s: %s\n",where,cudaGetErrorString(e)); std::exit(20); }
}
static ResidentRelationalSequenceBody body(std::uint64_t logical, std::uint64_t revision,
                                           std::uint64_t relation, bool join) {
  ResidentRelationalSequenceBody b{};
  b.logical_recipe_id=logical; b.revision_identity=revision; b.relation_identity=relation;
  b.support=1; b.active=1;
  if (!join) {
    b.port_count=2; b.piece_count=5;
    b.pieces[0]={101u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::fixed_unit),0u};
    b.pieces[1]={0u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::port),0u};
    b.pieces[2]={102u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::fixed_unit),0u};
    b.pieces[3]={1u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::port),0u};
    b.pieces[4]={103u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::fixed_unit),0u};
  } else {
    b.port_count=2; b.piece_count=3;
    b.pieces[0]={0u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::port),0u};
    b.pieces[1]={104u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::fixed_unit),0u};
    b.pieces[2]={1u,static_cast<std::uint16_t>(DirectRelSeqPieceKind::port),0u};
  }
  return b;
}
static bool make_occ(const ResidentRecipeCell& cell, std::uint64_t oid,
                     std::uint32_t a, std::uint32_t b, ResidentRecipeOccurrence* out) {
  ResidentRecipeDerivation d{}; d.logical_recipe_id=cell.logical_recipe_id;
  d.revision_identity=cell.revision_identity; d.port_count=2;
  std::uint32_t vars[2]={a,b};
  return bind_resident_recipe_occurrence(cell,d,vars,2,oid,oid+1000,77,1,
      ResidentOccurrenceLineageKind::endogenous,DirectParticipationAuthority::none,
      9,1,100,0,out);
}
int main(){
  ResidentRecipeCell cells[3]{};
  cells[0].logical_recipe_id=0xA1;cells[0].revision_identity=0xB1;
  cells[1].logical_recipe_id=0xA1;cells[1].revision_identity=0xB1;
  cells[2].logical_recipe_id=0xA2;cells[2].revision_identity=0xB2;
  ResidentRecipeOccurrence canonical[3]{};
  if(!make_occ(cells[0],0xC1,11,12,&canonical[0])) return 2;
  if(!make_occ(cells[1],0xC2,21,22,&canonical[1])) return 3;
  if(!make_occ(cells[2],0xC3,31,32,&canonical[2])) return 4;
  const auto leaf_body=body(0xA1,0xB1,0xD1,false);
  const auto join_body=body(0xA2,0xB2,0xD2,true);
  ResidentRelationalSequenceOccurrenceSidecar leaf_sc0{};leaf_sc0.occurrence_identity=0xC1;
  ResidentRelationalSequenceOccurrenceSidecar leaf_sc1{};leaf_sc1.occurrence_identity=0xC2;
  ResidentRelationalSequenceOccurrenceSidecar join_sc{};join_sc.occurrence_identity=0xC3;
  join_sc.child_occurrence_identities[0]=0xC1;join_sc.child_occurrence_identities[1]=0xC2;
  DirectRelSeqRecipe lowered_recipes[2]{}; DirectRelSeqOccurrence lowered_occ[3]{};
  if(!lower_canonical_resident_relseq(cells[0],canonical[0],leaf_body,&leaf_sc0,&lowered_recipes[0],&lowered_occ[0])) return 5;
  DirectRelSeqRecipe duplicate_leaf{};
  if(!lower_canonical_resident_relseq(cells[1],canonical[1],leaf_body,&leaf_sc1,&duplicate_leaf,&lowered_occ[1])) return 6;
  if(duplicate_leaf.revision_identity!=lowered_recipes[0].revision_identity) return 7;
  if(!lower_canonical_resident_relseq(cells[2],canonical[2],join_body,&join_sc,&lowered_recipes[1],&lowered_occ[2])) return 8;
  DirectRelSeqRecipe *dr=nullptr; DirectRelSeqOccurrence *doo=nullptr; DirectRelSeqOutput *dout=nullptr; int *dok=nullptr;
  ck(cudaMalloc(&dr,sizeof(lowered_recipes)),"malloc recipe");ck(cudaMalloc(&doo,sizeof(lowered_occ)),"malloc occ");ck(cudaMalloc(&dout,sizeof(DirectRelSeqOutput)),"malloc out");ck(cudaMalloc(&dok,sizeof(int)),"malloc ok");
  ck(cudaMemcpy(dr,lowered_recipes,sizeof(lowered_recipes),cudaMemcpyHostToDevice),"copy recipe");ck(cudaMemcpy(doo,lowered_occ,sizeof(lowered_occ),cudaMemcpyHostToDevice),"copy occ");ck(cudaMemset(dout,0,sizeof(DirectRelSeqOutput)),"zero out");ck(cudaMemset(dok,0,sizeof(int)),"zero ok");
  bridge_eval_kernel<<<1,1>>>(dr,2,doo,3,0xC3,dout,dok);ck(cudaGetLastError(),"launch");ck(cudaDeviceSynchronize(),"sync");
  DirectRelSeqOutput out{};int ok=0;ck(cudaMemcpy(&out,dout,sizeof(out),cudaMemcpyDeviceToHost),"out");ck(cudaMemcpy(&ok,dok,sizeof(ok),cudaMemcpyDeviceToHost),"ok");
  const std::uint32_t expected[]={101,11,102,12,103,104,101,21,102,22,103};
  if(!ok || out.count!=11) return 9;for(unsigned i=0;i<11;++i)if(out.units[i]!=expected[i])return 10;
  std::printf("DIRECT_ADULT_RELSEQ_BRIDGE CUDA GREEN canonical_recipe=1 canonical_occurrence=1 sidecar_child_binding=1 units=%u depth=%u\n",out.count,out.depth_peak);
  return 0;
}

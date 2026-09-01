#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hardware_native/direct_foundry_condensation_workspace.cuh"
#include "hardware_native/direct_resident_recipe_experience_revision.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

static void ck(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA %s: %s\n", where, cudaGetErrorString(e));
    std::exit(20);
  }
}

static bool seed_parent(ResidentRecipeCell* cells, ResidentRecipeDerivation* ders,
                        ResidentPostbirthConstructorState* state, std::uint32_t* cell_count,
                        DirectWhiteboxCondensationV1* witness) {
  DirectWhiteboxReductionSourceV1 source{};
  source.kind = DirectWhiteboxReductionKindV1::affine_composition;
  source.coefficient_count = 4u;
  source.coefficients_q16[0] = 2 << 16;
  source.coefficients_q16[1] = 1 << 16;
  source.coefficients_q16[2] = 3 << 16;
  source.coefficients_q16[3] = -(1 << 15);
  source.source_identity = direct_whitebox_source_identity(source);
  if (!direct_whitebox_reduce_exact(source, witness)) return false;
  cells[0] = {};
  cells[0].logical_recipe_id = 0xabc001ull;
  cells[0].revision = 1u;
  cells[0].support_q16 = 1 << 16;
  cells[0].revision_identity = resident_recipe_revision_identity(
      cells[0].logical_recipe_id, 0u, 1u, 0xabc002ull, cells[0].support_q16, 0);
  ders[0] = {};
  ders[0].logical_recipe_id = cells[0].logical_recipe_id;
  ders[0].revision_identity = cells[0].revision_identity;
  ders[0].recipe_cell = 0u;
  ders[0].generation = 1u;
  ders[0].port_count = 2u;
  ders[0].ports[0] = ResidentRecipePort{
      10u, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::input, 1u};
  ders[0].ports[1] = ResidentRecipePort{
      20u, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::output, 1u};
  *state = {};
  state->derivation_count = 1u;
  state->derivation_capacity = 4u;
  state->recipe_cell_capacity = 4u;
  state->port_capacity = 16u;
  state->relation_capacity = 16u;
  state->parameter_capacity = 16u;
  state->ports_used = 2u;
  *cell_count = 1u;
  return true;
}

static bool mint(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                 ResidentRecipeDerivation* ders, ResidentPostbirthConstructorState* state,
                 const DirectWhiteboxCondensationV1& witness) {
  FoundryCondensationWorkspace workspace{};
  workspace.cells = cells;
  workspace.cell_count = cell_count;
  workspace.cell_capacity = 4u;
  workspace.derivations = ders;
  workspace.state = state;
  workspace.parent_cell = 0u;
  workspace.parent_logical = cells[0].logical_recipe_id;
  workspace.witness = &witness;
  return foundry_materialize_from_workspace(&workspace, nullptr);
}

__global__ void condense_one(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                             ResidentRecipeDerivation* ders,
                             ResidentPostbirthConstructorState* state,
                             DirectWhiteboxCondensationV1 witness, int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  FoundryCondensationWorkspace workspace{};
  workspace.cells = cells;
  workspace.cell_count = cell_count;
  workspace.cell_capacity = 4u;
  workspace.derivations = ders;
  workspace.state = state;
  workspace.parent_cell = 0u;
  workspace.parent_logical = cells[0].logical_recipe_id;
  workspace.witness = &witness;
  *ok = foundry_materialize_from_workspace(&workspace, nullptr) &&
                replay_resident_whitebox_recipe(witness, cells[1], ders[1])
            ? 1
            : 0;
}

__global__ void condense_again(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                               ResidentRecipeDerivation* ders,
                               ResidentPostbirthConstructorState* state,
                               DirectWhiteboxCondensationV1 witness, int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  FoundryCondensationWorkspace workspace{};
  workspace.cells = cells;
  workspace.cell_count = cell_count;
  workspace.cell_capacity = 4u;
  workspace.derivations = ders;
  workspace.state = state;
  workspace.parent_cell = 0u;
  workspace.parent_logical = cells[0].logical_recipe_id;
  workspace.witness = &witness;
  *ok = foundry_materialize_from_workspace(&workspace, nullptr) ? 1 : 0;
}

__global__ void rematerialize_one(DirectWhiteboxCondensationV1 witness, std::uint64_t* sid,
                                 int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  DirectWhiteboxReductionSourceV1 restored{};
  *ok = direct_whitebox_rematerialize(witness, &restored) ? 1 : 0;
  *sid = restored.source_identity;
}

__global__ void expire_one(ResidentRecipeCell* cells, const ResidentRecipeDerivation* ders,
                           ResidentRecipeOccurrence* occ, int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t vars[2] = {1u, 2u};
  *ok = bind_resident_recipe_occurrence(
            cells[1], ders[1], vars, 2u, 0xC60ull, 0xC60ull + 1000u, 77u, 1u,
            ResidentOccurrenceLineageKind::actual,
            DirectParticipationAuthority::independent_external, 9u, 1u, 1u, 0, occ) &&
                commit_resident_causal_difference_revision(occ, &cells[1], 1u, true, 1, 0,
                                                           2u)
            ? 1
            : 0;
}

__global__ void two_occ_isolate(ResidentRecipeCell* cells, const ResidentRecipeDerivation* ders,
                                int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t vars[2] = {1u, 2u};
  ResidentRecipeOccurrence a{}, b{};
  if (!bind_resident_recipe_occurrence(
          cells[1], ders[1], vars, 2u, 0xC70ull, 0xC70ull + 1000u, 77u, 1u,
          ResidentOccurrenceLineageKind::actual,
          DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, &a) ||
      !bind_resident_recipe_occurrence(
          cells[1], ders[1], vars, 2u, 0xC71ull, 0xC71ull + 1000u, 77u, 1u,
          ResidentOccurrenceLineageKind::actual,
          DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, &b)) {
    *ok = 0;
    return;
  }
  const std::uint64_t shared = a.revision_identity;
  if (!commit_resident_causal_difference_revision(&a, &cells[1], 1u, true, 1, 0)) {
    *ok = 0;
    return;
  }
  *ok = a.state == kResidentRecipeOccurrenceSettled &&
                b.state == kResidentRecipeOccurrenceLive &&
                b.revision_identity == shared &&
                cells[1].revision_identity != shared &&
                !commit_resident_causal_difference_revision(&b, &cells[1], 1u, true, 1, 0)
            ? 1
            : 0;
}

__global__ void settle_one(ResidentRecipeCell* cells, const ResidentRecipeDerivation* ders,
                           ResidentRecipeOccurrence* occ, int independent, int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  const std::uint32_t vars[2] = {1u, 2u};
  *ok = bind_resident_recipe_occurrence(
            cells[1], ders[1], vars, 2u, 0xC50ull, 0xC50ull + 1000u, 77u, 1u,
            ResidentOccurrenceLineageKind::actual,
            DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, occ) &&
                commit_resident_causal_difference_revision(occ, &cells[1], 1u,
                                                           independent != 0, 1, 0)
            ? 1
            : 0;
}

int main() {
  ResidentRecipeCell host_cells[4]{};
  ResidentRecipeDerivation host_ders[4]{};
  ResidentPostbirthConstructorState host_state{};
  DirectWhiteboxCondensationV1 witness{};
  std::uint32_t host_count = 0;
  if (!seed_parent(host_cells, host_ders, &host_state, &host_count, &witness)) return 2;
  ResidentRecipeCell expect_cells[4] = {host_cells[0]};
  ResidentRecipeDerivation expect_ders[4] = {host_ders[0]};
  ResidentPostbirthConstructorState expect_state = host_state;
  std::uint32_t expect_count = 1;
  if (!mint(expect_cells, &expect_count, expect_ders, &expect_state, witness) ||
      !replay_resident_whitebox_recipe(witness, expect_cells[1], expect_ders[1]))
    return 3;

  ResidentRecipeCell* dcells = nullptr;
  ResidentRecipeDerivation* dders = nullptr;
  ResidentPostbirthConstructorState* dstate = nullptr;
  std::uint32_t* dcount = nullptr;
  int* dok = nullptr;
  ck(cudaMalloc(&dcells, sizeof(host_cells)), "cells");
  ck(cudaMalloc(&dders, sizeof(host_ders)), "ders");
  ck(cudaMalloc(&dstate, sizeof(host_state)), "state");
  ck(cudaMalloc(&dcount, sizeof(host_count)), "count");
  ck(cudaMalloc(&dok, sizeof(int)), "ok");
  ck(cudaMemcpy(dcells, host_cells, sizeof(host_cells), cudaMemcpyHostToDevice), "h2d cells");
  ck(cudaMemcpy(dders, host_ders, sizeof(host_ders), cudaMemcpyHostToDevice), "h2d ders");
  ck(cudaMemcpy(dstate, &host_state, sizeof(host_state), cudaMemcpyHostToDevice), "h2d state");
  ck(cudaMemcpy(dcount, &host_count, sizeof(host_count), cudaMemcpyHostToDevice), "h2d count");
  condense_one<<<1, 1>>>(dcells, dcount, dders, dstate, witness, dok);
  ck(cudaGetLastError(), "launch");
  ck(cudaDeviceSynchronize(), "sync");
  int ok = 0;
  std::uint32_t dev_count = 0;
  ResidentRecipeCell dev_cells[4]{};
  ResidentRecipeDerivation dev_ders[4]{};
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "ok");
  ck(cudaMemcpy(&dev_count, dcount, sizeof(dev_count), cudaMemcpyDeviceToHost), "count");
  ck(cudaMemcpy(dev_cells, dcells, sizeof(dev_cells), cudaMemcpyDeviceToHost), "d2h cells");
  ck(cudaMemcpy(dev_ders, dders, sizeof(dev_ders), cudaMemcpyDeviceToHost), "d2h ders");
  if (!ok || dev_count != expect_count ||
      dev_cells[1].logical_recipe_id != expect_cells[1].logical_recipe_id ||
      dev_cells[1].revision_identity != expect_cells[1].revision_identity ||
      dev_ders[1].generation != expect_ders[1].generation)
    return 4;

  condense_again<<<1, 1>>>(dcells, dcount, dders, dstate, witness, dok);
  ck(cudaGetLastError(), "launch dup");
  ck(cudaDeviceSynchronize(), "sync dup");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "dup ok");
  if (ok) return 5;

  ResidentRecipeOccurrence host_occ{};
  const std::uint32_t vars[2] = {1u, 2u};
  if (!bind_resident_recipe_occurrence(
          expect_cells[1], expect_ders[1], vars, 2u, 0xC50ull, 0xC50ull + 1000u, 77u, 1u,
          ResidentOccurrenceLineageKind::actual,
          DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, &host_occ) ||
      !commit_resident_causal_difference_revision(&host_occ, &expect_cells[1], 1u, true, 1,
                                                  0))
    return 6;
  ResidentRecipeOccurrence* docc = nullptr;
  ck(cudaMalloc(&docc, sizeof(ResidentRecipeOccurrence)), "occ");
  settle_one<<<1, 1>>>(dcells, dders, docc, 1, dok);
  ck(cudaGetLastError(), "launch settle");
  ck(cudaDeviceSynchronize(), "sync settle");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "settle ok");
  ck(cudaMemcpy(dev_cells, dcells, sizeof(dev_cells), cudaMemcpyDeviceToHost), "d2h settled");
  if (!ok || dev_cells[1].credit_q16 != expect_cells[1].credit_q16 ||
      dev_cells[1].revision_identity != expect_cells[1].revision_identity ||
      dev_cells[1].credit_q16 != substrate::direct_adult_core::kQ16One)
    return 7;
  const std::uint64_t settled_rev = dev_cells[1].revision_identity;
  settle_one<<<1, 1>>>(dcells, dders, docc, 0, dok);
  ck(cudaGetLastError(), "launch yoked");
  ck(cudaDeviceSynchronize(), "sync yoked");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "yoked ok");
  ck(cudaMemcpy(dev_cells, dcells, sizeof(dev_cells), cudaMemcpyDeviceToHost), "d2h yoked");
  if (!ok || dev_cells[1].revision_identity != settled_rev) return 8;

  expire_one<<<1, 1>>>(dcells, dders, docc, dok);
  ck(cudaGetLastError(), "launch expire");
  ck(cudaDeviceSynchronize(), "sync expire");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "expire ok");
  ck(cudaMemcpy(dev_cells, dcells, sizeof(dev_cells), cudaMemcpyDeviceToHost), "d2h expire");
  if (!ok || dev_cells[1].revision_identity != settled_rev) return 9;

  std::uint64_t* dsid = nullptr;
  ck(cudaMalloc(&dsid, sizeof(std::uint64_t)), "sid");
  rematerialize_one<<<1, 1>>>(witness, dsid, dok);
  ck(cudaGetLastError(), "launch remat");
  ck(cudaDeviceSynchronize(), "sync remat");
  std::uint64_t sid = 0;
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "remat ok");
  ck(cudaMemcpy(&sid, dsid, sizeof(sid), cudaMemcpyDeviceToHost), "sid");
  if (!ok || sid != witness.source.source_identity) return 10;
  DirectWhiteboxCondensationV1 tampered = witness;
  tampered.result_q16[0] ^= 1;
  rematerialize_one<<<1, 1>>>(tampered, dsid, dok);
  ck(cudaGetLastError(), "launch tamper");
  ck(cudaDeviceSynchronize(), "sync tamper");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "tamper ok");
  if (ok) return 11;

  two_occ_isolate<<<1, 1>>>(dcells, dders, dok);
  ck(cudaGetLastError(), "launch two");
  ck(cudaDeviceSynchronize(), "sync two");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "two ok");
  if (!ok) return 12;

  std::printf(
      "FOUNDRY_WHITEBOX_CONDENSE CUDA GREEN host_device_match=1 replay=1 "
      "duplicate_refuse=1 pn1_bind=1 settle_match=1 yoked_no_revision=1 "
      "eligibility_expiry=1 rematerialize=1 tampered_refuse=1 "
      "same_recipe_two_occ=1 stale_head_refuse=1 "
      "generation=%llu logical=%llu graph_flip=0\n",
      static_cast<unsigned long long>(dev_ders[1].generation),
      static_cast<unsigned long long>(dev_cells[1].logical_recipe_id));
  return 0;
}

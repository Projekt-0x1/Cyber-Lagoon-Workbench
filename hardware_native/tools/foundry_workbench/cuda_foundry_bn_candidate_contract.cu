#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_foundry_condensation_workspace.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

static void ck(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA %s: %s\n", where, cudaGetErrorString(e));
    std::exit(20);
  }
}

static bool seed_rank(ResidentRecipeCell* cell, ResidentRecipeDerivation* der,
                      std::uint32_t recipe_cell, std::uint64_t logical, std::uint64_t rev,
                      std::uint64_t generation) {
  *cell = {};
  cell->logical_recipe_id = logical;
  cell->revision_identity = rev;
  cell->revision = 1u;
  cell->support_q16 = 1 << 16;
  *der = {};
  der->logical_recipe_id = logical;
  der->revision_identity = rev;
  der->recipe_cell = recipe_cell;
  der->generation = generation;
  der->port_count = 2u;
  der->ports[0] = ResidentRecipePort{
      10u, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::input, 1u};
  der->ports[1] = ResidentRecipePort{
      20u, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::output, 1u};
  return true;
}

static bool bind_live(const ResidentRecipeCell& cell, const ResidentRecipeDerivation& der,
                      const std::uint32_t* vars, std::uint64_t oid,
                      ResidentRecipeOccurrence* out) {
  return bind_resident_recipe_occurrence(
      cell, der, vars, 2u, oid, oid + 1000u, 77u, 1u,
      ResidentOccurrenceLineageKind::actual,
      DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, out);
}

static bool seed_affine(DirectWhiteboxCondensationV1* witness, std::int32_t a, std::int32_t b,
                        std::int32_t c, std::int32_t d) {
  DirectWhiteboxReductionSourceV1 source{};
  source.kind = DirectWhiteboxReductionKindV1::affine_composition;
  source.coefficient_count = 4u;
  source.coefficients_q16[0] = a;
  source.coefficients_q16[1] = b;
  source.coefficients_q16[2] = c;
  source.coefficients_q16[3] = d;
  source.source_identity = direct_whitebox_source_identity(source);
  return direct_whitebox_reduce_exact(source, witness);
}

static bool host_mint(ResidentRecipeCell* rec, std::uint32_t* cell_count,
                      ResidentRecipeDerivation* der, ResidentPostbirthConstructorState* state,
                      std::uint32_t parent_cell, std::uint64_t parent_logical,
                      const DirectWhiteboxCondensationV1& witness) {
  FoundryCondensationWorkspace workspace{};
  workspace.cells = rec;
  workspace.cell_count = cell_count;
  workspace.cell_capacity = 5u;
  workspace.derivations = der;
  workspace.state = state;
  workspace.parent_cell = parent_cell;
  workspace.parent_logical = parent_logical;
  workspace.witness = &witness;
  return foundry_materialize_from_workspace(&workspace, nullptr);
}

static bool admit(const ResidentRelationalNetworkClosure& claimed,
                  const ResidentRecipeCell* rec, const ResidentRecipeDerivation* der,
                  const ResidentRecipeOccurrence* occ, const ResidentOccurrenceCoupling* edges,
                  std::uint64_t parent_logical) {
  ResidentRelationalNetworkClosure rebuilt{};
  if (!bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &rebuilt) ||
      rebuilt.identity != claimed.identity || claimed.identity == 0u ||
      claimed.boundary_count < 2u)
    return false;
  for (std::uint16_t i = 0; i < rebuilt.occurrence_count; ++i)
    if (rebuilt.members[i].logical_recipe_id == parent_logical) return true;
  return false;
}

__global__ void admit_and_condense(const ResidentRecipeCell* net_rec,
                                   const ResidentRecipeDerivation* net_der,
                                   const ResidentRecipeOccurrence* occ, std::uint32_t n,
                                   const ResidentOccurrenceCoupling* edges, std::uint32_t en,
                                   std::uint64_t claimed_identity, std::uint64_t parent_logical,
                                   ResidentRecipeCell* bank, ResidentRecipeDerivation* bank_der,
                                   ResidentPostbirthConstructorState* state,
                                   std::uint32_t* cell_count, std::uint32_t parent_cell,
                                   std::uint32_t capacity, DirectWhiteboxCondensationV1 witness,
                                   int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  ResidentRelationalNetworkClosure rebuilt{};
  if (!bind_resident_relational_network_closure(net_rec, net_der, occ, n, edges, en,
                                               &rebuilt) ||
      rebuilt.identity != claimed_identity || claimed_identity == 0u ||
      rebuilt.boundary_count < 2u || parent_cell >= *cell_count ||
      bank[parent_cell].logical_recipe_id != parent_logical) {
    *ok = 0;
    return;
  }
  bool member = false;
  for (std::uint16_t i = 0; i < rebuilt.occurrence_count; ++i)
    if (rebuilt.members[i].logical_recipe_id == parent_logical) member = true;
  if (!member) {
    *ok = 0;
    return;
  }
  FoundryCondensationWorkspace workspace{};
  workspace.cells = bank;
  workspace.cell_count = cell_count;
  workspace.cell_capacity = capacity;
  workspace.derivations = bank_der;
  workspace.state = state;
  workspace.parent_cell = parent_cell;
  workspace.parent_logical = parent_logical;
  workspace.witness = &witness;
  *ok = foundry_materialize_from_workspace(&workspace, nullptr) ? 1 : 0;
}

__global__ void bind_n(const ResidentRecipeCell* rec, const ResidentRecipeDerivation* der,
                       const ResidentRecipeOccurrence* occ, std::uint32_t n,
                       const ResidentOccurrenceCoupling* edges, std::uint32_t en,
                       ResidentRelationalNetworkClosure* out, int* ok) {
  if (blockIdx.x || threadIdx.x) return;
  *ok = bind_resident_relational_network_closure(rec, der, occ, n, edges, en, out) ? 1 : 0;
}

int main() {
  ResidentRecipeCell rec[5]{};
  ResidentRecipeDerivation der[5]{};
  if (!seed_rank(&rec[0], &der[0], 0u, 0xA0, 0xB0, 1u) ||
      !seed_rank(&rec[1], &der[1], 1u, 0xA1, 0xB1, 2u) ||
      !seed_rank(&rec[2], &der[2], 2u, 0xA2, 0xB2, 3u))
    return 2;
  const std::uint32_t v0[] = {10u, 20u};
  const std::uint32_t v1[] = {20u, 30u};
  const std::uint32_t v2[] = {30u, 40u};
  ResidentRecipeOccurrence occ[3]{};
  if (!bind_live(rec[0], der[0], v0, 0xD20, &occ[0]) ||
      !bind_live(rec[1], der[1], v1, 0xD21, &occ[1]) ||
      !bind_live(rec[2], der[2], v2, 0xD22, &occ[2]))
    return 3;
  ResidentOccurrenceCoupling edges[2]{};
  if (!bind_resident_occurrence_coupling(occ[0], der[0], 1u, occ[1], der[1], 0u, &edges[0]) ||
      !bind_resident_occurrence_coupling(occ[1], der[1], 1u, occ[2], der[2], 0u, &edges[1]))
    return 4;
  ResidentRelationalNetworkClosure claimed{};
  if (!bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &claimed))
    return 5;
  ResidentRelationalNetworkClosure stale = claimed;
  stale.identity ^= 1ull;
  if (admit(stale, rec, der, occ, edges, rec[2].logical_recipe_id) ||
      admit(claimed, rec, der, occ, edges, 0xDEADull) ||
      !admit(claimed, rec, der, occ, edges, rec[2].logical_recipe_id))
    return 6;

  DirectWhiteboxCondensationV1 witness{};
  DirectWhiteboxCondensationV1 witness2{};
  DirectWhiteboxCondensationV1 witness3{};
  if (!seed_affine(&witness, 5 << 16, 0, 1 << 16, 1 << 16) ||
      !seed_affine(&witness2, 4 << 16, 0, 2 << 16, 1 << 16) ||
      !seed_affine(&witness3, 6 << 16, 0, 1 << 16, 0))
    return 7;
  ResidentPostbirthConstructorState state{};
  state.derivation_count = 3u;
  state.derivation_capacity = 5u;
  state.recipe_cell_capacity = 5u;
  state.port_capacity = 16u;
  state.relation_capacity = 16u;
  state.parameter_capacity = 16u;
  state.ports_used = 6u;
  std::uint32_t cell_count = 3u;

  ResidentRecipeCell expect_rec[5] = {rec[0], rec[1], rec[2]};
  ResidentRecipeDerivation expect_der[5] = {der[0], der[1], der[2]};
  ResidentPostbirthConstructorState expect_state = state;
  std::uint32_t expect_count = 3u;
  if (!host_mint(expect_rec, &expect_count, expect_der, &expect_state, 2u,
                 rec[2].logical_recipe_id, witness) ||
      expect_count != 4u ||
      !replay_resident_whitebox_recipe(witness, expect_rec[3], expect_der[3]) ||
      expect_der[3].parent_logical_recipe_id != rec[2].logical_recipe_id)
    return 8;

  ResidentRecipeCell* dbank = nullptr;
  ResidentRecipeDerivation* dbank_der = nullptr;
  ResidentRecipeCell* dnet = nullptr;
  ResidentRecipeDerivation* dnet_der = nullptr;
  ResidentRecipeOccurrence* docc = nullptr;
  ResidentOccurrenceCoupling* dedge = nullptr;
  ResidentPostbirthConstructorState* dstate = nullptr;
  std::uint32_t* dcount = nullptr;
  int* dok = nullptr;
  ck(cudaMalloc(&dbank, sizeof(rec)), "bank");
  ck(cudaMalloc(&dbank_der, sizeof(der)), "bank der");
  ck(cudaMalloc(&dnet, 3u * sizeof(ResidentRecipeCell)), "net");
  ck(cudaMalloc(&dnet_der, 3u * sizeof(ResidentRecipeDerivation)), "net der");
  ck(cudaMalloc(&docc, sizeof(occ)), "occ");
  ck(cudaMalloc(&dedge, sizeof(edges)), "edge");
  ck(cudaMalloc(&dstate, sizeof(state)), "state");
  ck(cudaMalloc(&dcount, sizeof(cell_count)), "count");
  ck(cudaMalloc(&dok, sizeof(int)), "ok");

  auto upload_bank = [&]() {
    ck(cudaMemcpy(dbank, rec, sizeof(rec), cudaMemcpyHostToDevice), "h2d bank");
    ck(cudaMemcpy(dbank_der, der, sizeof(der), cudaMemcpyHostToDevice), "h2d bank der");
    ck(cudaMemcpy(dstate, &state, sizeof(state), cudaMemcpyHostToDevice), "h2d state");
    ck(cudaMemcpy(dcount, &cell_count, sizeof(cell_count), cudaMemcpyHostToDevice),
       "h2d count");
  };
  auto upload_net = [&](const ResidentRecipeCell* net, const ResidentRecipeDerivation* netd,
                        const ResidentRecipeOccurrence* o, const ResidentOccurrenceCoupling* e) {
    ck(cudaMemcpy(dnet, net, 3u * sizeof(*net), cudaMemcpyHostToDevice), "h2d net");
    ck(cudaMemcpy(dnet_der, netd, 3u * sizeof(*netd), cudaMemcpyHostToDevice), "h2d net der");
    ck(cudaMemcpy(docc, o, 3u * sizeof(*o), cudaMemcpyHostToDevice), "h2d occ");
    ck(cudaMemcpy(dedge, e, 2u * sizeof(*e), cudaMemcpyHostToDevice), "h2d edge");
  };
  auto launch = [&](std::uint64_t identity, std::uint64_t parent, std::uint32_t parent_cell,
                    DirectWhiteboxCondensationV1 w) {
    admit_and_condense<<<1, 1>>>(dnet, dnet_der, docc, 3u, dedge, 2u, identity, parent, dbank,
                                 dbank_der, dstate, dcount, parent_cell, 5u, w, dok);
    ck(cudaGetLastError(), "launch");
    ck(cudaDeviceSynchronize(), "sync");
    int ok = 0;
    ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "ok");
    return ok;
  };

  upload_bank();
  upload_net(rec, der, occ, edges);
  if (launch(stale.identity, rec[2].logical_recipe_id, 2u, witness)) return 9;
  std::uint32_t after_stale = 0;
  ck(cudaMemcpy(&after_stale, dcount, sizeof(after_stale), cudaMemcpyDeviceToHost),
     "stale count");
  if (after_stale != 3u) return 9;

  upload_bank();
  if (launch(claimed.identity, 0xDEADull, 2u, witness)) return 10;
  std::uint32_t after_foreign = 0;
  ck(cudaMemcpy(&after_foreign, dcount, sizeof(after_foreign), cudaMemcpyDeviceToHost),
     "foreign count");
  if (after_foreign != 3u) return 10;

  upload_bank();
  if (launch(claimed.identity, rec[2].logical_recipe_id, 1u, witness)) return 10;
  std::uint32_t after_mismatch = 0;
  ck(cudaMemcpy(&after_mismatch, dcount, sizeof(after_mismatch), cudaMemcpyDeviceToHost),
     "mismatch count");
  if (after_mismatch != 3u) return 10;

  upload_bank();
  if (!launch(claimed.identity, rec[2].logical_recipe_id, 2u, witness)) return 11;
  ResidentRecipeCell got[5]{};
  ResidentRecipeDerivation got_der[5]{};
  std::uint32_t got_count = 0;
  ck(cudaMemcpy(got, dbank, sizeof(got), cudaMemcpyDeviceToHost), "cells");
  ck(cudaMemcpy(got_der, dbank_der, sizeof(got_der), cudaMemcpyDeviceToHost), "ders");
  ck(cudaMemcpy(&got_count, dcount, sizeof(got_count), cudaMemcpyDeviceToHost), "count");
  if (got_count != 4u || got[3].logical_recipe_id != expect_rec[3].logical_recipe_id ||
      got[3].revision_identity != expect_rec[3].revision_identity ||
      got_der[3].generation != expect_der[3].generation ||
      got_der[3].parent_logical_recipe_id != rec[2].logical_recipe_id ||
      !replay_resident_whitebox_recipe(witness, got[3], got_der[3]))
    return 12;

  if (got_der[0].generation == got_der[3].generation ||
      got_der[3].generation != got_der[2].generation + 1u)
    return 13;
  ResidentRecipeCell mix_rec[3] = {got[0], got[1], got[3]};
  ResidentRecipeDerivation mix_der[3] = {got_der[0], got_der[1], got_der[3]};
  const std::uint32_t vm0[] = {10u, 20u};
  const std::uint32_t vm1[] = {50u, 60u};
  const std::uint32_t vm3[] = {20u, 50u};
  ResidentRecipeOccurrence mix_occ[3]{};
  if (!bind_live(mix_rec[0], mix_der[0], vm0, 0xE00, &mix_occ[0]) ||
      !bind_live(mix_rec[1], mix_der[1], vm1, 0xE01, &mix_occ[1]) ||
      !bind_live(mix_rec[2], mix_der[2], vm3, 0xE03, &mix_occ[2]))
    return 13;
  ResidentOccurrenceCoupling mix_edges[2]{};
  if (!bind_resident_occurrence_coupling(mix_occ[0], mix_der[0], 1u, mix_occ[2], mix_der[2],
                                         0u, &mix_edges[0]) ||
      !bind_resident_occurrence_coupling(mix_occ[2], mix_der[2], 1u, mix_occ[1], mix_der[1],
                                         0u, &mix_edges[1]))
    return 13;
  ResidentRelationalNetworkClosure mix_host{};
  if (!bind_resident_relational_network_closure(mix_rec, mix_der, mix_occ, 3u, mix_edges, 2u,
                                                &mix_host) ||
      mix_host.identity == 0u || mix_host.occurrence_count != 3u)
    return 13;

  ResidentRelationalNetworkClosure* dcl = nullptr;
  ck(cudaMalloc(&dcl, sizeof(ResidentRelationalNetworkClosure)), "cl");
  upload_net(mix_rec, mix_der, mix_occ, mix_edges);
  bind_n<<<1, 1>>>(dnet, dnet_der, docc, 3u, dedge, 2u, dcl, dok);
  ck(cudaGetLastError(), "mix launch");
  ck(cudaDeviceSynchronize(), "mix sync");
  int mix_ok = 0;
  ResidentRelationalNetworkClosure mix_dev{};
  ck(cudaMemcpy(&mix_ok, dok, sizeof(mix_ok), cudaMemcpyDeviceToHost), "mix ok");
  ck(cudaMemcpy(&mix_dev, dcl, sizeof(mix_dev), cudaMemcpyDeviceToHost), "mix cl");
  if (!mix_ok || mix_dev.identity != mix_host.identity) return 14;

  if (!host_mint(expect_rec, &expect_count, expect_der, &expect_state, 3u,
                 expect_rec[3].logical_recipe_id, witness2) ||
      expect_count != 5u ||
      !replay_resident_whitebox_recipe(witness2, expect_rec[4], expect_der[4]) ||
      expect_der[4].parent_logical_recipe_id != expect_rec[3].logical_recipe_id)
    return 16;
  if (!launch(mix_host.identity, got[3].logical_recipe_id, 3u, witness2)) return 16;
  ck(cudaMemcpy(got, dbank, sizeof(got), cudaMemcpyDeviceToHost), "pn2 cells");
  ck(cudaMemcpy(got_der, dbank_der, sizeof(got_der), cudaMemcpyDeviceToHost), "pn2 ders");
  ck(cudaMemcpy(&got_count, dcount, sizeof(got_count), cudaMemcpyDeviceToHost), "pn2 count");
  if (got_count != 5u || got[4].logical_recipe_id != expect_rec[4].logical_recipe_id ||
      got[4].revision_identity != expect_rec[4].revision_identity ||
      got_der[4].generation != expect_der[4].generation ||
      got_der[4].parent_logical_recipe_id != got[3].logical_recipe_id ||
      !replay_resident_whitebox_recipe(witness2, got[4], got_der[4]))
    return 17;
  if (launch(mix_host.identity, got[3].logical_recipe_id, 3u, witness3)) return 18;
  std::uint32_t after_cap = 0;
  ck(cudaMemcpy(&after_cap, dcount, sizeof(after_cap), cudaMemcpyDeviceToHost), "cap count");
  if (after_cap != 5u) return 18;

  mix_occ[2].state = kResidentRecipeOccurrenceSettled;
  ck(cudaMemcpy(docc, mix_occ, sizeof(mix_occ), cudaMemcpyHostToDevice), "settled pn1");
  bind_n<<<1, 1>>>(dnet, dnet_der, docc, 3u, dedge, 2u, dcl, dok);
  ck(cudaGetLastError(), "settled launch");
  ck(cudaDeviceSynchronize(), "settled sync");
  ck(cudaMemcpy(&mix_ok, dok, sizeof(mix_ok), cudaMemcpyDeviceToHost), "settled ok");
  if (mix_ok) return 15;

  std::printf(
      "FOUNDRY_BN_CANDIDATE CUDA GREEN host_device_match=1 "
      "stale_closure_refuse=1 non_member_refuse=1 parent_cell_mismatch=1 "
      "bn_candidate=1 replay=1 pn1_mixed_rank=1 ancestor_direct_bind=1 "
      "pn2_repeat=1 capacity_refuse=1 settled_pn1_refuse=1 "
      "generation=%llu logical=%llu graph_flip=0\n",
      static_cast<unsigned long long>(got_der[4].generation),
      static_cast<unsigned long long>(got[4].logical_recipe_id));
  return 0;
}

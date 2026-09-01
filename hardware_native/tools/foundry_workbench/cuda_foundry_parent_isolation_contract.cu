#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hardware_native/direct_adult_language_recipe_opportunity.cuh"
#include "hardware_native/direct_adult_language_recipe_opportunity_bank.cuh"
#include "hardware_native/direct_foundry_condensation_workspace.cuh"
#include "hardware_native/direct_resident_recipe_experience_revision.cuh"
#include "hardware_native/direct_adult_surface_sequence.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

static void ck(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA %s: %s\n", where, cudaGetErrorString(e));
    std::exit(20);
  }
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

static bool bind_live(const ResidentRecipeCell& cell, const ResidentRecipeDerivation& der,
                      const std::uint32_t* vars, std::uint64_t oid, ResidentRecipeOccurrence* out,
                      std::uint32_t incarnation = 1u, std::int32_t activation_q16 = 0) {
  if (!bind_resident_recipe_occurrence(cell, der, vars, 2u, oid, oid + 1000u, 77u, incarnation,
                                       ResidentOccurrenceLineageKind::actual,
                                       DirectParticipationAuthority::independent_external, 9u, 1u,
                                       100u, 0, out))
    return false;
  out->activation_q16 = activation_q16;
  return true;
}

// clang-format off: language_after_pn2 calls the helper defined by the first splice.
#include "nominate_source_network_credit.inl"
#include "language_after_pn2.inl"
// clang-format on

static bool seed_bank(ResidentRecipeCell* cells, ResidentRecipeDerivation* ders,
                      ResidentPostbirthConstructorState* state, std::uint32_t* cell_count) {
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
  ders[0].ports[0] = ResidentRecipePort{10u, ResidentRecipePortDomain::q16_scalar,
                                        ResidentRecipePortDirection::input, 1u};
  ders[0].ports[1] = ResidentRecipePort{20u, ResidentRecipePortDomain::q16_scalar,
                                        ResidentRecipePortDirection::output, 1u};
  *state = {};
  state->derivation_count = 1u;
  state->derivation_capacity = 4u;
  state->recipe_cell_capacity = 4u;
  state->port_capacity = 16u;
  state->relation_capacity = 16u;
  state->parameter_capacity = 16u;
  state->ports_used = 2u;
  *cell_count = 1u;
  DirectWhiteboxCondensationV1 first{};
  return seed_affine(&first, 2 << 16, 1 << 16, 3 << 16, -(1 << 15)) &&
         materialize_resident_whitebox_recipe(cells, cell_count, 4u, ders, state, 0u, first, 0u);
}

__global__ void from_workspace(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                               ResidentRecipeDerivation* ders,
                               ResidentPostbirthConstructorState* state, ResidentRecipeCell episode,
                               std::uint32_t parent_cell, std::uint64_t parent_logical,
                               DirectWhiteboxCondensationV1 witness, int* ok) {
  if (blockIdx.x || threadIdx.x)
    return;
  FoundryCondensationWorkspace workspace{};
  workspace.cells = cells;
  workspace.cell_count = cell_count;
  workspace.cell_capacity = 4u;
  workspace.derivations = ders;
  workspace.state = state;
  workspace.parent_cell = parent_cell;
  workspace.parent_logical = parent_logical;
  workspace.witness = &witness;
  *ok = foundry_materialize_from_workspace(&workspace, &episode) ? 1 : 0;
}

__global__ void observe_n(const ResidentRecipeCell* rec, const ResidentRecipeDerivation* der,
                          const ResidentRecipeOccurrence* first,
                          const ResidentRecipeOccurrence* second,
                          const ResidentOccurrenceCoupling* e1,
                          const ResidentOccurrenceCoupling* e2, DirectWhiteboxCondensationV1* out,
                          int* ok) {
  if (blockIdx.x || threadIdx.x)
    return;
  *ok = observe_resident_mixed_rank_whitebox(rec, der, first, second, 3u, e1, e2, 2u, out) ? 1 : 0;
}

__global__ void observe_ev(const ResidentRecipeCell* rec, const ResidentRecipeDerivation* der,
                           const ResidentRecipeOccurrence* first,
                           const ResidentRecipeOccurrence* second,
                           const ResidentOccurrenceCoupling* e1,
                           const ResidentOccurrenceCoupling* e2,
                           DirectWhiteboxCondensationV1* whitebox,
                           ResidentNetworkCondensationEvidence* evidence, int* ok) {
  if (blockIdx.x || threadIdx.x)
    return;
  *ok = observe_resident_mixed_rank_evidence(rec, der, first, second, 3u, e1, e2, 2u, whitebox,
                                             evidence)
            ? 1
            : 0;
}

__global__ void replay_ev(const ResidentRecipeCell* cells, std::uint32_t cell_count,
                          const ResidentRecipeDerivation* ders, std::uint32_t der_count,
                          ResidentNetworkCondensationEvidence evidence,
                          DirectWhiteboxCondensationV1 witness, std::uint64_t* logical,
                          std::uint64_t* revision, int* ok) {
  if (blockIdx.x || threadIdx.x)
    return;
  *ok = replay_resident_mixed_rank_evidence(cells, cell_count, ders, der_count, evidence, witness,
                                            logical, revision)
            ? 1
            : 0;
}

int main() {
  ResidentRecipeCell cells[4]{};
  ResidentRecipeDerivation ders[4]{};
  ResidentPostbirthConstructorState state{};
  std::uint32_t cell_count = 0;
  if (!seed_bank(cells, ders, &state, &cell_count) || cell_count != 2u)
    return 2;
  const ResidentRecipeCell parent = cells[1];
  ResidentRecipeCell foreign{};
  foreign.logical_recipe_id = 0xA2;
  foreign.revision_identity = 0xB2;
  foreign.credit_q16 = 99;
  ResidentRecipeCell matching = parent;
  matching.revision_identity = 0xFEEDull;
  matching.credit_q16 = 99;
  DirectWhiteboxCondensationV1 next{};
  DirectWhiteboxCondensationV1 stamped{};
  if (!seed_affine(&next, 7 << 16, 0, 1 << 16, 1 << 16) ||
      !seed_affine(&stamped, 8 << 16, 0, 1 << 16, 0))
    return 3;
  ResidentRecipeCell expect[4] = {cells[0], cells[1]};
  ResidentRecipeDerivation expect_der[4] = {ders[0], ders[1]};
  ResidentPostbirthConstructorState expect_state = state;
  std::uint32_t expect_count = 2u;
  FoundryCondensationWorkspace host_ws{};
  host_ws.cells = expect;
  host_ws.cell_count = &expect_count;
  host_ws.cell_capacity = 4u;
  host_ws.derivations = expect_der;
  host_ws.state = &expect_state;
  host_ws.parent_cell = 1u;
  host_ws.parent_logical = parent.logical_recipe_id;
  host_ws.witness = &next;
  if (!foundry_materialize_from_workspace(&host_ws, &foreign) || expect_count != 3u ||
      expect[1].logical_recipe_id != parent.logical_recipe_id ||
      expect[1].revision_identity != parent.revision_identity || expect[1].credit_q16 != 0 ||
      expect_der[2].parent_logical_recipe_id != parent.logical_recipe_id)
    return 4;
  ResidentRecipeCell rec[3] = {expect[0], expect[1], expect[2]};
  ResidentRecipeDerivation der[3] = {expect_der[0], expect_der[1], expect_der[2]};
  for (std::uint32_t i = 0; i < 3u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
  }
  const std::uint32_t v0[] = {10u, 20u};
  const std::uint32_t v1[] = {20u, 30u};
  const std::uint32_t v2[] = {30u, 40u};
  ResidentRecipeOccurrence occ[3]{};
  ResidentOccurrenceCoupling edges[2]{};
  ResidentRelationalNetworkClosure claimed{};
  DirectWhiteboxCondensationV1 admit_w{};
  if (!bind_live(rec[0], der[0], v0, 0xE10, &occ[0], 1u, 1 << 16) ||
      !bind_live(rec[1], der[1], v1, 0xE11, &occ[1], 1u, 1 << 16) ||
      !bind_live(rec[2], der[2], v2, 0xE12, &occ[2], 1u, 1 << 16) ||
      !bind_resident_occurrence_coupling(occ[0], der[0], 1u, occ[1], der[1], 0u, &edges[0]) ||
      !bind_resident_occurrence_coupling(occ[1], der[1], 1u, occ[2], der[2], 0u, &edges[1]) ||
      !bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &claimed) ||
      !admit_resident_network_parent(rec, der, occ, 3u, edges, 2u, claimed.identity,
                                     rec[1].logical_recipe_id) ||
      admit_resident_network_parent(rec, der, occ, 3u, edges, 2u, claimed.identity ^ 1ull,
                                    rec[1].logical_recipe_id) ||
      !seed_affine(&admit_w, 9 << 16, 0, 1 << 16, 1 << 16))
    return 4;
  host_ws.claimed_identity = claimed.identity;
  host_ws.network_cells = rec;
  host_ws.network_ders = der;
  host_ws.occurrences = occ;
  host_ws.couplings = edges;
  host_ws.occurrence_count = 3u;
  host_ws.coupling_count = 2u;
  host_ws.witness = &admit_w;
  if (!foundry_workspace_ready(&host_ws))
    return 4;
  host_ws.claimed_identity ^= 1ull;
  if (foundry_workspace_ready(&host_ws))
    return 4;
  host_ws.claimed_identity = 0u;
  host_ws.witness = &next;
  ResidentRecipeOccurrence occ2[3]{};
  ResidentRecipeOccurrence lesion_occ[3]{};
  ResidentOccurrenceCoupling edges2[2]{};
  ResidentOccurrenceCoupling lesion[2]{};
  const std::uint32_t w1[] = {20u, 50u};
  const std::uint32_t w2[] = {50u, 60u};
  DirectWhiteboxCondensationV1 observed{};
  ResidentNetworkCondensationEvidence evidence{};
  if (!bind_live(rec[0], der[0], v0, 0xE20, &occ2[0], 2u, 2 << 16) ||
      !bind_live(rec[1], der[1], v1, 0xE21, &occ2[1], 2u, 2 << 16) ||
      !bind_live(rec[2], der[2], v2, 0xE22, &occ2[2], 2u, 2 << 16) ||
      !bind_live(rec[0], der[0], v0, 0xE30, &lesion_occ[0], 3u, 3 << 16) ||
      !bind_live(rec[1], der[1], w1, 0xE31, &lesion_occ[1], 3u, 3 << 16) ||
      !bind_live(rec[2], der[2], w2, 0xE32, &lesion_occ[2], 3u, 3 << 16) ||
      !bind_resident_occurrence_coupling(occ2[0], der[0], 1u, occ2[1], der[1], 0u, &edges2[0]) ||
      !bind_resident_occurrence_coupling(occ2[1], der[1], 1u, occ2[2], der[2], 0u, &edges2[1]) ||
      !bind_resident_occurrence_coupling(lesion_occ[0], der[0], 1u, lesion_occ[1], der[1], 0u,
                                         &lesion[0]) ||
      !bind_resident_occurrence_coupling(lesion_occ[1], der[1], 1u, lesion_occ[2], der[2], 0u,
                                         &lesion[1]) ||
      observe_resident_mixed_rank_whitebox(rec, der, occ, occ, 3u, edges, edges, 2u, &observed) ||
      observe_resident_mixed_rank_whitebox(rec, der, occ, lesion_occ, 3u, edges, lesion, 2u,
                                           &observed) ||
      !observe_resident_mixed_rank_whitebox(rec, der, occ, occ2, 3u, edges, edges2, 2u,
                                            &observed) ||
      observed.witness_identity == 0u ||
      observe_resident_mixed_rank_evidence(rec, der, occ, occ, 3u, edges, edges, 2u, &observed,
                                           &evidence) ||
      observe_resident_mixed_rank_evidence(rec, der, occ, lesion_occ, 3u, edges, lesion, 2u,
                                           &observed, &evidence) ||
      !observe_resident_mixed_rank_evidence(rec, der, occ, occ2, 3u, edges, edges2, 2u, &observed,
                                            &evidence) ||
      evidence.witness_identity == 0u ||
      evidence.witness_identity != resident_network_condensation_witness(evidence) ||
      evidence.boundary_identity != resident_condensation_boundary_identity(evidence) ||
      evidence.maximum_error_q16 != 0 || evidence.source_count != 2u ||
      evidence.sources[0].relation !=
          direct_relation_algebra_resident_word(DirectRelationAlgebraFamilyV1::linear) ||
      evidence.sources[0].relation == static_cast<std::uint32_t>(ResidentRecipeRelation::trigger))
    return 4;
  const std::uint64_t witness_before = observed.witness_identity;
  const std::int64_t credit1 = rec[1].credit_q16;
  rec[1].credit_q16 ^= 99;
  occ[1].eligibility_q16 = 17;
  DirectWhiteboxCondensationV1 smashed{};
  ResidentNetworkCondensationEvidence smashed_ev{};
  ResidentRelationalNetworkClosure n_elig{};
  if (!observe_resident_mixed_rank_whitebox(rec, der, occ, occ2, 3u, edges, edges2, 2u, &smashed) ||
      smashed.witness_identity != witness_before ||
      !observe_resident_mixed_rank_evidence(rec, der, occ, occ2, 3u, edges, edges2, 2u, &smashed,
                                            &smashed_ev) ||
      smashed_ev.witness_identity != evidence.witness_identity ||
      smashed_ev.boundary_identity != evidence.boundary_identity ||
      !bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &n_elig) ||
      n_elig.identity == claimed.identity ||
      n_elig.eligibility_signed_q16 == claimed.eligibility_signed_q16)
    return 4;
  rec[1].credit_q16 = credit1;
  occ[1].eligibility_q16 = 0;
  const std::uint32_t vb0[] = {70u, 80u};
  const std::uint32_t vb1[] = {80u, 90u};
  ResidentRecipeOccurrence occ_b[2]{};
  ResidentOccurrenceCoupling edge_b{};
  ResidentRecipeCell rec_b[2] = {rec[0], rec[0]};
  ResidentRecipeDerivation der_b[2] = {der[0], der[0]};
  ResidentRecipeCell rec5[5] = {rec[0], rec[1], rec[2], rec[0], rec[0]};
  ResidentRecipeDerivation der5[5] = {der[0], der[1], der[2], der[0], der[0]};
  ResidentRelationalNetworkClosure n_b{};
  ResidentRelationalNetworkClosure whole_n{};
  ResidentRelationalNetworkSet nets{};
  DirectWhiteboxCondensationV1 whole_w{};
  if (!bind_live(rec[0], der[0], vb0, 0xB10, &occ_b[0], 4u, 4 << 16) ||
      !bind_live(rec[0], der[0], vb1, 0xB11, &occ_b[1], 4u, 4 << 16) ||
      !bind_resident_occurrence_coupling(occ_b[0], der[0], 1u, occ_b[1], der[0], 0u, &edge_b) ||
      !bind_resident_relational_network_closure(rec_b, der_b, occ_b, 2u, &edge_b, 1u, &n_b) ||
      n_b.identity == 0u || n_b.identity == claimed.identity)
    return 4;
  ResidentRecipeOccurrence occ5[5] = {occ[0], occ[1], occ[2], occ_b[0], occ_b[1]};
  ResidentOccurrenceCoupling edges5[3] = {edges[0], edges[1], edge_b};
  occ_b[0].eligibility_q16 = 31;
  occ5[3].eligibility_q16 = 31;
  if (!bind_resident_relational_network_set(rec5, der5, occ5, 5u, edges5, 3u, &nets) ||
      nets.network_count != 2u || nets.networked_occurrence_count != 5u ||
      nets.isolated_occurrence_count != 0u ||
      (nets.networks[0].identity != claimed.identity &&
       nets.networks[1].identity != claimed.identity) ||
      bind_resident_relational_network_closure(rec5, der5, occ5, 5u, edges5, 3u, &whole_n) ||
      observe_resident_mixed_rank_whitebox(rec5, der5, occ5, occ5, 5u, edges5, edges5, 3u,
                                           &whole_w) ||
      !observe_resident_mixed_rank_whitebox(rec, der, occ, occ2, 3u, edges, edges2, 2u, &whole_w) ||
      whole_w.witness_identity != observed.witness_identity)
    return 4;
  std::uint64_t replay_logical = 0, replay_rev = 0;
  ResidentNetworkCondensationEvidence tampered = evidence;
  tampered.witness_identity ^= 1ull;
  DirectWhiteboxCondensationV1 bad_w = observed;
  bad_w.witness_identity ^= 1ull;
  if (!replay_resident_network_candidate(rec, 3u, der, 3u, evidence, &observed, &replay_logical,
                                         &replay_rev) ||
      replay_logical != resident_whitebox_recipe_logical_identity(observed) ||
      replay_resident_network_candidate(rec, 3u, der, 3u, evidence, nullptr, &replay_logical,
                                        &replay_rev) ||
      replay_resident_network_candidate(rec, 3u, der, 3u, tampered, &observed, &replay_logical,
                                        &replay_rev) ||
      replay_resident_network_candidate(rec, 3u, der, 3u, evidence, &bad_w, &replay_logical,
                                        &replay_rev))
    return 4;
  host_ws.witness = &observed;
  host_ws.evidence = &evidence;
  host_ws.claimed_identity = claimed.identity;
  if (!foundry_workspace_ready(&host_ws))
    return 4;
  host_ws.evidence = &tampered;
  if (foundry_workspace_ready(&host_ws))
    return 4;
  host_ws.evidence = nullptr;
  host_ws.claimed_identity = 0u;
  host_ws.witness = &next;
  ResidentRecipeCell stamp_expect[7] = {cells[0], cells[1]};
  ResidentRecipeDerivation stamp_der[7] = {ders[0], ders[1]};
  ResidentPostbirthConstructorState stamp_state = state;
  std::uint32_t stamp_count = 2u;
  host_ws.cells = stamp_expect;
  host_ws.cell_count = &stamp_count;
  host_ws.derivations = stamp_der;
  host_ws.state = &stamp_state;
  host_ws.witness = &stamped;
  if (!foundry_materialize_from_workspace(&host_ws, &matching) || stamp_count != 3u ||
      stamp_expect[1].revision_identity != 0xFEEDull || stamp_expect[1].credit_q16 != 99 ||
      stamp_der[2].parent_logical_recipe_id != parent.logical_recipe_id)
    return 4;
  const std::uint32_t held_a[] = {50u, 60u};
  const std::uint32_t held_b[] = {70u, 80u};
  ResidentRecipeOccurrence surface_a{}, surface_b{};
  if (!bind_live(stamp_expect[2], stamp_der[2], held_a, 0xF50, &surface_a) ||
      !bind_live(stamp_expect[2], stamp_der[2], held_b, 0xF70, &surface_b) ||
      surface_a.logical_recipe_id != stamp_expect[2].logical_recipe_id ||
      surface_b.logical_recipe_id != stamp_expect[2].logical_recipe_id ||
      surface_a.bindings[0].variable_identity != 50u ||
      surface_b.bindings[0].variable_identity != 70u)
    return 4;
  const std::uint32_t va[] = {50u, 60u};
  const std::uint32_t vp[] = {60u, 70u};
  ResidentRecipeOccurrence q_anc{}, q_pn{};
  ResidentOccurrenceCoupling edge_next{};
  ResidentRelationalNetworkClosure n_next{};
  ResidentRecipeCell rec_next[2] = {stamp_expect[0], stamp_expect[2]};
  ResidentRecipeDerivation der_next[2] = {stamp_der[0], stamp_der[2]};
  der_next[0].logical_recipe_id = stamp_expect[0].logical_recipe_id;
  der_next[0].revision_identity = stamp_expect[0].revision_identity;
  der_next[1].logical_recipe_id = stamp_expect[2].logical_recipe_id;
  der_next[1].revision_identity = stamp_expect[2].revision_identity;
  if (!bind_live(stamp_expect[0], der_next[0], va, 0xA50, &q_anc, 5u, 5 << 16) ||
      !bind_live(stamp_expect[2], der_next[1], vp, 0xA60, &q_pn, 5u, 5 << 16) ||
      !bind_resident_occurrence_coupling(q_anc, der_next[0], 1u, q_pn, der_next[1], 0u, &edge_next))
    return 4;
  ResidentRecipeOccurrence occ_next[2] = {q_anc, q_pn};
  if (!bind_resident_relational_network_closure(rec_next, der_next, occ_next, 2u, &edge_next, 1u,
                                                &n_next) ||
      n_next.identity == 0u || n_next.identity == claimed.identity || n_next.actual_count != 2u ||
      !admit_resident_network_parent(rec_next, der_next, occ_next, 2u, &edge_next, 1u,
                                     n_next.identity, stamp_expect[2].logical_recipe_id) ||
      admit_resident_network_parent(rec_next, der_next, occ_next, 2u, &edge_next, 1u,
                                    claimed.identity, stamp_expect[2].logical_recipe_id))
    return 4;
  ResidentRecipeCell rec_r[2] = {stamp_expect[1], stamp_expect[2]};
  ResidentRecipeDerivation der_r[2] = {stamp_der[1], stamp_der[2]};
  der_r[0].logical_recipe_id = stamp_expect[1].logical_recipe_id;
  der_r[0].revision_identity = stamp_expect[1].revision_identity;
  der_r[1].logical_recipe_id = stamp_expect[2].logical_recipe_id;
  der_r[1].revision_identity = stamp_expect[2].revision_identity;
  const std::uint32_t r0[] = {50u, 60u};
  const std::uint32_t r1[] = {60u, 70u};
  ResidentRecipeOccurrence occ_r1[2]{}, occ_r2[2]{};
  ResidentOccurrenceCoupling e_r1[1]{}, e_r2[1]{};
  DirectWhiteboxCondensationV1 again{};
  ResidentNetworkCondensationEvidence ev2{};
  std::uint64_t again_logical = 0, again_rev = 0;
  if (!bind_live(stamp_expect[1], der_r[0], r0, 0xC10, &occ_r1[0], 6u, 6 << 16) ||
      !bind_live(stamp_expect[2], der_r[1], r1, 0xC11, &occ_r1[1], 6u, 6 << 16) ||
      !bind_live(stamp_expect[1], der_r[0], r0, 0xC20, &occ_r2[0], 7u, 7 << 16) ||
      !bind_live(stamp_expect[2], der_r[1], r1, 0xC21, &occ_r2[1], 7u, 7 << 16) ||
      !bind_resident_occurrence_coupling(occ_r1[0], der_r[0], 1u, occ_r1[1], der_r[1], 0u,
                                         &e_r1[0]) ||
      !bind_resident_occurrence_coupling(occ_r2[0], der_r[0], 1u, occ_r2[1], der_r[1], 0u,
                                         &e_r2[0]) ||
      observe_resident_mixed_rank_whitebox(rec_r, der_r, occ_r1, occ_r1, 2u, e_r1, e_r1, 1u,
                                           &again) ||
      !observe_resident_mixed_rank_whitebox(rec_r, der_r, occ_r1, occ_r2, 2u, e_r1, e_r2, 1u,
                                            &again) ||
      again.witness_identity == 0u || again.witness_identity == observed.witness_identity ||
      !observe_resident_mixed_rank_evidence(rec_r, der_r, occ_r1, occ_r2, 2u, e_r1, e_r2, 1u,
                                            &again, &ev2) ||
      !replay_resident_network_candidate(stamp_expect, stamp_count, stamp_der,
                                         stamp_state.derivation_count, ev2, &again, &again_logical,
                                         &again_rev) ||
      again_logical == 0u || again_logical == stamp_expect[2].logical_recipe_id)
    return 4;
  ResidentRelationalNetworkClosure n_r{};
  if (!bind_resident_relational_network_closure(rec_r, der_r, occ_r1, 2u, e_r1, 1u, &n_r) ||
      n_r.identity == 0u)
    return 4;
  const ResidentRecipeCell later_parent = stamp_expect[2];
  host_ws.cells = stamp_expect;
  host_ws.cell_count = &stamp_count;
  host_ws.cell_capacity = 4u;
  host_ws.derivations = stamp_der;
  host_ws.state = &stamp_state;
  host_ws.claimed_identity = n_r.identity;
  host_ws.network_cells = rec_r;
  host_ws.network_ders = der_r;
  host_ws.occurrences = occ_r1;
  host_ws.couplings = e_r1;
  host_ws.occurrence_count = 2u;
  host_ws.coupling_count = 1u;
  host_ws.parent_cell = 2u;
  host_ws.parent_logical = later_parent.logical_recipe_id;
  host_ws.witness = &again;
  host_ws.evidence = &ev2;
  ResidentRecipeOccurrence surface_pn2{};
  const std::uint32_t held_pn2[] = {90u, 100u};
  if (!foundry_workspace_ready(&host_ws) ||
      !foundry_materialize_from_workspace(&host_ws, nullptr) || stamp_count != 4u ||
      stamp_expect[2].logical_recipe_id != later_parent.logical_recipe_id ||
      stamp_expect[2].revision_identity != later_parent.revision_identity ||
      stamp_expect[3].logical_recipe_id != again_logical ||
      stamp_der[3].parent_logical_recipe_id != later_parent.logical_recipe_id ||
      !replay_resident_whitebox_recipe(again, stamp_expect[3], stamp_der[3]) ||
      !bind_live(stamp_expect[3], stamp_der[3], held_pn2, 0xF90, &surface_pn2) ||
      surface_pn2.logical_recipe_id != stamp_expect[3].logical_recipe_id)
    return 4;
  LanguageAfterPn2 after{};
  after.cells = stamp_expect;
  after.cell_count = stamp_count;
  after.ders = stamp_der;
  after.der_count = stamp_state.derivation_count;
  after.i_anc = 0u;
  after.i_src = 1u;
  after.i_pn1 = 2u;
  after.i_pn2 = 3u;
  after.rec_r = rec_r;
  after.der_r = der_r;
  after.occ_r1 = occ_r1;
  after.occ_r2 = occ_r2;
  after.e_r1 = e_r1;
  after.e_r2 = e_r2;
  after.n_r = n_r;
  after.n_next = n_next;
  after.again = &again;
  after.ev2 = &ev2;
  after.again_logical = again_logical;
  after.workspace = &host_ws;
  stamp_state.recipe_cell_capacity = 5u;
  stamp_state.derivation_capacity = 5u;
  host_ws.cell_capacity = 5u;
  if (!language_after_pn2(after))
    return 4;

  ResidentRecipeCell* dcells = nullptr;
  ResidentRecipeDerivation* dders = nullptr;
  ResidentPostbirthConstructorState* dstate = nullptr;
  std::uint32_t* dcount = nullptr;
  int* dok = nullptr;
  ck(cudaMalloc(&dcells, sizeof(cells)), "cells");
  ck(cudaMalloc(&dders, sizeof(ders)), "ders");
  ck(cudaMalloc(&dstate, sizeof(state)), "state");
  ck(cudaMalloc(&dcount, sizeof(cell_count)), "count");
  ck(cudaMalloc(&dok, sizeof(int)), "ok");
  ck(cudaMemcpy(dcells, cells, sizeof(cells), cudaMemcpyHostToDevice), "h2d cells");
  ck(cudaMemcpy(dders, ders, sizeof(ders), cudaMemcpyHostToDevice), "h2d ders");
  ck(cudaMemcpy(dstate, &state, sizeof(state), cudaMemcpyHostToDevice), "h2d state");
  ck(cudaMemcpy(dcount, &cell_count, sizeof(cell_count), cudaMemcpyHostToDevice), "h2d count");

  int ok = 0;
  std::uint32_t after_mismatch = 0;
  from_workspace<<<1, 1>>>(dcells, dcount, dders, dstate, foreign, 1u, 0u, next, dok);
  ck(cudaGetLastError(), "unnamed launch");
  ck(cudaDeviceSynchronize(), "unnamed sync");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "unnamed ok");
  ck(cudaMemcpy(&after_mismatch, dcount, sizeof(after_mismatch), cudaMemcpyDeviceToHost),
     "unnamed count");
  if (ok || after_mismatch != 2u)
    return 5;
  from_workspace<<<1, 1>>>(dcells, dcount, dders, dstate, foreign, 1u, 0xDEADull, next, dok);
  ck(cudaGetLastError(), "mismatch launch");
  ck(cudaDeviceSynchronize(), "mismatch sync");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "mismatch ok");
  ck(cudaMemcpy(&after_mismatch, dcount, sizeof(after_mismatch), cudaMemcpyDeviceToHost),
     "mismatch count");
  if (ok || after_mismatch != 2u)
    return 5;

  from_workspace<<<1, 1>>>(dcells, dcount, dders, dstate, foreign, 1u, parent.logical_recipe_id,
                           next, dok);
  ck(cudaGetLastError(), "iso launch");
  ck(cudaDeviceSynchronize(), "iso sync");
  ResidentRecipeCell got[4]{};
  ResidentRecipeDerivation got_der[4]{};
  std::uint32_t got_count = 0;
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "iso ok");
  ck(cudaMemcpy(got, dcells, sizeof(got), cudaMemcpyDeviceToHost), "iso cells");
  ck(cudaMemcpy(got_der, dders, sizeof(got_der), cudaMemcpyDeviceToHost), "iso ders");
  ck(cudaMemcpy(&got_count, dcount, sizeof(got_count), cudaMemcpyDeviceToHost), "iso count");
  if (!ok || got_count != 3u || got[1].logical_recipe_id != parent.logical_recipe_id ||
      got[1].revision_identity != parent.revision_identity || got[1].credit_q16 != 0 ||
      got[2].logical_recipe_id != expect[2].logical_recipe_id ||
      got_der[2].parent_logical_recipe_id != parent.logical_recipe_id)
    return 6;

  ck(cudaMemcpy(dcells, cells, sizeof(cells), cudaMemcpyHostToDevice), "stamp cells");
  ck(cudaMemcpy(dders, ders, sizeof(ders), cudaMemcpyHostToDevice), "stamp ders");
  ck(cudaMemcpy(dstate, &state, sizeof(state), cudaMemcpyHostToDevice), "stamp state");
  ck(cudaMemcpy(dcount, &cell_count, sizeof(cell_count), cudaMemcpyHostToDevice), "stamp count");
  from_workspace<<<1, 1>>>(dcells, dcount, dders, dstate, matching, 1u, parent.logical_recipe_id,
                           stamped, dok);
  ck(cudaGetLastError(), "stamp launch");
  ck(cudaDeviceSynchronize(), "stamp sync");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "stamp ok");
  ck(cudaMemcpy(got, dcells, sizeof(got), cudaMemcpyDeviceToHost), "stamp got");
  ck(cudaMemcpy(&got_count, dcount, sizeof(got_count), cudaMemcpyDeviceToHost), "stamp got count");
  if (!ok || got_count != 3u || got[1].revision_identity != 0xFEEDull || got[1].credit_q16 != 99 ||
      got[2].logical_recipe_id != stamp_expect[2].logical_recipe_id)
    return 7;

  ck(cudaMemcpy(dcells, cells, sizeof(cells), cudaMemcpyHostToDevice), "bad cells");
  ck(cudaMemcpy(dders, ders, sizeof(ders), cudaMemcpyHostToDevice), "bad ders");
  ck(cudaMemcpy(dstate, &state, sizeof(state), cudaMemcpyHostToDevice), "bad state");
  ck(cudaMemcpy(dcount, &cell_count, sizeof(cell_count), cudaMemcpyHostToDevice), "bad count");
  DirectWhiteboxCondensationV1 empty{};
  from_workspace<<<1, 1>>>(dcells, dcount, dders, dstate, matching, 1u, parent.logical_recipe_id,
                           empty, dok);
  ck(cudaGetLastError(), "bad launch");
  ck(cudaDeviceSynchronize(), "bad sync");
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "bad ok");
  ck(cudaMemcpy(got, dcells, sizeof(got), cudaMemcpyDeviceToHost), "bad got");
  ck(cudaMemcpy(&got_count, dcount, sizeof(got_count), cudaMemcpyDeviceToHost), "bad got count");
  if (ok || got_count != 2u || got[1].revision_identity != parent.revision_identity ||
      got[1].credit_q16 != parent.credit_q16)
    return 8;

  ResidentRecipeCell* drec = nullptr;
  ResidentRecipeDerivation* dder = nullptr;
  ResidentRecipeOccurrence* docc = nullptr;
  ResidentRecipeOccurrence* docc2 = nullptr;
  ResidentOccurrenceCoupling* dedge = nullptr;
  ResidentOccurrenceCoupling* dedge2 = nullptr;
  DirectWhiteboxCondensationV1* dwit = nullptr;
  ck(cudaMalloc(&drec, sizeof(rec)), "drec");
  ck(cudaMalloc(&dder, sizeof(der)), "dder");
  ck(cudaMalloc(&docc, sizeof(occ)), "docc");
  ck(cudaMalloc(&docc2, sizeof(occ2)), "docc2");
  ck(cudaMalloc(&dedge, sizeof(edges)), "dedge");
  ck(cudaMalloc(&dedge2, sizeof(edges2)), "dedge2");
  ck(cudaMalloc(&dwit, sizeof(observed)), "dwit");
  ck(cudaMemcpy(drec, rec, sizeof(rec), cudaMemcpyHostToDevice), "h2d rec");
  ck(cudaMemcpy(dder, der, sizeof(der), cudaMemcpyHostToDevice), "h2d der");
  ck(cudaMemcpy(docc, occ, sizeof(occ), cudaMemcpyHostToDevice), "h2d occ");
  ck(cudaMemcpy(docc2, occ2, sizeof(occ2), cudaMemcpyHostToDevice), "h2d occ2");
  ck(cudaMemcpy(dedge, edges, sizeof(edges), cudaMemcpyHostToDevice), "h2d edge");
  ck(cudaMemcpy(dedge2, edges2, sizeof(edges2), cudaMemcpyHostToDevice), "h2d edge2");
  observe_n<<<1, 1>>>(drec, dder, docc, docc2, dedge, dedge2, dwit, dok);
  ck(cudaGetLastError(), "observe launch");
  ck(cudaDeviceSynchronize(), "observe sync");
  DirectWhiteboxCondensationV1 device_w{};
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "observe ok");
  ck(cudaMemcpy(&device_w, dwit, sizeof(device_w), cudaMemcpyDeviceToHost), "observe wit");
  if (!ok || device_w.witness_identity != observed.witness_identity)
    return 9;
  ResidentNetworkCondensationEvidence* dev = nullptr;
  DirectWhiteboxCondensationV1* dwit2 = nullptr;
  ck(cudaMalloc(&dev, sizeof(evidence)), "dev");
  ck(cudaMalloc(&dwit2, sizeof(observed)), "dwit2");
  observe_ev<<<1, 1>>>(drec, dder, docc, docc2, dedge, dedge2, dwit2, dev, dok);
  ck(cudaGetLastError(), "evidence launch");
  ck(cudaDeviceSynchronize(), "evidence sync");
  ResidentNetworkCondensationEvidence device_ev{};
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "evidence ok");
  ck(cudaMemcpy(&device_ev, dev, sizeof(device_ev), cudaMemcpyDeviceToHost), "evidence ev");
  if (!ok || device_ev.witness_identity != evidence.witness_identity ||
      device_ev.boundary_identity != evidence.boundary_identity)
    return 9;
  std::uint64_t* dlogical = nullptr;
  std::uint64_t* drev = nullptr;
  ck(cudaMalloc(&dlogical, sizeof(std::uint64_t)), "dlogical");
  ck(cudaMalloc(&drev, sizeof(std::uint64_t)), "drev");
  replay_ev<<<1, 1>>>(drec, 3u, dder, 3u, evidence, observed, dlogical, drev, dok);
  ck(cudaGetLastError(), "replay launch");
  ck(cudaDeviceSynchronize(), "replay sync");
  std::uint64_t device_logical = 0;
  ck(cudaMemcpy(&ok, dok, sizeof(ok), cudaMemcpyDeviceToHost), "replay ok");
  ck(cudaMemcpy(&device_logical, dlogical, sizeof(device_logical), cudaMemcpyDeviceToHost),
     "replay logical");
  if (!ok || device_logical != replay_logical)
    return 9;

  std::printf(
      "FOUNDRY_PARENT_ISOLATION CUDA GREEN host_device_match=1 "
      "parent_isolation=1 parent_logical_refuse=1 unnamed_parent_refuse=1 "
      "matching_stamp=1 invalid_witness_no_stamp=1 "
      "network_admit=1 stale_network_refuse=1 "
      "mixed_rank_observe=1 observe_repeat=1 observe_bn_mismatch_refuse=1 "
      "observe_single_refuse=1 mixed_rank_evidence=1 evidence_witness=1 "
      "evidence_boundary=1 mixed_rank_replay=1 replay_match_mint=1 "
      "replay_tamper_refuse=1 replay_dispatch=1 evidence_ready=1 "
      "evidence_ready_tamper=1 affine_relation_tag=1 dispatch_by_tag=1 "
      "credit_eligibility_not_witness=1 n_eligibility_identity=1 "
      "heldout_surface_rebind=1 "
      "coactive_construction_n=1 whole_frontier_one_n_refuse=1 "
      "n_of_n_bind=1 ancestor_cross_rank=1 "
      "recurse_observe=1 recurse_replay=1 recurse_mint=1 "
      "pn2_cross_rank=1 pn2_stale_source_n_refuse=1 "
      "source_n_survives_condense=1 pn2_lesion_admit_refuse=1 "
      "ambiguous_bn=1 nominate_alt_not_source=1 "
      "withdraw_alt_leaves_source=1 withdraw_source_leaves_alt=1 "
      "same_recipe_two_surfaces=1 surface_bn_differs=1 fold_ignores_surface=1 "
      "surface_replay_same_logical=1 surface_alt_witness_refuse=1 "
      "utterance_obs_differ=1 utterance_evidence_not_recipe=1 "
      "express_recipe_new_surface=1 "
      "nominate_credit_source=1 alt_bn_credit_zero=1 elig_refresh_keeps_n_credit=1 "
      "withdraw_alt_select_source=1 withdraw_source_select_none=1 "
      "equal_credit_select_none=1 break_tie_selects_source=1 "
      "surface_reuses_recruitment=1 surface_inherits_credit=1 "
      "hybrid_new_recruitment=1 hybrid_credit_zero=1 "
      "reafferent_no_couple=1 reafferent_no_construction_n=1 "
      "reafferent_not_world_observe=1 "
      "world_observe_after_express=1 lesion_new_surface_keeps_old=1 "
      "chained_express_is_n=1 chained_express_new_rid=1 "
      "chained_express_not_source_observe=1 "
      "motor_whole_construction=1 motor_fragment_zero=1 "
      "triple_linear_fold=1 triple_linear_evidence_refuse=1 "
      "triple_fold_not_minted=1 triple_fold_evidence_replay_refuse=1 "
      "imagine_endogenous_n=1 imagine_actual_count_zero=1 "
      "imagine_observe_refuse=1 imagine_same_rid=1 "
      "imagine_coactive_not_one_n=1 imagine_authority_fake_refuse=1 "
      "heard_settle_n_refuse=1 settle_keeps_recipe=1 "
      "imagine_after_settle=1 fresh_utterance_after_settle=1 "
      "settle_observe_refuse=1 settle_evidence_replay=1 "
      "imagine_admit_refuse=1 heard_admit_member=1 "
      "causal_cross_ecology_n=1 causal_reconverge=1 "
      "causal_fold_refuse=1 formal_two_n=1 "
      "asif_cross_n=1 asif_actual_count_one=1 "
      "asif_admit_refuse=1 asif_fold_refuse=1 "
      "two_cue_same_node=1 two_cue_two_n=1 "
      "two_cue_coalition_admit_refuse=1 two_cue_coalition_fold_refuse=1 "
      "c4_linear_fold_refuse=1 c4_linear_observe_refuse=1 "
      "w5_c2_fold=1 w5_c2_observe=1 w5_c2_same_depth=1 "
      "multilingual_two_n=1 multilingual_same_depth=1 multilingual_shared_recruitment=1 "
      "multilingual_shared_world=1 "
      "discourse_settle_cross_refuse=1 discourse_later_fold=1 discourse_later_world=1 "
      "recursive_wide_fold=1 recursive_wide_observe=1 recursive_wide_new_logical=1 "
      "recursive_wide_mint=1 recursive_wide_unfold=1 "
      "recursive_wide_settle_refuse=1 recursive_pn3_after_settle=1 recursive_again_observe=1 "
      "pn3_world_n=1 pn3_world_fold_refuse=1 "
      "pn3_two_surfaces=1 pn3_two_n=1 "
      "pn3_lesion_a_refuse=1 pn3_surface_b_keeps_world=1 "
      "pn3_both_surfaces_dead=1 world_after_language_lesion=1 "
      "pn3_returns=1 pn3_returns_world=1 "
      "pn3_imagine=1 pn3_imagine_actual_one=1 pn3_imagine_admit_refuse=1 "
      "pn3_heard_imagine_two_n=1 pn3_coalition_admit_refuse=1 "
      "pn4_mint=1 pn4_unfold=1 pn4_parent_pn3=1 "
      "phrase_compose_fold=1 phrase_compose_observe=1 phrase_compose_new_logical=1 "
      "phrase_compose_deep_fold_refuse=1 "
      "phrase_compose_mint=1 phrase_compose_unfold=1 phrase_compose_parent_pn2=1 "
      "phrase_recipe_n=1 phrase_recipe_fold_matches_observe=1 "
      "phrase_opportunity=1 phrase_opportunity_stale_bank_refuse=1 "
      "phrase_two_surfaces=1 phrase_two_surfaces_same_logical=1 phrase_cross_surface_refuse=1 "
      "phrase_bank_record=1 phrase_bank_after_express_gone=1 phrase_bank_wrong_refuse=1 "
      "phrase_bank_idempotent=1 phrase_bank_two_surfaces_one_recipe=1 "
      "phrase_bank_nominate_cannot_write=1 phrase_bank_dead_express_refuse=1 "
      "phrase_bank_map_survives_write_refuse=1 "
      "phrase_bank_ambiguous_refuse=1 phrase_bank_unique_still_nominates=1 "
      "phrase_world_bank_selects=1 phrase_world_n=1 phrase_world_no_teach=1 "
      "phrase_world_two_n=1 phrase_world_two_same_recipe=1 phrase_world_two_rid_differs=1 "
      "phrase_world_lesion_dead_n=1 phrase_world_lesion_fresh_n=1 "
      "phrase_world_lesion_bank_survives=1 "
      "phrase_bank_lesion_nominate_refuse=1 phrase_bank_lesion_recipe_n=1 "
      "phrase_bank_relearn=1 phrase_bank_relearn_same_recipe=1 phrase_bank_relearn_no_mint=1 "
      "phrase_world_revise=1 phrase_world_revise_nom_refuse=1 phrase_world_revise_bank_stale=1 "
      "phrase_revise_rerecord=1 phrase_revise_rerecord_nom_refuse=1 phrase_revise_der_stale=1 "
      "phrase_revise_world_bind=1 phrase_revise_world_couple_refuse=1 "
      "phrase_revise_world_nom_refuse=1 "
      "phrase_revise_pair=1 phrase_revise_pair_nominate=1 phrase_revise_pair_n=1 "
      "phrase_revise_authority_experience=1 phrase_revise_authority_nom_none=1 "
      "adult_include=0 graph_flip=0\n");
  return 0;
}

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_language_recipe_opportunity.cuh"
#include "hardware_native/direct_adult_language_recipe_opportunity_bank.cuh"
#include "hardware_native/direct_adult_surface_sequence.cuh"
#include "relseq_hierarchy_fixture.inl"

using namespace relseq_hierarchy;
using substrate::direct_adult_core::admit_resident_network_parent;
using substrate::direct_adult_core::apply_resident_recruited_network_credit;
using substrate::direct_adult_core::bind_resident_occurrence_causal_intersection_coupling;
using substrate::direct_adult_core::bind_resident_occurrence_coupling;
using substrate::direct_adult_core::bind_resident_relational_network_closure;
using substrate::direct_adult_core::bind_resident_relational_network_set;
using substrate::direct_adult_core::DirectParticipationDescriptor;
using substrate::direct_adult_core::fold_resident_mixed_rank_affine;
using substrate::direct_adult_core::foundry_materialize_from_workspace;
using substrate::direct_adult_core::foundry_workspace_ready;
using substrate::direct_adult_core::observe_resident_mixed_rank_evidence;
using substrate::direct_adult_core::observe_resident_mixed_rank_whitebox;
using substrate::direct_adult_core::plan_resident_recruited_network_credit;
using substrate::direct_adult_core::recruit_resident_relational_network_set;
using substrate::direct_adult_core::replay_resident_mixed_rank_evidence;
using substrate::direct_adult_core::replay_resident_network_candidate;
using substrate::direct_adult_core::replay_resident_whitebox_recipe;
using substrate::direct_adult_core::resident_network_boundary_relation_equal;
using substrate::direct_adult_core::resident_relational_network_recruitment_identity;
using substrate::direct_adult_core::resident_whitebox_recipe_logical_identity;
using substrate::direct_adult_core::ResidentOccurrenceCoupling;
using substrate::direct_adult_core::ResidentOccurrenceCouplingKind;
using substrate::direct_adult_core::ResidentRecruitedNetworkCreditPlan;
using substrate::direct_adult_core::ResidentRelationalNetworkClosure;
using substrate::direct_adult_core::ResidentRelationalNetworkSet;
using substrate::direct_adult_core::select_resident_recruited_network;
using substrate::direct_network::direct_relation_algebra_resident_word;
using substrate::direct_network::direct_relation_algebra_resident_word_parse;
using substrate::direct_network::DirectRelationAlgebraFamilyV1;
using substrate::direct_network::resident_condensation_boundary_identity;
using substrate::direct_network::resident_network_condensation_witness;
using substrate::direct_network::ResidentNetworkCondensationEvidence;
using substrate::direct_network::ResidentRecipePortDirection;
using substrate::direct_network::ResidentRecipeRelation;

static bool bind_live(const ResidentRecipeCell& cell, const ResidentRecipeDerivation& derivation,
                      const std::uint32_t* vars, std::uint64_t oid, ResidentRecipeOccurrence* out,
                      std::uint32_t incarnation = 1u, std::int32_t activation_q16 = 0) {
  if (!bind_resident_recipe_occurrence(cell, derivation, vars, 2u, oid, oid + 1000u, 77u,
                                       incarnation, ResidentOccurrenceLineageKind::actual,
                                       DirectParticipationAuthority::independent_external, 9u, 1u,
                                       100u, 0, out))
    return false;
  out->activation_q16 = activation_q16;
  return true;
}

static bool boundary_has(const ResidentRelationalNetworkClosure& closure, std::uint32_t var,
                         ResidentRecipePortDirection direction) {
  for (std::uint32_t i = 0; i < closure.boundary_count; ++i)
    if (closure.boundary[i].variable_identity == var && closure.boundary[i].direction == direction)
      return true;
  return false;
}

// clang-format off: language_after_pn2 calls the helper defined by the first splice.
#include "nominate_source_network_credit.inl"
using substrate::direct_adult_core::resident_motor_candidate_network_credit_q16;
#include "language_after_pn2.inl"
// clang-format on

static bool bind_mixed_rank_network(ResidentRecipeCell* cells, std::uint32_t cell_count,
                                    ResidentRecipeDerivation* derivations) {
  if (!cells || !derivations || cell_count < 3u)
    return false;
  ResidentRecipeCell rec[3] = {cells[0], cells[1], cells[2]};
  ResidentRecipeDerivation der[3] = {derivations[0], derivations[1], derivations[2]};
  for (std::uint32_t i = 0; i < 3u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
  }
  if (der[0].generation == der[1].generation || der[1].generation == der[2].generation)
    return false;
  const std::uint32_t v0[] = {10u, 20u};
  const std::uint32_t v1[] = {20u, 30u};
  const std::uint32_t v2[] = {30u, 40u};
  ResidentRecipeOccurrence occ[3]{};
  if (!bind_live(rec[0], der[0], v0, 0xD10, &occ[0]) ||
      !bind_live(rec[1], der[1], v1, 0xD11, &occ[1]) ||
      !bind_live(rec[2], der[2], v2, 0xD12, &occ[2]))
    return false;
  ResidentOccurrenceCoupling edges[2]{};
  if (!bind_resident_occurrence_coupling(occ[0], der[0], 1u, occ[1], der[1], 0u, &edges[0]) ||
      !bind_resident_occurrence_coupling(occ[1], der[1], 1u, occ[2], der[2], 0u, &edges[1]))
    return false;
  ResidentRelationalNetworkClosure closure{};
  if (!bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &closure) ||
      closure.identity == 0u || closure.occurrence_count != 3u || closure.actual_count != 3u ||
      closure.boundary_count != 2u ||
      !boundary_has(closure, 10u, ResidentRecipePortDirection::input) ||
      !boundary_has(closure, 40u, ResidentRecipePortDirection::output) ||
      boundary_has(closure, 20u, ResidentRecipePortDirection::output) ||
      boundary_has(closure, 30u, ResidentRecipePortDirection::input))
    return false;
  ResidentRecipeCell rec_rev[3] = {rec[2], rec[0], rec[1]};
  ResidentRecipeDerivation der_rev[3] = {der[2], der[0], der[1]};
  ResidentRecipeOccurrence occ_rev[3] = {occ[2], occ[0], occ[1]};
  ResidentRelationalNetworkClosure again{};
  if (!bind_resident_relational_network_closure(rec_rev, der_rev, occ_rev, 3u, edges, 2u, &again) ||
      again.identity != closure.identity)
    return false;
  if (bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 1u, &again))
    return false;
  ResidentRecipeCell rec4[4] = {rec[0], rec[1], rec[2], rec[0]};
  ResidentRecipeDerivation der4[4] = {der[0], der[1], der[2], der[0]};
  ResidentRecipeOccurrence occ4[4] = {occ[0], occ[1], occ[2], {}};
  const std::uint32_t v_share[] = {20u, 40u};
  if (!bind_live(rec4[3], der4[3], v_share, 0xD13, &occ4[3]))
    return false;
  ResidentOccurrenceCoupling share_edges[3] = {edges[0], edges[1], {}};
  if (!bind_resident_occurrence_coupling(occ4[0], der4[0], 1u, occ4[3], der4[3], 0u,
                                         &share_edges[2]))
    return false;
  if (bind_resident_relational_network_closure(rec4, der4, occ4, 4u, share_edges, 3u, &again))
    return false;
  rec4[3] = rec[2];
  der4[3] = der[2];
  const std::uint32_t v3[] = {40u, 50u};
  if (!bind_live(rec4[3], der4[3], v3, 0xD14, &occ4[3]))
    return false;
  ResidentOccurrenceCoupling edges3[3] = {edges[0], edges[1], {}};
  if (!bind_resident_occurrence_coupling(occ4[2], der4[2], 1u, occ4[3], der4[3], 0u, &edges3[2]))
    return false;
  ResidentRelationalNetworkClosure lesioned{};
  if (!bind_resident_relational_network_closure(rec4, der4, occ4, 4u, edges3, 3u, &lesioned) ||
      lesioned.identity == closure.identity || lesioned.boundary_count != 2u ||
      !boundary_has(lesioned, 10u, ResidentRecipePortDirection::input) ||
      !boundary_has(lesioned, 50u, ResidentRecipePortDirection::output) ||
      boundary_has(lesioned, 40u, ResidentRecipePortDirection::output))
    return false;
  if (!commit_resident_causal_difference_revision(&occ[1], &rec[1], 1u, true, 1, 0))
    return false;
  return !bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &again);
}

static bool closure_has_logical(const ResidentRelationalNetworkClosure& closure,
                                std::uint64_t logical) {
  for (std::uint32_t i = 0; i < closure.occurrence_count; ++i)
    if (closure.members[i].logical_recipe_id == logical)
      return true;
  return false;
}

static bool admit_network_candidate(const ResidentRelationalNetworkClosure& claimed,
                                    const ResidentRecipeCell* rec,
                                    const ResidentRecipeDerivation* der,
                                    const ResidentRecipeOccurrence* occ,
                                    const ResidentOccurrenceCoupling* edges,
                                    std::uint64_t parent_logical) {
  ResidentRelationalNetworkClosure rebuilt{};
  return bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &rebuilt) &&
         rebuilt.identity == claimed.identity && claimed.identity != 0u &&
         claimed.boundary_count >= 2u && closure_has_logical(claimed, parent_logical);
}

static bool condense_from_validated_network(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                                            ResidentRecipeDerivation* derivations,
                                            ResidentPostbirthConstructorState* state) {
  if (!cells || !cell_count || !derivations || !state || *cell_count < 3u)
    return false;
  ResidentRecipeCell rec[3] = {cells[0], cells[1], cells[2]};
  ResidentRecipeDerivation der[3] = {derivations[0], derivations[1], derivations[2]};
  for (std::uint32_t i = 0; i < 3u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
  }
  const std::uint32_t v0[] = {10u, 20u};
  const std::uint32_t v1[] = {20u, 30u};
  const std::uint32_t v2[] = {30u, 40u};
  ResidentRecipeOccurrence occ[3]{};
  if (!bind_live(rec[0], der[0], v0, 0xD20, &occ[0]) ||
      !bind_live(rec[1], der[1], v1, 0xD21, &occ[1]) ||
      !bind_live(rec[2], der[2], v2, 0xD22, &occ[2]))
    return false;
  ResidentOccurrenceCoupling edges[2]{};
  if (!bind_resident_occurrence_coupling(occ[0], der[0], 1u, occ[1], der[1], 0u, &edges[0]) ||
      !bind_resident_occurrence_coupling(occ[1], der[1], 1u, occ[2], der[2], 0u, &edges[1]))
    return false;
  ResidentRelationalNetworkClosure closure{};
  if (!bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &closure))
    return false;
  ResidentRelationalNetworkClosure stale = closure;
  stale.identity ^= 1ull;
  if (admit_network_candidate(stale, rec, der, occ, edges, rec[2].logical_recipe_id) ||
      admit_network_candidate(closure, rec, der, occ, edges, 0xDEADull) ||
      !admit_network_candidate(closure, rec, der, occ, edges, rec[2].logical_recipe_id))
    return false;
  DirectWhiteboxCondensationV1 next{};
  if (!seed_affine_witness(&next, 5 << 16, 0, 1 << 16, 1 << 16))
    return false;
  const std::uint32_t before = *cell_count;
  if (!condense_from_parent(cells, cell_count, derivations, state, 2u, next) ||
      *cell_count != before + 1u ||
      !replay_resident_whitebox_recipe(next, cells[before], derivations[before]))
    return false;
  return derivations[before].parent_logical_recipe_id == rec[2].logical_recipe_id;
}

static bool parent_bank_isolation() {
  Built tree{};
  ResidentRecipeCell cells[8]{};
  ResidentRecipeDerivation derivations[8]{};
  ResidentPostbirthConstructorState state{};
  DirectWhiteboxCondensationV1 witness{};
  DirectWhiteboxCondensationV1 next{};
  ResidentNetworkCondensationEvidence evidence{};
  FoundryCondensationWorkspace workspace{};
  std::uint32_t cell_count = 0;
  FoundryPhaseLowering lowered{};
  if (!build(&tree) ||
      !seed_condensation_workspace(cells, &cell_count, derivations, &state, &witness, &workspace) ||
      !run_phase_episode(&tree, &lowered, &workspace) ||
      !recurse_mixed_rank(cells, &cell_count, derivations, &state) || cell_count != 3u)
    return false;
  ResidentRecipeCell rec[3] = {cells[0], cells[1], cells[2]};
  ResidentRecipeDerivation der[3] = {derivations[0], derivations[1], derivations[2]};
  for (std::uint32_t i = 0; i < 3u; ++i) {
    der[i].logical_recipe_id = rec[i].logical_recipe_id;
    der[i].revision_identity = rec[i].revision_identity;
  }
  const std::uint32_t v0[] = {10u, 20u};
  const std::uint32_t v1[] = {20u, 30u};
  const std::uint32_t v2[] = {30u, 40u};
  const std::uint32_t w1[] = {20u, 50u};
  const std::uint32_t w2[] = {50u, 60u};
  ResidentRecipeOccurrence occ[3]{};
  ResidentRecipeOccurrence occ2[3]{};
  ResidentRecipeOccurrence lesion_occ[3]{};
  ResidentOccurrenceCoupling edges[2]{};
  ResidentOccurrenceCoupling edges2[2]{};
  ResidentOccurrenceCoupling lesion[2]{};
  ResidentRelationalNetworkClosure claimed{};
  DirectRelationAlgebraFamilyV1 family{};
  if (!bind_live(rec[0], der[0], v0, 0xE10, &occ[0], 1u, 1 << 16) ||
      !bind_live(rec[1], der[1], v1, 0xE11, &occ[1], 1u, 1 << 16) ||
      !bind_live(rec[2], der[2], v2, 0xE12, &occ[2], 1u, 1 << 16) ||
      !bind_live(rec[0], der[0], v0, 0xE20, &occ2[0], 2u, 2 << 16) ||
      !bind_live(rec[1], der[1], v1, 0xE21, &occ2[1], 2u, 2 << 16) ||
      !bind_live(rec[2], der[2], v2, 0xE22, &occ2[2], 2u, 2 << 16) ||
      !bind_live(rec[0], der[0], v0, 0xE30, &lesion_occ[0], 3u, 3 << 16) ||
      !bind_live(rec[1], der[1], w1, 0xE31, &lesion_occ[1], 3u, 3 << 16) ||
      !bind_live(rec[2], der[2], w2, 0xE32, &lesion_occ[2], 3u, 3 << 16) ||
      !bind_resident_occurrence_coupling(occ[0], der[0], 1u, occ[1], der[1], 0u, &edges[0]) ||
      !bind_resident_occurrence_coupling(occ[1], der[1], 1u, occ[2], der[2], 0u, &edges[1]) ||
      !bind_resident_occurrence_coupling(occ2[0], der[0], 1u, occ2[1], der[1], 0u, &edges2[0]) ||
      !bind_resident_occurrence_coupling(occ2[1], der[1], 1u, occ2[2], der[2], 0u, &edges2[1]) ||
      !bind_resident_occurrence_coupling(lesion_occ[0], der[0], 1u, lesion_occ[1], der[1], 0u,
                                         &lesion[0]) ||
      !bind_resident_occurrence_coupling(lesion_occ[1], der[1], 1u, lesion_occ[2], der[2], 0u,
                                         &lesion[1]) ||
      observe_resident_mixed_rank_whitebox(rec, der, occ, occ, 3u, edges, edges, 2u, &next) ||
      observe_resident_mixed_rank_whitebox(rec, der, occ, lesion_occ, 3u, edges, lesion, 2u,
                                           &next) ||
      !observe_resident_mixed_rank_whitebox(rec, der, occ, occ2, 3u, edges, edges2, 2u, &next) ||
      next.witness_identity == 0u ||
      observe_resident_mixed_rank_evidence(rec, der, occ, occ, 3u, edges, edges, 2u, &next,
                                           &evidence) ||
      observe_resident_mixed_rank_evidence(rec, der, occ, lesion_occ, 3u, edges, lesion, 2u, &next,
                                           &evidence) ||
      !observe_resident_mixed_rank_evidence(rec, der, occ, occ2, 3u, edges, edges2, 2u, &next,
                                            &evidence) ||
      evidence.witness_identity == 0u ||
      evidence.witness_identity != resident_network_condensation_witness(evidence) ||
      evidence.boundary_identity != resident_condensation_boundary_identity(evidence) ||
      evidence.maximum_error_q16 != 0 || evidence.source_count != 2u ||
      evidence.sources[0].relation !=
          direct_relation_algebra_resident_word(DirectRelationAlgebraFamilyV1::linear) ||
      evidence.sources[0].relation == static_cast<std::uint32_t>(ResidentRecipeRelation::trigger) ||
      !direct_relation_algebra_resident_word_parse(evidence.sources[0].relation, &family,
                                                   nullptr) ||
      family != DirectRelationAlgebraFamilyV1::linear ||
      !bind_resident_relational_network_closure(rec, der, occ, 3u, edges, 2u, &claimed) ||
      !admit_resident_network_parent(rec, der, occ, 3u, edges, 2u, claimed.identity,
                                     rec[2].logical_recipe_id) ||
      admit_resident_network_parent(rec, der, occ, 3u, edges, 2u, claimed.identity ^ 1ull,
                                    rec[2].logical_recipe_id))
    return false;
  const std::uint64_t witness_before = next.witness_identity;
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
    return false;
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
    return false;
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
      whole_w.witness_identity != next.witness_identity)
    return false;
  std::uint64_t replay_logical = 0, replay_rev = 0;
  ResidentNetworkCondensationEvidence tampered = evidence;
  tampered.witness_identity ^= 1ull;
  DirectWhiteboxCondensationV1 bad_w = next;
  bad_w.witness_identity ^= 1ull;
  if (!replay_resident_network_candidate(cells, cell_count, derivations, state.derivation_count,
                                         evidence, &next, &replay_logical, &replay_rev) ||
      replay_logical == 0u ||
      replay_resident_network_candidate(cells, cell_count, derivations, state.derivation_count,
                                        evidence, nullptr, &replay_logical, &replay_rev) ||
      replay_resident_network_candidate(cells, cell_count, derivations, state.derivation_count,
                                        tampered, &next, &replay_logical, &replay_rev) ||
      replay_resident_network_candidate(cells, cell_count, derivations, state.derivation_count,
                                        evidence, &bad_w, &replay_logical, &replay_rev))
    return false;
  workspace.claimed_identity = claimed.identity;
  workspace.network_cells = rec;
  workspace.network_ders = der;
  workspace.occurrences = occ;
  workspace.couplings = edges;
  workspace.occurrence_count = 3u;
  workspace.coupling_count = 2u;
  workspace.parent_cell = 2u;
  workspace.parent_logical = rec[2].logical_recipe_id;
  workspace.witness = &next;
  workspace.evidence = &evidence;
  if (!foundry_workspace_ready(&workspace))
    return false;
  workspace.evidence = &tampered;
  if (foundry_workspace_ready(&workspace))
    return false;
  workspace.evidence = &evidence;
  workspace.claimed_identity ^= 1ull;
  if (foundry_workspace_ready(&workspace))
    return false;
  workspace.claimed_identity = claimed.identity;
  const ResidentRecipeCell parent = cells[2];
  workspace.parent_cell = 2u;
  workspace.parent_logical = parent.logical_recipe_id;
  workspace.witness = &next;
  if (!run_phase_episode(&tree, &lowered, &workspace) || lowered.condense != 1u ||
      cell_count != 4u || cells[2].logical_recipe_id != parent.logical_recipe_id ||
      cells[2].revision_identity != parent.revision_identity ||
      derivations[3].parent_logical_recipe_id != parent.logical_recipe_id ||
      cells[3].logical_recipe_id != replay_logical ||
      !replay_resident_whitebox_recipe(next, cells[3], derivations[3]))
    return false;
  workspace.parent_logical = 0xDEADull;
  if (foundry_materialize_from_workspace(&workspace, nullptr) || cell_count != 4u)
    return false;
  workspace.parent_logical = 0u;
  const std::uint32_t held_a[] = {50u, 60u};
  const std::uint32_t held_b[] = {70u, 80u};
  const std::uint32_t va[] = {50u, 60u};
  const std::uint32_t vp[] = {60u, 70u};
  ResidentRecipeOccurrence surface_a{}, surface_b{}, q_anc{}, q_pn{};
  ResidentOccurrenceCoupling edge_next{};
  ResidentRelationalNetworkClosure n_next{};
  ResidentRecipeCell rec_next[2] = {cells[0], cells[3]};
  ResidentRecipeDerivation der_next[2] = {derivations[0], derivations[3]};
  der_next[0].logical_recipe_id = cells[0].logical_recipe_id;
  der_next[0].revision_identity = cells[0].revision_identity;
  der_next[1].logical_recipe_id = cells[3].logical_recipe_id;
  der_next[1].revision_identity = cells[3].revision_identity;
  if (foundry_materialize_from_workspace(&workspace, nullptr) || cell_count != 4u ||
      !bind_live(cells[3], derivations[3], held_a, 0xF50, &surface_a) ||
      !bind_live(cells[3], derivations[3], held_b, 0xF70, &surface_b) ||
      surface_a.logical_recipe_id != cells[3].logical_recipe_id ||
      surface_b.logical_recipe_id != cells[3].logical_recipe_id ||
      surface_a.bindings[0].variable_identity != 50u ||
      surface_b.bindings[0].variable_identity != 70u ||
      !bind_live(cells[0], der_next[0], va, 0xA50, &q_anc, 5u, 5 << 16) ||
      !bind_live(cells[3], der_next[1], vp, 0xA60, &q_pn, 5u, 5 << 16) ||
      !bind_resident_occurrence_coupling(q_anc, der_next[0], 1u, q_pn, der_next[1], 0u, &edge_next))
    return false;
  ResidentRecipeOccurrence occ_next[2] = {q_anc, q_pn};
  if (!bind_resident_relational_network_closure(rec_next, der_next, occ_next, 2u, &edge_next, 1u,
                                                &n_next) ||
      n_next.identity == 0u || n_next.identity == claimed.identity || n_next.actual_count != 2u ||
      !boundary_has(n_next, 50u, ResidentRecipePortDirection::input) ||
      !boundary_has(n_next, 70u, ResidentRecipePortDirection::output) ||
      !admit_resident_network_parent(rec_next, der_next, occ_next, 2u, &edge_next, 1u,
                                     n_next.identity, cells[3].logical_recipe_id) ||
      admit_resident_network_parent(rec_next, der_next, occ_next, 2u, &edge_next, 1u,
                                    claimed.identity, cells[3].logical_recipe_id))
    return false;
  ResidentRecipeCell rec_r[2] = {cells[1], cells[3]};
  ResidentRecipeDerivation der_r[2] = {derivations[1], derivations[3]};
  der_r[0].logical_recipe_id = cells[1].logical_recipe_id;
  der_r[0].revision_identity = cells[1].revision_identity;
  der_r[1].logical_recipe_id = cells[3].logical_recipe_id;
  der_r[1].revision_identity = cells[3].revision_identity;
  const std::uint32_t r0[] = {50u, 60u};
  const std::uint32_t r1[] = {60u, 70u};
  ResidentRecipeOccurrence occ_r1[2]{}, occ_r2[2]{};
  ResidentOccurrenceCoupling e_r1[1]{}, e_r2[1]{};
  DirectWhiteboxCondensationV1 again{};
  ResidentNetworkCondensationEvidence ev2{};
  std::uint64_t again_logical = 0, again_rev = 0;
  if (!bind_live(cells[1], der_r[0], r0, 0xC10, &occ_r1[0], 6u, 6 << 16) ||
      !bind_live(cells[3], der_r[1], r1, 0xC11, &occ_r1[1], 6u, 6 << 16) ||
      !bind_live(cells[1], der_r[0], r0, 0xC20, &occ_r2[0], 7u, 7 << 16) ||
      !bind_live(cells[3], der_r[1], r1, 0xC21, &occ_r2[1], 7u, 7 << 16) ||
      !bind_resident_occurrence_coupling(occ_r1[0], der_r[0], 1u, occ_r1[1], der_r[1], 0u,
                                         &e_r1[0]) ||
      !bind_resident_occurrence_coupling(occ_r2[0], der_r[0], 1u, occ_r2[1], der_r[1], 0u,
                                         &e_r2[0]) ||
      observe_resident_mixed_rank_whitebox(rec_r, der_r, occ_r1, occ_r1, 2u, e_r1, e_r1, 1u,
                                           &again) ||
      !observe_resident_mixed_rank_whitebox(rec_r, der_r, occ_r1, occ_r2, 2u, e_r1, e_r2, 1u,
                                            &again) ||
      again.witness_identity == 0u || again.witness_identity == next.witness_identity ||
      !observe_resident_mixed_rank_evidence(rec_r, der_r, occ_r1, occ_r2, 2u, e_r1, e_r2, 1u,
                                            &again, &ev2) ||
      !replay_resident_network_candidate(cells, cell_count, derivations, state.derivation_count,
                                         ev2, &again, &again_logical, &again_rev) ||
      again_logical == 0u || again_logical == cells[3].logical_recipe_id)
    return false;
  ResidentRelationalNetworkClosure n_r{};
  if (!bind_resident_relational_network_closure(rec_r, der_r, occ_r1, 2u, e_r1, 1u, &n_r) ||
      n_r.identity == 0u)
    return false;
  const ResidentRecipeCell later_parent = cells[3];
  state.recipe_cell_capacity = 5u;
  state.derivation_capacity = 5u;
  workspace.cell_capacity = 5u;
  workspace.claimed_identity = n_r.identity;
  workspace.network_cells = rec_r;
  workspace.network_ders = der_r;
  workspace.occurrences = occ_r1;
  workspace.couplings = e_r1;
  workspace.occurrence_count = 2u;
  workspace.coupling_count = 1u;
  workspace.parent_cell = 3u;
  workspace.parent_logical = later_parent.logical_recipe_id;
  workspace.witness = &again;
  workspace.evidence = &ev2;
  ResidentRecipeOccurrence surface_pn2{};
  const std::uint32_t held_pn2[] = {90u, 100u};
  if (!foundry_workspace_ready(&workspace) ||
      !foundry_materialize_from_workspace(&workspace, nullptr) || cell_count != 5u ||
      cells[3].logical_recipe_id != later_parent.logical_recipe_id ||
      cells[3].revision_identity != later_parent.revision_identity ||
      cells[4].logical_recipe_id != again_logical ||
      derivations[4].parent_logical_recipe_id != later_parent.logical_recipe_id ||
      !replay_resident_whitebox_recipe(again, cells[4], derivations[4]) ||
      !bind_live(cells[4], derivations[4], held_pn2, 0xF90, &surface_pn2) ||
      surface_pn2.logical_recipe_id != cells[4].logical_recipe_id)
    return false;
  LanguageAfterPn2 after{};
  after.cells = cells;
  after.cell_count = cell_count;
  after.ders = derivations;
  after.der_count = state.derivation_count;
  after.i_anc = 0u;
  after.i_src = 1u;
  after.i_pn1 = 3u;
  after.i_pn2 = 4u;
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
  after.workspace = &workspace;
  state.recipe_cell_capacity = 6u;
  state.derivation_capacity = 6u;
  workspace.cell_capacity = 6u;
  return language_after_pn2(after);
}

static bool authority_firewall_and_capacity(ResidentRecipeCell* cells, std::uint32_t cell_count,
                                            ResidentRecipeDerivation* derivations,
                                            ResidentPostbirthConstructorState* state) {
  if (!cells || !derivations || !state || cell_count != 4u)
    return false;
  derivations[3].logical_recipe_id = cells[3].logical_recipe_id;
  derivations[3].revision_identity = cells[3].revision_identity;
  const std::uint32_t vars[2] = {1u, 2u};
  ResidentRecipeOccurrence expired{};
  if (!bind_resident_recipe_occurrence(cells[3], derivations[3], vars, 2u, 0xD30, 0xD30 + 1000u,
                                       77u, 1u, ResidentOccurrenceLineageKind::actual,
                                       DirectParticipationAuthority::independent_external, 9u, 1u,
                                       1u, 0, &expired))
    return false;
  const std::uint64_t rev = cells[3].revision_identity;
  const std::int64_t credit = cells[3].credit_q16;
  if (!commit_resident_causal_difference_revision(&expired, &cells[3], 3u, true, 1, 0, 2u) ||
      expired.state != kResidentRecipeOccurrenceSettled || cells[3].revision_identity != rev ||
      cells[3].credit_q16 != credit)
    return false;
  ResidentRecipeOccurrence yoked{};
  if (!bind_condensed_actual(cells[3], derivations[3], 0xD31, &yoked) ||
      !commit_resident_causal_difference_revision(&yoked, &cells[3], 3u, false, 1, 0) ||
      yoked.state != kResidentRecipeOccurrenceSettled || cells[3].revision_identity != rev ||
      cells[3].credit_q16 != credit)
    return false;
  DirectWhiteboxCondensationV1 extra{};
  std::uint32_t refused = cell_count;
  return seed_affine_witness(&extra, 6 << 16, 0, 1 << 16, 0) &&
         !condense_from_parent(cells, &refused, derivations, state, 3u, extra) &&
         refused == cell_count;
}

int main() {
  Built skip_tree{};
  FoundryPhaseLowering skip{};
  if (!build(&skip_tree) || !run_phase_episode(&skip_tree, &skip) || skip.condense != 0u)
    return 1;

  Built tree{};
  ResidentRecipeCell cells[4]{};
  ResidentRecipeDerivation derivations[4]{};
  ResidentPostbirthConstructorState state{};
  DirectWhiteboxCondensationV1 witness{};
  FoundryCondensationWorkspace workspace{};
  std::uint32_t cell_count = 0;
  FoundryPhaseLowering lowered{};
  if (!build(&tree) ||
      !seed_condensation_workspace(cells, &cell_count, derivations, &state, &witness, &workspace) ||
      !run_phase_episode(&tree, &lowered, &workspace) || lowered.condense != 1u ||
      cell_count != 2u || state.condensed != 1u ||
      lowered.condensed_revision_identity != cells[1].revision_identity ||
      !replay_resident_whitebox_recipe(witness, cells[1], derivations[1]) ||
      !reuse_condensed_recipe(cells, cell_count, derivations, witness) ||
      !recurse_mixed_rank(cells, &cell_count, derivations, &state) || cell_count != 3u ||
      !deopt_rematerialize_detail(cells, cell_count, derivations, &state, witness) ||
      !same_recipe_two_occurrences(cells, cell_count, derivations) ||
      !bind_mixed_rank_network(cells, cell_count, derivations) ||
      !condense_from_validated_network(cells, &cell_count, derivations, &state) ||
      cell_count != 4u ||
      !authority_firewall_and_capacity(cells, cell_count, derivations, &state) ||
      !parent_bank_isolation())
    return 2;

  Built refuse_tree{};
  ResidentRecipeCell refuse_cells[4]{};
  ResidentRecipeDerivation refuse_ders[4]{};
  ResidentPostbirthConstructorState refuse_state{};
  DirectWhiteboxCondensationV1 good{};
  DirectWhiteboxCondensationV1 bad{};
  FoundryCondensationWorkspace refuse_ws{};
  std::uint32_t refuse_count = 0;
  FoundryPhaseLowering refused{};
  if (!build(&refuse_tree) || !seed_condensation_workspace(refuse_cells, &refuse_count, refuse_ders,
                                                           &refuse_state, &good, &refuse_ws))
    return 3;
  refuse_ws.witness = &bad;
  const ResidentRecipeCell refuse_parent = refuse_cells[0];
  if (run_phase_episode(&refuse_tree, &refused, &refuse_ws) || refused.settle != 0u ||
      refused.condense != 0u ||
      refuse_cells[0].revision_identity != refuse_parent.revision_identity ||
      refuse_cells[0].credit_q16 != refuse_parent.credit_q16)
    return 4;

  std::printf(
      "DIRECT_FOUNDRY_PHASE_CONDENSE_HOST GREEN skip_condense=1 condense=1 "
      "replay=1 pn1_bind=1 isolation=1 birth_replay=1 live_revised_replay_refuse=1 "
      "pn2_condense=1 mixed_rank_bind=1 pn2_settle_isolation=1 "
      "deopt_rematerialize=1 contradicted_replay_refuse=1 ancestor_still_bindable=1 "
      "same_recipe_two_occ=1 settle_one_isolates=1 historical_participation_frozen=1 "
      "mixed_rank_network=1 order_canon=1 settled_member_refuse=1 "
      "boundary_bn=1 boundary_lesion=1 uncoupled_shared_refuse=1 "
      "bn_candidate=1 closure_validate=1 stale_closure_refuse=1 "
      "eligibility_expiry=1 non_independent_refuse=1 capacity_refuse=1 "
      "parent_isolation=1 parent_logical_refuse=1 unnamed_parent_refuse=1 "
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
      "express_recipe_new_surface=1 language_opportunity=1 "
      "language_opportunity_wrong_surface_refuse=1 language_opportunity_stale_revision_refuse=1 "
      "language_opportunity_endogenous_zero_credit=1 "
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
      "invalid_witness_refuse=1 invalid_witness_no_stamp=1 invalid_witness_no_settle=1 "
      "generation=%llu logical=%llu "
      "graph_flip=0 translation_status=UNDEFINED\n",
      static_cast<unsigned long long>(lowered.condensed_generation),
      static_cast<unsigned long long>(lowered.condensed_logical_id));
  return 0;
}

#ifndef RELSEQ_HIERARCHY_FIXTURE_INL
#define RELSEQ_HIERARCHY_FIXTURE_INL

#include <cstdint>
#include <cstring>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_resident_recipe_experience_revision.cuh"
#include "hardware_native/direct_adult_recipe_ir.cuh"
#include "hardware_native/direct_foundry_phase_lower.cuh"
#include "hardware_native/direct_adult_relational_sequence_bridge.cuh"

namespace relseq_hierarchy {
using substrate::direct_network::DirectRelSeqOccurrence;
using substrate::direct_network::DirectRelSeqOutput;
using substrate::direct_network::DirectRelSeqRecipe;
using substrate::direct_network::direct_relseq_induce_flat_recipe;
using substrate::direct_adult_core::DirectParticipationAuthority;
using substrate::direct_adult_core::ResidentOccurrenceLineageKind;
using substrate::direct_adult_core::ResidentRecipeOccurrence;
using substrate::direct_adult_core::ResidentRelationalSequenceBody;
using substrate::direct_adult_core::ResidentRelationalSequenceOccurrenceSidecar;
using substrate::direct_adult_core::bind_resident_recipe_occurrence;
using substrate::direct_adult_core::commit_resident_causal_difference_revision;
using substrate::direct_adult_core::FoundryCondensationWorkspace;
using substrate::direct_adult_core::FoundryPhaseLowering;
using substrate::direct_adult_core::lower_foundry_phase_episode;
using substrate::direct_adult_core::foundry_materialize_from_workspace;
using substrate::direct_adult_core::materialize_resident_whitebox_recipe;
using substrate::direct_adult_core::replay_resident_whitebox_recipe;
using substrate::direct_network::DirectWhiteboxCondensationV1;
using substrate::direct_network::DirectWhiteboxReductionKindV1;
using substrate::direct_network::DirectWhiteboxReductionSourceV1;
using substrate::direct_network::ResidentPostbirthConstructorState;
using substrate::direct_network::ResidentRecipeDerivation;
using substrate::direct_network::ResidentRecipePort;
using substrate::direct_network::ResidentRecipePortDomain;
using substrate::direct_network::ResidentRecipePortDirection;
using substrate::direct_network::direct_whitebox_condensation_valid;
using substrate::direct_network::direct_whitebox_reduce_exact;
using substrate::direct_network::direct_whitebox_rematerialize;
using substrate::direct_network::direct_whitebox_source_identity;
using substrate::direct_adult_core::encode_resident_recipe_ir;
using substrate::direct_adult_core::execute_resident_recipe_ir;
using substrate::direct_adult_core::ResidentRecipeIrOp;
using substrate::direct_adult_core::ResidentRecipeIrProgram;
using substrate::direct_network::ResidentRecipeCell;
using substrate::direct_network::ResidentRecipeRevisionAuthority;
using substrate::direct_network::stage_resident_recipe_revision_event;
using substrate::direct_network::apply_resident_recipe_revision_event;
using substrate::direct_adult_core::kResidentRecipeOccurrenceLive;
using substrate::direct_adult_core::kResidentRecipeOccurrenceSettled;
using substrate::direct_adult_core::lower_canonical_resident_relseq;

struct RecipeCellView {
  std::uint64_t logical_recipe_id, revision_identity;
};
struct DerivationView {
  std::uint64_t logical_recipe_id;
  std::uint16_t port_count;
};

struct Built {
  DirectRelSeqRecipe recipes[2];
  DirectRelSeqOccurrence occ[9];
  std::uint64_t leaf_ids[4];
  std::uint64_t join_ids[5];
};

inline bool same_output(const DirectRelSeqOutput& a, const DirectRelSeqOutput& b) {
  return a.count == b.count && a.depth_peak == b.depth_peak &&
         std::memcmp(a.units, b.units, a.count * sizeof(std::uint32_t)) == 0;
}

inline bool bind(const RecipeCellView& cell, std::uint64_t oid, std::uint32_t a, std::uint32_t b,
                 ResidentRecipeOccurrence* out) {
  DerivationView d{cell.logical_recipe_id, 2u};
  const std::uint32_t vars[2] = {a, b};
  return bind_resident_recipe_occurrence(
      cell, d, vars, 2u, oid, oid + 1000u, 77u, 1u, ResidentOccurrenceLineageKind::endogenous,
      DirectParticipationAuthority::none, 9u, 1u, 100u, 0, out);
}

inline bool bind_actual(const RecipeCellView& cell, std::uint64_t oid, std::uint32_t a,
                        std::uint32_t b, ResidentRecipeOccurrence* out) {
  DerivationView d{cell.logical_recipe_id, 2u};
  const std::uint32_t vars[2] = {a, b};
  return bind_resident_recipe_occurrence(
      cell, d, vars, 2u, oid, oid + 1000u, 77u, 1u, ResidentOccurrenceLineageKind::actual,
      DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, out);
}

inline bool induce_body(std::uint64_t logical, std::uint64_t revision, std::uint64_t relation,
                        const std::uint32_t* observed, std::uint32_t n, const std::uint32_t* ports,
                        ResidentRelationalSequenceBody* out) {
  DirectRelSeqRecipe r{};
  if (!out || !direct_relseq_induce_flat_recipe(logical, relation, observed, n, ports, 2u, nullptr,
                                               0u, &r))
    return false;
  *out = {};
  out->logical_recipe_id = logical;
  out->revision_identity = revision;
  out->relation_identity = relation;
  out->support = 1u;
  out->active = 1u;
  out->port_count = r.port_count;
  out->piece_count = r.piece_count;
  for (std::uint32_t i = 0; i < r.piece_count; ++i) out->pieces[i] = r.pieces[i];
  return true;
}

inline bool build(Built* out) {
  if (!out) return false;
  *out = {};
  constexpr std::uint64_t L = 0xA1, J = 0xA2, LR = 0xB1, JR = 0xB2;
  RecipeCellView leaf_cell{L, LR}, join_cell{J, JR};
  ResidentRecipeOccurrence canon[9]{};
  const std::uint32_t leaf_vars[4][2] = {{11, 12}, {21, 22}, {31, 32}, {41, 42}};
  out->leaf_ids[0] = 0xC10;
  out->leaf_ids[1] = 0xC11;
  out->leaf_ids[2] = 0xC12;
  out->leaf_ids[3] = 0xC13;
  for (unsigned i = 0; i < 4; ++i)
    if (!bind(leaf_cell, out->leaf_ids[i], leaf_vars[i][0], leaf_vars[i][1], &canon[i]))
      return false;
  out->join_ids[0] = 0xC20;
  out->join_ids[1] = 0xC21;
  out->join_ids[2] = 0xC22;
  out->join_ids[3] = 0xC23;
  out->join_ids[4] = 0xC24;
  for (unsigned i = 0; i < 5; ++i)
    if (!bind(join_cell, out->join_ids[i], 1u, 2u, &canon[4 + i])) return false;

  const std::uint32_t leaf_obs[] = {101u, 11u, 102u, 12u, 103u};
  const std::uint32_t leaf_ports[] = {11u, 12u};
  const std::uint32_t leaf1_obs[] = {101u, 21u, 102u, 22u, 103u};
  const std::uint32_t leaf1_ports[] = {21u, 22u};
  const std::uint32_t join_obs[] = {31u, 104u, 32u};
  const std::uint32_t join_ports[] = {31u, 32u};
  ResidentRelationalSequenceBody leaf_body{}, leaf1_body{}, join_body{};
  if (!induce_body(L, LR, 0xD1, leaf_obs, 5u, leaf_ports, &leaf_body)) return false;
  if (!induce_body(L, LR, 0xD1, leaf1_obs, 5u, leaf1_ports, &leaf1_body)) return false;
  if (leaf_body.piece_count != leaf1_body.piece_count) return false;
  for (std::uint32_t i = 0; i < leaf_body.piece_count; ++i)
    if (leaf_body.pieces[i].value != leaf1_body.pieces[i].value ||
        leaf_body.pieces[i].kind != leaf1_body.pieces[i].kind)
      return false;
  if (!induce_body(J, JR, 0xD2, join_obs, 3u, join_ports, &join_body)) return false;
  DirectRelSeqRecipe miss{};
  const std::uint32_t missing_obs[] = {101u, 102u, 103u};
  if (direct_relseq_induce_flat_recipe(L, 0xD1, missing_obs, 3u, leaf_ports, 2u, nullptr, 0u, &miss))
    return false;

  ResidentRelationalSequenceOccurrenceSidecar sc[9]{};
  for (unsigned i = 0; i < 9; ++i) sc[i].occurrence_identity = canon[i].occurrence_identity;
  sc[4].child_occurrence_identities[0] = out->leaf_ids[0];
  sc[4].child_occurrence_identities[1] = out->leaf_ids[1];
  sc[5].child_occurrence_identities[0] = out->join_ids[0];
  sc[5].child_occurrence_identities[1] = out->leaf_ids[2];
  sc[6].child_occurrence_identities[0] = out->join_ids[1];
  sc[6].child_occurrence_identities[1] = out->leaf_ids[3];
  sc[7].child_occurrence_identities[0] = out->leaf_ids[1];
  sc[7].child_occurrence_identities[1] = out->leaf_ids[2];
  sc[8].child_occurrence_identities[0] = out->leaf_ids[0];
  sc[8].child_occurrence_identities[1] = out->join_ids[3];

  for (unsigned i = 0; i < 4; ++i) {
    DirectRelSeqRecipe r{};
    if (!lower_canonical_resident_relseq(leaf_cell, canon[i], leaf_body, &sc[i], &r, &out->occ[i]))
      return false;
    if (i == 0)
      out->recipes[0] = r;
    else if (r.revision_identity != out->recipes[0].revision_identity)
      return false;
  }
  for (unsigned i = 4; i < 9; ++i) {
    DirectRelSeqRecipe r{};
    if (!lower_canonical_resident_relseq(join_cell, canon[i], join_body, &sc[i], &r, &out->occ[i]))
      return false;
    if (i == 4)
      out->recipes[1] = r;
    else if (r.revision_identity != out->recipes[1].revision_identity)
      return false;
  }
  return sc[5].child_occurrence_identities[0] != sc[8].child_occurrence_identities[0];
}

inline bool seed_affine_witness(DirectWhiteboxCondensationV1* witness,
                                std::int32_t a = 2 << 16, std::int32_t b = 1 << 16,
                                std::int32_t c = 3 << 16, std::int32_t d = -(1 << 15)) {
  if (!witness) return false;
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

inline bool seed_condensation_workspace(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                                        ResidentRecipeDerivation* derivations,
                                        ResidentPostbirthConstructorState* state,
                                        DirectWhiteboxCondensationV1* witness,
                                        FoundryCondensationWorkspace* out) {
  if (!cells || !cell_count || !derivations || !state || !witness || !out ||
      !seed_affine_witness(witness))
    return false;
  cells[0] = {};
  cells[0].logical_recipe_id = 0xA2;
  cells[0].revision_identity = 0xB2;
  cells[0].revision = 1u;
  cells[0].support_q16 = 1 << 16;
  derivations[0] = {};
  derivations[0].logical_recipe_id = 0xA2;
  derivations[0].revision_identity = 0xB2;
  derivations[0].recipe_cell = 0u;
  derivations[0].generation = 1u;
  derivations[0].port_count = 2u;
  derivations[0].ports[0] = ResidentRecipePort{
      10u, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::input, 1u};
  derivations[0].ports[1] = ResidentRecipePort{
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
  *out = {};
  out->cells = cells;
  out->cell_count = cell_count;
  out->cell_capacity = 4u;
  out->derivations = derivations;
  out->state = state;
  out->parent_cell = 0u;
  out->parent_logical = cells[0].logical_recipe_id;
  out->witness = witness;
  return true;
}

inline bool run_phase_episode(Built* h, FoundryPhaseLowering* out,
                              FoundryCondensationWorkspace* condensation = nullptr) {
  if (!h || !out) return false;
  h->occ[0].context_feature_count = 1;
  h->occ[0].context_features[0] = 0xFAu;
  DirectRelSeqRecipe morphs[2]{};
  const std::uint32_t jobs[] = {31u, 104u, 32u};
  const std::uint32_t jp[] = {31u, 32u};
  const std::uint32_t ca[] = {0xFAu};
  const std::uint32_t cb[] = {0xFBu};
  if (!direct_relseq_induce_flat_recipe(0xE1, 0xF2, jobs, 3u, jp, 2u, ca, 1u, &morphs[0]) ||
      !direct_relseq_induce_flat_recipe(0xE2, 0xF2, jobs, 3u, jp, 2u, cb, 1u, &morphs[1]))
    return false;
  ResidentRecipeCell cell{};
  cell.logical_recipe_id = 0xA2;
  cell.revision_identity = 0xB2;
  ResidentRecipeOccurrence actual{};
  if (!bind_actual({0xA2, 0xB2}, 0xC50, 1u, 2u, &actual)) return false;
  const std::uint32_t expect_condense = condensation == nullptr ? 0u : 1u;
  return lower_foundry_phase_episode(h->recipes, 2u, h->occ, 9u, morphs, 2u, 4u, 0u,
                                    h->join_ids[0], &actual, &cell, out, condensation) &&
         out->units == 11u && out->depth == 1u &&
         out->credit_q16 == substrate::direct_adult_core::kQ16One &&
         out->condense == expect_condense;
}

inline bool bind_condensed_actual(const ResidentRecipeCell& cell,
                                  const ResidentRecipeDerivation& derivation,
                                  std::uint64_t oid, ResidentRecipeOccurrence* out) {
  const std::uint32_t vars[2] = {1u, 2u};
  return bind_resident_recipe_occurrence(
      cell, derivation, vars, 2u, oid, oid + 1000u, 77u, 1u,
      ResidentOccurrenceLineageKind::actual,
      DirectParticipationAuthority::independent_external, 9u, 1u, 100u, 0, out);
}

inline bool reuse_condensed_recipe(ResidentRecipeCell* cells, std::uint32_t cell_count,
                                   const ResidentRecipeDerivation* derivations,
                                   const DirectWhiteboxCondensationV1& witness) {
  if (!cells || !derivations || cell_count < 2u) return false;
  const ResidentRecipeCell birth = cells[1];
  const std::int64_t parent_credit = cells[0].credit_q16;
  const std::uint64_t parent_rev = cells[0].revision_identity;
  if (!replay_resident_whitebox_recipe(witness, birth, derivations[1])) return false;
  ResidentRecipeOccurrence child{};
  if (!bind_condensed_actual(cells[1], derivations[1], 0xC60, &child)) return false;
  const std::uint32_t endogenous_vars[2] = {1u, 2u};
  ResidentRecipeOccurrence endogenous{};
  if (bind_resident_recipe_occurrence(
          cells[1], derivations[1], endogenous_vars, 2u, 0xC61, 0xC61 + 1000u, 77u, 1u,
          ResidentOccurrenceLineageKind::endogenous, DirectParticipationAuthority::none, 9u,
          1u, 100u, 0, &endogenous) &&
      commit_resident_causal_difference_revision(&endogenous, &cells[1], 1u, true, 1, 0))
    return false;
  if (!commit_resident_causal_difference_revision(&child, &cells[1], 1u, true, 1, 0))
    return false;
  DirectWhiteboxReductionSourceV1 restored{};
  return cells[1].credit_q16 == substrate::direct_adult_core::kQ16One &&
         cells[1].revision_identity != birth.revision_identity &&
         cells[0].credit_q16 == parent_credit &&
         cells[0].revision_identity == parent_rev &&
         replay_resident_whitebox_recipe(witness, birth, derivations[1]) &&
         !replay_resident_whitebox_recipe(witness, cells[1], derivations[1]) &&
         direct_whitebox_rematerialize(witness, &restored) &&
         restored.source_identity == witness.source.source_identity;
}

inline bool condense_from_parent(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                                 ResidentRecipeDerivation* derivations,
                                 ResidentPostbirthConstructorState* state,
                                 std::uint32_t parent,
                                 const DirectWhiteboxCondensationV1& witness) {
  if (!cells || !cell_count || !derivations || !state || parent >= *cell_count)
    return false;
  FoundryCondensationWorkspace workspace{};
  workspace.cells = cells;
  workspace.cell_count = cell_count;
  workspace.cell_capacity = 4u;
  workspace.derivations = derivations;
  workspace.state = state;
  workspace.parent_cell = parent;
  workspace.parent_logical = cells[parent].logical_recipe_id;
  workspace.witness = &witness;
  return foundry_materialize_from_workspace(&workspace, nullptr);
}

inline bool recurse_mixed_rank(ResidentRecipeCell* cells, std::uint32_t* cell_count,
                               ResidentRecipeDerivation* derivations,
                               ResidentPostbirthConstructorState* state) {
  if (!cells || !cell_count || !derivations || !state || *cell_count < 2u) return false;
  DirectWhiteboxCondensationV1 next{};
  if (!seed_affine_witness(&next, 4 << 16, 0, 2 << 16, 1 << 16) ||
      !condense_from_parent(cells, cell_count, derivations, state, 1u, next) ||
      *cell_count != 3u || derivations[2].generation != derivations[1].generation + 1u ||
      derivations[2].parent_logical_recipe_id != cells[1].logical_recipe_id ||
      !replay_resident_whitebox_recipe(next, cells[2], derivations[2]))
    return false;
  ResidentRecipeOccurrence p0{}, pn2{};
  if (!bind_condensed_actual(cells[0], derivations[0], 0xC70, &p0) ||
      !bind_condensed_actual(cells[2], derivations[2], 0xC71, &pn2) ||
      p0.logical_recipe_id == pn2.logical_recipe_id)
    return false;
  const std::int64_t c0 = cells[0].credit_q16, c1 = cells[1].credit_q16;
  const std::uint64_t r0 = cells[0].revision_identity, r1 = cells[1].revision_identity;
  return commit_resident_causal_difference_revision(&pn2, &cells[2], 2u, true, 1, 0) &&
         cells[2].credit_q16 == substrate::direct_adult_core::kQ16One &&
         cells[0].credit_q16 == c0 && cells[0].revision_identity == r0 &&
         cells[1].credit_q16 == c1 && cells[1].revision_identity == r1 &&
         p0.revision_identity == r0 &&
         p0.state == kResidentRecipeOccurrenceLive &&
         pn2.state == kResidentRecipeOccurrenceSettled;
}

inline bool deopt_rematerialize_detail(ResidentRecipeCell* cells, std::uint32_t cell_count,
                                       ResidentRecipeDerivation* derivations,
                                       ResidentPostbirthConstructorState* state,
                                       const DirectWhiteboxCondensationV1& witness) {
  if (!cells || !derivations || !state || cell_count < 3u) return false;
  DirectWhiteboxCondensationV1 tampered = witness;
  tampered.result_q16[0] ^= 1;
  DirectWhiteboxReductionSourceV1 junk{};
  if (direct_whitebox_condensation_valid(tampered) ||
      direct_whitebox_rematerialize(tampered, &junk))
    return false;
  DirectWhiteboxReductionSourceV1 restored{};
  if (!direct_whitebox_rematerialize(witness, &restored) ||
      restored.source_identity != witness.source.source_identity)
    return false;
  ResidentRecipeOccurrence ancestor{};
  if (!bind_condensed_actual(cells[0], derivations[0], 0xC80, &ancestor)) return false;
  const std::int64_t child_credit = cells[1].credit_q16;
  const std::uint64_t child_rev = cells[1].revision_identity;
  const std::uint64_t ancestor_rev = cells[0].revision_identity;
  ResidentRecipeOccurrence counter{};
  if (!bind_condensed_actual(cells[1], derivations[1], 0xC81, &counter) ||
      !commit_resident_causal_difference_revision(&counter, &cells[1], 1u, true, 0, 1))
    return false;
  std::uint32_t refused_count = cell_count;
  if (condense_from_parent(cells, &refused_count, derivations, state, 0u, witness))
    return false;
  return !replay_resident_whitebox_recipe(witness, cells[1], derivations[1]) &&
         cells[1].credit_q16 != child_credit &&
         cells[1].revision_identity != child_rev &&
         cells[0].revision_identity == ancestor_rev &&
         ancestor.state == kResidentRecipeOccurrenceLive &&
         ancestor.revision_identity == ancestor_rev &&
         refused_count == cell_count;
}

inline bool same_recipe_two_occurrences(ResidentRecipeCell* cells, std::uint32_t cell_count,
                                        const ResidentRecipeDerivation* derivations) {
  if (!cells || !derivations || cell_count < 2u) return false;
  ResidentRecipeOccurrence a{}, b{};
  if (!bind_condensed_actual(cells[1], derivations[1], 0xC90, &a) ||
      !bind_condensed_actual(cells[1], derivations[1], 0xC91, &b) ||
      a.logical_recipe_id != b.logical_recipe_id ||
      a.revision_identity != b.revision_identity ||
      a.occurrence_identity == b.occurrence_identity)
    return false;
  const std::uint64_t shared_rev = a.revision_identity;
  const std::int32_t va[2] = {3, 4}, vb[2] = {5, 6};
  a.bindings[0].variable_identity = static_cast<std::uint32_t>(va[0]);
  a.bindings[1].variable_identity = static_cast<std::uint32_t>(va[1]);
  b.bindings[0].variable_identity = static_cast<std::uint32_t>(vb[0]);
  b.bindings[1].variable_identity = static_cast<std::uint32_t>(vb[1]);
  if (!commit_resident_causal_difference_revision(&a, &cells[1], 1u, true, 1, 0))
    return false;
  return a.state == kResidentRecipeOccurrenceSettled &&
         b.state == kResidentRecipeOccurrenceLive &&
         b.revision_identity == shared_rev &&
         b.bindings[0].variable_identity == 5u &&
         b.bindings[1].variable_identity == 6u &&
         cells[1].revision_identity != shared_rev &&
         !commit_resident_causal_difference_revision(&b, &cells[1], 1u, true, 1, 0);
}
}  // namespace relseq_hierarchy

#endif

#include <cstdint>
#include <cstdio>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_relational_sequence_bridge.cuh"

using namespace substrate::direct_network;
using namespace substrate::direct_adult_core;

namespace {
bool induce_body(std::uint64_t logical, std::uint64_t revision, std::uint64_t relation,
                 const std::uint32_t* observed, std::uint32_t n, const std::uint32_t* ports,
                 std::uint32_t pn, ResidentRelationalSequenceBody* out) {
  DirectRelSeqRecipe induced{};
  if (!out || !direct_relseq_induce_flat_recipe(logical, relation, observed, n, ports, pn,
                                               nullptr, 0u, &induced))
    return false;
  *out = {};
  out->logical_recipe_id = logical;
  out->revision_identity = revision;
  out->relation_identity = relation;
  out->support = induced.support;
  out->active = induced.active;
  out->port_count = induced.port_count;
  out->piece_count = induced.piece_count;
  for (std::uint32_t i = 0; i < induced.piece_count; ++i) out->pieces[i] = induced.pieces[i];
  return true;
}

bool make_occ(const ResidentRecipeCell& cell, std::uint64_t oid, std::uint32_t a,
              std::uint32_t b, ResidentRecipeOccurrence* out) {
  ResidentRecipeDerivation derivation{};
  derivation.logical_recipe_id = cell.logical_recipe_id;
  derivation.revision_identity = cell.revision_identity;
  derivation.port_count = 2;
  const std::uint32_t vars[2] = {a, b};
  return bind_resident_recipe_occurrence(
      cell, derivation, vars, 2u, oid, oid + 1000u, 77u, 1u,
      ResidentOccurrenceLineageKind::endogenous,
      DirectParticipationAuthority::none, 9u, 1u, 100u, 0, out);
}

int fail(const char* why) {
  std::fprintf(stderr, "DIRECT_ADULT_RELSEQ_BRIDGE_HOST RED %s\n", why);
  return 1;
}
}  // namespace

int main() {
  ResidentRecipeCell cells[3]{};
  cells[0].logical_recipe_id = 0xA1;
  cells[0].revision_identity = 0xB1;
  cells[1].logical_recipe_id = 0xA1;
  cells[1].revision_identity = 0xB1;
  cells[2].logical_recipe_id = 0xA2;
  cells[2].revision_identity = 0xB2;
  ResidentRecipeOccurrence canonical[3]{};
  if (!make_occ(cells[0], 0xC1, 11u, 12u, &canonical[0]))
    return fail("bind leaf 0");
  if (!make_occ(cells[1], 0xC2, 21u, 22u, &canonical[1]))
    return fail("bind leaf 1");
  if (!make_occ(cells[2], 0xC3, 31u, 32u, &canonical[2]))
    return fail("bind join");

  const std::uint32_t leaf_obs[] = {101u, 11u, 102u, 12u, 103u};
  const std::uint32_t leaf_ports[] = {11u, 12u};
  const std::uint32_t join_obs[] = {31u, 104u, 32u};
  const std::uint32_t join_ports[] = {31u, 32u};
  ResidentRelationalSequenceBody leaf_body{};
  ResidentRelationalSequenceBody join_body{};
  if (!induce_body(0xA1, 0xB1, 0xD1, leaf_obs, 5u, leaf_ports, 2u, &leaf_body))
    return fail("induce leaf");
  if (!induce_body(0xA2, 0xB2, 0xD2, join_obs, 3u, join_ports, 2u, &join_body))
    return fail("induce join");
  DirectRelSeqRecipe miss{};
  const std::uint32_t miss_obs[] = {101u, 102u, 103u};
  const std::uint32_t miss_ports[] = {11u};
  if (direct_relseq_induce_flat_recipe(0xA1, 0xD1, miss_obs, 3u, miss_ports, 1u, nullptr, 0u,
                                      &miss))
    return fail("missing port induce accepted");

  ResidentRelationalSequenceOccurrenceSidecar leaf0{};
  leaf0.occurrence_identity = 0xC1;
  ResidentRelationalSequenceOccurrenceSidecar leaf1{};
  leaf1.occurrence_identity = 0xC2;
  ResidentRelationalSequenceOccurrenceSidecar join{};
  join.occurrence_identity = 0xC3;
  join.child_occurrence_identities[0] = 0xC1;
  join.child_occurrence_identities[1] = 0xC2;

  DirectRelSeqRecipe recipes[2]{};
  DirectRelSeqOccurrence occ[3]{};
  if (!lower_canonical_resident_relseq(cells[0], canonical[0], leaf_body, &leaf0,
                                       &recipes[0], &occ[0]))
    return fail("lower leaf 0");
  DirectRelSeqRecipe dup{};
  if (!lower_canonical_resident_relseq(cells[1], canonical[1], leaf_body, &leaf1,
                                       &dup, &occ[1]))
    return fail("lower leaf 1");
  if (dup.revision_identity != recipes[0].revision_identity)
    return fail("same RecipeRevision lowered to two bodies");
  if (!lower_canonical_resident_relseq(cells[2], canonical[2], join_body, &join,
                                       &recipes[1], &occ[2]))
    return fail("lower join");

  ResidentRecipeOccurrence mismatched = canonical[0];
  mismatched.logical_recipe_id = 0xDEAD;
  DirectRelSeqRecipe junk{};
  DirectRelSeqOccurrence junk_o{};
  if (lower_canonical_resident_relseq(cells[0], mismatched, leaf_body, &leaf0,
                                      &junk, &junk_o))
    return fail("identity mismatch was accepted");

  DirectRelSeqOutput out{};
  if (!direct_relseq_evaluate_occurrence(recipes, 2u, occ, 3u, 0xC3, &out))
    return fail("host evaluate refused a lawful nested Occurrence");
  const std::uint32_t expected[] = {101u, 11u, 102u, 12u, 103u, 104u,
                                    101u, 21u, 102u, 22u, 103u};
  if (out.count != 11u)
    return fail("unit count diverged");
  for (unsigned i = 0; i < 11u; ++i)
    if (out.units[i] != expected[i]) return fail("unit identity diverged");

  DirectRelSeqRecipe morphs[2]{};
  const std::uint32_t obs_a[] = {11u};
  const std::uint32_t ports_a[] = {11u};
  const std::uint32_t ctx_a[] = {0xFAu};
  const std::uint32_t obs_b[] = {11u, 4u};
  const std::uint32_t ports_b[] = {11u};
  const std::uint32_t ctx_b[] = {0xFBu};
  if (!direct_relseq_induce_flat_recipe(0xE1, 0xF1, obs_a, 1u, ports_a, 1u, ctx_a, 1u,
                                       &morphs[0]))
    return fail("induce morph a");
  if (!direct_relseq_induce_flat_recipe(0xE2, 0xF1, obs_b, 2u, ports_b, 1u, ctx_b, 1u,
                                       &morphs[1]))
    return fail("induce morph b");
  DirectRelSeqOccurrence pick{};
  pick.occurrence_identity = 0x99;
  pick.logical_recipe_id = 0xE1;
  pick.revision_identity = morphs[0].revision_identity;
  pick.binding_count = 1;
  pick.bindings[0] = {0u, 11u, 0u};
  pick.context_feature_count = 1;
  pick.context_features[0] = 0xFAu;
  if (direct_relseq_select_recipe(morphs, 2u, 0xF1, pick) != &morphs[0])
    return fail("context select");
  pick.context_feature_count = 2;
  pick.context_features[1] = 0xFBu;
  if (direct_relseq_select_recipe(morphs, 2u, 0xF1, pick))
    return fail("tie accepted");

  std::printf(
      "DIRECT_ADULT_RELSEQ_BRIDGE_HOST GREEN fake_occurrence=0 induced=1 "
      "missing_port_refuse=1 select=1 tie_refuse=1 "
      "canonical_recipe=1 canonical_occurrence=1 sidecar_child_binding=1 "
      "identity_mismatch_refuse=1 host_evaluate=1 units=%u depth_peak=%u "
      "gpu=0 graph_flip=0\n",
      out.count, out.depth_peak);
  return 0;
}

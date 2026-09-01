#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RELATIONAL_SEQUENCE_BRIDGE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RELATIONAL_SEQUENCE_BRIDGE_CUH

#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_relational_sequence_composition.cuh"

#if defined(__CUDACC__)
#define DIRECT_ADULT_RELSEQ_HD __host__ __device__
#else
#define DIRECT_ADULT_RELSEQ_HD
#endif

namespace substrate::direct_adult_core {

// Domain-general sidecar for executable relation bodies whose boundary is an
// ordered sequence of fixed resident units and formal ports. Canonical
// RecipeRevision/Occurrence identity remains owned by ResidentRecipeCell and
// ResidentRecipeOccurrence. This sidecar contributes relation-body data only.
inline constexpr std::uint32_t kResidentRelSeqMaxPieces =
    direct_network::kDirectRelSeqMaxPieces;
inline constexpr std::uint32_t kResidentRelSeqMaxPorts =
    direct_network::kDirectRelSeqMaxPorts;

struct ResidentRelationalSequenceBody {
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::uint64_t relation_identity;
  direct_network::DirectRelSeqPiece pieces[kResidentRelSeqMaxPieces];
  std::uint32_t piece_count;
  std::uint32_t port_count;
  std::uint32_t context_features[direct_network::kDirectRelSeqMaxContextFeatures];
  std::uint32_t context_feature_count;
  std::uint32_t support;
  std::uint32_t active;
};
static_assert(std::is_trivially_copyable_v<ResidentRelationalSequenceBody>);

// Ordinary ResidentRecipeOccurrence stores canonical formal-port variable
// bindings. Recursive composition additionally needs to say that one formal
// port is supplied by the boundary of another current Occurrence. That is
// occurrence-local state, therefore it lives in a sidecar keyed by the exact
// canonical occurrence identity rather than in the persistent Recipe.
struct ResidentRelationalSequenceOccurrenceSidecar {
  std::uint64_t occurrence_identity;
  std::uint64_t child_occurrence_identities[kResidentRelSeqMaxPorts];
  std::uint32_t context_features[direct_network::kDirectRelSeqMaxContextFeatures];
  std::uint32_t context_feature_count;
};
static_assert(std::is_trivially_copyable_v<ResidentRelationalSequenceOccurrenceSidecar>);

DIRECT_ADULT_RELSEQ_HD inline bool resident_relseq_body_valid(
    const ResidentRelationalSequenceBody& body) {
  if (!body.logical_recipe_id || !body.revision_identity ||
      !body.relation_identity || !body.active || !body.piece_count ||
      body.piece_count > kResidentRelSeqMaxPieces ||
      body.port_count > kResidentRelSeqMaxPorts ||
      body.context_feature_count > direct_network::kDirectRelSeqMaxContextFeatures)
    return false;
  direct_network::DirectRelSeqRecipe lowered{};
  lowered.logical_recipe_id = body.logical_recipe_id;
  lowered.revision_identity = body.revision_identity;
  lowered.relation_identity = body.relation_identity;
  lowered.piece_count = body.piece_count;
  lowered.port_count = body.port_count;
  lowered.context_feature_count = body.context_feature_count;
  lowered.support = body.support;
  lowered.active = body.active;
  for (std::uint32_t i = 0; i < body.piece_count; ++i)
    lowered.pieces[i] = body.pieces[i];
  for (std::uint32_t i = 0; i < body.context_feature_count; ++i)
    lowered.context_features[i] = body.context_features[i];
  // Body revision identity is canonical Adult identity, not DirectRelSeq's
  // independently computed helper identity, so validate structural fields here.
  for (std::uint32_t i = 0; i < lowered.piece_count; ++i) {
    const auto kind = static_cast<direct_network::DirectRelSeqPieceKind>(lowered.pieces[i].kind);
    if (kind == direct_network::DirectRelSeqPieceKind::fixed_unit) {
      if (!lowered.pieces[i].value || lowered.pieces[i].trim_right) return false;
    } else if (kind == direct_network::DirectRelSeqPieceKind::port) {
      if (lowered.pieces[i].value >= lowered.port_count) return false;
    } else return false;
  }
  return true;
}

template <typename RecipeCellT, typename OccurrenceT>
DIRECT_ADULT_RELSEQ_HD inline bool lower_canonical_resident_relseq(
    const RecipeCellT& recipe, const OccurrenceT& occurrence,
    const ResidentRelationalSequenceBody& body,
    const ResidentRelationalSequenceOccurrenceSidecar* sidecar,
    direct_network::DirectRelSeqRecipe* out_recipe,
    direct_network::DirectRelSeqOccurrence* out_occurrence) {
  if (!out_recipe || !out_occurrence || !resident_relseq_body_valid(body) ||
      recipe.logical_recipe_id != body.logical_recipe_id ||
      recipe.revision_identity != body.revision_identity ||
      occurrence.logical_recipe_id != recipe.logical_recipe_id ||
      occurrence.revision_identity != recipe.revision_identity ||
      occurrence.occurrence_identity == 0u ||
      occurrence.binding_count != body.port_count ||
      occurrence.binding_count > kResidentRelSeqMaxPorts)
    return false;
  direct_network::DirectRelSeqRecipe r{};
  r.logical_recipe_id = body.logical_recipe_id;
  r.relation_identity = body.relation_identity;
  r.piece_count = body.piece_count;
  r.port_count = body.port_count;
  r.context_feature_count = body.context_feature_count;
  r.support = body.support;
  r.active = body.active;
  for (std::uint32_t i = 0; i < body.piece_count; ++i) r.pieces[i] = body.pieces[i];
  for (std::uint32_t i = 0; i < body.context_feature_count; ++i)
    r.context_features[i] = body.context_features[i];
  // DirectRelSeq's evaluator identifies the physical body by this helper
  // revision. Canonical revision identity was checked above and remains the
  // authority; this is an execution-lowering identity only.
  r.revision_identity = direct_network::direct_relseq_recipe_revision_identity(r);

  direct_network::DirectRelSeqOccurrence o{};
  o.occurrence_identity = occurrence.occurrence_identity;
  o.logical_recipe_id = r.logical_recipe_id;
  o.revision_identity = r.revision_identity;
  o.binding_count = occurrence.binding_count;
  if (sidecar) {
    if (sidecar->occurrence_identity != occurrence.occurrence_identity ||
        sidecar->context_feature_count > direct_network::kDirectRelSeqMaxContextFeatures)
      return false;
    o.context_feature_count = sidecar->context_feature_count;
    for (std::uint32_t i = 0; i < sidecar->context_feature_count; ++i)
      o.context_features[i] = sidecar->context_features[i];
  } else if (body.context_feature_count) {
    o.context_feature_count = body.context_feature_count;
    for (std::uint32_t i = 0; i < body.context_feature_count; ++i)
      o.context_features[i] = body.context_features[i];
  }
  for (std::uint32_t i = 0; i < occurrence.binding_count; ++i) {
    if (occurrence.bindings[i].formal_port_index != i ||
        occurrence.bindings[i].variable_identity == 0u)
      return false;
    const std::uint64_t child = sidecar ? sidecar->child_occurrence_identities[i] : 0u;
    o.bindings[i] = direct_network::DirectRelSeqPortBinding{
        i, child ? 0u : occurrence.bindings[i].variable_identity, child};
  }
  *out_recipe = r;
  *out_occurrence = o;
  return true;
}

}  // namespace substrate::direct_adult_core

#undef DIRECT_ADULT_RELSEQ_HD
#endif

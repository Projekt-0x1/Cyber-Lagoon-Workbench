#ifndef HARDWARE_NATIVE_DIRECT_RESIDENT_OCCURRENCE_COUPLING_CUH
#define HARDWARE_NATIVE_DIRECT_RESIDENT_OCCURRENCE_COUPLING_CUH

#include "hardware_native/direct_adult_core.cuh"

#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CORE_CUH
#include "hardware_native/direct_network_brain.cuh"
namespace substrate::direct_adult_core {
#endif

enum class ResidentOccurrenceCouplingKind : std::uint16_t {
  formal_variable = 0u,
  causal_intersection = 1u,
  // Transient unordered co-membership of adjacent authenticated contacts on the
  // same physical boundary source. This is candidate episode structure, not
  // semantic truth and not persistent Recipe morphology.
  boundary_coepisode = 2u,
  // A transient completion edge reconstructed from positively credited resident
  // recruitment morphology. `reserved` carries the original structural coupling
  // kind for recruitment-identity matching. It never asserts external
  // participation for an endogenous endpoint.
  recruited_completion = 3u,
};
inline constexpr std::uint16_t kResidentOccurrenceNoFormalPort = 0xffffu;

struct alignas(8) ResidentOccurrenceCoupling {
  std::uint64_t source_occurrence_identity;
  std::uint64_t target_occurrence_identity;
  std::uint64_t source_revision_identity;
  std::uint64_t target_revision_identity;
  std::uint64_t source_derivation_rank;
  std::uint64_t target_derivation_rank;
  std::uint64_t source_identity;
  std::uint64_t target_identity;
  std::uint64_t source_route_incarnation;
  std::uint64_t target_route_incarnation;
  std::uint32_t variable_identity;
  std::uint32_t source_incarnation;
  std::uint32_t target_incarnation;
  std::uint16_t source_port_index;
  std::uint16_t target_port_index;
  ResidentOccurrenceCouplingKind kind;
  std::uint16_t reserved;
  std::uint32_t reserved2;
};
static_assert(std::is_standard_layout_v<ResidentOccurrenceCoupling> &&
              std::is_trivial_v<ResidentOccurrenceCoupling> &&
              std::has_unique_object_representations_v<ResidentOccurrenceCoupling>);
DIRECT_ADULT_HD inline bool bind_resident_occurrence_coupling(
    const ResidentRecipeOccurrence& source,
    const direct_network::ResidentRecipeDerivation& source_derivation,
    std::uint16_t source_port_index,
    const ResidentRecipeOccurrence& target,
    const direct_network::ResidentRecipeDerivation& target_derivation,
    std::uint16_t target_port_index, ResidentOccurrenceCoupling* out) {
  if (out == nullptr || source.state != kResidentRecipeOccurrenceLive ||
      target.state != kResidentRecipeOccurrenceLive ||
      source.occurrence_identity == 0u || target.occurrence_identity == 0u ||
      source.occurrence_identity == target.occurrence_identity ||
      source.logical_recipe_id != source_derivation.logical_recipe_id ||
      source.revision_identity != source_derivation.revision_identity ||
      target.logical_recipe_id != target_derivation.logical_recipe_id ||
      target.revision_identity != target_derivation.revision_identity ||
      source_port_index >= source.binding_count ||
      source_port_index >= source_derivation.port_count ||
      target_port_index >= target.binding_count ||
      target_port_index >= target_derivation.port_count)
    return false;
  const ResidentOccurrencePortBinding& source_binding =
      source.bindings[source_port_index];
  const ResidentOccurrencePortBinding& target_binding =
      target.bindings[target_port_index];
  if (source_binding.formal_port_index != source_port_index ||
      target_binding.formal_port_index != target_port_index ||
      source_binding.variable_identity == 0u ||
      source_binding.variable_identity != target_binding.variable_identity ||
      !direct_network::resident_recipe_ports_compatible(
          source_derivation.ports[source_port_index],
          target_derivation.ports[target_port_index]))
    return false;
  *out = ResidentOccurrenceCoupling{
      source.occurrence_identity, target.occurrence_identity,
      source.revision_identity, target.revision_identity,
      source_derivation.generation, target_derivation.generation,
      source.source_identity, target.source_identity,
      source.route_incarnation, target.route_incarnation,
      source_binding.variable_identity, source.source_incarnation,
      target.source_incarnation, source_port_index, target_port_index,
      ResidentOccurrenceCouplingKind::formal_variable, 0u, 0u};
  return true;
}

DIRECT_ADULT_HD inline std::uint64_t resident_occurrence_persistent_morphology_key(
    const ResidentRecipeOccurrence& occurrence,
    const direct_network::ResidentRecipeDerivation& derivation) {
  using direct_network::exact_history_fold_word;
  std::uint64_t key = exact_history_fold_word(
      0x6f63636d6f727068ull, occurrence.logical_recipe_id);
  key = exact_history_fold_word(key, occurrence.revision_identity);
  key = exact_history_fold_word(key, derivation.generation);
  return key == 0u ? 1u : key;
}

DIRECT_ADULT_HD inline bool bind_resident_occurrence_boundary_coepisode_coupling(
    const ResidentRecipeOccurrence& left,
    const direct_network::ResidentRecipeDerivation& left_derivation,
    const ResidentRecipeOccurrence& right,
    const direct_network::ResidentRecipeDerivation& right_derivation,
    ResidentOccurrenceCoupling* out) {
  using direct_network::exact_history_fold_word;
  if (out == nullptr || left.source_identity == 0u ||
      right.source_identity == 0u ||
      left.state != kResidentRecipeOccurrenceLive ||
      right.state != kResidentRecipeOccurrenceLive ||
      left.occurrence_identity == 0u || right.occurrence_identity == 0u ||
      left.occurrence_identity == right.occurrence_identity ||
      left.participation_identity == 0u || right.participation_identity == 0u ||
      left.participation_identity == right.participation_identity ||
      left.logical_recipe_id != left_derivation.logical_recipe_id ||
      left.revision_identity != left_derivation.revision_identity ||
      right.logical_recipe_id != right_derivation.logical_recipe_id ||
      right.revision_identity != right_derivation.revision_identity ||
      left.authority != DirectParticipationAuthority::independent_external ||
      right.authority != DirectParticipationAuthority::independent_external)
    return false;

  const ResidentRecipeOccurrence* source = &left;
  const ResidentRecipeOccurrence* target = &right;
  const direct_network::ResidentRecipeDerivation* source_derivation = &left_derivation;
  const direct_network::ResidentRecipeDerivation* target_derivation = &right_derivation;
  const std::uint64_t left_key =
      resident_occurrence_persistent_morphology_key(left, left_derivation);
  const std::uint64_t right_key =
      resident_occurrence_persistent_morphology_key(right, right_derivation);
  if (left_key > right_key ||
      (left_key == right_key && left.occurrence_identity > right.occurrence_identity)) {
    source = &right; target = &left;
    source_derivation = &right_derivation; target_derivation = &left_derivation;
  }
  // Episode incidence can span authenticated body sources (for example sound
  // plus sight). Canonicalize the physical source pair; temporal adjacency is
  // checked by the caller and this edge remains transient/non-crediting.
  const std::uint64_t first_source =
      left.source_identity < right.source_identity
          ? left.source_identity : right.source_identity;
  const std::uint64_t second_source =
      left.source_identity < right.source_identity
          ? right.source_identity : left.source_identity;
  std::uint64_t relation = exact_history_fold_word(
      0x636f657069736f64ull, first_source);
  relation = exact_history_fold_word(relation, second_source);
  std::uint32_t relation_identity = static_cast<std::uint32_t>(relation) ^
      static_cast<std::uint32_t>(relation >> 32u);
  if (relation_identity == 0u) relation_identity = 1u;
  *out = ResidentOccurrenceCoupling{
      source->occurrence_identity, target->occurrence_identity,
      source->revision_identity, target->revision_identity,
      source_derivation->generation, target_derivation->generation,
      source->source_identity, target->source_identity,
      source->route_incarnation, target->route_incarnation,
      relation_identity, source->source_incarnation, target->source_incarnation,
      kResidentOccurrenceNoFormalPort, kResidentOccurrenceNoFormalPort,
      ResidentOccurrenceCouplingKind::boundary_coepisode, 0u, 0u};
  return true;
}

DIRECT_ADULT_HD inline std::uint32_t
resident_occurrence_causal_intersection_relation_identity(
    std::uint32_t intersection_node) {
  using direct_network::exact_history_fold_word;
  if (intersection_node == direct_network::kInvalidIndex) return 0u;
  const std::uint64_t relation = exact_history_fold_word(
      0x63617573616c6978ull, intersection_node);
  std::uint32_t identity = static_cast<std::uint32_t>(relation) ^
      static_cast<std::uint32_t>(relation >> 32u);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline bool bind_resident_occurrence_causal_intersection_coupling(
    const ResidentRecipeOccurrence& left,
    const direct_network::ResidentRecipeDerivation& left_derivation,
    const ResidentRecipeOccurrence& right,
    const direct_network::ResidentRecipeDerivation& right_derivation,
    std::uint32_t intersection_node, ResidentOccurrenceCoupling* out) {
  using direct_network::exact_history_fold_word;
  if (out == nullptr || intersection_node == direct_network::kInvalidIndex ||
      left.state != kResidentRecipeOccurrenceLive ||
      right.state != kResidentRecipeOccurrenceLive ||
      left.occurrence_identity == 0u || right.occurrence_identity == 0u ||
      left.occurrence_identity == right.occurrence_identity ||
      left.logical_recipe_id != left_derivation.logical_recipe_id ||
      left.revision_identity != left_derivation.revision_identity ||
      right.logical_recipe_id != right_derivation.logical_recipe_id ||
      right.revision_identity != right_derivation.revision_identity)
    return false;
  // This relation belongs to the unfolded computation, not to either persistent
  // Recipe's formal ports. Canonical orientation makes frontier order irrelevant.
  const ResidentRecipeOccurrence* source = &left;
  const ResidentRecipeOccurrence* target = &right;
  const direct_network::ResidentRecipeDerivation* source_derivation = &left_derivation;
  const direct_network::ResidentRecipeDerivation* target_derivation = &right_derivation;
  if (source->occurrence_identity > target->occurrence_identity) {
    source = &right; target = &left;
    source_derivation = &right_derivation; target_derivation = &left_derivation;
  }
  const std::uint32_t relation_identity =
      resident_occurrence_causal_intersection_relation_identity(intersection_node);
  if (relation_identity == 0u) return false;
  *out = ResidentOccurrenceCoupling{
      source->occurrence_identity, target->occurrence_identity,
      source->revision_identity, target->revision_identity,
      source_derivation->generation, target_derivation->generation,
      source->source_identity, target->source_identity,
      source->route_incarnation, target->route_incarnation,
      relation_identity, source->source_incarnation, target->source_incarnation,
      kResidentOccurrenceNoFormalPort, kResidentOccurrenceNoFormalPort,
      ResidentOccurrenceCouplingKind::causal_intersection, 0u, intersection_node};
  return true;
}

#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CORE_CUH
}
#endif
#endif

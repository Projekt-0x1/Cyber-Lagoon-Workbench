#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_BOUNDARY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_BOUNDARY_CUH
// Mid-include splice for substrate::direct_adult_core (see direct_adult_core.cuh).
// Owns public Network boundary relation derived from a validated closure.

inline constexpr std::uint16_t kResidentNetworkBoundaryMaxMembers = 4u;
struct ResidentNetworkBoundaryMemberWitness {
  std::uint64_t occurrence_identity, logical_recipe_id, revision_identity;
  std::uint32_t input_variable_identity, output_variable_identity;
  std::uint32_t relation;
  std::int32_t parameter_q16;
};
struct ResidentNetworkBoundaryRelation {
  std::uint64_t closure_identity, witness_identity, behavior_identity;
  ResidentRelationalNetworkBoundary input_boundary, output_boundary;
  ResidentNetworkBoundaryMemberWitness members[
      kResidentNetworkBoundaryMaxMembers];
  std::int32_t composed_parameter_q16;
  std::uint16_t member_count, relation;
  std::uint32_t reserved, reserved2;
};
static_assert(std::is_standard_layout_v<ResidentNetworkBoundaryMemberWitness> &&
              std::is_trivial_v<ResidentNetworkBoundaryMemberWitness> &&
              std::has_unique_object_representations_v<
                  ResidentNetworkBoundaryMemberWitness>);
static_assert(std::is_standard_layout_v<ResidentNetworkBoundaryRelation> &&
              std::is_trivial_v<ResidentNetworkBoundaryRelation> &&
              std::has_unique_object_representations_v<
                  ResidentNetworkBoundaryRelation>);

DIRECT_ADULT_HD inline std::uint64_t resident_network_boundary_witness(
    const ResidentNetworkBoundaryRelation& relation) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x626f756e64617279ull, relation.closure_identity);
  identity = exact_history_fold_word(identity, relation.member_count);
  identity = exact_history_fold_word(identity, relation.relation);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(relation.composed_parameter_q16));
  const ResidentRelationalNetworkBoundary boundaries[2] = {
      relation.input_boundary, relation.output_boundary};
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    identity = exact_history_fold_word(
        identity, boundaries[i].occurrence_identity);
    identity = exact_history_fold_word(identity, boundaries[i].variable_identity);
    identity = exact_history_fold_word(identity, boundaries[i].formal_port_index);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(boundaries[i].domain));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(boundaries[i].direction));
    identity = exact_history_fold_word(identity, boundaries[i].arity);
  }
  for (std::uint32_t i = 0u; i < relation.member_count; ++i) {
    const auto& member = relation.members[i];
    identity = exact_history_fold_word(identity, member.occurrence_identity);
    identity = exact_history_fold_word(identity, member.logical_recipe_id);
    identity = exact_history_fold_word(identity, member.revision_identity);
    identity = exact_history_fold_word(identity, member.input_variable_identity);
    identity = exact_history_fold_word(identity, member.output_variable_identity);
    identity = exact_history_fold_word(identity, member.relation);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint64_t>(member.parameter_q16));
  }
  return identity == 0u ? 1u : identity;
}

// Derive the public relation from a validated active closure.  This bounded
// lowering handles one trigger-relation chain; neither rank nor host-supplied
// expected output participates in discovery or evaluation.
DIRECT_ADULT_HD inline bool compose_resident_network_boundary_relation(
    const direct_network::ResidentRecipeCell* recipes,
    const direct_network::ResidentRecipeDerivation* derivations,
    const ResidentRecipeOccurrence* occurrences,
    std::uint32_t occurrence_count,
    const ResidentOccurrenceCoupling* couplings,
    std::uint32_t coupling_count,
    const ResidentRelationalNetworkClosure& closure,
    ResidentNetworkBoundaryRelation* out) {
  using namespace direct_network;
  if (recipes == nullptr || derivations == nullptr || occurrences == nullptr ||
      couplings == nullptr || out == nullptr || occurrence_count < 2u ||
      occurrence_count > kResidentNetworkBoundaryMaxMembers ||
      coupling_count + 1u != occurrence_count || closure.boundary_count != 2u)
    return false;
  ResidentRelationalNetworkClosure rebuilt{};
  if (!bind_resident_relational_network_closure(
          recipes, derivations, occurrences, occurrence_count, couplings,
          coupling_count, &rebuilt))
    return false;
  const auto* expected_bytes =
      reinterpret_cast<const unsigned char*>(&closure);
  const auto* rebuilt_bytes =
      reinterpret_cast<const unsigned char*>(&rebuilt);
  for (std::uint32_t i = 0u; i < sizeof(closure); ++i)
    if (expected_bytes[i] != rebuilt_bytes[i]) return false;
  std::uint32_t input_boundary = 2u, output_boundary = 2u;
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    const auto direction = closure.boundary[i].direction;
    if (direction == ResidentRecipePortDirection::input)
      input_boundary = input_boundary == 2u ? i : 3u;
    if (direction == ResidentRecipePortDirection::output)
      output_boundary = output_boundary == 2u ? i : 3u;
  }
  if (input_boundary >= 2u || output_boundary >= 2u)
    return false;
  std::uint32_t current = occurrence_count;
  for (std::uint32_t i = 0u; i < occurrence_count; ++i)
    if (occurrences[i].occurrence_identity ==
        closure.boundary[input_boundary].occurrence_identity)
      current = i;
  if (current == occurrence_count) return false;
  ResidentNetworkBoundaryRelation candidate{};
  candidate.closure_identity = closure.identity;
  candidate.input_boundary = closure.boundary[input_boundary];
  candidate.output_boundary = closure.boundary[output_boundary];
  candidate.member_count = static_cast<std::uint16_t>(occurrence_count);
  candidate.relation =
      static_cast<std::uint16_t>(ResidentRecipeRelation::trigger);
  bool visited[kResidentNetworkBoundaryMaxMembers]{};
  std::int64_t composed = 0;
  for (std::uint32_t step = 0u; step < occurrence_count; ++step) {
    if (current >= occurrence_count || visited[current]) return false;
    visited[current] = true;
    const auto& derivation = derivations[current];
    const auto& occurrence = occurrences[current];
    if (derivation.port_count != 2u || derivation.relation_count != 1u ||
        derivation.parameter_count != 1u ||
        derivation.relations[0] !=
            static_cast<std::uint32_t>(ResidentRecipeRelation::trigger) ||
        derivation.ports[0].direction != ResidentRecipePortDirection::input ||
        derivation.ports[1].direction != ResidentRecipePortDirection::output ||
        derivation.ports[0].domain != ResidentRecipePortDomain::q16_scalar ||
        derivation.ports[1].domain != ResidentRecipePortDomain::q16_scalar ||
        derivation.ports[0].arity != 1u || derivation.ports[1].arity != 1u)
      return false;
    candidate.members[step] = ResidentNetworkBoundaryMemberWitness{
        occurrence.occurrence_identity, occurrence.logical_recipe_id,
        occurrence.revision_identity,
        occurrence.bindings[0].variable_identity,
        occurrence.bindings[1].variable_identity,
        derivation.relations[0], derivation.parameters_q16[0]};
    composed += derivation.parameters_q16[0];
    if (composed < -0x80000000ll || composed > 0x7fffffffll)
      return false;
    std::uint32_t next = occurrence_count, outgoing = 0u;
    for (std::uint32_t edge = 0u; edge < coupling_count; ++edge) {
      if (couplings[edge].source_occurrence_identity !=
          occurrence.occurrence_identity)
        continue;
      if (couplings[edge].source_port_index != 1u ||
          couplings[edge].target_port_index != 0u)
        return false;
      for (std::uint32_t target = 0u; target < occurrence_count; ++target)
        if (occurrences[target].occurrence_identity ==
            couplings[edge].target_occurrence_identity)
          next = target;
      ++outgoing;
    }
    if (step + 1u == occurrence_count) {
      if (outgoing != 0u || occurrence.occurrence_identity !=
              candidate.output_boundary.occurrence_identity)
        return false;
    } else if (outgoing != 1u || next == occurrence_count) {
      return false;
    }
    current = next;
  }
  candidate.composed_parameter_q16 = static_cast<std::int32_t>(composed);
  std::uint64_t behavior = exact_history_fold_word(
      0x626f756e64626568ull, candidate.relation);
  behavior = exact_history_fold_word(
      behavior, static_cast<std::uint64_t>(candidate.composed_parameter_q16));
  behavior = exact_history_fold_word(
      behavior, candidate.input_boundary.variable_identity);
  behavior = exact_history_fold_word(
      behavior, candidate.output_boundary.variable_identity);
  behavior = exact_history_fold_word(
      behavior, static_cast<std::uint32_t>(candidate.input_boundary.domain));
  behavior = exact_history_fold_word(
      behavior, static_cast<std::uint32_t>(candidate.output_boundary.domain));
  behavior = exact_history_fold_word(behavior, candidate.input_boundary.arity);
  behavior = exact_history_fold_word(behavior, candidate.output_boundary.arity);
  candidate.behavior_identity = behavior == 0u ? 1u : behavior;
  candidate.witness_identity = resident_network_boundary_witness(candidate);
  *out = candidate;
  return true;
}

DIRECT_ADULT_HD inline bool evaluate_resident_network_boundary_relation_q16(
    const ResidentNetworkBoundaryRelation& relation,
    std::int32_t input_q16, std::int32_t* output_q16) {
  if (relation.member_count < 2u ||
      relation.member_count > kResidentNetworkBoundaryMaxMembers ||
      relation.witness_identity == 0u ||
      relation.witness_identity != resident_network_boundary_witness(relation) ||
      relation.relation !=
          static_cast<std::uint16_t>(
              direct_network::ResidentRecipeRelation::trigger))
    return false;
  return direct_network::evaluate_resident_recipe_boundary_q16(
      relation.relation, relation.composed_parameter_q16, input_q16,
      output_q16);
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_NETWORK_BOUNDARY_CUH

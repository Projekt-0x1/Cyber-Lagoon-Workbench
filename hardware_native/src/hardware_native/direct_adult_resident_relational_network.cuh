#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_RELATIONAL_NETWORK_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_RELATIONAL_NETWORK_CUH
// Mid-include splice for substrate::direct_adult_core (see direct_adult_core.cuh).
// Owns transient Network closure over an explicit Occurrence/coupling frontier.

inline constexpr std::uint16_t kResidentRelationalNetworkMaxOccurrences = 8u;
inline constexpr std::uint16_t kResidentRelationalNetworkMinOccurrences = 2u;
inline constexpr std::uint16_t kResidentRelationalNetworkMaxCouplings = 24u;
inline constexpr std::uint16_t kResidentRelationalNetworkMaxBoundary = 8u;
// Upper bound on coactive Networks from one frontier (one coupled pair each).
inline constexpr std::uint16_t kResidentRelationalNetworkMaxCoactive = 4u;
struct ResidentRelationalNetworkMember {
  std::uint64_t occurrence_identity, logical_recipe_id, revision_identity;
  std::uint64_t participation_identity, derivation_rank;
  std::uint64_t morphology_identity;
  std::int32_t eligibility_q16;
  std::uint32_t context_signature;
  std::uint16_t binding_count;
  ResidentOccurrenceLineageKind lineage_kind;
  DirectParticipationAuthority authority;
};
struct ResidentRelationalNetworkBoundary {
  std::uint64_t occurrence_identity;
  std::uint32_t variable_identity;
  std::uint16_t formal_port_index;
  direct_network::ResidentRecipePortDomain domain;
  direct_network::ResidentRecipePortDirection direction;
  std::uint16_t arity, reserved;
  std::uint32_t reserved2;
};
struct ResidentRelationalNetworkClosure {
  std::uint64_t identity;
  std::int64_t eligibility_signed_q16;
  std::uint64_t eligibility_l1_q16;
  std::uint16_t occurrence_count, coupling_count, boundary_count;
  std::uint16_t overlap_count, reconvergence_count, actual_count;
  std::uint16_t reserved[2];
  ResidentRelationalNetworkMember members[
      kResidentRelationalNetworkMaxOccurrences];
  ResidentOccurrenceCoupling couplings[
      kResidentRelationalNetworkMaxCouplings];
  ResidentRelationalNetworkBoundary boundary[
      kResidentRelationalNetworkMaxBoundary];
};
struct ResidentRelationalNetworkSet {
  std::uint64_t identity;
  std::uint16_t network_count, networked_occurrence_count;
  std::uint16_t isolated_occurrence_count, coupling_count;
  ResidentRelationalNetworkClosure networks[
      kResidentRelationalNetworkMaxCoactive];
};
static_assert(std::is_standard_layout_v<ResidentRelationalNetworkMember> &&
              std::is_trivial_v<ResidentRelationalNetworkMember> &&
              std::has_unique_object_representations_v<
                  ResidentRelationalNetworkMember>);
static_assert(std::is_standard_layout_v<ResidentRelationalNetworkBoundary> &&
              std::is_trivial_v<ResidentRelationalNetworkBoundary> &&
              std::has_unique_object_representations_v<
                  ResidentRelationalNetworkBoundary>);
static_assert(std::is_standard_layout_v<ResidentRelationalNetworkClosure> &&
              std::is_trivial_v<ResidentRelationalNetworkClosure> &&
              std::has_unique_object_representations_v<
                  ResidentRelationalNetworkClosure>);
static_assert(std::is_standard_layout_v<ResidentRelationalNetworkSet> &&
              std::is_trivial_v<ResidentRelationalNetworkSet> &&
              std::has_unique_object_representations_v<
                  ResidentRelationalNetworkSet>);

DIRECT_ADULT_HD inline bool resident_relational_network_contains_occurrences(
    const ResidentRelationalNetworkClosure& closure,
    const std::uint64_t* occurrence_identities,
    std::uint32_t occurrence_count) {
  if (occurrence_identities == nullptr || closure.identity == 0u ||
      occurrence_count == 0u || occurrence_count > closure.occurrence_count ||
      closure.occurrence_count > kResidentRelationalNetworkMaxOccurrences)
    return false;
  for (std::uint32_t i = 0u; i < occurrence_count; ++i) {
    if (occurrence_identities[i] == 0u) return false;
    bool found = false;
    for (std::uint16_t m = 0u; m < closure.occurrence_count; ++m)
      found |= closure.members[m].occurrence_identity ==
          occurrence_identities[i];
    if (!found) return false;
    for (std::uint32_t prior = 0u; prior < i; ++prior)
      if (occurrence_identities[prior] == occurrence_identities[i])
        return false;
  }
  return true;
}

DIRECT_ADULT_HD inline bool resident_relational_coupling_less(
    const ResidentOccurrenceCoupling& left,
    const ResidentOccurrenceCoupling& right) {
  if (left.source_occurrence_identity != right.source_occurrence_identity)
    return left.source_occurrence_identity < right.source_occurrence_identity;
  if (left.target_occurrence_identity != right.target_occurrence_identity)
    return left.target_occurrence_identity < right.target_occurrence_identity;
  if (left.variable_identity != right.variable_identity)
    return left.variable_identity < right.variable_identity;
  if (left.source_port_index != right.source_port_index)
    return left.source_port_index < right.source_port_index;
  return left.target_port_index < right.target_port_index;
}

// Parameter values and physical node placement can change while the same
// reusable computation rematerializes. Port law, relation law and arity cannot.
DIRECT_ADULT_HD inline std::uint64_t resident_recipe_morphology_identity(
    const direct_network::ResidentRecipeDerivation& derivation) {
  using namespace direct_network;
  if (derivation.port_count == 0u ||
      derivation.port_count > kResidentDerivationWidth ||
      derivation.relation_count > kResidentDerivationWidth ||
      derivation.parameter_count > kResidentDerivationWidth)
    return 0u;
  std::uint64_t identity = exact_history_fold_word(
      0x6e7265636d6f7270ull, derivation.port_count);
  identity = exact_history_fold_word(identity, derivation.relation_count);
  identity = exact_history_fold_word(identity, derivation.parameter_count);
  for (std::uint16_t port = 0u; port < derivation.port_count; ++port) {
    const auto& formal = derivation.ports[port];
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(formal.domain));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(formal.direction));
    identity = exact_history_fold_word(identity, formal.arity);
  }
  for (std::uint16_t relation = 0u;
       relation < derivation.relation_count; ++relation)
    identity = exact_history_fold_word(
        identity, derivation.relations[relation]);
  return identity == 0u ? 1u : identity;
}

// A Network is a transient closure over an explicitly supplied active
// frontier. Canonicalization depends on exact Occurrence/incidence identity,
// never array order or derivation-rank order, and commits only after the whole
// bounded closure validates.
DIRECT_ADULT_HD inline bool bind_resident_relational_network_closure(
    const direct_network::ResidentRecipeCell* recipes,
    const direct_network::ResidentRecipeDerivation* derivations,
    const ResidentRecipeOccurrence* occurrences,
    std::uint32_t occurrence_count,
    const ResidentOccurrenceCoupling* couplings,
    std::uint32_t coupling_count,
    ResidentRelationalNetworkClosure* out) {
  using namespace direct_network;
  if (recipes == nullptr || derivations == nullptr || occurrences == nullptr ||
      couplings == nullptr || out == nullptr ||
      occurrence_count < kResidentRelationalNetworkMinOccurrences ||
      occurrence_count > kResidentRelationalNetworkMaxOccurrences ||
      coupling_count < occurrence_count - 1u ||
      coupling_count > kResidentRelationalNetworkMaxCouplings)
    return false;
  ResidentRelationalNetworkClosure candidate{};
  candidate.occurrence_count = static_cast<std::uint16_t>(occurrence_count);
  candidate.coupling_count = static_cast<std::uint16_t>(coupling_count);
  std::uint8_t adjacency[kResidentRelationalNetworkMaxOccurrences]{};
  std::uint8_t incoming[kResidentRelationalNetworkMaxOccurrences]{};
  bool coupled_port[kResidentRelationalNetworkMaxOccurrences]
                   [kResidentDerivationWidth]{};
  for (std::uint32_t i = 0u; i < occurrence_count; ++i) {
    const auto& recipe = recipes[i];
    const auto& derivation = derivations[i];
    const auto& occurrence = occurrences[i];
    if (recipe.logical_recipe_id == 0u || recipe.revision_identity == 0u ||
        occurrence.state != kResidentRecipeOccurrenceLive ||
        occurrence.occurrence_identity == 0u ||
        occurrence.participation_identity == 0u ||
        occurrence.logical_recipe_id != recipe.logical_recipe_id ||
        occurrence.revision_identity != recipe.revision_identity ||
        derivation.logical_recipe_id != recipe.logical_recipe_id ||
        derivation.revision_identity != recipe.revision_identity ||
        occurrence.binding_count == 0u ||
        occurrence.binding_count != derivation.port_count ||
        occurrence.binding_count > kResidentDerivationWidth)
      return false;
    const bool actual =
        occurrence.lineage_kind == ResidentOccurrenceLineageKind::actual;
    if (occurrence.lineage_kind == ResidentOccurrenceLineageKind::none ||
        (actual && occurrence.authority == DirectParticipationAuthority::none) ||
        (!actual && occurrence.authority != DirectParticipationAuthority::none))
      return false;
    for (std::uint32_t j = 0u; j < i; ++j)
      if (occurrences[j].occurrence_identity == occurrence.occurrence_identity ||
          occurrences[j].participation_identity ==
              occurrence.participation_identity)
        return false;
    for (std::uint32_t port = 0u; port < occurrence.binding_count; ++port)
      if (occurrence.bindings[port].formal_port_index != port ||
          occurrence.bindings[port].variable_identity == 0u)
        return false;
    const std::uint64_t morphology_identity =
        resident_recipe_morphology_identity(derivation);
    if (morphology_identity == 0u) return false;
    candidate.members[i] = ResidentRelationalNetworkMember{
        occurrence.occurrence_identity, occurrence.logical_recipe_id,
        occurrence.revision_identity, occurrence.participation_identity,
        derivation.generation, morphology_identity, occurrence.eligibility_q16,
        occurrence.context_signature,
        occurrence.binding_count, occurrence.lineage_kind,
        occurrence.authority};
    candidate.eligibility_signed_q16 += occurrence.eligibility_q16;
    const std::int64_t signed_eligibility = occurrence.eligibility_q16;
    candidate.eligibility_l1_q16 += static_cast<std::uint64_t>(
        signed_eligibility < 0 ? -signed_eligibility : signed_eligibility);
    if (actual) ++candidate.actual_count;
    for (std::uint32_t j = 0u; j < i; ++j)
      if (occurrences[j].logical_recipe_id == occurrence.logical_recipe_id &&
          occurrences[j].revision_identity == occurrence.revision_identity)
        ++candidate.overlap_count;
  }
  for (std::uint32_t i = 1u; i < occurrence_count; ++i) {
    const auto value = candidate.members[i];
    std::uint32_t j = i;
    while (j != 0u && candidate.members[j - 1u].occurrence_identity >
                            value.occurrence_identity) {
      candidate.members[j] = candidate.members[j - 1u];
      --j;
    }
    candidate.members[j] = value;
  }
  for (std::uint32_t i = 0u; i < coupling_count; ++i) {
    const auto& coupling = couplings[i];
    std::uint32_t source = occurrence_count, target = occurrence_count;
    for (std::uint32_t member = 0u; member < occurrence_count; ++member) {
      if (occurrences[member].occurrence_identity ==
          coupling.source_occurrence_identity) source = member;
      if (occurrences[member].occurrence_identity ==
          coupling.target_occurrence_identity) target = member;
    }
    if (source == occurrence_count || target == occurrence_count ||
        source == target || coupling.variable_identity == 0u ||
        coupling.source_revision_identity != occurrences[source].revision_identity ||
        coupling.target_revision_identity != occurrences[target].revision_identity ||
        coupling.source_derivation_rank != derivations[source].generation ||
        coupling.target_derivation_rank != derivations[target].generation ||
        coupling.source_port_index >= occurrences[source].binding_count ||
        coupling.target_port_index >= occurrences[target].binding_count ||
        occurrences[source].bindings[coupling.source_port_index]
                .variable_identity != coupling.variable_identity ||
        occurrences[target].bindings[coupling.target_port_index]
                .variable_identity != coupling.variable_identity ||
        !resident_recipe_ports_compatible(
            derivations[source].ports[coupling.source_port_index],
            derivations[target].ports[coupling.target_port_index]))
      return false;
    for (std::uint32_t j = 0u; j < i; ++j)
      if (couplings[j].source_occurrence_identity ==
              coupling.source_occurrence_identity &&
          couplings[j].target_occurrence_identity ==
              coupling.target_occurrence_identity &&
          couplings[j].variable_identity == coupling.variable_identity &&
          couplings[j].source_port_index == coupling.source_port_index &&
          couplings[j].target_port_index == coupling.target_port_index)
        return false;
    coupled_port[source][coupling.source_port_index] = true;
    coupled_port[target][coupling.target_port_index] = true;
    adjacency[source] |= static_cast<std::uint8_t>(1u << target);
    adjacency[target] |= static_cast<std::uint8_t>(1u << source);
    ++incoming[target];
    candidate.couplings[i] = coupling;
  }
  for (std::uint32_t i = 1u; i < coupling_count; ++i) {
    const auto value = candidate.couplings[i];
    std::uint32_t j = i;
    while (j != 0u &&
           resident_relational_coupling_less(value, candidate.couplings[j - 1u])) {
      candidate.couplings[j] = candidate.couplings[j - 1u];
      --j;
    }
    candidate.couplings[j] = value;
  }
  std::uint8_t reached = 1u;
  for (std::uint32_t pass = 0u; pass < occurrence_count; ++pass)
    for (std::uint32_t member = 0u; member < occurrence_count; ++member)
      if ((reached & static_cast<std::uint8_t>(1u << member)) != 0u)
        reached |= adjacency[member];
  if (reached != static_cast<std::uint8_t>((1u << occurrence_count) - 1u))
    return false;
  for (std::uint32_t i = 0u; i < occurrence_count; ++i) {
    if (incoming[i] > 1u) ++candidate.reconvergence_count;
    for (std::uint32_t port = 0u; port < occurrences[i].binding_count; ++port) {
      const std::uint32_t variable =
          occurrences[i].bindings[port].variable_identity;
      std::uint32_t incidence = 0u;
      for (std::uint32_t other = 0u; other < occurrence_count; ++other)
        for (std::uint32_t other_port = 0u;
             other_port < occurrences[other].binding_count; ++other_port)
          incidence += occurrences[other].bindings[other_port].variable_identity ==
              variable;
      if (incidence > 1u && !coupled_port[i][port]) return false;
      if (incidence != 1u) continue;
      if (candidate.boundary_count >= kResidentRelationalNetworkMaxBoundary)
        return false;
      candidate.boundary[candidate.boundary_count++] =
          ResidentRelationalNetworkBoundary{
              occurrences[i].occurrence_identity, variable,
              static_cast<std::uint16_t>(port),
              derivations[i].ports[port].domain,
              derivations[i].ports[port].direction,
              derivations[i].ports[port].arity, 0u, 0u};
    }
  }
  for (std::uint32_t i = 1u; i < candidate.boundary_count; ++i) {
    const auto value = candidate.boundary[i];
    std::uint32_t j = i;
    while (j != 0u &&
           (candidate.boundary[j - 1u].variable_identity > value.variable_identity ||
            (candidate.boundary[j - 1u].variable_identity == value.variable_identity &&
             candidate.boundary[j - 1u].occurrence_identity >
                 value.occurrence_identity))) {
      candidate.boundary[j] = candidate.boundary[j - 1u];
      --j;
    }
    candidate.boundary[j] = value;
  }
  std::uint64_t identity = exact_history_fold_word(
      0x6d6978656472616eull, candidate.occurrence_count);
  identity = exact_history_fold_word(identity, candidate.coupling_count);
  identity = exact_history_fold_word(identity, candidate.boundary_count);
  identity = exact_history_fold_word(identity, candidate.overlap_count);
  identity = exact_history_fold_word(identity, candidate.reconvergence_count);
  identity = exact_history_fold_word(identity, candidate.actual_count);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(candidate.eligibility_signed_q16));
  identity = exact_history_fold_word(identity, candidate.eligibility_l1_q16);
  for (std::uint32_t i = 0u; i < candidate.occurrence_count; ++i) {
    const auto& member = candidate.members[i];
    identity = exact_history_fold_word(identity, member.occurrence_identity);
    identity = exact_history_fold_word(identity, member.logical_recipe_id);
    identity = exact_history_fold_word(identity, member.revision_identity);
    identity = exact_history_fold_word(identity, member.participation_identity);
    identity = exact_history_fold_word(identity, member.derivation_rank);
    identity = exact_history_fold_word(identity, member.morphology_identity);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(member.eligibility_q16));
    identity = exact_history_fold_word(identity, member.context_signature);
    identity = exact_history_fold_word(identity, member.binding_count);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(member.lineage_kind));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(member.authority));
  }
  for (std::uint32_t i = 0u; i < candidate.coupling_count; ++i) {
    const auto& coupling = candidate.couplings[i];
    identity = exact_history_fold_word(identity,
                                       coupling.source_occurrence_identity);
    identity = exact_history_fold_word(identity,
                                       coupling.target_occurrence_identity);
    identity = exact_history_fold_word(identity,
                                       coupling.source_revision_identity);
    identity = exact_history_fold_word(identity,
                                       coupling.target_revision_identity);
    identity = exact_history_fold_word(identity,
                                       coupling.source_derivation_rank);
    identity = exact_history_fold_word(identity,
                                       coupling.target_derivation_rank);
    identity = exact_history_fold_word(identity, coupling.variable_identity);
    identity = exact_history_fold_word(identity, coupling.source_port_index);
    identity = exact_history_fold_word(identity, coupling.target_port_index);
  }
  for (std::uint32_t i = 0u; i < candidate.boundary_count; ++i) {
    identity = exact_history_fold_word(identity,
        candidate.boundary[i].occurrence_identity);
    identity = exact_history_fold_word(identity,
        candidate.boundary[i].variable_identity);
    identity = exact_history_fold_word(identity,
        candidate.boundary[i].formal_port_index);
    identity = exact_history_fold_word(identity,
        static_cast<std::uint32_t>(candidate.boundary[i].domain));
    identity = exact_history_fold_word(identity,
        static_cast<std::uint32_t>(candidate.boundary[i].direction));
    identity = exact_history_fold_word(identity,
        candidate.boundary[i].arity);
  }
  candidate.identity = identity == 0u ? 1u : identity;
  *out = candidate;
  return true;
}

// Derive every connected formal-port incidence component as its own Network.
// Isolated Occurrences are preserved without manufacturing singleton Networks.
// A coupled pair is already a finite causal Network. Publication is atomic:
// any Network-sized component that fails bind refuses the whole frontier.
// Output order is identity-canonical, never frontier slot order.
DIRECT_ADULT_HD inline bool bind_resident_coactive_relational_network_closures(
    const direct_network::ResidentRecipeCell* recipes,
    const direct_network::ResidentRecipeDerivation* derivations,
    const ResidentRecipeOccurrence* occurrences,
    std::uint32_t occurrence_count,
    const ResidentOccurrenceCoupling* couplings,
    std::uint32_t coupling_count,
    ResidentRelationalNetworkClosure* outs,
    std::uint32_t max_outs,
    std::uint32_t* out_count) {
  if (out_count == nullptr) return false;
  *out_count = 0u;
  if (occurrence_count == 0u) return true;
  if (recipes == nullptr || derivations == nullptr || occurrences == nullptr ||
      (coupling_count != 0u && couplings == nullptr) || outs == nullptr ||
      occurrence_count > kResidentRelationalNetworkMaxOccurrences ||
      coupling_count > kResidentRelationalNetworkMaxCouplings || max_outs == 0u)
    return false;

  std::uint8_t adjacency[kResidentRelationalNetworkMaxOccurrences]{};
  for (std::uint32_t i = 0u; i < coupling_count; ++i) {
    const auto& coupling = couplings[i];
    std::uint32_t source = kResidentRelationalNetworkMaxOccurrences;
    std::uint32_t target = kResidentRelationalNetworkMaxOccurrences;
    for (std::uint32_t j = 0u; j < occurrence_count; ++j) {
      if (occurrences[j].occurrence_identity ==
          coupling.source_occurrence_identity)
        source = j;
      if (occurrences[j].occurrence_identity ==
          coupling.target_occurrence_identity)
        target = j;
    }
    if (source >= occurrence_count || target >= occurrence_count ||
        source == target)
      return false;
    adjacency[source] |= static_cast<std::uint8_t>(1u << target);
    adjacency[target] |= static_cast<std::uint8_t>(1u << source);
  }

  // Stage component bitmasks first so we never park multiple full Network
  // closures on the CUDA stack (each closure embeds MaxCouplings).
  std::uint8_t seen = 0u;
  std::uint8_t component_masks[kResidentRelationalNetworkMaxCoactive]{};
  std::uint32_t published_count = 0u;
  for (std::uint32_t seed = 0u; seed < occurrence_count; ++seed) {
    if ((seen & static_cast<std::uint8_t>(1u << seed)) != 0u) continue;
    std::uint8_t component = static_cast<std::uint8_t>(1u << seed);
    for (std::uint32_t pass = 0u; pass < occurrence_count; ++pass)
      for (std::uint32_t member = 0u; member < occurrence_count; ++member)
        if ((component & static_cast<std::uint8_t>(1u << member)) != 0u)
          component = static_cast<std::uint8_t>(component | adjacency[member]);
    seen = static_cast<std::uint8_t>(seen | component);

    std::uint32_t component_size = 0u;
    for (std::uint32_t i = 0u; i < occurrence_count; ++i)
      if ((component & static_cast<std::uint8_t>(1u << i)) != 0u)
        ++component_size;

    // Below Network floor: keep singleton Occurrences live.
    if (component_size < kResidentRelationalNetworkMinOccurrences) continue;
    if (published_count >= kResidentRelationalNetworkMaxCoactive ||
        published_count >= max_outs)
      return false;
    component_masks[published_count++] = component;
  }

  direct_network::ResidentRecipeCell sub_recipes[
      kResidentRelationalNetworkMaxOccurrences]{};
  direct_network::ResidentRecipeDerivation sub_derivations[
      kResidentRelationalNetworkMaxOccurrences]{};
  ResidentRecipeOccurrence sub_occurrences[
      kResidentRelationalNetworkMaxOccurrences]{};
  ResidentOccurrenceCoupling sub_couplings[
      kResidentRelationalNetworkMaxCouplings]{};

  for (std::uint32_t n = 0u; n < published_count; ++n) {
    const std::uint8_t component = component_masks[n];
    std::uint32_t component_size = 0u;
    std::uint32_t index_map[kResidentRelationalNetworkMaxOccurrences]{};
    for (std::uint32_t i = 0u; i < occurrence_count; ++i)
      if ((component & static_cast<std::uint8_t>(1u << i)) != 0u)
        index_map[component_size++] = i;

    for (std::uint32_t i = 0u; i < component_size; ++i) {
      const std::uint32_t src = index_map[i];
      sub_recipes[i] = recipes[src];
      sub_derivations[i] = derivations[src];
      sub_occurrences[i] = occurrences[src];
    }
    std::uint32_t sub_coupling_count = 0u;
    for (std::uint32_t i = 0u; i < coupling_count; ++i) {
      const auto& coupling = couplings[i];
      std::uint32_t source = kResidentRelationalNetworkMaxOccurrences;
      std::uint32_t target = kResidentRelationalNetworkMaxOccurrences;
      for (std::uint32_t j = 0u; j < component_size; ++j) {
        if (sub_occurrences[j].occurrence_identity ==
            coupling.source_occurrence_identity)
          source = j;
        if (sub_occurrences[j].occurrence_identity ==
            coupling.target_occurrence_identity)
          target = j;
      }
      if (source >= component_size || target >= component_size) continue;
      if (sub_coupling_count >= kResidentRelationalNetworkMaxCouplings) {
        *out_count = 0u;
        return false;
      }
      sub_couplings[sub_coupling_count++] = coupling;
    }

    if (!bind_resident_relational_network_closure(
            sub_recipes, sub_derivations, sub_occurrences, component_size,
            sub_couplings, sub_coupling_count, &outs[n])) {
      *out_count = 0u;
      return false;
    }
  }

  // Identity-canonical order (independent of frontier discovery order).
  for (std::uint32_t i = 1u; i < published_count; ++i) {
    const auto value = outs[i];
    std::uint32_t j = i;
    while (j != 0u && outs[j - 1u].identity > value.identity) {
      outs[j] = outs[j - 1u];
      --j;
    }
    outs[j] = value;
  }
  *out_count = published_count;
  return true;
}

// Fold only reusable Recipe morphology. Exact Occurrence, participation and
// current eligibility identities deliberately remain transient.
DIRECT_ADULT_HD inline std::uint64_t
resident_relational_network_recruitment_identity(
    const ResidentRelationalNetworkClosure& closure) {
  using direct_network::exact_history_fold_word;
  if (closure.identity == 0u || closure.occurrence_count < 2u ||
      closure.occurrence_count > kResidentRelationalNetworkMaxOccurrences ||
      closure.coupling_count > kResidentRelationalNetworkMaxCouplings ||
      closure.boundary_count > kResidentRelationalNetworkMaxBoundary)
    return 0u;
  std::uint64_t members[kResidentRelationalNetworkMaxOccurrences]{};
  for (std::uint16_t i = 0u; i < closure.occurrence_count; ++i) {
    const auto& member = closure.members[i];
    // Recruitment is reusable morphology, not one transient revision head.
    // Exact Network and Action identities retain RecipeRevision identity;
    // the compact recruitment row survives an ordinary parameter revision.
    std::uint64_t h = exact_history_fold_word(
        0x6e7265636d656d62ull, member.logical_recipe_id);
    h = exact_history_fold_word(h, member.derivation_rank);
    h = exact_history_fold_word(h, member.morphology_identity);
    h = exact_history_fold_word(h, member.binding_count);
    members[i] = h;
  }
  for (std::uint16_t i = 1u; i < closure.occurrence_count; ++i) {
    const auto value = members[i];
    std::uint16_t j = i;
    while (j != 0u && members[j - 1u] > value) {
      members[j] = members[j - 1u];
      --j;
    }
    members[j] = value;
  }
  std::uint64_t edges[kResidentRelationalNetworkMaxCouplings]{};
  for (std::uint16_t e = 0u; e < closure.coupling_count; ++e) {
    const auto& coupling = closure.couplings[e];
    const ResidentRelationalNetworkMember* source = nullptr;
    const ResidentRelationalNetworkMember* target = nullptr;
    for (std::uint16_t m = 0u; m < closure.occurrence_count; ++m) {
      if (closure.members[m].occurrence_identity ==
          coupling.source_occurrence_identity) source = &closure.members[m];
      if (closure.members[m].occurrence_identity ==
          coupling.target_occurrence_identity) target = &closure.members[m];
    }
    if (source == nullptr || target == nullptr) return 0u;
    std::uint64_t h = exact_history_fold_word(
        0x6e72656365646765ull, source->logical_recipe_id);
    h = exact_history_fold_word(h, source->derivation_rank);
    h = exact_history_fold_word(h, coupling.source_port_index);
    h = exact_history_fold_word(h, target->logical_recipe_id);
    h = exact_history_fold_word(h, target->derivation_rank);
    h = exact_history_fold_word(h, coupling.target_port_index);
    edges[e] = h;
  }
  for (std::uint16_t i = 1u; i < closure.coupling_count; ++i) {
    const auto value = edges[i];
    std::uint16_t j = i;
    while (j != 0u && edges[j - 1u] > value) {
      edges[j] = edges[j - 1u];
      --j;
    }
    edges[j] = value;
  }
  std::uint64_t boundaries[kResidentRelationalNetworkMaxBoundary]{};
  for (std::uint16_t b = 0u; b < closure.boundary_count; ++b) {
    const auto& boundary = closure.boundary[b];
    const ResidentRelationalNetworkMember* member = nullptr;
    for (std::uint16_t m = 0u; m < closure.occurrence_count; ++m)
      if (closure.members[m].occurrence_identity == boundary.occurrence_identity)
        member = &closure.members[m];
    if (member == nullptr) return 0u;
    std::uint64_t h = exact_history_fold_word(
        0x6e726563626e6479ull, member->logical_recipe_id);
    h = exact_history_fold_word(h, member->derivation_rank);
    h = exact_history_fold_word(h, boundary.formal_port_index);
    h = exact_history_fold_word(
        h, static_cast<std::uint32_t>(boundary.domain));
    h = exact_history_fold_word(
        h, static_cast<std::uint32_t>(boundary.direction));
    h = exact_history_fold_word(h, boundary.arity);
    boundaries[b] = h;
  }
  for (std::uint16_t i = 1u; i < closure.boundary_count; ++i) {
    const auto value = boundaries[i];
    std::uint16_t j = i;
    while (j != 0u && boundaries[j - 1u] > value) {
      boundaries[j] = boundaries[j - 1u];
      --j;
    }
    boundaries[j] = value;
  }
  std::uint64_t identity = exact_history_fold_word(
      0x6e72656372756974ull, closure.occurrence_count);
  identity = exact_history_fold_word(identity, closure.coupling_count);
  identity = exact_history_fold_word(identity, closure.boundary_count);
  identity = exact_history_fold_word(identity, closure.overlap_count);
  identity = exact_history_fold_word(identity, closure.reconvergence_count);
  for (std::uint16_t i = 0u; i < closure.occurrence_count; ++i)
    identity = exact_history_fold_word(identity, members[i]);
  for (std::uint16_t i = 0u; i < closure.coupling_count; ++i)
    identity = exact_history_fold_word(identity, edges[i]);
  for (std::uint16_t i = 0u; i < closure.boundary_count; ++i)
    identity = exact_history_fold_word(identity, boundaries[i]);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline bool resident_relational_network_eligibility_current(
    const ResidentRelationalNetworkClosure& closure) {
  if (closure.occurrence_count < 2u ||
      closure.occurrence_count > kResidentRelationalNetworkMaxOccurrences)
    return false;
  std::int64_t signed_q16 = 0;
  std::uint64_t l1_q16 = 0u;
  for (std::uint16_t i = 0u; i < closure.occurrence_count; ++i) {
    const std::int64_t value = closure.members[i].eligibility_q16;
    signed_q16 += value;
    l1_q16 += static_cast<std::uint64_t>(value < 0 ? -value : value);
  }
  return signed_q16 == closure.eligibility_signed_q16 &&
      l1_q16 == closure.eligibility_l1_q16;
}

DIRECT_ADULT_HD inline bool resident_recruited_network_incidence_matches(
    const direct_network::ResidentRecruitedNetworkIncidence& incidence,
    const ResidentRelationalNetworkClosure& closure) {
  return incidence.recruitment_identity ==
             resident_relational_network_recruitment_identity(closure) &&
      incidence.member_count == closure.occurrence_count &&
      incidence.coupling_count == closure.coupling_count &&
      incidence.boundary_count == closure.boundary_count &&
      incidence.actual_count == closure.actual_count;
}

// Single-closure sparse nomination avoids constructing a four-closure set in
// hot motor code. Capacity is checked before the only possible row mutation.
DIRECT_ADULT_HD inline bool recruit_resident_relational_network(
    direct_network::ResidentDevelopmentState* development,
    const ResidentRelationalNetworkClosure& closure, std::uint32_t tick) {
  using namespace direct_network;
  if (development == nullptr ||
      development->recruited_networks.incidence_count >
          kResidentRecruitedNetworkCapacity)
    return false;
  const std::uint64_t rid =
      resident_relational_network_recruitment_identity(closure);
  if (rid == 0u || closure.occurrence_count < 2u ||
      closure.occurrence_count > kResidentRecruitedNetworkMaxMembers)
    return false;
  auto& state = development->recruited_networks;
  std::uint32_t slot = state.incidence_count, matches = 0u;
  for (std::uint32_t i = 0u; i < state.incidence_count; ++i)
    if (state.incidences[i].recruitment_identity == rid) {
      slot = i;
      ++matches;
    }
  if (matches > 1u ||
      (matches == 1u && !resident_recruited_network_incidence_matches(
                           state.incidences[slot], closure)))
    return false;
  if (matches == 0u && state.incidence_count ==
                           kResidentRecruitedNetworkCapacity) {
    ++state.capacity_refusals;
    return false;
  }
  if (matches == 0u) ++state.incidence_count;
  auto& incidence = state.incidences[slot];
  const bool fresh_tick = incidence.recruitment_identity == 0u ||
      incidence.last_active_tick != tick;
  const std::uint64_t activation_count =
      incidence.recruitment_identity == 0u ? 1u :
      incidence.activation_count + (fresh_tick ? 1u : 0u);
  const std::int64_t retained_credit =
      incidence.recruitment_identity == rid ? incidence.credit_q16 : 0;
  const std::uint32_t retained_credit_tick =
      incidence.recruitment_identity == rid ? incidence.last_credit_tick : 0u;
  incidence = ResidentRecruitedNetworkIncidence{};
  incidence.recruitment_identity = rid;
  incidence.activation_count = activation_count;
  incidence.credit_q16 = retained_credit;
  incidence.last_active_tick = tick;
  incidence.last_credit_tick = retained_credit_tick;
  incidence.member_count = closure.occurrence_count;
  incidence.coupling_count = closure.coupling_count;
  incidence.boundary_count = closure.boundary_count;
  incidence.actual_count = closure.actual_count;
  ++state.nominations;
  if (fresh_tick) ++state.activations;
  return true;
}

// Sparse and atomic: refuse capacity before mutating any row. Re-observation
// of the same morphology in one tick does not manufacture another activation.
DIRECT_ADULT_HD inline bool recruit_resident_relational_network_set(
    direct_network::ResidentDevelopmentState* development,
    const ResidentRelationalNetworkSet& set, std::uint32_t tick) {
  using namespace direct_network;
  if (development == nullptr ||
      development->recruited_networks.incidence_count >
          kResidentRecruitedNetworkCapacity ||
      set.network_count > kResidentRelationalNetworkMaxCoactive)
    return false;
  auto& state = development->recruited_networks;
  std::uint32_t new_count = 0u;
  for (std::uint16_t n = 0u; n < set.network_count; ++n) {
    const auto& closure = set.networks[n];
    const std::uint64_t rid =
        resident_relational_network_recruitment_identity(closure);
    if (rid == 0u || closure.occurrence_count < 2u ||
        closure.occurrence_count > kResidentRecruitedNetworkMaxMembers)
      return false;
    bool found = false;
    for (std::uint32_t i = 0u; i < state.incidence_count; ++i)
      if (state.incidences[i].recruitment_identity == rid) {
        if (!resident_recruited_network_incidence_matches(
                state.incidences[i], closure)) return false;
        found = true;
      }
    bool prior_nomination = false;
    for (std::uint16_t prior = 0u; prior < n; ++prior)
      if (resident_relational_network_recruitment_identity(
              set.networks[prior]) == rid)
        prior_nomination = true;
    new_count += found || prior_nomination ? 0u : 1u;
  }
  if (new_count > kResidentRecruitedNetworkCapacity - state.incidence_count) {
    ++state.capacity_refusals;
    return false;
  }
  state.nominations += set.network_count;
  for (std::uint16_t n = 0u; n < set.network_count; ++n) {
    const auto& closure = set.networks[n];
    const std::uint64_t rid =
        resident_relational_network_recruitment_identity(closure);
    std::uint32_t slot = state.incidence_count;
    for (std::uint32_t i = 0u; i < state.incidence_count; ++i)
      if (state.incidences[i].recruitment_identity == rid) {
        slot = i;
        break;
      }
    if (slot == state.incidence_count) ++state.incidence_count;
    auto& incidence = state.incidences[slot];
    const bool fresh_tick = incidence.recruitment_identity == 0u ||
        incidence.last_active_tick != tick;
    const std::uint64_t activation_count =
        incidence.recruitment_identity == 0u ? 1u :
        incidence.activation_count + (fresh_tick ? 1u : 0u);
    const std::int64_t retained_credit =
        incidence.recruitment_identity == rid ? incidence.credit_q16 : 0;
    const std::uint32_t retained_credit_tick =
        incidence.recruitment_identity == rid ? incidence.last_credit_tick : 0u;
    incidence = ResidentRecruitedNetworkIncidence{};
    incidence.recruitment_identity = rid;
    incidence.activation_count = activation_count;
    incidence.credit_q16 = retained_credit;
    incidence.last_active_tick = tick;
    incidence.last_credit_tick = retained_credit_tick;
    incidence.member_count = closure.occurrence_count;
    incidence.coupling_count = closure.coupling_count;
    incidence.boundary_count = closure.boundary_count;
    incidence.actual_count = closure.actual_count;
    if (fresh_tick) ++state.activations;
  }
  return true;
}

struct ResidentRecruitedNetworkCreditPlan {
  std::uint64_t recruitment_identity, active_network_identity;
  std::int64_t prior_credit_q16, next_credit_q16, applied_delta_q16;
  std::uint32_t incidence_slot, valid;
};

DIRECT_ADULT_HD inline std::int64_t
resident_recruited_network_saturating_add(std::int64_t left,
                                          std::int64_t right) {
  constexpr std::int64_t kMax = 0x7fffffffffffffffll;
  constexpr std::int64_t kMin = -kMax - 1;
  if (right > 0 && left > kMax - right) return kMax;
  if (right < 0 && left < kMin - right) return kMin;
  return left + right;
}

DIRECT_ADULT_HD inline bool plan_resident_recruited_network_credit(
    const direct_network::ResidentDevelopmentState* development,
    std::uint64_t recruitment_identity, std::uint64_t active_network_identity,
    std::int64_t requested_delta_q16,
    ResidentRecruitedNetworkCreditPlan* out) {
  if (out == nullptr) return false;
  *out = ResidentRecruitedNetworkCreditPlan{};
  if ((recruitment_identity == 0u) != (active_network_identity == 0u))
    return false;
  if (recruitment_identity == 0u) return requested_delta_q16 == 0;
  if (requested_delta_q16 == 0) return true;
  if (development == nullptr ||
      development->recruited_networks.incidence_count >
          direct_network::kResidentRecruitedNetworkCapacity)
    return false;
  const auto& state = development->recruited_networks;
  std::uint32_t slot = state.incidence_count, matches = 0u;
  for (std::uint32_t i = 0u; i < state.incidence_count; ++i)
    if (state.incidences[i].recruitment_identity == recruitment_identity) {
      slot = i;
      ++matches;
    }
  if (matches != 1u || state.incidences[slot].activation_count == 0u)
    return false;
  const std::int64_t prior = state.incidences[slot].credit_q16;
  const std::int64_t next = resident_recruited_network_saturating_add(
      prior, requested_delta_q16);
  out->recruitment_identity = recruitment_identity;
  out->active_network_identity = active_network_identity;
  out->prior_credit_q16 = prior;
  out->next_credit_q16 = next;
  out->applied_delta_q16 = next - prior;
  out->incidence_slot = slot;
  out->valid = out->applied_delta_q16 != 0 ? 1u : 0u;
  return true;
}

DIRECT_ADULT_HD inline bool resident_recruited_network_credit_plan_current(
    const direct_network::ResidentDevelopmentState* development,
    const ResidentRecruitedNetworkCreditPlan& plan) {
  if (plan.valid == 0u) return true;
  if (development == nullptr ||
      plan.incidence_slot >= development->recruited_networks.incidence_count)
    return false;
  const auto& incidence =
      development->recruited_networks.incidences[plan.incidence_slot];
  return incidence.recruitment_identity == plan.recruitment_identity &&
      incidence.credit_q16 == plan.prior_credit_q16;
}

// Call only after currentness preflight in the same single-owner settlement
// transaction. This post-history publication is assignment-only and cannot
// introduce a new refusal after other causal state has begun to commit.
DIRECT_ADULT_HD inline void commit_resident_recruited_network_credit(
    direct_network::ResidentDevelopmentState* development,
    const ResidentRecruitedNetworkCreditPlan& plan, std::uint32_t tick) {
  if (plan.valid == 0u) return;
  auto& incidence =
      development->recruited_networks.incidences[plan.incidence_slot];
  incidence.credit_q16 = plan.next_credit_q16;
  incidence.last_credit_tick = tick;
}

DIRECT_ADULT_HD inline bool apply_resident_recruited_network_credit(
    direct_network::ResidentDevelopmentState* development,
    const ResidentRecruitedNetworkCreditPlan& plan, std::uint32_t tick) {
  if (!resident_recruited_network_credit_plan_current(development, plan))
    return false;
  commit_resident_recruited_network_credit(development, plan, tick);
  return true;
}

// Persistent credit may rank only complete current closures. Current signed
// eligibility remains a veto; equal learned alternatives stay unresolved.
DIRECT_ADULT_HD inline bool select_resident_recruited_network(
    const direct_network::ResidentRecruitedNetworkState& state,
    const ResidentRelationalNetworkSet& active,
    std::uint64_t* active_network_identity,
    std::uint64_t* recruitment_identity, std::int64_t* credit_q16) {
  if (active_network_identity == nullptr || recruitment_identity == nullptr ||
      credit_q16 == nullptr ||
      state.incidence_count > direct_network::kResidentRecruitedNetworkCapacity ||
      active.network_count > kResidentRelationalNetworkMaxCoactive)
    return false;
  *active_network_identity = 0u;
  *recruitment_identity = 0u;
  *credit_q16 = 0;
  bool have = false, tied = false;
  std::int64_t best = 0;
  for (std::uint16_t n = 0u; n < active.network_count; ++n) {
    const auto& closure = active.networks[n];
    if (closure.identity == 0u ||
        !resident_relational_network_eligibility_current(closure))
      return false;
    if (closure.eligibility_l1_q16 == 0u ||
        closure.eligibility_signed_q16 <= 0)
      continue;
    const std::uint64_t rid =
        resident_relational_network_recruitment_identity(closure);
    const direct_network::ResidentRecruitedNetworkIncidence* incidence = nullptr;
    std::uint32_t matches = 0u;
    for (std::uint32_t i = 0u; i < state.incidence_count; ++i)
      if (state.incidences[i].recruitment_identity == rid) {
        incidence = &state.incidences[i];
        ++matches;
      }
    if (rid == 0u || matches > 1u) return false;
    if (matches == 0u || incidence->credit_q16 <= 0) continue;
    if (!have || incidence->credit_q16 > best) {
      have = true;
      tied = false;
      best = incidence->credit_q16;
      *active_network_identity = closure.identity;
      *recruitment_identity = rid;
      *credit_q16 = best;
    } else if (incidence->credit_q16 == best) {
      tied = true;
    }
  }
  if (!have || tied) {
    *active_network_identity = 0u;
    *recruitment_identity = 0u;
    *credit_q16 = 0;
  }
  return true;
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_RELATIONAL_NETWORK_CUH

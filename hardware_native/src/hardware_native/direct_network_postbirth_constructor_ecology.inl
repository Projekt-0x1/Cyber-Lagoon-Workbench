#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_POSTBIRTH_CONSTRUCTOR_TYPES
#define HARDWARE_NATIVE_DIRECT_NETWORK_POSTBIRTH_CONSTRUCTOR_TYPES
inline constexpr std::uint32_t kResidentPostbirthRecipeReserve = 32u;
inline constexpr std::uint32_t kResidentDerivationWidth = 4u;
enum class ResidentRecipePortDomain : std::uint8_t {
  q16_scalar = 1u,
  discrete_event = 2u,
};
enum class ResidentRecipePortDirection : std::uint8_t {
  input = 1u,
  output = 2u,
  bidirectional = 3u,
};
struct ResidentRecipePort {
  std::uint32_t node;
  ResidentRecipePortDomain domain;
  ResidentRecipePortDirection direction;
  std::uint16_t arity;
};
static_assert(std::is_trivial_v<ResidentRecipePort> &&
              std::is_standard_layout_v<ResidentRecipePort> &&
              std::has_unique_object_representations_v<ResidentRecipePort>);
DIRECT_NETWORK_HD inline bool resident_recipe_ports_compatible(
    const ResidentRecipePort& left, const ResidentRecipePort& right) {
  const bool left_writes =
      left.direction == ResidentRecipePortDirection::output ||
      left.direction == ResidentRecipePortDirection::bidirectional;
  const bool right_reads =
      right.direction == ResidentRecipePortDirection::input ||
      right.direction == ResidentRecipePortDirection::bidirectional;
  return left_writes && right_reads && left.domain == right.domain &&
         left.arity != 0u && left.arity == right.arity;
}
inline constexpr std::uint16_t kResidentCondensationSourceCount = 2u;
inline constexpr std::uint16_t kResidentBoundaryCondensationMaxRoots = 4u;
inline constexpr std::uint16_t kResidentBoundaryCondensationProbeCount = 3u;
inline constexpr std::uint16_t kResidentDerivationCondensedNetwork = 1u << 0;
inline constexpr std::uint16_t kResidentDerivationRepresentationProbation = 1u << 1;
inline constexpr std::uint16_t kResidentDerivationBoundaryCondensedNetwork = 1u << 2;
struct ResidentCondensationSourceSnapshot {
  std::uint64_t logical_recipe_id, revision_identity, generation;
  std::uint64_t route_incarnation, source_identity;
  std::uint32_t recipe_cell, relation, route_index;
  std::int32_t parameter_q16;
  ResidentRecipePort input_port, output_port;
};
static_assert(std::is_trivial_v<ResidentCondensationSourceSnapshot> &&
              std::is_standard_layout_v<ResidentCondensationSourceSnapshot> &&
              std::has_unique_object_representations_v<ResidentCondensationSourceSnapshot>);
struct ResidentNetworkCondensationEvidence {
  std::uint64_t witness_identity;
  ResidentCondensationSourceSnapshot sources[kResidentCondensationSourceCount];
  std::uint64_t observation_identities[kResidentCondensationSourceCount];
  std::uint64_t boundary_identity, proof_identity;
  std::uint64_t refinement_logical_recipe_id, refinement_revision_identity;
  std::uint32_t observation_source_incarnations[
      kResidentCondensationSourceCount][kResidentCondensationSourceCount];
  std::uint32_t observation_variable_identities[
      kResidentCondensationSourceCount];
  std::uint32_t refinement_recipe_cell;
  std::uint32_t recurring_observations, guard_min_q16, guard_max_q16;
  std::int32_t maximum_error_q16;
  std::uint16_t source_count, reserved;
  std::uint32_t reserved2, reserved3;
};
static_assert(std::is_trivial_v<ResidentNetworkCondensationEvidence> &&
              std::is_standard_layout_v<ResidentNetworkCondensationEvidence> &&
              std::has_unique_object_representations_v<ResidentNetworkCondensationEvidence>);
struct ResidentRecipeDerivation {
  std::uint64_t logical_recipe_id, revision_identity, parent_logical_recipe_id;
  std::uint64_t parent_revision_identity, witness_identity, generation, route_incarnations[2];
  std::uint32_t recipe_cell, parent_recipe_cell, source_node, route_index;
  std::uint16_t port_count, relation_count, parameter_count, territory_index;
  ResidentRecipePort ports[kResidentDerivationWidth];
  std::uint32_t relations[kResidentDerivationWidth];
  std::int32_t parameters_q16[kResidentDerivationWidth];
  ResidentCondensationSourceSnapshot condensation_sources[
      kResidentCondensationSourceCount];
  std::uint64_t condensation_observation_identities[
      kResidentCondensationSourceCount];
  std::uint64_t condensation_boundary_identity, condensation_proof_identity;
  std::uint64_t condensation_refinement_logical_recipe_id;
  std::uint64_t condensation_refinement_revision_identity;
  std::uint32_t condensation_observation_source_incarnations[
      kResidentCondensationSourceCount][kResidentCondensationSourceCount];
  std::uint32_t condensation_observation_variable_identities[
      kResidentCondensationSourceCount];
  std::uint32_t condensation_refinement_recipe_cell;
  std::uint32_t condensation_refinement_reserved;
  std::uint32_t condensation_observations, condensation_guard_min_q16;
  std::uint32_t condensation_guard_max_q16;
  std::int32_t condensation_maximum_error_q16;
  std::uint16_t condensation_source_count, condensation_flags;
  std::uint32_t condensation_reserved;
  std::uint64_t boundary_condensation_root_logical_recipe_ids[
      kResidentBoundaryCondensationMaxRoots];
  std::uint64_t boundary_condensation_root_revision_identities[
      kResidentBoundaryCondensationMaxRoots];
  ResidentCondensationSourceSnapshot boundary_condensation_sources[
      kResidentBoundaryCondensationMaxRoots];
  std::uint64_t boundary_condensation_probe_participation_identities[
      kResidentBoundaryCondensationProbeCount];
  std::int32_t boundary_condensation_probe_inputs_q16[
      kResidentBoundaryCondensationProbeCount];
  std::int32_t boundary_condensation_probe_outputs_q16[
      kResidentBoundaryCondensationProbeCount];
  std::uint64_t boundary_condensation_proof_identity;
  std::uint64_t boundary_condensation_refinement_logical_recipe_id;
  std::uint64_t boundary_condensation_refinement_revision_identity;
  std::uint32_t boundary_condensation_refinement_recipe_cell;
  std::uint32_t boundary_condensation_guard_min_q16;
  std::uint32_t boundary_condensation_guard_max_q16;
  std::uint16_t boundary_condensation_root_count;
  std::uint16_t boundary_condensation_probe_count;
  std::uint16_t boundary_condensation_eliminated_variable_count;
  std::uint16_t boundary_condensation_version;
  std::uint32_t boundary_condensation_reserved;
};
struct ResidentCondensationDeoptimizationState {
  std::uint64_t witness_identity, rematerialization_identity;
  ResidentCondensationSourceSnapshot sources[kResidentCondensationSourceCount];
  std::uint32_t source_count, active;
};
static_assert(std::is_trivial_v<ResidentCondensationDeoptimizationState> &&
              std::is_standard_layout_v<ResidentCondensationDeoptimizationState> &&
              std::has_unique_object_representations_v<
                  ResidentCondensationDeoptimizationState>);
inline constexpr std::uint32_t kResidentRecipeIncidenceCapacity = 64u;
struct ResidentRawContactKey {
  std::uint32_t node, channel, word, context;
};
inline constexpr std::uint32_t kResidentSensoryWitnessVersion = 1u;
struct ResidentSensoryAuthorityReceipt {
  std::uint64_t history_sequence;
  std::uint64_t history_content_root;
  std::uint64_t receipt_content_root;
  std::uint64_t contact_identity;
  std::uint64_t boundary_session_epoch;
  std::uint64_t ingress_sequence;
  std::uint32_t resident_tick;
  std::uint32_t event_tick;
  std::uint32_t history_flags;
  std::uint32_t authority_incarnation;
  std::uint32_t witness_version;
  std::uint32_t claim_incarnation;
};
struct ResidentRawContactBinding {
  std::uint64_t ticket_id;
  ResidentRawContactKey key;
  ResidentSensoryAuthorityReceipt authority;
};
struct ResidentRecipeIncidence {
  std::uint64_t revision_identity;
  std::uint64_t route_incarnation;
  ResidentRawContactKey contact;
  std::uint32_t node;
  std::uint32_t derivation_index;
  // Compact resident morphology charge. Zero is a directly lived surface;
  // emitted compound surfaces carry one plus their deepest constituent.
  std::uint32_t composition_depth;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<ResidentRawContactBinding> &&
              std::is_standard_layout_v<ResidentRawContactBinding> &&
              std::has_unique_object_representations_v<
                  ResidentRawContactBinding>);
static_assert(std::is_trivial_v<ResidentSensoryAuthorityReceipt> &&
              std::is_standard_layout_v<ResidentSensoryAuthorityReceipt> &&
              std::has_unique_object_representations_v<
                  ResidentSensoryAuthorityReceipt>);
static_assert(std::is_trivial_v<ResidentRecipeIncidence> &&
              std::is_standard_layout_v<ResidentRecipeIncidence> &&
              std::has_unique_object_representations_v<
                  ResidentRecipeIncidence>);
struct ResidentPostbirthConstructorState {
  std::uint32_t derivation_count, derivation_capacity, recipe_cell_capacity;
  std::uint32_t port_capacity, relation_capacity, parameter_capacity;
  std::uint32_t ports_used, relations_used, parameters_used, history_scan_cursor, trace_scan_cursor;
  std::uint32_t condensed, no_history, capacity_refusals, shape_refusals;
  std::uint64_t highest_derivation_rank;
  ResidentCondensationDeoptimizationState condensation_deoptimization;
  ResidentRecipeIncidence recipe_incidence[
      kResidentRecipeIncidenceCapacity];
  std::uint32_t recipe_incidence_count;
  std::uint32_t recipe_incidence_overflow;
  ResidentRawContactBinding raw_contact_bindings[
      kResidentRecipeIncidenceCapacity];
  std::uint32_t raw_contact_binding_count;
  std::uint32_t raw_contact_binding_overflow;
  std::uint32_t macro_param_refusals;
  // Uses the structure's former tail padding: together with trace_scan_cursor
  // this stores a versioned 64-bit bounded recurrent-trace frontier without
  // changing the checkpointed arena ABI.
  std::uint32_t trace_frontier_state_hi;
};
static_assert(std::is_trivial_v<ResidentRecipeDerivation> &&
              std::is_standard_layout_v<ResidentRecipeDerivation> &&
              std::has_unique_object_representations_v<ResidentRecipeDerivation>);
static_assert(std::is_trivial_v<ResidentPostbirthConstructorState> &&
              std::is_standard_layout_v<ResidentPostbirthConstructorState>);
static_assert(sizeof(ResidentPostbirthConstructorState) == 9480u);
inline constexpr std::uint64_t kPostbirthTraceFrontierMarker = 1ull << 63u;
inline constexpr std::uint64_t kPostbirthTraceHighWaterMask = (1ull << 15u) - 1u;
inline constexpr std::uint64_t kPostbirthTraceIndexMask = (1ull << 14u) - 1u;
DIRECT_NETWORK_HD inline std::uint64_t resident_postbirth_trace_frontier_raw(
    const ResidentPostbirthConstructorState& state) {
  return (static_cast<std::uint64_t>(state.trace_frontier_state_hi) << 32u) |
      state.trace_scan_cursor;
}
DIRECT_NETWORK_HD inline std::uint32_t resident_postbirth_trace_frontier_high_water(
    const ResidentPostbirthConstructorState& state) {
  const std::uint64_t raw = resident_postbirth_trace_frontier_raw(state);
  return (raw & kPostbirthTraceFrontierMarker) != 0u
      ? static_cast<std::uint32_t>(raw & kPostbirthTraceHighWaterMask)
      : state.trace_scan_cursor;
}
DIRECT_NETWORK_HD inline std::uint32_t resident_postbirth_trace_frontier_carry_count(
    const ResidentPostbirthConstructorState& state) {
  const std::uint64_t raw = resident_postbirth_trace_frontier_raw(state);
  return (raw & kPostbirthTraceFrontierMarker) != 0u
      ? static_cast<std::uint32_t>((raw >> 15u) & 0x3u) : 0u;
}
DIRECT_NETWORK_HD inline ResidentRawContactKey resident_raw_contact_key(
    std::uint32_t node, std::uint32_t channel, std::uint32_t word,
    std::uint32_t context) {
  return ResidentRawContactKey{node, channel, word, context};
}
DIRECT_NETWORK_HD inline bool resident_raw_contact_key_equal(
    const ResidentRawContactKey& left, const ResidentRawContactKey& right) {
  return left.node == right.node && left.channel == right.channel &&
      left.word == right.word && left.context == right.context;
}
DIRECT_NETWORK_HD inline bool resident_raw_contact_key_empty(
    const ResidentRawContactKey& key) {
  return key.node == 0u && key.channel == 0u && key.word == 0u &&
      key.context == 0u;
}
DIRECT_NETWORK_HD inline std::uint32_t resident_bootstrap_incidence_count(
    const ResidentPostbirthConstructorState& state, std::uint32_t node) {
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < state.recipe_incidence_count; ++i)
    count += state.recipe_incidence[i].node == node &&
        resident_raw_contact_key_empty(state.recipe_incidence[i].contact);
  return count;
}

// Exact opaque/surface incidences already bound on this node. When any exist,
// novel words must match them — leftover bootstrap wildcards must not absorb
// untaught constituents (#1610 agreement/lesion). Outcome-as-ground motor
// Words are exempted at the frontier via motor_output history, not here.
DIRECT_NETWORK_HD inline std::uint32_t resident_exact_surface_incidence_count(
    const ResidentPostbirthConstructorState& state, std::uint32_t node) {
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < state.recipe_incidence_count; ++i)
    count += state.recipe_incidence[i].node == node &&
        !resident_raw_contact_key_empty(state.recipe_incidence[i].contact);
  return count;
}
DIRECT_NETWORK_HD inline std::uint64_t resident_sensory_history_content_root(
    const DirectExactHistoryRecord& record) {
  return exact_history_fold_record(1469598103934665603ull, record);
}
DIRECT_NETWORK_HD inline std::uint64_t resident_sensory_receipt_content_root(
    std::uint64_t ticket_id, const ResidentRawContactKey& key,
    const ResidentSensoryAuthorityReceipt& authority) {
  std::uint64_t root = authority.history_content_root;
  root = exact_history_fold_word(root, ticket_id);
  root = exact_history_fold_word(root, key.node);
  root = exact_history_fold_word(root, key.channel);
  root = exact_history_fold_word(root, key.word);
  root = exact_history_fold_word(root, key.context);
  root = exact_history_fold_word(root, authority.contact_identity);
  root = exact_history_fold_word(root, authority.boundary_session_epoch);
  root = exact_history_fold_word(root, authority.ingress_sequence);
  root = exact_history_fold_word(root, authority.authority_incarnation);
  root = exact_history_fold_word(root, authority.witness_version);
  return exact_history_fold_word(root, authority.claim_incarnation);
}
DIRECT_NETWORK_HD inline bool resident_sensory_authority_receipt_valid(
    std::uint64_t ticket_id, const ResidentRawContactKey& key,
    const ResidentSensoryAuthorityReceipt& authority,
    std::uint32_t expected_origin) {
  if (ticket_id == 0u || authority.history_sequence == 0u ||
      authority.contact_identity == 0u || authority.boundary_session_epoch == 0u ||
      authority.ingress_sequence == 0u || authority.authority_incarnation == 0u ||
      authority.claim_incarnation == 0u ||
      authority.witness_version != kResidentSensoryWitnessVersion ||
      authority.event_tick > authority.resident_tick ||
      (authority.history_flags & kDirectHistoryVerifiedObservation) == 0u ||
      (authority.history_flags & kDirectHistoryPayloadFlags) != expected_origin)
    return false;
  DirectExactHistoryRecord record{};
  record.sequence = authority.history_sequence;
  record.identity = ticket_id;
  record.resident_tick = authority.resident_tick;
  record.event_tick = authority.event_tick;
  record.kind = DirectExactHistoryKind::sensory_contact;
  record.source = key.node;
  record.subject = key.channel;
  record.value = key.word;
  record.context = key.context;
  record.flags = authority.history_flags;
  return authority.history_content_root ==
             resident_sensory_history_content_root(record) &&
      authority.receipt_content_root ==
             resident_sensory_receipt_content_root(ticket_id, key, authority);
}
DIRECT_NETWORK_HD inline bool record_resident_raw_contact_binding(
    ResidentPostbirthConstructorState* state, std::uint64_t ticket_id,
    const ResidentRawContactKey& key,
    const ResidentSensoryAuthorityReceipt& authority,
    std::uint32_t expected_origin) {
  if (state == nullptr ||
      state->raw_contact_binding_count > kResidentRecipeIncidenceCapacity ||
      !resident_sensory_authority_receipt_valid(
          ticket_id, key, authority, expected_origin))
    return false;
  for (std::uint32_t i = 0u; i < state->raw_contact_binding_count; ++i) {
    const auto& binding = state->raw_contact_bindings[i];
    if (binding.ticket_id == ticket_id)
      return resident_raw_contact_key_equal(binding.key, key) &&
          binding.authority.history_content_root == authority.history_content_root &&
          binding.authority.receipt_content_root == authority.receipt_content_root &&
          binding.authority.history_sequence == authority.history_sequence &&
          binding.authority.contact_identity == authority.contact_identity &&
          binding.authority.boundary_session_epoch == authority.boundary_session_epoch &&
          binding.authority.ingress_sequence == authority.ingress_sequence &&
          binding.authority.resident_tick == authority.resident_tick &&
          binding.authority.event_tick == authority.event_tick &&
          binding.authority.history_flags == authority.history_flags &&
          binding.authority.authority_incarnation == authority.authority_incarnation &&
          binding.authority.witness_version == authority.witness_version &&
          binding.authority.claim_incarnation == authority.claim_incarnation;
  }
  if (state->raw_contact_binding_count >= kResidentRecipeIncidenceCapacity) {
    ++state->raw_contact_binding_overflow;
    return false;
  }
  state->raw_contact_bindings[state->raw_contact_binding_count++] =
      ResidentRawContactBinding{ticket_id, key, authority};
  return true;
}
DIRECT_NETWORK_HD inline bool can_record_resident_raw_contact_binding(
    const ResidentPostbirthConstructorState* state, std::uint64_t ticket_id,
    const ResidentRawContactKey& key,
    const ResidentSensoryAuthorityReceipt& authority,
    std::uint32_t expected_origin) {
  if (state == nullptr ||
      !resident_sensory_authority_receipt_valid(
          ticket_id, key, authority, expected_origin) ||
      state->raw_contact_binding_count > kResidentRecipeIncidenceCapacity)
    return false;
  for (std::uint32_t i = 0u; i < state->raw_contact_binding_count; ++i) {
    const auto& binding = state->raw_contact_bindings[i];
    if (binding.ticket_id != ticket_id) continue;
    return resident_raw_contact_key_equal(binding.key, key) &&
        binding.authority.history_content_root == authority.history_content_root &&
        binding.authority.receipt_content_root == authority.receipt_content_root &&
        binding.authority.history_sequence == authority.history_sequence &&
        binding.authority.contact_identity == authority.contact_identity &&
        binding.authority.boundary_session_epoch == authority.boundary_session_epoch &&
        binding.authority.ingress_sequence == authority.ingress_sequence &&
        binding.authority.resident_tick == authority.resident_tick &&
        binding.authority.event_tick == authority.event_tick &&
        binding.authority.history_flags == authority.history_flags &&
        binding.authority.authority_incarnation == authority.authority_incarnation &&
        binding.authority.witness_version == authority.witness_version &&
        binding.authority.claim_incarnation == authority.claim_incarnation;
  }
  return state->raw_contact_binding_count < kResidentRecipeIncidenceCapacity;
}
DIRECT_NETWORK_HD inline ResidentRawContactBinding resident_raw_contact_authority(
    const ResidentPostbirthConstructorState& state, std::uint64_t ticket_id) {
  for (std::uint32_t i = 0u; i < state.raw_contact_binding_count; ++i)
    if (state.raw_contact_bindings[i].ticket_id == ticket_id)
      return state.raw_contact_bindings[i];
  return ResidentRawContactBinding{};
}
DIRECT_NETWORK_HD inline ResidentRawContactKey resident_raw_contact_binding(
    const ResidentPostbirthConstructorState& state, std::uint64_t ticket_id) {
  for (std::uint32_t i = 0u; i < state.raw_contact_binding_count; ++i)
    if (state.raw_contact_bindings[i].ticket_id == ticket_id)
      return state.raw_contact_bindings[i].key;
  return ResidentRawContactKey{};
}
DIRECT_NETWORK_HD inline void record_resident_recipe_incidence(
    ResidentPostbirthConstructorState* state, std::uint32_t derivation_index,
    const ResidentRecipeDerivation& derivation,
    ResidentRawContactKey contact = {}, std::uint32_t composition_depth = 0u) {
  if (state == nullptr || derivation.port_count == 0u ||
      derivation.ports[0].direction != ResidentRecipePortDirection::input)
    return;
  const std::uint32_t node = derivation.ports[0].node;
  if (!resident_raw_contact_key_empty(contact) &&
      (contact.node != node || derivation.source_node != node))
    return;
  if (!resident_raw_contact_key_empty(contact))
    for (std::uint32_t i = 0u; i < state->recipe_incidence_count; ++i)
      if (resident_raw_contact_key_equal(
              state->recipe_incidence[i].contact, contact)) {
        if (composition_depth > state->recipe_incidence[i].composition_depth)
          state->recipe_incidence[i].composition_depth = composition_depth;
        return;
      }
  if (resident_raw_contact_key_empty(contact))
    for (std::uint32_t i = 0u; i < state->recipe_incidence_count; ++i) {
      auto& incidence = state->recipe_incidence[i];
      if (!resident_raw_contact_key_empty(incidence.contact) ||
          incidence.node != node)
        continue;
      if (incidence.derivation_index == derivation_index)
        incidence = ResidentRecipeIncidence{
            derivation.revision_identity, derivation.route_incarnations[0],
            contact, node, derivation_index, composition_depth, 0u};
      return;
    }
  if (state->recipe_incidence_count >=
      kResidentRecipeIncidenceCapacity) {
    ++state->recipe_incidence_overflow;
    return;
  }
  state->recipe_incidence[state->recipe_incidence_count++] =
      ResidentRecipeIncidence{
          derivation.revision_identity, derivation.route_incarnations[0],
          contact, node, derivation_index, composition_depth, 0u};
}

// Outcome-as-ground: once a motor Word is publicly emitted, register it as an
// exact surface alongside existing opaque incidences so later nesting can
// exact-match instead of falling through leftover wildcards (#1610).
// Uses node/channel/context from an existing exact surface (no boundary-port
// walk) so checkpointed brains without live port pointers still register.
DIRECT_NETWORK_HD inline void resident_register_motor_ground_surfaces(
    ResidentPostbirthConstructorState* state,
    const DirectBoundaryPort* /*ports*/, std::uint32_t /*port_count*/,
    const ResidentRecipeDerivation* derivations, std::uint32_t derivation_count,
    std::uint32_t motor_word, std::uint32_t composition_depth = 0u) {
  if (state == nullptr || derivations == nullptr || motor_word == 0u) return;
  for (std::uint32_t i = 0u; i < state->recipe_incidence_count; ++i) {
    const ResidentRecipeIncidence& existing = state->recipe_incidence[i];
    if (resident_raw_contact_key_empty(existing.contact) ||
        existing.derivation_index >= derivation_count)
      continue;
    const ResidentRecipeDerivation& derivation =
        derivations[existing.derivation_index];
    if (derivation.port_count == 0u ||
        derivation.ports[0].direction != ResidentRecipePortDirection::input ||
        derivation.ports[0].node != existing.node ||
        derivation.source_node != existing.node)
      continue;
    const ResidentRawContactKey contact = resident_raw_contact_key(
        existing.contact.node, existing.contact.channel, motor_word,
        existing.contact.context);
    record_resident_recipe_incidence(
        state, existing.derivation_index, derivation, contact,
        composition_depth);
  }
}

DIRECT_NETWORK_HD inline std::uint64_t resident_recipe_derivation_rank(
    const ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count, std::uint64_t logical_recipe_id) {
  if (derivations == nullptr || logical_recipe_id == 0u) return 0u;
  for (std::uint32_t i = 0u; i < derivation_count; ++i)
    if (derivations[i].logical_recipe_id == logical_recipe_id)
      return derivations[i].generation;
  return 0u;
}
DIRECT_NETWORK_HD inline std::uint64_t resident_recipe_child_derivation_rank(
    const ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count, std::uint64_t parent_logical_recipe_id) {
  if (parent_logical_recipe_id == 0u) return 0u;
  const std::uint64_t parent_rank = resident_recipe_derivation_rank(
      derivations, derivation_count, parent_logical_recipe_id);
  return parent_rank == ~std::uint64_t{0} ? 0u : parent_rank + 1u;
}
DIRECT_NETWORK_HD inline std::uint64_t resident_condensation_boundary_identity(
    const ResidentNetworkCondensationEvidence& evidence) {
  std::uint64_t identity = 0x636f6e64626f756eull;
  for (std::uint32_t i = 0u; i < evidence.source_count &&
       i < kResidentCondensationSourceCount; ++i) {
    const auto& source = evidence.sources[i];
    identity = exact_history_fold_word(identity, source.logical_recipe_id);
    identity = exact_history_fold_word(identity, source.revision_identity);
    identity = exact_history_fold_word(identity, source.relation);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint64_t>(source.parameter_q16));
    identity = exact_history_fold_word(identity, source.input_port.node);
    identity = exact_history_fold_word(identity, source.output_port.node);
    identity = exact_history_fold_word(identity, source.input_port.arity);
    identity = exact_history_fold_word(identity, source.output_port.arity);
  }
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline std::uint64_t resident_condensation_proof_identity(
    const ResidentNetworkCondensationEvidence& evidence) {
  std::uint64_t identity = exact_history_fold_word(
      0x636f6e6470726f6full, evidence.boundary_identity);
  identity = exact_history_fold_word(identity, evidence.recurring_observations);
  identity = exact_history_fold_word(identity, evidence.guard_min_q16);
  identity = exact_history_fold_word(identity, evidence.guard_max_q16);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(evidence.maximum_error_q16));
  for (std::uint32_t i = 0u; i < evidence.source_count &&
       i < kResidentCondensationSourceCount; ++i)
    identity = exact_history_fold_word(
        identity, evidence.observation_identities[i]);
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline std::uint64_t resident_network_condensation_witness(
    const ResidentNetworkCondensationEvidence& evidence) {
  std::uint64_t identity = 0x6e65746f666e6574ull;
  identity = exact_history_fold_word(identity, evidence.source_count);
  identity = exact_history_fold_word(identity, evidence.recurring_observations);
  identity = exact_history_fold_word(identity, evidence.guard_min_q16);
  identity = exact_history_fold_word(identity, evidence.guard_max_q16);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(evidence.maximum_error_q16));
  identity = exact_history_fold_word(identity, evidence.boundary_identity);
  identity = exact_history_fold_word(identity, evidence.proof_identity);
  identity = exact_history_fold_word(
      identity, evidence.refinement_logical_recipe_id);
  identity = exact_history_fold_word(
      identity, evidence.refinement_revision_identity);
  identity = exact_history_fold_word(
      identity, evidence.refinement_recipe_cell);
  for (std::uint32_t i = 0u; i < evidence.source_count &&
       i < kResidentCondensationSourceCount; ++i) {
    const auto& source = evidence.sources[i];
    identity = exact_history_fold_word(identity, source.logical_recipe_id);
    identity = exact_history_fold_word(identity, source.revision_identity);
    identity = exact_history_fold_word(identity, source.generation);
    identity = exact_history_fold_word(identity, source.recipe_cell);
    identity = exact_history_fold_word(identity, source.route_index);
    identity = exact_history_fold_word(identity, source.route_incarnation);
    identity = exact_history_fold_word(identity, source.source_identity);
    identity = exact_history_fold_word(identity, source.relation);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint64_t>(source.parameter_q16));
    identity = exact_history_fold_word(identity, source.input_port.node);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(source.input_port.domain));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(source.input_port.direction));
    identity = exact_history_fold_word(identity, source.input_port.arity);
    identity = exact_history_fold_word(identity, source.output_port.node);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(source.output_port.domain));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(source.output_port.direction));
    identity = exact_history_fold_word(identity, source.output_port.arity);
    identity = exact_history_fold_word(
        identity, evidence.observation_identities[i]);
    identity = exact_history_fold_word(
        identity, evidence.observation_source_incarnations[i][0]);
    identity = exact_history_fold_word(
        identity, evidence.observation_source_incarnations[i][1]);
    identity = exact_history_fold_word(
        identity, evidence.observation_variable_identities[i]);
  }
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline std::int32_t resident_level_c_saturating_q16(
    std::int64_t value) {
  return value < -0x80000000ll ? static_cast<std::int32_t>(-0x80000000ll) :
      value > 0x7fffffffll ? static_cast<std::int32_t>(0x7fffffffll) :
      static_cast<std::int32_t>(value);
}
DIRECT_NETWORK_HD inline ResidentRecipeReceptorState
resident_level_c_child_receptor(
    const ResidentRecipeReceptorState& left,
    const ResidentRecipeReceptorState& right,
    const ResidentNetworkCondensationEvidence& evidence) {
  const std::int64_t parameter =
      static_cast<std::int64_t>(evidence.sources[0].parameter_q16) +
      evidence.sources[1].parameter_q16;
  ResidentRecipeReceptorState child{};
  child.activation_q16 = resident_level_c_saturating_q16(
      (static_cast<std::int64_t>(left.activation_q16) +
       right.activation_q16) / 2 + parameter);
  child.plasticity_q16 = resident_level_c_saturating_q16(
      (static_cast<std::int64_t>(left.plasticity_q16) +
       right.plasticity_q16) / 2);
  std::uint64_t identity = exact_history_fold_word(
      0x6c6576656c636463ull, left.causal_identity);
  identity = exact_history_fold_word(identity, left.revision_identity);
  identity = exact_history_fold_word(identity, right.causal_identity);
  identity = exact_history_fold_word(identity, right.revision_identity);
  identity = exact_history_fold_word(identity, evidence.boundary_identity);
  identity = exact_history_fold_word(identity, evidence.witness_identity);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(child.activation_q16));
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(child.plasticity_q16));
  child.causal_identity = identity == 0u ? 1u : identity;
  return child;
}
DIRECT_NETWORK_HD inline bool resident_condensation_source_matches(
    const ResidentCondensationSourceSnapshot& source,
    const ResidentRecipeCell* cells, std::uint32_t cell_count,
    const ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count) {
  if (cells == nullptr || derivations == nullptr ||
      source.recipe_cell >= cell_count || source.logical_recipe_id == 0u ||
      source.revision_identity == 0u || source.source_identity == 0u ||
      cells[source.recipe_cell].logical_recipe_id != source.logical_recipe_id ||
      cells[source.recipe_cell].revision_identity != source.revision_identity)
    return false;
  for (std::uint32_t i = 0u; i < derivation_count; ++i) {
    const auto& derivation = derivations[i];
    if (derivation.logical_recipe_id != source.logical_recipe_id ||
        derivation.revision_identity != source.revision_identity)
      continue;
    return derivation.generation == source.generation &&
           derivation.recipe_cell == source.recipe_cell &&
           derivation.route_index == source.route_index &&
           derivation.route_incarnations[0] == source.route_incarnation &&
           derivation.relation_count == 1u &&
           derivation.parameter_count == 1u && derivation.port_count == 2u &&
           derivation.relations[0] == source.relation &&
           derivation.parameters_q16[0] == source.parameter_q16 &&
           derivation.ports[0].node == source.input_port.node &&
           derivation.ports[0].domain == source.input_port.domain &&
           derivation.ports[0].direction == source.input_port.direction &&
           derivation.ports[0].arity == source.input_port.arity &&
           derivation.ports[1].node == source.output_port.node &&
           derivation.ports[1].domain == source.output_port.domain &&
           derivation.ports[1].direction == source.output_port.direction &&
           derivation.ports[1].arity == source.output_port.arity;
  }
  return false;
}
DIRECT_NETWORK_HD inline bool replay_resident_condensation_witness(
    const ResidentRecipeCell* cells, std::uint32_t cell_count,
    const ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count,
    const ResidentNetworkCondensationEvidence& evidence,
    std::uint64_t* logical_recipe_id, std::uint64_t* revision_identity) {
  if (cells == nullptr || derivations == nullptr ||
      logical_recipe_id == nullptr || revision_identity == nullptr ||
      evidence.source_count != kResidentCondensationSourceCount ||
      evidence.recurring_observations < 2u ||
      evidence.guard_min_q16 >= evidence.guard_max_q16 ||
      evidence.maximum_error_q16 != 0 || evidence.boundary_identity == 0u ||
      evidence.proof_identity == 0u ||
      evidence.refinement_logical_recipe_id == 0u ||
      evidence.refinement_revision_identity == 0u ||
      evidence.refinement_recipe_cell >= cell_count ||
      evidence.boundary_identity !=
          resident_condensation_boundary_identity(evidence) ||
      evidence.proof_identity != resident_condensation_proof_identity(evidence) ||
      evidence.witness_identity == 0u ||
      evidence.witness_identity != resident_network_condensation_witness(evidence) ||
      cells[evidence.refinement_recipe_cell].logical_recipe_id !=
          evidence.refinement_logical_recipe_id ||
      cells[evidence.refinement_recipe_cell].revision_identity !=
          evidence.refinement_revision_identity)
    return false;
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i)
    if (evidence.observation_identities[i] == 0u ||
        evidence.observation_variable_identities[i] == 0u ||
        evidence.observation_source_incarnations[i][0] == 0u ||
        evidence.observation_source_incarnations[i][1] == 0u ||
        !resident_condensation_source_matches(
            evidence.sources[i], cells, cell_count, derivations,
            derivation_count))
      return false;
  if (evidence.observation_source_incarnations[0][0] ==
          evidence.observation_source_incarnations[1][0] ||
      evidence.observation_source_incarnations[0][1] ==
          evidence.observation_source_incarnations[1][1])
    return false;
  const auto& left = evidence.sources[0];
  const auto& right = evidence.sources[1];
  if (left.recipe_cell == right.recipe_cell || left.relation != right.relation ||
      (left.relation != static_cast<std::uint32_t>(ResidentRecipeRelation::trigger) &&
       left.relation != static_cast<std::uint32_t>(ResidentRecipeRelation::route_gain)) ||
      !resident_recipe_ports_compatible(left.output_port, right.input_port))
    return false;
  const std::int64_t parameter =
      left.relation == static_cast<std::uint32_t>(ResidentRecipeRelation::trigger)
          ? static_cast<std::int64_t>(left.parameter_q16) + right.parameter_q16
          : (static_cast<std::int64_t>(left.parameter_q16) * right.parameter_q16) >> 16;
  if (parameter < -0x80000000ll || parameter > 0x7fffffffll) return false;
  std::uint64_t logical = exact_history_fold_word(
      0x6d65746172656369ull, evidence.witness_identity);
  logical = exact_history_fold_word(logical, left.logical_recipe_id);
  logical = exact_history_fold_word(logical, right.logical_recipe_id);
  if (logical == 0u) return false;
  const std::int64_t support =
      cells[left.recipe_cell].support_q16 < cells[right.recipe_cell].support_q16
          ? cells[left.recipe_cell].support_q16
          : cells[right.recipe_cell].support_q16;
  *logical_recipe_id = logical;
  *revision_identity = resident_recipe_revision_identity(
      logical, 0u, 1u, evidence.witness_identity, support, 0);
  return true;
}
DIRECT_NETWORK_HD inline bool evaluate_resident_recipe_boundary_q16(
    std::uint32_t relation, std::int32_t parameter_q16,
    std::int32_t input_q16, std::int32_t* output_q16) {
  if (output_q16 == nullptr) return false;
  std::int64_t result = 0;
  if (relation == static_cast<std::uint32_t>(ResidentRecipeRelation::trigger))
    result = static_cast<std::int64_t>(input_q16) + parameter_q16;
  else if (relation == static_cast<std::uint32_t>(ResidentRecipeRelation::route_gain))
    result = (static_cast<std::int64_t>(input_q16) * parameter_q16) >> 16;
  else return false;
  if (result < -0x80000000ll || result > 0x7fffffffll) return false;
  *output_q16 = static_cast<std::int32_t>(result);
  return true;
}
DIRECT_NETWORK_HD inline bool condense_resident_recipe_network(
    ResidentRecipeCell* cells, std::uint32_t* cell_count,
    std::uint32_t cell_capacity, ResidentRecipeDerivation* derivations,
    ResidentPostbirthConstructorState* state,
    const ResidentNetworkCondensationEvidence& evidence,
    std::uint16_t territory_index) {
  if (cells == nullptr || cell_count == nullptr || derivations == nullptr ||
      state == nullptr || evidence.source_count != kResidentCondensationSourceCount ||
      evidence.recurring_observations < 2u || evidence.maximum_error_q16 != 0 ||
      evidence.guard_min_q16 >= evidence.guard_max_q16 ||
      evidence.witness_identity == 0u ||
      evidence.witness_identity != resident_network_condensation_witness(evidence) ||
      *cell_count >= cell_capacity || *cell_count >= state->recipe_cell_capacity ||
      state->derivation_count >= state->derivation_capacity ||
      state->ports_used + 2u > state->port_capacity ||
      state->relations_used + 1u > state->relation_capacity ||
      state->parameters_used + 1u > state->parameter_capacity)
    return false;
  std::uint64_t logical_id = 0u, expected_revision_identity = 0u;
  if (!replay_resident_condensation_witness(
          cells, *cell_count, derivations, state->derivation_count, evidence,
          &logical_id, &expected_revision_identity))
    return false;
  const auto& left = evidence.sources[0];
  const auto& right = evidence.sources[1];
  std::uint32_t source_derivations[kResidentCondensationSourceCount] = {
      kInvalidIndex, kInvalidIndex};
  for (std::uint32_t source = 0u; source < kResidentCondensationSourceCount;
       ++source)
    for (std::uint32_t i = 0u; i < state->derivation_count; ++i)
      if (source_derivations[source] == kInvalidIndex &&
          derivations[i].recipe_cell == evidence.sources[source].recipe_cell &&
          derivations[i].logical_recipe_id ==
              evidence.sources[source].logical_recipe_id &&
          derivations[i].revision_identity ==
              evidence.sources[source].revision_identity &&
          derivations[i].generation == evidence.sources[source].generation)
        source_derivations[source] = i;
  if (source_derivations[0] == kInvalidIndex ||
      source_derivations[1] == kInvalidIndex)
    return false;
  const auto& left_derivation = derivations[source_derivations[0]];
  const auto& right_derivation = derivations[source_derivations[1]];
  if (left_derivation.route_index == kInvalidIndex ||
      right_derivation.route_index == kInvalidIndex ||
      left_derivation.route_incarnations[0] == 0u ||
      right_derivation.route_incarnations[0] == 0u)
    return false;
  const std::int64_t parameter =
      left.relation == static_cast<std::uint32_t>(ResidentRecipeRelation::trigger)
          ? static_cast<std::int64_t>(left.parameter_q16) + right.parameter_q16
          : (static_cast<std::int64_t>(left.parameter_q16) * right.parameter_q16) >> 16;
  if (parameter < -0x80000000ll || parameter > 0x7fffffffll) return false;
  for (std::uint32_t i = 0u; i < *cell_count; ++i)
    if (cells[i].logical_recipe_id == logical_id) return false;

  const std::uint32_t new_cell = *cell_count;
  ResidentRecipeCell cell{};
  cell.logical_recipe_id = logical_id;
  cell.support_q16 = cells[left.recipe_cell].support_q16 <
          cells[right.recipe_cell].support_q16
      ? cells[left.recipe_cell].support_q16
      : cells[right.recipe_cell].support_q16;
  cell.revision = 1u;
  cell.rule_index = cells[left.recipe_cell].rule_index;
  cell.revision_identity = expected_revision_identity;
  if (!initialize_resident_recipe_update_ir(&cell)) return false;
  cell.receptor_state = resident_level_c_child_receptor(
      cells[left.recipe_cell].receptor_state,
      cells[right.recipe_cell].receptor_state, evidence);
  cell.receptor_state.revision_identity = cell.revision_identity;
  ResidentRecipeDerivation derivation{};
  derivation.logical_recipe_id = logical_id;
  derivation.revision_identity = cell.revision_identity;
  derivation.parent_logical_recipe_id = left.logical_recipe_id;
  derivation.parent_revision_identity = left.revision_identity;
  derivation.witness_identity = evidence.witness_identity;
  derivation.generation = (left.generation > right.generation
                               ? left.generation : right.generation) + 1u;
  derivation.recipe_cell = new_cell;
  derivation.parent_recipe_cell = left.recipe_cell;
  derivation.source_node = left.input_port.node;
  derivation.route_index = left.route_index;
  derivation.route_incarnations[0] = left.route_incarnation;
  derivation.route_incarnations[1] = right.route_incarnation;
  derivation.territory_index = territory_index;
  derivation.port_count = 2u;
  derivation.relation_count = 1u;
  derivation.parameter_count = 1u;
  derivation.ports[0] = left.input_port;
  derivation.ports[1] = right.output_port;
  derivation.relations[0] = left.relation;
  derivation.parameters_q16[0] = static_cast<std::int32_t>(parameter);
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i)
    derivation.condensation_sources[i] = evidence.sources[i];
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i)
    derivation.condensation_observation_identities[i] =
        evidence.observation_identities[i];
  for (std::uint32_t observation = 0u;
       observation < kResidentCondensationSourceCount; ++observation) {
    derivation.condensation_observation_variable_identities[observation] =
        evidence.observation_variable_identities[observation];
    for (std::uint32_t source = 0u;
         source < kResidentCondensationSourceCount; ++source)
      derivation.condensation_observation_source_incarnations[observation][source] =
          evidence.observation_source_incarnations[observation][source];
  }
  derivation.condensation_boundary_identity = evidence.boundary_identity;
  derivation.condensation_proof_identity = evidence.proof_identity;
  derivation.condensation_refinement_logical_recipe_id =
      evidence.refinement_logical_recipe_id;
  derivation.condensation_refinement_revision_identity =
      evidence.refinement_revision_identity;
  derivation.condensation_refinement_recipe_cell =
      evidence.refinement_recipe_cell;
  derivation.condensation_observations = evidence.recurring_observations;
  derivation.condensation_guard_min_q16 = evidence.guard_min_q16;
  derivation.condensation_guard_max_q16 = evidence.guard_max_q16;
  derivation.condensation_maximum_error_q16 = evidence.maximum_error_q16;
  derivation.condensation_source_count = evidence.source_count;
  derivation.condensation_flags = kResidentDerivationCondensedNetwork;

  cells[new_cell] = cell;
  derivations[state->derivation_count] = derivation;
  record_resident_recipe_incidence(
      state, state->derivation_count, derivation);
  ++*cell_count;
  ++state->derivation_count;
  state->ports_used += 2u;
  ++state->relations_used;
  ++state->parameters_used;
  ++state->condensed;
  if (derivation.generation > state->highest_derivation_rank)
    state->highest_derivation_rank = derivation.generation;
  return true;
}
DIRECT_NETWORK_HD inline std::uint64_t resident_level_c_constructor_evidence(
    const ResidentRecipeCell& cell,
    const ResidentRecipeDerivation& derivation,
    const DirectExactHistoryRecord* consequences,
    std::uint32_t consequence_count) {
  if ((derivation.condensation_flags &
       (kResidentDerivationCondensedNetwork |
        kResidentDerivationBoundaryCondensedNetwork)) == 0u ||
      derivation.recipe_cell == kInvalidIndex ||
      cell.logical_recipe_id != derivation.logical_recipe_id ||
      cell.receptor_state.causal_identity == 0u || consequence_count < 2u ||
      consequences == nullptr)
    return 0u;
  std::uint64_t prior = derivation.revision_identity;
  std::uint64_t identity = exact_history_fold_word(
      0x6c6576656c636374ull, derivation.witness_identity);
  identity = exact_history_fold_word(identity, cell.receptor_state.causal_identity);
  for (std::uint32_t i = 0u; i < consequence_count; ++i) {
    const auto& event = consequences[i];
    if (event.kind != DirectExactHistoryKind::recipe_revision ||
        event.source != derivation.recipe_cell || event.parent_identity != prior ||
        event.identity == 0u || event.incarnation_before == 0u ||
        event.incarnation_after == 0u || event.resource_delta == 0)
      return 0u;
    identity = exact_history_fold_word(identity, event.identity);
    identity = exact_history_fold_word(identity, event.incarnation_before);
    identity = exact_history_fold_word(identity, event.incarnation_after);
    prior = event.identity;
  }
  if (cell.revision_identity != prior ||
      cell.receptor_state.revision_identity != prior)
    return 0u;
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline bool promote_resident_level_c_constructor(
    ResidentRecipeCell* cell, const ResidentRecipeDerivation& derivation,
    const DirectExactHistoryRecord* consequences,
    std::uint32_t consequence_count) {
  if (cell == nullptr) return false;
  const std::uint64_t evidence = resident_level_c_constructor_evidence(
      *cell, derivation, consequences, consequence_count);
  if (evidence == 0u) return false;
  cell->constructor_evidence_identity = evidence;
  cell->flags |= kResidentRecipeLevelCConstructor;
  return true;
}
DIRECT_NETWORK_HD inline bool rematerialize_resident_condensation(
    const ResidentRecipeDerivation& derivation,
    ResidentNetworkCondensationEvidence* evidence) {
  if (evidence == nullptr ||
      (derivation.condensation_flags & kResidentDerivationCondensedNetwork) == 0u ||
      derivation.condensation_source_count != kResidentCondensationSourceCount ||
      derivation.witness_identity == 0u)
    return false;
  ResidentNetworkCondensationEvidence restored{};
  restored.witness_identity = derivation.witness_identity;
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i)
    restored.sources[i] = derivation.condensation_sources[i];
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i)
    restored.observation_identities[i] =
        derivation.condensation_observation_identities[i];
  for (std::uint32_t observation = 0u;
       observation < kResidentCondensationSourceCount; ++observation) {
    restored.observation_variable_identities[observation] =
        derivation.condensation_observation_variable_identities[observation];
    for (std::uint32_t source = 0u;
         source < kResidentCondensationSourceCount; ++source)
      restored.observation_source_incarnations[observation][source] =
          derivation.condensation_observation_source_incarnations[observation][source];
  }
  restored.boundary_identity = derivation.condensation_boundary_identity;
  restored.proof_identity = derivation.condensation_proof_identity;
  restored.refinement_logical_recipe_id =
      derivation.condensation_refinement_logical_recipe_id;
  restored.refinement_revision_identity =
      derivation.condensation_refinement_revision_identity;
  restored.refinement_recipe_cell =
      derivation.condensation_refinement_recipe_cell;
  restored.recurring_observations = derivation.condensation_observations;
  restored.guard_min_q16 = derivation.condensation_guard_min_q16;
  restored.guard_max_q16 = derivation.condensation_guard_max_q16;
  restored.maximum_error_q16 = derivation.condensation_maximum_error_q16;
  restored.source_count = derivation.condensation_source_count;
  if (resident_network_condensation_witness(restored) != restored.witness_identity)
    return false;
  *evidence = restored;
  return true;
}
DIRECT_NETWORK_HD inline std::uint64_t
resident_condensation_rematerialization_identity(
    const ResidentCondensationDeoptimizationState& state) {
  std::uint64_t identity = exact_history_fold_word(
      0x64656f7074636f6eull, state.witness_identity);
  identity = exact_history_fold_word(identity, state.source_count);
  for (std::uint32_t i = 0u; i < state.source_count &&
       i < kResidentCondensationSourceCount; ++i) {
    identity = exact_history_fold_word(
        identity, state.sources[i].logical_recipe_id);
    identity = exact_history_fold_word(
        identity, state.sources[i].revision_identity);
    identity = exact_history_fold_word(identity, state.sources[i].generation);
    identity = exact_history_fold_word(identity, state.sources[i].recipe_cell);
  }
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline bool deoptimize_resident_condensation(
    const ResidentRecipeDerivation& derivation, std::int32_t input_q16,
    ResidentPostbirthConstructorState* state) {
  if (state == nullptr || input_q16 < 0) return false;
  ResidentNetworkCondensationEvidence evidence{};
  if (!rematerialize_resident_condensation(derivation, &evidence) ||
      (static_cast<std::uint32_t>(input_q16) >= evidence.guard_min_q16 &&
       static_cast<std::uint32_t>(input_q16) <= evidence.guard_max_q16))
    return false;
  ResidentCondensationDeoptimizationState rematerialized{};
  rematerialized.witness_identity = evidence.witness_identity;
  rematerialized.source_count = evidence.source_count;
  rematerialized.active = 1u;
  for (std::uint32_t i = 0u; i < evidence.source_count; ++i)
    rematerialized.sources[i] = evidence.sources[i];
  rematerialized.rematerialization_identity =
      resident_condensation_rematerialization_identity(rematerialized);
  state->condensation_deoptimization = rematerialized;
  return true;
}
DIRECT_NETWORK_HD inline bool execute_deoptimized_resident_condensation(
    const ResidentRecipeDerivation& derivation, std::int32_t input_q16,
    ResidentPostbirthConstructorState* state, std::int32_t* output_q16,
    std::uint32_t* work_units) {
  if (state == nullptr || output_q16 == nullptr || work_units == nullptr ||
      !deoptimize_resident_condensation(derivation, input_q16, state))
    return false;
  std::int32_t value = input_q16;
  const auto rematerialized = state->condensation_deoptimization;
  if (rematerialized.active != 1u ||
      rematerialized.source_count != kResidentCondensationSourceCount ||
      rematerialized.rematerialization_identity !=
          resident_condensation_rematerialization_identity(rematerialized))
    return false;
  for (std::uint32_t i = 0u; i < rematerialized.source_count; ++i)
    if (!evaluate_resident_recipe_boundary_q16(
            rematerialized.sources[i].relation,
            rematerialized.sources[i].parameter_q16, value, &value))
      return false;
  *output_q16 = value;
  *work_units = rematerialized.source_count;
  return true;
}
#endif

#if defined(DIRECT_NETWORK_POSTBIRTH_CONSTRUCTOR_IMPLEMENTATION)
__device__ bool postbirth_identity_unused(DirectBrain, std::uint64_t);
struct PostbirthTraceStep {
  DirectExactHistoryRecord record;
  std::uint32_t parent_cell;
  std::int32_t conductance_q16;
};

__device__ void load_postbirth_trace_frontier(
    const ResidentPostbirthConstructorState& state,
    const DirectExactHistoryHotPage& history, std::uint32_t indices[3],
    std::uint32_t* count, std::uint32_t* high_water) {
  const std::uint64_t raw = resident_postbirth_trace_frontier_raw(state);
  if ((raw & kPostbirthTraceFrontierMarker) == 0u) {
    *count = 0u;
    *high_water = state.trace_scan_cursor <= history.committed_slots
        ? state.trace_scan_cursor : 0u;
    return;
  }
  const std::uint32_t archived_mod = static_cast<std::uint32_t>((raw >> 59u) & 0xfu);
  *high_water = static_cast<std::uint32_t>(raw & kPostbirthTraceHighWaterMask);
  *count = static_cast<std::uint32_t>((raw >> 15u) & 0x3u);
  if (archived_mod != (history.archived_pages & 0xfu) ||
      *high_water > history.committed_slots || *count > 3u) {
    *count = 0u;
    *high_water = 0u;
    return;
  }
  indices[0] = static_cast<std::uint32_t>((raw >> 17u) & kPostbirthTraceIndexMask);
  indices[1] = static_cast<std::uint32_t>((raw >> 31u) & kPostbirthTraceIndexMask);
  indices[2] = static_cast<std::uint32_t>((raw >> 45u) & kPostbirthTraceIndexMask);
}

__device__ void store_postbirth_trace_frontier(
    ResidentPostbirthConstructorState* state,
    const DirectExactHistoryHotPage& history, const std::uint32_t indices[3],
    std::uint32_t count, std::uint32_t high_water) {
  count = count > 3u ? 3u : count;
  high_water = high_water > kDirectExactHistoryHotPageCapacity
      ? kDirectExactHistoryHotPageCapacity : high_water;
  std::uint64_t raw = kPostbirthTraceFrontierMarker |
      static_cast<std::uint64_t>(high_water) |
      (static_cast<std::uint64_t>(count) << 15u) |
      (static_cast<std::uint64_t>(history.archived_pages & 0xfu) << 59u);
  if (count > 0u) raw |= static_cast<std::uint64_t>(indices[0] & kPostbirthTraceIndexMask) << 17u;
  if (count > 1u) raw |= static_cast<std::uint64_t>(indices[1] & kPostbirthTraceIndexMask) << 31u;
  if (count > 2u) raw |= static_cast<std::uint64_t>(indices[2] & kPostbirthTraceIndexMask) << 45u;
  state->trace_scan_cursor = static_cast<std::uint32_t>(raw);
  state->trace_frontier_state_hi = static_cast<std::uint32_t>(raw >> 32u);
}

}  // namespace
__device__ bool materialize_postbirth_route_recipe_parent_external(
    DirectBrain brain, DirectRoute& route, std::uint32_t route_index,
    std::uint64_t route_incarnation, std::uint32_t* parent_cell) {
  auto* state = brain.postbirth_constructor;
  if (parent_cell == nullptr || state == nullptr || brain.development == nullptr || brain.resource_ecology == nullptr ||
      brain.recipe_cells == nullptr || brain.postbirth_derivations == nullptr || !route_is_active(route) ||
      route_index >= brain.route_capacity || route_incarnation == 0u) return false;
  const std::uint32_t builder = decode_route_recipe_builder(route.flags);
  if (builder < brain.development->recipe_cell_count) { *parent_cell = builder; return true; }
  if (builder != kInvalidIndex || state->derivation_count >= state->derivation_capacity ||
      brain.development->recipe_cell_count >= state->recipe_cell_capacity || state->ports_used + 2u > state->port_capacity ||
      state->relations_used + 1u > state->relation_capacity || state->parameters_used + 1u > state->parameter_capacity) return false;
  std::uint32_t inherited_parent = kInvalidIndex, inherited_front = kInvalidIndex, matches = 0u;
  if (brain.construction_fronts != nullptr && brain.construction_front_count != nullptr) {
    const std::uint32_t count = *brain.construction_front_count < brain.construction_front_capacity ? *brain.construction_front_count : brain.construction_front_capacity;
    for (std::uint32_t i = 0u; i < count; ++i) {
      const ResidentConstructionFront& front = brain.construction_fronts[i];
      if (front.state != kConstructionFrontLive || front.source_node != route.source || front.recipe_cell >= brain.development->recipe_cell_count) continue;
      const ResidentRecipeCell& parent = brain.recipe_cells[front.recipe_cell];
      if (parent.rule_index >= brain.resident_rule_count || front.rule_index != parent.rule_index) continue;
      inherited_parent = front.recipe_cell; inherited_front = i; if (++matches > 1u) break;
    }
  }
  if (matches != 1u) inherited_parent = inherited_front = kInvalidIndex;
  ResidentRecipeCell inherited{}; std::uint64_t inherited_generation = 0u;
  if (inherited_parent < brain.development->recipe_cell_count) {
    inherited = brain.recipe_cells[inherited_parent]; inherited_generation = resident_recipe_child_derivation_rank(
        brain.postbirth_derivations, state->derivation_count, inherited.logical_recipe_id);
    if (inherited_generation == 0u) return false;
  }
  using substrate::direct_adult::DirectResourcePoolKind;
  if (!substrate::direct_adult::device_commit_pool_units(brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u)) return false;
  const bool has_parent = inherited_parent < brain.development->recipe_cell_count;
  if (has_parent && !substrate::direct_adult::device_commit_pool_units(brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge, 1u)) {
    substrate::direct_adult::device_uncommit_pool_units(brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u); return false;
  }
  std::uint64_t logical = exact_history_fold_word(0x726f757465703000ull, route_index);
  logical = exact_history_fold_word(exact_history_fold_word(exact_history_fold_word(exact_history_fold_word(
      logical, route_incarnation), route.source), route.target), static_cast<std::uint32_t>(route.conductance_q16));
  if (logical == 0u) logical = 1u; const std::uint64_t witness = exact_history_fold_word(logical, route_incarnation);
  ResidentRecipeCell cell{}; cell.logical_recipe_id = logical; cell.rule_index = has_parent ? inherited.rule_index : kInvalidIndex;
  cell.support_q16 = has_parent ? inherited.support_q16 : (1 << 15); cell.revision = has_parent ? 1u : 0u;
  cell.revision_identity = resident_recipe_revision_identity(logical, has_parent ? inherited.revision_identity : 0u,
      cell.revision, witness, cell.support_q16, 0);
  if (!initialize_resident_recipe_update_ir(&cell)) return false;
  ResidentRecipeDerivation derivation{}; derivation.logical_recipe_id = logical; derivation.revision_identity = cell.revision_identity;
  derivation.parent_logical_recipe_id = has_parent ? inherited.logical_recipe_id : 0u; derivation.parent_revision_identity = has_parent ? inherited.revision_identity : 0u;
  derivation.witness_identity = witness; derivation.generation = inherited_generation; derivation.recipe_cell = brain.development->recipe_cell_count;
  derivation.parent_recipe_cell = inherited_parent; derivation.source_node = route.source; derivation.route_index = route_index;
  derivation.route_incarnations[0] = route_incarnation; derivation.territory_index = brain.nodes[route.source].territory_index;
  derivation.port_count = 2u; derivation.relation_count = 1u; derivation.parameter_count = 1u;
  derivation.ports[0] = ResidentRecipePort{route.source, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::input, 1u};
  derivation.ports[1] = ResidentRecipePort{route.target, ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::output, 1u};
  derivation.relations[0] = static_cast<std::uint32_t>(ResidentRecipeRelation::route_gain); derivation.parameters_q16[0] = route.conductance_q16;
  const std::uint32_t derivation_index = state->derivation_count, cell_index = brain.development->recipe_cell_count, unowned_flags = route.flags;
  const std::uint32_t claimed_flags = encode_route_recipe_builder(unowned_flags, cell_index);
  const std::uint32_t observed_flags = atomicCAS(reinterpret_cast<unsigned int*>(&route.flags), unowned_flags, claimed_flags);
  if (observed_flags != unowned_flags) {
    if (has_parent) substrate::direct_adult::device_uncommit_pool_units(brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge, 1u);
    substrate::direct_adult::device_uncommit_pool_units(brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u);
    const std::uint32_t observed_builder = decode_route_recipe_builder(observed_flags);
    if (observed_builder < brain.development->recipe_cell_count) { *parent_cell = observed_builder; return true; } return false;
  }
  brain.recipe_cells[cell_index] = cell; brain.postbirth_derivations[derivation_index] = derivation;
  record_resident_recipe_incidence(state, derivation_index, derivation); __threadfence();
  state->ports_used += 2u; state->relations_used += 1u; state->parameters_used += 1u; ++state->derivation_count; ++brain.development->recipe_cell_count;
  if (inherited_front != kInvalidIndex) {
    ResidentConstructionFront& front = brain.construction_fronts[inherited_front];
    if (front.state == kConstructionFrontLive && front.source_node == route.source && front.recipe_cell == inherited_parent && front.rule_index == cell.rule_index) {
      const std::uint64_t generation = next_construction_front_generation(brain.construction_front_generation_by_node[front.source_node]);
      if (generation != 0u) { front.recipe_cell = cell_index; front.generation = generation; brain.construction_front_generation_by_node[front.source_node] = generation; }
    }
  }
  *parent_cell = cell_index; return true;
}
namespace {

__device__ bool decode_postbirth_trace_step(
    DirectBrain brain, const DirectExactHistoryRecord& record,
    PostbirthTraceStep* out) {
  if (out == nullptr || record.kind != DirectExactHistoryKind::sparse_credit ||
      record.identity == 0u || record.parent_identity == 0u ||
      record.resource_delta == 0 || record.subject >= brain.route_capacity ||
      record.source >= brain.node_count || record.value >= brain.node_count)
    return false;
  DirectRoute& route = brain.routes[record.subject];
  if (!route_is_active(route) || route.source != record.source ||
      route.target != record.value || brain.route_incarnations == nullptr ||
      brain.route_incarnations[record.subject] != record.incarnation_before)
    return false;
  std::uint32_t parent = decode_route_recipe_builder(route.flags);
  if (parent >= brain.development->recipe_cell_count &&
      !materialize_postbirth_route_recipe_parent_external(
          brain, route, record.subject, record.incarnation_before, &parent))
    return false;
  *out = PostbirthTraceStep{record, parent, route.conductance_q16};
  return true;
}

}  // namespace

__device__ bool materialize_raw_contact_sparse_credit(
    DirectBrain brain, const DirectExactHistoryRecord& witness,
    const ResidentRawContactKey& contact,
    ResidentDevelopmentCounters* counters) {
  auto* state = brain.postbirth_constructor;
  if (state == nullptr || resident_raw_contact_key_empty(contact)) return false;
  if (witness.source != contact.node) return false;
  for (std::uint32_t i = 0u; i < state->recipe_incidence_count; ++i)
    if (resident_raw_contact_key_equal(
            state->recipe_incidence[i].contact, contact))
      return true;
  PostbirthTraceStep step{};
  if (!decode_postbirth_trace_step(brain, witness, &step)) return false;
  constexpr std::uint32_t ports = 2u, relations = 1u, parameters = 1u;
  if (state->derivation_count >= state->derivation_capacity ||
      brain.development->recipe_cell_count >= state->recipe_cell_capacity) {
    ++state->capacity_refusals;
    return false;
  }
  if (state->ports_used + ports > state->port_capacity ||
      state->relations_used + relations > state->relation_capacity ||
      state->parameters_used + parameters > state->parameter_capacity) {
    ++state->shape_refusals;
    return false;
  }
  using substrate::direct_adult::DirectResourcePoolKind;
  auto* recipe_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_record);
  auto* parent_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge);
  if (recipe_pool == nullptr || parent_pool == nullptr ||
      recipe_pool->reserved_units == 0u || parent_pool->reserved_units == 0u) {
    ++state->capacity_refusals;
    return false;
  }
  const std::uint32_t derivation_index = state->derivation_count;
  const std::uint32_t cell_index = brain.development->recipe_cell_count;
  const ResidentRecipeCell parent = brain.recipe_cells[step.parent_cell];
  const std::uint64_t generation = resident_recipe_child_derivation_rank(
      brain.postbirth_derivations, state->derivation_count,
      parent.logical_recipe_id);
  if (generation == 0u) {
    ++state->capacity_refusals;
    return false;
  }
  std::uint64_t logical_id = exact_history_fold_word(
      0x706f737462697274ull, parent.logical_recipe_id);
  logical_id = exact_history_fold_word(logical_id, witness.identity);
  logical_id = exact_history_fold_word(logical_id, witness.parent_identity);
  logical_id = exact_history_fold_word(logical_id, generation);
  if (logical_id == 0u || !postbirth_identity_unused(brain, logical_id)) {
    ++state->capacity_refusals;
    return false;
  }
  ResidentRecipeCell cell{};
  cell.logical_recipe_id = logical_id;
  cell.rule_index = parent.rule_index;
  cell.support_q16 = parent.support_q16;
  cell.credit_q16 = parent.credit_q16;
  cell.revision = 1u;
  cell.revision_identity = resident_recipe_revision_identity(
      logical_id, parent.revision_identity, 1u, witness.identity,
      cell.support_q16, cell.credit_q16);
  if (!initialize_resident_recipe_update_ir(&cell)) return false;
  ResidentRecipeDerivation derivation{};
  derivation.logical_recipe_id = logical_id;
  derivation.revision_identity = cell.revision_identity;
  derivation.parent_logical_recipe_id = parent.logical_recipe_id;
  derivation.parent_revision_identity = parent.revision_identity;
  derivation.witness_identity = witness.identity;
  derivation.generation = generation;
  derivation.recipe_cell = cell_index;
  derivation.parent_recipe_cell = step.parent_cell;
  derivation.source_node = witness.source;
  derivation.route_index = witness.subject;
  derivation.route_incarnations[0] = witness.incarnation_before;
  derivation.territory_index = brain.nodes[witness.source].territory_index;
  derivation.port_count = ports;
  derivation.relation_count = relations;
  derivation.parameter_count = parameters;
  derivation.ports[0] = ResidentRecipePort{
      witness.source, ResidentRecipePortDomain::q16_scalar,
      ResidentRecipePortDirection::input, 1u};
  derivation.ports[1] = ResidentRecipePort{
      witness.value, ResidentRecipePortDomain::q16_scalar,
      ResidentRecipePortDirection::output, 1u};
  derivation.relations[0] =
      static_cast<std::uint32_t>(ResidentRecipeRelation::trigger);
  derivation.parameters_q16[0] =
      static_cast<std::int32_t>(witness.resource_delta);
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u)) {
    ++state->capacity_refusals;
    return false;
  }
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology,
          DirectResourcePoolKind::derivation_parent_edge, 1u)) {
    substrate::direct_adult::device_uncommit_pool_units(
        brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u);
    ++state->capacity_refusals;
    return false;
  }
  brain.recipe_cells[cell_index] = cell;
  brain.postbirth_derivations[derivation_index] = derivation;
  record_resident_recipe_incidence(
      state, derivation_index, derivation, contact);
  state->ports_used += ports;
  state->relations_used += relations;
  state->parameters_used += parameters;
  ++state->derivation_count;
  ++brain.development->recipe_cell_count;
  ++state->condensed;
  if (generation > state->highest_derivation_rank)
    state->highest_derivation_rank = generation;
  DirectRoute& route = brain.routes[witness.subject];
  const std::uint32_t prior_flags = route.flags;
  if (decode_route_recipe_builder(prior_flags) == step.parent_cell) {
    const std::uint32_t child_flags =
        encode_route_recipe_builder(prior_flags, cell_index);
    atomicCAS(reinterpret_cast<unsigned int*>(&route.flags), prior_flags,
              child_flags);
  }
  const std::uint32_t front_count = brain.construction_front_count != nullptr
      ? *brain.construction_front_count : 0u;
  for (std::uint32_t i = 0u; i < front_count; ++i) {
    ResidentConstructionFront& front = brain.construction_fronts[i];
    if (front.state != kConstructionFrontLive ||
        front.source_node != witness.source ||
        front.recipe_cell != step.parent_cell)
      continue;
    front.recipe_cell = cell_index;
    front.rule_index = cell.rule_index;
    const std::uint64_t next = next_construction_front_generation(
        brain.construction_front_generation_by_node[front.source_node]);
    if (next != 0u) {
      front.generation = next;
      brain.construction_front_generation_by_node[front.source_node] = next;
    }
    break;
  }
  if (counters != nullptr) ++counters->postbirth_recipes_condensed;
  __threadfence();
  return true;
}

namespace {

__device__ bool same_postbirth_trace_step(
    const PostbirthTraceStep& left, const PostbirthTraceStep& right) {
  return left.record.source == right.record.source &&
      left.record.subject == right.record.subject &&
      left.record.value == right.record.value &&
      left.record.context == right.record.context &&
      left.record.incarnation_before == right.record.incarnation_before &&
      left.parent_cell == right.parent_cell;
}

__device__ bool condense_recurrent_postbirth_trace(
    DirectBrain brain, ResidentDevelopmentCounters* counters) {
  auto* state = brain.postbirth_constructor;
  DirectExactHistoryHotPage& history = brain.development->exact_history;
  PostbirthTraceStep steps[4]{};
  std::uint32_t carry_indices[3]{};
  std::uint32_t carry_count = 0u, high_water = 0u;
  load_postbirth_trace_frontier(
      *state, history, carry_indices, &carry_count, &high_water);
  std::uint32_t found = 0u;
  for (std::uint32_t i = 0u; i < carry_count; ++i) {
    if (carry_indices[i] >= history.committed_slots) continue;
    PostbirthTraceStep step{};
    if (!decode_postbirth_trace_step(brain, history.records[carry_indices[i]], &step))
      continue;
    steps[found++] = step;
    carry_indices[found - 1u] = carry_indices[i];
  }
  for (std::uint32_t cursor = high_water;
       cursor < history.committed_slots && found < 4u; ++cursor) {
    high_water = cursor + 1u;
    PostbirthTraceStep step{};
    if (!decode_postbirth_trace_step(brain, history.records[cursor], &step)) continue;
    steps[found] = step;
    if (found < 3u) carry_indices[found] = cursor;
    ++found;
  }
  if (found < 4u) {
    store_postbirth_trace_frontier(
        state, history, carry_indices, found, high_water);
    return false;
  }
  const bool recurrent = steps[0].record.identity == steps[1].record.identity &&
      steps[2].record.identity == steps[3].record.identity &&
      steps[0].record.identity != steps[2].record.identity &&
      steps[0].record.subject != steps[1].record.subject &&
      steps[0].record.value == steps[1].record.source &&
      steps[2].record.value == steps[3].record.source &&
      same_postbirth_trace_step(steps[0], steps[2]) &&
      same_postbirth_trace_step(steps[1], steps[3]) &&
      steps[0].parent_cell == steps[1].parent_cell;
  if (!recurrent) {
    // Slide the bounded qualifying-step window by one. The fourth step is at
    // high_water-1 because the scan stops immediately after admitting it.
    carry_indices[0] = carry_indices[1];
    carry_indices[1] = carry_indices[2];
    carry_indices[2] = high_water - 1u;
    store_postbirth_trace_frontier(state, history, carry_indices, 3u, high_water);
    ++state->shape_refusals;
    return false;
  }
  constexpr std::uint32_t ports = 2u, relations = 2u, parameters = 4u;
  if (state->derivation_count >= state->derivation_capacity ||
      brain.development->recipe_cell_count >= state->recipe_cell_capacity ||
      state->ports_used + ports > state->port_capacity ||
      state->relations_used + relations > state->relation_capacity ||
      state->parameters_used + parameters > state->parameter_capacity) {
    ++state->capacity_refusals;
    return false;
  }
  using substrate::direct_adult::DirectResourcePoolKind;
  auto* recipe_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_record);
  auto* parent_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge);
  if (recipe_pool == nullptr || parent_pool == nullptr ||
      recipe_pool->reserved_units == 0u || parent_pool->reserved_units == 0u) {
    ++state->capacity_refusals;
    return false;
  }
  const std::uint32_t parent_cell = steps[0].parent_cell;
  const ResidentRecipeCell parent = brain.recipe_cells[parent_cell];
  std::uint64_t witness = exact_history_fold_word(
      0x73656c66636f6d70ull, parent.revision_identity);
  for (const auto& step : steps) {
    witness = exact_history_fold_word(witness, step.record.identity);
    witness = exact_history_fold_word(witness, step.record.parent_identity);
    witness = exact_history_fold_word(witness, step.record.incarnation_before);
  }
  std::uint64_t logical_id = exact_history_fold_word(witness, parent.logical_recipe_id);
  if (logical_id == 0u || !postbirth_identity_unused(brain, logical_id)) {
    ++state->capacity_refusals;
    return false;
  }
  const std::uint64_t generation = resident_recipe_child_derivation_rank(
      brain.postbirth_derivations, state->derivation_count,
      parent.logical_recipe_id);
  if (generation == 0u) { ++state->capacity_refusals; return false; }
  ResidentRecipeCell cell{};
  cell.logical_recipe_id = logical_id;
  cell.rule_index = parent.rule_index;
  cell.support_q16 = parent.support_q16;
  cell.credit_q16 = parent.credit_q16;
  cell.revision = 1u;
  cell.revision_identity = resident_recipe_revision_identity(
      logical_id, parent.revision_identity, 1u, witness, cell.support_q16,
      cell.credit_q16);
  if (!initialize_resident_recipe_update_ir(&cell)) return false;
  ResidentRecipeDerivation derivation{};
  derivation.logical_recipe_id = logical_id;
  derivation.revision_identity = cell.revision_identity;
  derivation.parent_logical_recipe_id = parent.logical_recipe_id;
  derivation.parent_revision_identity = parent.revision_identity;
  derivation.witness_identity = witness;
  derivation.generation = generation;
  derivation.recipe_cell = brain.development->recipe_cell_count;
  derivation.parent_recipe_cell = parent_cell;
  derivation.source_node = steps[0].record.source;
  derivation.route_index = steps[0].record.subject;
  derivation.territory_index = brain.nodes[steps[0].record.source].territory_index;
  derivation.port_count = ports;
  derivation.relation_count = relations;
  derivation.parameter_count = parameters;
  derivation.ports[0] = ResidentRecipePort{steps[0].record.source,
      ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::input, 1u};
  derivation.ports[1] = ResidentRecipePort{steps[1].record.value,
      ResidentRecipePortDomain::q16_scalar, ResidentRecipePortDirection::output, 1u};
  derivation.relations[0] = steps[0].record.subject;
  derivation.relations[1] = steps[1].record.subject;
  derivation.route_incarnations[0] = steps[0].record.incarnation_before;
  derivation.route_incarnations[1] = steps[1].record.incarnation_before;
  derivation.parameters_q16[0] = static_cast<std::int32_t>(
      (static_cast<std::int64_t>(steps[0].conductance_q16) *
       steps[1].conductance_q16) >> 16);
  derivation.parameters_q16[1] = steps[0].conductance_q16;
  derivation.parameters_q16[2] = steps[1].conductance_q16;
  derivation.parameters_q16[3] = 2;
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u))
    { ++state->capacity_refusals; return false; }
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge, 1u)) {
    substrate::direct_adult::device_uncommit_pool_units(
        brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u);
    ++state->capacity_refusals;
    return false;
  }
  const std::uint32_t derivation_index = state->derivation_count;
  brain.recipe_cells[derivation.recipe_cell] = cell;
  brain.postbirth_derivations[derivation_index] = derivation;
  record_resident_recipe_incidence(
      state, derivation_index, derivation);
  __threadfence();
  state->highest_derivation_rank = generation > state->highest_derivation_rank
      ? generation : state->highest_derivation_rank;
  state->ports_used += ports;
  state->relations_used += relations;
  state->parameters_used += parameters;
  ++state->derivation_count;
  ++brain.development->recipe_cell_count;
  ++state->condensed;
  const std::uint32_t empty_indices[3]{};
  store_postbirth_trace_frontier(state, history, empty_indices, 0u, high_water);
  if (counters != nullptr) ++counters->postbirth_recipes_condensed;
  return true;
}

__device__ bool postbirth_identity_unused(DirectBrain brain, std::uint64_t logical_id) {
  for (std::uint32_t i = 0u; i < brain.development->recipe_cell_count; ++i)
    if (brain.recipe_cells[i].logical_recipe_id == logical_id) return false;
  return true;
}

__device__ bool postbirth_cause_kind(DirectExactHistoryKind kind) {
  return kind == DirectExactHistoryKind::topology_growth ||
      kind == DirectExactHistoryKind::topology_retraction;
}
// Cause resolution for a condensed Recipe witness: earliest committed
// growth/retraction carrying the parent identity. The journal is append-only
// between cold archives, so a cause resolved once for a witness position
// stays valid until the page is archived away; the per-position memo makes
// repeated sweeps of a lagging window constant-time while the first
// resolution pays the exact first-match scan. The memo is runtime scratch,
// never resident identity: slots hold 0 for unresolved, 1 for
// resolved-without-cause, and position + 2 otherwise, and any stale or
// mismatched entry recomputes -- eviction costs time, never truth.
__device__ DirectExactHistoryRecord postbirth_history_cause(
    DirectBrain brain, std::uint32_t* cause_memo,
    const DirectExactHistoryRecord& witness, std::uint32_t witness_position) {
  if (witness.kind != DirectExactHistoryKind::recipe_commit) return witness;
  DirectExactHistoryRecord cause{};
  DirectExactHistoryHotPage& history = brain.development->exact_history;
  std::uint32_t memo = cause_memo[witness_position];
  if (memo >= 2u) {
    const DirectExactHistoryRecord candidate = history.records[memo - 2u];
    if (candidate.identity == witness.parent_identity &&
        postbirth_cause_kind(candidate.kind))
      return candidate;
    memo = 0u;
  }
  if (memo == 0u) {
    for (std::uint32_t p = 0u; p < witness_position; ++p) {
      const DirectExactHistoryRecord candidate = history.records[p];
      if (postbirth_cause_kind(candidate.kind) &&
          candidate.identity == witness.parent_identity) {
        cause_memo[witness_position] = p + 2u;
        return candidate;
      }
    }
    cause_memo[witness_position] = 1u;
  }
  return cause;
}
__device__ std::uint32_t preferred_postbirth_history_cursor(
    DirectBrain brain, std::uint32_t* cause_memo,
    std::uint32_t begin, std::uint32_t end) {
  std::uint32_t preferred = kInvalidIndex;
  std::uint64_t preferred_rank = 0u;
  const auto& history = brain.development->exact_history;
  for (std::uint32_t cursor = begin; cursor < end; ++cursor) {
    const DirectExactHistoryRecord witness = history.records[cursor];
    const DirectExactHistoryRecord cause =
        postbirth_history_cause(brain, cause_memo, witness, cursor);
    const bool causal_credit = witness.kind == DirectExactHistoryKind::sparse_credit &&
        witness.identity != 0u && witness.parent_identity != 0u &&
        witness.resource_delta != 0;
    const bool causal_construction = witness.kind == DirectExactHistoryKind::recipe_commit &&
        witness.identity != 0u && cause.kind == DirectExactHistoryKind::topology_growth;
    if ((!causal_credit && !causal_construction) || cause.subject >= brain.route_capacity ||
        cause.source >= brain.node_count || cause.value >= brain.node_count) continue;
    const DirectRoute& route = brain.routes[cause.subject];
    const std::uint64_t expected = causal_construction
        ? cause.incarnation_after : cause.incarnation_before;
    if (!route_is_active(route) || brain.route_incarnations[cause.subject] != expected)
      continue;
    const std::uint32_t parent_cell = causal_construction
        ? witness.subject : decode_route_recipe_builder(route.flags);
    if (parent_cell >= brain.development->recipe_cell_count) continue;
    const std::uint64_t rank = resident_recipe_derivation_rank(
        brain.postbirth_derivations, brain.postbirth_constructor->derivation_count,
        brain.recipe_cells[parent_cell].logical_recipe_id);
    if (preferred == kInvalidIndex || rank > preferred_rank) {
      preferred = cursor;
      preferred_rank = rank;
    }
  }
  return preferred;
}


__device__ void condense_postbirth_recipe_impl(
    DirectBrain brain, ResidentDevelopmentCounters* counters,
    std::uint32_t* cause_memo) {
  auto* state = brain.postbirth_constructor;
  if (state == nullptr || brain.development == nullptr || brain.resource_ecology == nullptr)
    return;
  if (condense_recurrent_postbirth_trace(brain, counters)) return;
  DirectExactHistoryHotPage& history = brain.development->exact_history;
  const std::uint32_t end = history.committed_slots;
  if (cause_memo[kDirectExactHistoryHotPageCapacity] != history.archived_pages) {
    for (std::uint32_t i = 0u; i < kDirectExactHistoryHotPageCapacity; ++i)
      cause_memo[i] = 0u;
    cause_memo[kDirectExactHistoryHotPageCapacity] = history.archived_pages;
  }
  const std::uint32_t preferred = preferred_postbirth_history_cursor(
      brain, cause_memo, state->history_scan_cursor, end);
  bool found_history = false;
  for (std::uint32_t cursor = state->history_scan_cursor; cursor < end; ++cursor) {
    if (cursor != preferred) continue;
    const DirectExactHistoryRecord witness = history.records[cursor];
    const DirectExactHistoryRecord cause =
        postbirth_history_cause(brain, cause_memo, witness, cursor);
    const bool causal_credit = witness.kind == DirectExactHistoryKind::sparse_credit &&
        witness.identity != 0u && witness.parent_identity != 0u && witness.resource_delta != 0;
    const bool causal_construction = witness.kind == DirectExactHistoryKind::recipe_commit &&
        witness.identity != 0u && cause.kind == DirectExactHistoryKind::topology_growth;
    if ((!causal_credit && !causal_construction) || cause.subject >= brain.route_capacity ||
        cause.source >= brain.node_count || cause.value >= brain.node_count) continue;
    found_history = true;
    const DirectRoute& route = brain.routes[cause.subject];
    const std::uint64_t expected_incarnation = causal_construction
        ? cause.incarnation_after : cause.incarnation_before;
    if (!route_is_active(route) ||
        brain.route_incarnations[cause.subject] != expected_incarnation) continue;
    const ResidentRawContactKey raw_contact = causal_credit
        ? resident_raw_contact_binding(*state, witness.parent_identity)
        : ResidentRawContactKey{};
    if (!resident_raw_contact_key_empty(raw_contact)) {
      if (witness.source != raw_contact.node) {
        state->history_scan_cursor = cursor + 1u;
        return;
      }
      if (!materialize_raw_contact_sparse_credit(
              brain, witness, raw_contact, counters))
        break;
      state->history_scan_cursor = cursor + 1u;
      return;
    }
    const std::uint32_t parent_cell = causal_construction
        ? witness.subject : decode_route_recipe_builder(route.flags);
    if (parent_cell >= brain.development->recipe_cell_count) continue;
    constexpr std::uint32_t ports = 2u, relations = 1u, parameters = 1u;
    if (state->derivation_count >= state->derivation_capacity ||
        brain.development->recipe_cell_count >= state->recipe_cell_capacity) {
      ++state->capacity_refusals; break;
    }
    if (state->ports_used + ports > state->port_capacity ||
        state->relations_used + relations > state->relation_capacity ||
        state->parameters_used + parameters > state->parameter_capacity) {
      ++state->shape_refusals; break;
    }
    using substrate::direct_adult::DirectResourcePoolKind;
    auto* recipe_pool = substrate::direct_adult::direct_ecology_pool(
        brain.resource_ecology, DirectResourcePoolKind::derivation_record);
    auto* parent_pool = substrate::direct_adult::direct_ecology_pool(
        brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge);
    if (recipe_pool == nullptr || parent_pool == nullptr ||
        recipe_pool->reserved_units == 0u || parent_pool->reserved_units == 0u) {
      ++state->capacity_refusals; break;
    }
    const std::uint32_t derivation_index = state->derivation_count;
    const std::uint32_t cell_index = brain.development->recipe_cell_count;
    const ResidentRecipeCell parent = brain.recipe_cells[parent_cell];
    const std::uint64_t generation = resident_recipe_child_derivation_rank(
        brain.postbirth_derivations, state->derivation_count,
        parent.logical_recipe_id);
    if (generation == 0u) {
      ++state->capacity_refusals;
      break;
    }
    std::uint64_t logical_id = exact_history_fold_word(
        0x706f737462697274ull, parent.logical_recipe_id);
    logical_id = exact_history_fold_word(logical_id, witness.identity);
    logical_id = exact_history_fold_word(logical_id, witness.parent_identity);
    logical_id = exact_history_fold_word(logical_id, generation);
    if (logical_id == 0u || !postbirth_identity_unused(brain, logical_id)) {
      ++state->capacity_refusals; break;
    }
    ResidentRecipeCell cell{};
    cell.logical_recipe_id = logical_id;
    cell.rule_index = parent.rule_index;
    cell.support_q16 = parent.support_q16;
    cell.credit_q16 = parent.credit_q16;
    cell.revision = 1u;
    cell.revision_identity = resident_recipe_revision_identity(
        logical_id, parent.revision_identity, 1u, witness.identity,
        cell.support_q16, cell.credit_q16);
    if (!initialize_resident_recipe_update_ir(&cell)) {
      ++state->shape_refusals;
      break;
    }
    ResidentRecipeDerivation derivation{};
    derivation.logical_recipe_id = logical_id;
    derivation.revision_identity = cell.revision_identity;
    derivation.parent_logical_recipe_id = parent.logical_recipe_id;
    derivation.parent_revision_identity = parent.revision_identity;
    derivation.witness_identity = witness.identity;
    derivation.generation = generation;
    derivation.recipe_cell = cell_index; derivation.parent_recipe_cell = parent_cell;
    derivation.source_node = cause.source; derivation.route_index = cause.subject;
    derivation.route_incarnations[0] = expected_incarnation;
    derivation.territory_index = brain.nodes[cause.source].territory_index;
    derivation.port_count = ports; derivation.relation_count = relations;
    derivation.parameter_count = parameters;
    derivation.ports[0] = ResidentRecipePort{
        cause.source, ResidentRecipePortDomain::q16_scalar,
        ResidentRecipePortDirection::input, 1u};
    derivation.ports[1] = ResidentRecipePort{
        cause.value, ResidentRecipePortDomain::q16_scalar,
        ResidentRecipePortDirection::output, 1u};
    derivation.relations[0] = static_cast<std::uint32_t>(ResidentRecipeRelation::trigger);
    derivation.parameters_q16[0] = static_cast<std::int32_t>(
        causal_credit ? witness.resource_delta : cause.resource_delta);
    if (!substrate::direct_adult::device_commit_pool_units(
            brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u)) {
      ++state->capacity_refusals; break;
    }
    if (!substrate::direct_adult::device_commit_pool_units(
            brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge, 1u)) {
      substrate::direct_adult::device_uncommit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u);
      ++state->capacity_refusals; break;
    }
    brain.recipe_cells[cell_index] = cell;
    brain.postbirth_derivations[derivation_index] = derivation;
    const ResidentRawContactKey contact = causal_credit
        ? resident_raw_contact_binding(*state, witness.parent_identity)
        : ResidentRawContactKey{};
    record_resident_recipe_incidence(
        state, derivation_index, derivation, contact);
    __threadfence();
    if (generation > state->highest_derivation_rank)
      state->highest_derivation_rank = generation;
    state->ports_used += ports; state->relations_used += relations;
    state->parameters_used += parameters;
    ++state->derivation_count; ++brain.development->recipe_cell_count;
    ++state->condensed;
    if (cause.subject < brain.route_capacity) {
      DirectRoute& causal_route = brain.routes[cause.subject];
      const std::uint32_t prior_flags = causal_route.flags;
      if (route_is_active(causal_route) &&
          decode_route_recipe_builder(prior_flags) == parent_cell &&
          brain.route_incarnations[cause.subject] == expected_incarnation) {
        const std::uint32_t child_flags =
            encode_route_recipe_builder(prior_flags, cell_index);
        atomicCAS(reinterpret_cast<unsigned int*>(&causal_route.flags),
                  prior_flags, child_flags);
      }
    }
    const std::uint32_t front_count = brain.construction_front_count != nullptr
        ? *brain.construction_front_count : 0u;
    for (std::uint32_t i = 0u; i < front_count; ++i) {
      ResidentConstructionFront& front = brain.construction_fronts[i];
      if (front.state == kConstructionFrontLive &&
          front.source_node == cause.source && front.recipe_cell == parent_cell) {
        front.recipe_cell = cell_index; front.rule_index = cell.rule_index;
        const std::uint64_t generation = next_construction_front_generation(
            brain.construction_front_generation_by_node[front.source_node]);
        if (generation == 0u) break; front.generation = generation;
        brain.construction_front_generation_by_node[front.source_node] = generation;
        break;
      }
    }
    if (counters != nullptr) ++counters->postbirth_recipes_condensed;
    state->history_scan_cursor = cursor + 1u;
    return;
  }
  state->history_scan_cursor = end;
  if (!found_history) { ++state->no_history; if (counters) ++counters->postbirth_no_history; }
}

__global__ void condense_postbirth_recipe_kernel(
    DirectBrain brain, ResidentDevelopmentCounters* counters,
    std::uint32_t* cause_memo) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    condense_postbirth_recipe_impl(brain, counters, cause_memo);
}
#endif

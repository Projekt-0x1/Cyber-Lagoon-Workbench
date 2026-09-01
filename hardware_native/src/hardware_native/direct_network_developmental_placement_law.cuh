// Included inside direct_network_life_function.cu's anonymous namespace
// immediately after direct_network_genesis_stage_abi.cuh. Owns the pure
// __host__ __device__ developmental placement laws: field packing and
// influence, combined scores, deterministic candidate coordinates, competition
// priority, node-site selection, and route geometry scoring. Stateless
// evaluation only; kernels live in downstream stages.
// independent_decay_timescale (#1276/NET12): see field_decay_class's own
// comment below. Declared here, ahead of validate_direct_inputs, so that
// function's field.polarity bound can admit the packed decay class.
constexpr std::uint32_t kFieldDecayClassCount = 4u;
// distinct_effect_vector_over_gain (#1276/NET12 rung 3): a THIRD digit packed
// into the same `polarity` word, orthogonal to both field_kind and
// field_decay_class (see field_gain_participates below). Off (0) is exactly
// today's behavior for every existing field.
constexpr std::uint32_t kFieldGainParticipationCount = 2u;
// Spatial gradient participation (#1289 b.local_fields): a FOURTH digit in the
// same `polarity` word, orthogonal to kind, decay class and gain. When set,
// the field contributes its exact analytic gradient -- projected along a
// growth cone's extension -- to site placement and local route choice (see
// direct_network_field_ecology.inl). Off (0) reproduces the potential-only law
// exactly for every field authored before this rung.
constexpr std::uint32_t kFieldGradientParticipationCount = 2u;

DIRECT_NETWORK_HD inline std::uint64_t mix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

DIRECT_NETWORK_HD inline std::int32_t iabs32(std::int32_t value) {
  return value < 0 ? -value : value;
}

DIRECT_NETWORK_HD inline std::uint32_t min_u32(std::uint32_t a, std::uint32_t b) {
  return a < b ? a : b;
}

DIRECT_NETWORK_HD inline std::uint32_t max_u32(std::uint32_t a, std::uint32_t b) {
  return a > b ? a : b;
}

DIRECT_NETWORK_HD inline std::int64_t min_i64(std::int64_t a, std::int64_t b) {
  return a < b ? a : b;
}

DIRECT_NETWORK_HD inline std::uint32_t clamp_u32(std::uint32_t value, std::uint32_t lo,
                                                  std::uint32_t hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

DIRECT_NETWORK_HD inline std::int32_t clamp_i32(std::int32_t value, std::int32_t lo,
                                                 std::int32_t hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

DIRECT_NETWORK_HD inline bool chemistry_matches(std::uint32_t chemistry,
                                                 std::uint32_t require_mask,
                                                 std::uint32_t require_value) {
  return (chemistry & require_mask) == require_value;
}

DIRECT_NETWORK_HD inline DevelopmentFieldKind field_kind(const FieldBlock& field) {
  const std::uint32_t raw = field.polarity % kDevelopmentFieldKindCount;
  return static_cast<DevelopmentFieldKind>(raw);
}

// independent_decay_timescale (#1276/NET12): a decay CLASS packed into the
// same `polarity` word as `field_kind`, orthogonal to it
// (`polarity / kDevelopmentFieldKindCount`), rather than widening `FieldBlock`
// -- a BCC-shared POD (direct_network_recipe.hpp) that "cannot leave this
// struct until that lane retires". Class 0 is exactly the prior behavior:
// every existing NET00-NET02 field authors polarity < kDevelopmentFieldKindCount,
// so `field_decay_class` is always 0 for them and this is a strict no-op.
// (kFieldDecayClassCount itself is declared earlier in this file, ahead of
// validate_direct_inputs, whose field.polarity bound must admit it.)
DIRECT_NETWORK_HD inline std::uint32_t field_decay_class(const FieldBlock& field) {
  return (field.polarity / kDevelopmentFieldKindCount) % kFieldDecayClassCount;
}

// distinct_effect_vector_over_gain (#1276/NET12 rung 3): a bit packed above
// field_decay_class in the same `polarity` word, orthogonal to both prior
// digits. Off (0) is exactly the prior behavior -- every field authored
// before this rung has polarity < kDevelopmentFieldKindCount * kFieldDecayClassCount,
// so this is always false for them and a strict no-op. When true, the field's
// own influence score (which already carries its decay -- see
// field_decayed_strength_q16 above) is allowed to bias the initial
// conductance_q16 of the long-tract routes it wins, instead of every
// long-tract route landing on the flat kInitialConductanceQ16 regardless of
// which field, or how strong a field, chose it.
DIRECT_NETWORK_HD inline bool field_gain_participates(const FieldBlock& field) {
  return ((field.polarity / (kDevelopmentFieldKindCount * kFieldDecayClassCount)) %
          kFieldGainParticipationCount) != 0u;
}

// A plain switch rather than a device-side array: an `inline constexpr`
// array has no guaranteed device storage on this toolchain (nvcc rejected it
// as "undefined in device code"), while a switch over a compile-time-known
// small class count compiles to the same handful of immediates either way.
DIRECT_NETWORK_HD inline std::int32_t field_decay_magnitude_q16_per_tick(std::uint32_t decay_class) {
  switch (decay_class) {
    case 1u: return 1 << 4;
    case 2u: return 1 << 7;
    case 3u: return 1 << 10;
    default: return 0;
  }
}

DIRECT_NETWORK_HD inline std::int32_t field_strength_q16(const FieldBlock& field) {
  // Direct-network Gamma authors use write_value as a signed Q16 magnitude.
  // Zero intentionally means neutral/no force.
  return static_cast<std::int32_t>(field.write_value);
}

// Strength decayed toward zero by elapsed ticks since the field became
// active, at the field's own decay class. Never overshoots past zero, so a
// decaying field cannot flip sign and start pushing the opposite direction.
DIRECT_NETWORK_HD inline std::int32_t field_decayed_strength_q16(const FieldBlock& field,
                                                                  std::uint32_t logical_tick) {
  const std::int64_t strength = field_strength_q16(field);
  const std::uint32_t decay_class = field_decay_class(field);
  if (decay_class == 0u || logical_tick <= field.begin_tick) return static_cast<std::int32_t>(strength);
  const std::int64_t elapsed = static_cast<std::int64_t>(logical_tick) - static_cast<std::int64_t>(field.begin_tick);
  const std::int64_t decay = static_cast<std::int64_t>(field_decay_magnitude_q16_per_tick(decay_class)) * elapsed;
  if (strength >= 0) return static_cast<std::int32_t>(strength > decay ? strength - decay : 0);
  return static_cast<std::int32_t>(strength < -decay ? strength + decay : 0);
}

DIRECT_NETWORK_HD inline std::int32_t field_influence_q16(const FieldBlock* fields,
                                                           std::uint32_t field_count,
                                                           std::uint32_t field_index,
                                                           const std::int32_t coord[3],
                                                           std::uint32_t chemotype,
                                                           std::uint32_t logical_tick) {
  if (field_index == kInvalidField || field_index >= field_count) return 0;
  const FieldBlock& field = fields[field_index];
  if (logical_tick < field.begin_tick) return 0;
  if (field.end_tick != 0u && logical_tick >= field.end_tick) return 0;
  if (!chemistry_matches(chemotype, field.require_mask, field.require_value)) return 0;
  const std::int32_t dx = iabs32(coord[0] - static_cast<std::int32_t>(field.center[0]));
  const std::int32_t dy = iabs32(coord[1] - static_cast<std::int32_t>(field.center[1]));
  const std::int32_t dz = iabs32(coord[2] - static_cast<std::int32_t>(field.center[2]));
  const std::uint64_t distance = static_cast<std::uint64_t>(dx) + dy + dz;
  const std::uint64_t radius = static_cast<std::uint64_t>(field.radius) * 3ull;
  if (radius == 0u || distance > radius) return 0;
  const std::int64_t falloff_q16 =
      static_cast<std::int64_t>((radius - distance) << 16) / static_cast<std::int64_t>(radius);
  std::int64_t value =
      (static_cast<std::int64_t>(field_decayed_strength_q16(field, logical_tick)) * falloff_q16) >> 16;
  const DevelopmentFieldKind kind = field_kind(field);
  if (kind == DevelopmentFieldKind::repel || kind == DevelopmentFieldKind::inhibition) value = -value;
  return clamp_i32(static_cast<std::int32_t>(value), -(8 << 16), 8 << 16);
}

DIRECT_NETWORK_HD inline std::uint32_t logical_node_birth_tick(
    const GammaV1& gamma, const TerritoryPlan& plan, std::uint32_t local) {
  const SeedBlock& seed = gamma.seeds[plan.seed_index];
  const std::uint32_t begin = seed.begin_tick;
  const std::uint32_t handoff = max_u32(begin + 1u, gamma.header.development_end_tick);
  if (plan.node_count <= 1u || handoff <= begin + 1u) return begin;
  const std::uint64_t numerator =
      static_cast<std::uint64_t>(handoff - begin - 1u) * min_u32(local, plan.node_count - 1u);
  return begin + static_cast<std::uint32_t>(numerator / (plan.node_count - 1u));
}

DIRECT_NETWORK_HD inline std::int32_t bound_field_kind_influence_q16(
    const GammaV1& gamma, const TerritoryPlan& plan, const std::int32_t coord[3],
    std::uint32_t chemotype, std::uint32_t logical_tick,
    DevelopmentFieldKind kind) {
  std::int64_t sum = 0;
  for (std::uint32_t i = 0u; i < plan.bound_field_count; ++i) {
    const std::uint32_t field_index = plan.bound_field_indices[i];
    if (field_kind(gamma.fields[field_index]) != kind) continue;
    sum += field_influence_q16(gamma.fields, gamma.header.field_count, field_index,
                               coord, chemotype, logical_tick);
  }
  return clamp_i32(static_cast<std::int32_t>(sum), -(16 << 16), 16 << 16);
}

DIRECT_NETWORK_HD inline std::int32_t combined_developmental_score_q16(
    const GammaV1& gamma, const TerritoryPlan& plan, const std::int32_t coord[3],
    std::uint32_t logical_tick) {
  std::int64_t score = 0;
  score += bound_field_kind_influence_q16(gamma, plan, coord, plan.chemotype,
                                           logical_tick, DevelopmentFieldKind::attract);
  score += bound_field_kind_influence_q16(gamma, plan, coord, plan.chemotype,
                                           logical_tick, DevelopmentFieldKind::repel);
  score += bound_field_kind_influence_q16(gamma, plan, coord, plan.chemotype,
                                           logical_tick, DevelopmentFieldKind::resource);
  score += bound_field_kind_influence_q16(gamma, plan, coord, plan.chemotype,
                                           logical_tick, DevelopmentFieldKind::maturation) / 4;
  score += bound_field_kind_influence_q16(gamma, plan, coord, plan.chemotype,
                                           logical_tick, DevelopmentFieldKind::inhibition) / 4;
  return clamp_i32(static_cast<std::int32_t>(score), -(16 << 16), 16 << 16);
}

#include "direct_network_field_ecology.inl"

// gh #1268 / #1267: PARTNER AFFINITY, and it is deliberately not a new
// descriptor type.
//
// `combined_developmental_score_q16` above evaluates a territory's fields
// against `plan.chemotype` -- the SOURCE's own chemistry -- at whatever
// coordinate it is handed. That is the right reading for placing a node, and
// the wrong one for choosing the far end of a tract: it means corridor
// selection can express "grow toward my own kind" and "grow toward these
// coordinates", but never "grow toward THAT territory". The atlas asks for the
// third at every NETxx (`A<->C long_tract_affinity`), so no named corridor was
// authorable and every genome in the tree is homogeneous as a result.
//
// A growth cone carries the source's receptors and reads the TARGET's ligands.
// So this evaluates the source territory's own field set against the target
// territory's chemotype. No new struct, no ABI change: a FieldBlock already
// carries a chemistry predicate (`require_mask`/`require_value`), a spatial
// reach, a time window and a polarity, which is a complete partner-affinity
// descriptor. The only thing that was missing is whose chemistry it is tested
// against.
//
// INERT ON EVERY EXISTING GENOME, by construction rather than by a flag: every
// genome authored so far gives each territory fields requiring its OWN
// chemistry, so evaluating them against a DIFFERENT territory's chemotype
// fails `chemistry_matches` and contributes exactly 0. A field only becomes a
// partner affinity when an author writes one territory's field requiring
// another territory's chemistry -- which is exactly the declaration the atlas
// needs and nothing else in the tree does. The contract measures that
// inertness rather than trusting this paragraph.
DIRECT_NETWORK_HD inline std::int32_t partner_affinity_q16(const GammaV1& gamma,
                                                           const TerritoryPlan& source,
                                                           std::uint32_t target_chemotype,
                                                           const std::int32_t coord[3],
                                                           std::uint32_t logical_tick) {
  std::int64_t score = 0;
  score += bound_field_kind_influence_q16(gamma, source, coord, target_chemotype,
                                           logical_tick, DevelopmentFieldKind::attract);
  score += bound_field_kind_influence_q16(gamma, source, coord, target_chemotype,
                                           logical_tick, DevelopmentFieldKind::repel);
  score += bound_field_kind_influence_q16(gamma, source, coord, target_chemotype,
                                           logical_tick, DevelopmentFieldKind::resource);
  return clamp_i32(static_cast<std::int32_t>(score), -(16 << 16), 16 << 16);
}

// Does this territory ADDRESS someone else? True when one of its own fields
// carries a chemistry predicate that its own chemotype fails -- i.e. the author
// wrote it for a partner rather than for itself. Only such a territory gets the
// refusal semantics below; a territory that declares no partner keeps exactly
// today's geometric behaviour, which is what confines this change to genomes
// that opted in by authoring a corridor.
DIRECT_NETWORK_HD inline bool field_addresses_partner(const GammaV1& gamma,
                                                      std::uint32_t chemotype,
                                                      std::uint32_t field_index) {
  if (field_index == kInvalidField || field_index >= gamma.header.field_count) return false;
  const FieldBlock& field = gamma.fields[field_index];
  return !chemistry_matches(chemotype, field.require_mask, field.require_value);
}

DIRECT_NETWORK_HD inline bool declares_partner_affinity(const GammaV1& gamma,
                                                        const TerritoryPlan& plan) {
  for (std::uint32_t i = 0u; i < plan.bound_field_count; ++i) {
    const std::uint32_t field_index = plan.bound_field_indices[i];
    const DevelopmentFieldKind kind = field_kind(gamma.fields[field_index]);
    if (kind != DevelopmentFieldKind::attract && kind != DevelopmentFieldKind::repel &&
        kind != DevelopmentFieldKind::resource)
      continue;
    if (field_addresses_partner(gamma, plan.chemotype, field_index)) return true;
  }
  return false;
}

// gh #1359 / #1267 / #1290. Stable, domain-separated developmental priority for
// competition node selection. Uses territory-local identity (seed_index, local)
// rather than global allocation index so competitor identity is immune to
// unrelated earlier territory allocation shifts.
DIRECT_NETWORK_HD inline std::uint64_t competition_priority_key(
    const GammaV1& gamma, std::uint32_t seed_index, std::uint32_t local) {
  constexpr std::uint64_t kCompetitionPriorityDomain = 0x434f4d5050524930ull;  // "COMPPRI0"
  std::uint64_t key = mix64(gamma.header.development_seed ^ kCompetitionPriorityDomain);
  key ^= mix64((static_cast<std::uint64_t>(seed_index) << 32) | static_cast<std::uint64_t>(local));
  return mix64(key);
}

DIRECT_NETWORK_HD inline std::uint64_t candidate_coordinate_key(
    const GammaV1& gamma, const TerritoryPlan& plan, std::uint32_t local,
    std::uint32_t candidate) {
  return mix64(gamma.header.development_seed ^
              (static_cast<std::uint64_t>(plan.seed_index) << 48) ^
              (static_cast<std::uint64_t>(local) << 8) ^ static_cast<std::uint64_t>(candidate));
}

DIRECT_NETWORK_HD inline void candidate_coordinate(
    const GammaV1& gamma, const TerritoryPlan& plan, std::uint32_t local,
    std::uint32_t candidate, std::int32_t out[3]) {
  const SeedBlock& seed = gamma.seeds[plan.seed_index];
  const std::uint64_t h = candidate_coordinate_key(gamma, plan, local, candidate);
  const std::int32_t radius = static_cast<std::int32_t>(plan.radius);
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    const std::int32_t signed16 = static_cast<std::int32_t>((h >> (axis * 16u)) & 0xffffu) - 32768;
    const std::int32_t offset = radius == 0 ? 0 : (signed16 * radius) / 32768;
    out[axis] = static_cast<std::int32_t>(seed.coordinate[axis]) + offset;
  }
}

// The four-candidate argmax that decides where one node is placed. Extracted
// from materialize_nodes_kernel verbatim so the same code can be read on the
// host (direct_probe_node_site) instead of only inside a device launch --
// github #1319. Behaviour is unchanged: same candidates, same score, same
// strict `>` so ties keep the earlier candidate.
// gh #1348: HOW MANY EXTRA CANDIDATES A NODE MAY DRAW WHEN ALL FOUR ARE EXCLUDED.
//
// `kCandidateCoordinateCount` is where the argmax loop stops, not a property of
// the candidate set: `candidate_coordinate` hashes an arbitrary index through
// `candidate_coordinate_key` into `seed.coordinate +/- radius`, so candidates 4,
// 5, 6 ... are already deterministic, already Gamma-derived, and already inside
// the same territory. Drawing more of them is more of the same lawful set, not a
// second placement law and not authored geometry.
//
// ponytail: a fixed bound rather than an unbounded search, because this runs per
// node inside materialize_nodes_kernel. If a node still has no lawful site after
// the extension, it keeps the least-bad one and `environment_violating_nodes`
// counts it -- that residual is REPORTED, never assumed to be zero.
constexpr std::uint32_t kExtendedCandidateDraws = 32u;

// The four-candidate argmax, plus the refusal it could not previously express.
//
// gh #1348: `materialize_nodes_kernel` called `environment_hard_excludes` on the
// coordinate this function had ALREADY chosen, counted the violation, and wrote
// the node there anyway. Detection sat downstream of the decision it should have
// gated, so a hard exclusion was advisory: measured 13 of 32 nodes placed inside
// the exclusion with no individual salt in play, exactly the 13 whose four candidates were all
// covered. `null_inside == all_candidates_covered` was an equality, not a
// correlation.
//
// The repair follows the precedent `materialize_sparse_routes_kernel` already
// sets for routes -- detect that no candidate is legal, fall back to a named
// bounded mechanism, and count the fallback separately so it never masquerades
// as an ordinary placement. `out_extended_draws` is that third part.
//
// When any of the first four candidates is lawful this is byte-identical to what
// it did before: same candidates, same score, same strict `>` so ties keep the
// earlier candidate. Only the all-excluded node takes a different path.
DIRECT_NETWORK_HD inline std::int32_t select_node_site(
    const GammaV1& gamma, const DirectDevelopmentEnvironmentV1& environment,
    const TerritoryPlan& plan, std::uint32_t local, std::uint32_t logical_tick,
    std::int32_t out_coord[3],
    std::uint32_t* out_extended_draws = nullptr) {
  std::int32_t best_score = INT32_MIN;
  bool any_legal = false;
  bool best_legal = false;
  // b.local_fields: every placement is an extension from the territory's
  // founder site, so that is the cone origin the morphogen gradient is read
  // against.
  const SeedBlock& territory_seed = gamma.seeds[plan.seed_index];
  const std::int32_t cone_origin[3] = {
      static_cast<std::int32_t>(territory_seed.coordinate[0]),
      static_cast<std::int32_t>(territory_seed.coordinate[1]),
      static_cast<std::int32_t>(territory_seed.coordinate[2])};
  out_coord[0] = 0;
  out_coord[1] = 0;
  out_coord[2] = 0;
  if (out_extended_draws != nullptr) *out_extended_draws = 0u;
  for (std::uint32_t candidate = 0; candidate < kCandidateCoordinateCount; ++candidate) {
    std::int32_t coord[3];
    candidate_coordinate(gamma, plan, local, candidate, coord);
    const bool legal = !environment_hard_excludes(environment, coord);
    if (legal) any_legal = true;
    const std::int32_t score = clamp_i32(
        static_cast<std::int64_t>(combined_developmental_score_q16(gamma, plan, coord, logical_tick)) +
            environment_score_q16(environment, coord) +
            combined_gradient_tilt_q16(gamma, plan, coord, cone_origin, logical_tick),
        -(32 << 16), 32 << 16);
    // gh #1363: a hard exclusion clamps to the exact same score floor four
    // stacked soft penalties reach (proven by this file's own contract, ARM 4),
    // so score alone cannot tell an excluded candidate from a merely
    // maximally-penalized legal one -- on a tie the OLD strict `>` kept
    // whichever came first, which could be the excluded one even though a
    // legal candidate was scored in the same pass. Legality now gates the
    // comparison first: any legal candidate outranks any excluded one
    // regardless of score, and only among same-legality candidates does the
    // original strict `>` (earlier index wins ties) apply. This does not
    // touch the true total-obstruction case below, where no legal candidate
    // exists among the four to prefer.
    const bool better = (legal && !best_legal) || (legal == best_legal && score > best_score);
    if (better) {
      best_score = score;
      best_legal = legal;
      out_coord[0] = coord[0];
      out_coord[1] = coord[1];
      out_coord[2] = coord[2];
    }
  }
  if (any_legal) return best_score;

  // Every first-draw candidate is forbidden. Keep drawing from the SAME
  // generator and take the first lawful site; the argmax above already picked
  // the least-bad forbidden one, which is what we fall back to if the extension
  // finds nothing either.
  for (std::uint32_t extra = 0; extra < kExtendedCandidateDraws; ++extra) {
    std::int32_t coord[3];
    candidate_coordinate(gamma, plan, local, kCandidateCoordinateCount + extra, coord);
    if (environment_hard_excludes(environment, coord)) continue;
    out_coord[0] = coord[0];
    out_coord[1] = coord[1];
    out_coord[2] = coord[2];
    if (out_extended_draws != nullptr) *out_extended_draws = extra + 1u;
    return clamp_i32(
        static_cast<std::int64_t>(combined_developmental_score_q16(gamma, plan, coord, logical_tick)) +
            environment_score_q16(environment, coord) +
            combined_gradient_tilt_q16(gamma, plan, coord, cone_origin, logical_tick),
        -(32 << 16), 32 << 16);
  }
  return best_score;
}

DIRECT_NETWORK_HD inline std::int32_t geometry_route_score_q16(const DirectNode& source,
                                                                const DirectNode& target) {
  const std::int64_t distance = static_cast<std::int64_t>(iabs32(source.coordinate[0] - target.coordinate[0])) +
                                iabs32(source.coordinate[1] - target.coordinate[1]) +
                                iabs32(source.coordinate[2] - target.coordinate[2]);
  const std::int64_t locality_penalty = min_i64(distance << 7, 4ll << 16);
  return static_cast<std::int32_t>((2ll << 16) - locality_penalty);
}

DIRECT_NETWORK_HD inline std::uint32_t route_delay_from_geometry(const DirectNode& source,
                                                                  const DirectNode& target,
                                                                  bool long_tract) {
  const std::uint64_t distance = static_cast<std::uint64_t>(iabs32(source.coordinate[0] - target.coordinate[0])) +
                                 iabs32(source.coordinate[1] - target.coordinate[1]) +
                                 iabs32(source.coordinate[2] - target.coordinate[2]);
  const std::uint32_t divisor = long_tract ? 128u : 32u;
  return clamp_u32(1u + static_cast<std::uint32_t>(distance / divisor), 1u, long_tract ? 64u : 16u);
}

DIRECT_NETWORK_HD inline std::uint32_t candidate_local_target(const GammaV1& gamma,
                                                               const TerritoryPlan& plan,
                                                               std::uint32_t source_local,
                                                               std::uint32_t slot,
                                                               std::uint32_t candidate) {
  if (plan.node_count <= 1u) return source_local;
  const std::uint64_t h = mix64(gamma.header.development_seed ^
                                (static_cast<std::uint64_t>(plan.seed_index) << 48) ^
                                (static_cast<std::uint64_t>(source_local) << 24) ^
                                (static_cast<std::uint64_t>(slot) << 12) ^ candidate);
  const std::uint32_t step = 1u + static_cast<std::uint32_t>(h % (plan.node_count - 1u));
  return (source_local + step) % plan.node_count;
}

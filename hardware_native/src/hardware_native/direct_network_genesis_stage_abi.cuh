// Included inside direct_network_life_function.cu's anonymous namespace ahead
// of every genesis stage. This unit owns only the inter-stage data contract --
// territory plan layout, compile totals, and device-side diagnostics -- that
// planning, materialization, and the host receipt exchange. Every transform
// over these types lives downstream.
constexpr std::uint32_t kInvalidField = 0xffffffffu;
constexpr std::uint32_t kCandidateCoordinateCount = 4u;

constexpr std::uint32_t kTerritoryPlanConflictingCompetitionDensity = 1u << 0;
constexpr std::uint32_t kTerritoryPlanConflictingCompetitionMagnitude = 1u << 1;

// gh #1295: direct_adult_legacy_oracle.cu (a different translation unit,
// namespace substrate::direct_adult) names its own TU-local TerritoryPlan
// with 11 fields, a lean runtime topology plan -- not this file's 23-field
// genome-compilation territory descriptor. Six field names overlap (active,
// seed_index, lineage, node_count, node_offset, route_offset) at different
// struct offsets; the rest diverge entirely. Both copies are TU-local
// (anonymous namespace), so there is no ODR conflict -- only a naming trap
// for anyone grepping across files or copy-pasting between them.
struct TerritoryPlan {
  std::uint32_t active;
  std::uint32_t seed_index;
  std::uint32_t lineage;
  std::uint32_t chemotype;
  std::uint32_t flags;
  std::uint32_t node_count;
  std::uint32_t sparse_degree;
  std::uint32_t route_capacity_per_node;
  std::uint32_t long_tract_count;
  std::uint32_t dense_width;
  std::uint32_t node_offset;
  std::uint32_t route_offset;
  std::uint32_t active_route_estimate;
  std::uint32_t active_route_offset;
  std::uint32_t dense_block_offset;
  std::uint32_t dense_weight_offset;
  std::uint32_t attract_field;
  std::uint32_t repel_field;
  std::uint32_t resource_field;
  std::uint32_t maturation_field;
  std::uint32_t birth_maturation_field;
  std::uint32_t inhibition_field;
  std::uint32_t repair_field;
  std::uint16_t bound_field_indices[kDirectMaxFieldsV1];
  std::uint16_t bound_field_count;
  std::uint16_t bound_field_padding;
  std::uint32_t radius;
  // gh #1359 / #1267 / #1310. HOW STRONG, not only whether. `kRuleFlagInhibitoryBias`
  // says a territory competes. Authored magnitude lives on `threshold_q32 >> 16`
  // of the rule that grows the territory.
  std::uint32_t competition_strength_q16;
  std::uint32_t competition_magnitude_authored;
  // gh #1359 / #1267. HOW MANY, the other half of the same shape. The rate at
  // which a competing territory's nodes carry the disposition was `% 5ull` in
  // the genesis kernel -- one node in five, for every family that competes,
  // with no authored carrier -- so NET09's own atlas terms
  // (`parallel_competitive_loop_families`, `no_single_global_winner`), which are
  // statements about PROPORTION at least as much as about presence, were
  // unauthorable.
  //
  // The carrier is `minimum_age` on the rule that grows the territory. Its only
  // other consumer in the tree is `corridor_minimum_age` in the `long_tract`
  // case below, so the read is gated on the OPCODE as well as on the flag: a
  // per-opcode meaning has to be read per-opcode, or a corridor rule that
  // inherits the bias silently loses its growth window. 0 keeps `% 5` exactly.
  std::uint32_t inhibition_share_denominator;
  // gh #1359 / #1267 / #1290. Precomputed threshold for domain-separated
  // developmental priority comparison: (UINT64_MAX / denominator).
  std::uint64_t inhibition_priority_threshold;
  std::uint32_t competition_density_authored;
  std::uint32_t authoring_fault;
  // gh #1353 / #1267. Port capacity a Gamma rule ASKED for, as opposed to the
  // one DirectCompileOptions::route_reserve_per_node hands every territory
  // alike. #1353 seeded NET00 and named this as the gap it would not close:
  // `high_one_shot_port_capacity` is one of that network's four growth terms,
  // and until this field existed the only lever was the host option -- so two
  // organisms whose authored genomes agreed byte for byte still grew different
  // one-shot port capacity because a caller passed a different number. That is
  // an organism capacity chosen by the host rather than by the species.
  std::uint32_t port_reserve;
};
static_assert(std::is_trivial_v<TerritoryPlan> && std::is_standard_layout_v<TerritoryPlan>);

struct CompileTotals {
  std::uint32_t node_count;
  std::uint32_t route_capacity;
  std::uint32_t active_route_estimate;
  std::uint32_t dense_block_count;
  std::uint32_t dense_weight_count;
  std::uint32_t territory_count;
};

struct CompileDiagnostics {
  std::uint32_t invalid_boundary_bindings;
  std::uint32_t active_routes;
  std::uint32_t long_tracts;
  std::uint32_t maximum_in_degree;
  std::uint32_t pruned_routes;
  // gh #1309 / #1290 C0. Routes placed by the ring fallback below rather than by
  // a Gamma-scored candidate. They used to fall into `active_routes` alongside
  // every grown route, so a connectome that was 1% ring-wired and one that was
  // 40% ring-wired produced identical receipts -- and C0's "host does not choose
  // wiring" read green in both. This does not judge the fallback; it makes its
  // share visible so a policy can be chosen from a number.
  std::uint32_t fallback_wired_routes;
  // gh #1332 / #1319. Nodes placed inside a hard-excluded region. A hard exclude
  // is a fixed penalty, not a refusal, so when every candidate is excluded the
  // node lands inside anyway. This counts that; it decides nothing.
  std::uint32_t environment_violating_nodes;
  // gh #1348: nodes whose first four candidates were ALL hard-excluded and that
  // were placed from the bounded extended draw instead. Counted separately so an
  // extended placement never masquerades as an ordinary one -- the same reason
  // `fallback_wired_route_count` exists for routes.
  std::uint32_t extended_draw_nodes;
  // gh #1268 / #1267. A long tract that was REFUSED because its source territory
  // declares a partner affinity no territory in this organism satisfies. The
  // atlas authors named corridors ("A<->C"); a named affinity with no partner
  // must grow nothing rather than fall through to the geometric argmax, or the
  // organism silently grows a corridor the genome did not ask for and every
  // downstream disconnection probe measures an accident.
  std::uint32_t refused_partner_tracts;
  // Long tracts whose target was selected with a nonzero partner-affinity term.
  // This is the dose precondition for any claim that authored affinity steered
  // growth: zero here means the mechanism never fired and a passing direction
  // test is measuring geometry.
  std::uint32_t partner_steered_tracts;
  // gh #1268 / #1267. A long tract this node did not grow because its own
  // developmental birth tick fell outside the family's authored maturation
  // window (minimum_age/maximum_age). Nothing is written or committed --
  // an unrecruited tract must not charge the ledger for matter it never
  // took (#1178), the same restraint refused_partner_tracts already obeys.
  std::uint32_t immature_deferred_tracts;
};

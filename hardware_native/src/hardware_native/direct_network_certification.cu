#include "hardware_native/direct_network_certification.cuh"

#include <cuda_runtime.h>

#include <cstring>

#include "hardware_native/direct_network_genome_lowering.cuh"
#include "hardware_native/direct_seed_atlas.cuh"

namespace substrate::direct_network::certification {

ResidentAdultAdmissionReceipt observe_resident_adult_admission(
    const DirectBrain& brain) {
  ResidentAdultAdmissionReceipt receipt{};
  receipt.birth_root = brain.birth_root;
  if (brain.development != nullptr) {
    if (cudaMemcpy(&receipt.state, &brain.development->adult_admission,
                   sizeof(receipt.state), cudaMemcpyDeviceToHost) != cudaSuccess)
      receipt.state = ResidentAdultAdmissionState{};
  }
  return receipt;
}
namespace {

// Collision-conservative lineage census. An exact distinct-count over an
// unbounded key needs a device set; this ORs each lineage into one of 8192
// buckets, so collisions can only UNDERCOUNT. A measured lower bound of >= 2
// is still a proof of >= 2, which is the only direction t0 asks about.
inline constexpr std::uint32_t kLineageBucketWords = 256u;
inline constexpr std::uint32_t kLineageBuckets = kLineageBucketWords * 32u;

struct DeviceScan {
  unsigned int lineage_bits[kLineageBucketWords];
  unsigned long long active_routes;
  unsigned long long cross_lineage_routes;
  unsigned int max_out_degree;
  unsigned int branching_nodes;
  unsigned int overflow_nodes;
  unsigned int slice_count_mismatch;
  unsigned int foreign_source_routes;
  unsigned int out_of_range_targets;
  unsigned int min_lineage;
  unsigned int max_lineage;
  unsigned int lived_nodes;
  unsigned int credited_routes;
};

__global__ void scan_juvenile_kernel(const DirectNode* __restrict__ nodes,
                                     const DirectRoute* __restrict__ routes,
                                     std::uint32_t node_count,
                                     std::uint32_t route_capacity,
                                     DeviceScan* __restrict__ out) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;

  const DirectNode node = nodes[index];
  // Prenatal attractor/inhibition/maintenance and maturation fields are
  // generic construction state, not experience.  The lived boundary begins
  // at activation/EMA/credit and the actual/endogenous event clocks.
  if (node.activation_q16 != 0 || node.activity_ema_q16 != 0 ||
      node.credit_ema_q16 != 0 || node.last_actual_tick != 0u ||
      node.last_endogenous_tick != 0u)
    atomicAdd(&out->lived_nodes, 1u);
  const std::uint32_t bucket = node.lineage % kLineageBuckets;
  atomicOr(&out->lineage_bits[bucket >> 5], 1u << (bucket & 31u));
  atomicMin(&out->min_lineage, node.lineage);
  atomicMax(&out->max_lineage, node.lineage);

  // A slice that runs off the end of the route pool is an overflow whether or
  // not anything ever reads it, and it must be caught BEFORE the walk below --
  // otherwise the diagnostic pass is itself the out-of-bounds read.
  const std::uint32_t begin = node.route_offset;
  const std::uint64_t end = static_cast<std::uint64_t>(begin) + node.route_capacity;
  if (end > route_capacity || node.active_route_count > node.route_capacity) {
    atomicAdd(&out->overflow_nodes, 1u);
    return;
  }

  std::uint32_t active = 0u;
  std::uint32_t cross = 0u;
  for (std::uint32_t r = begin; r < static_cast<std::uint32_t>(end); ++r) {
    const DirectRoute route = routes[r];
    // Reserve slots are legitimately unowned matter; only a LIVE edge makes a
    // claim about who built it.
    if ((route.flags & kRouteFlagActive) == 0u) continue;
    // Developmental score/conductance/timing are Gamma-grown priors.  Only
    // eligibility and returned credit cross the experience boundary.
    if (route.eligibility_q16 != 0 || route.last_credit_q16 != 0 ||
        route.last_credit_ticket != kNoCreditTicket)
      atomicAdd(&out->credited_routes, 1u);
    ++active;
    if (route.source != index) atomicAdd(&out->foreign_source_routes, 1u);
    if (route.target >= node_count) {
      atomicAdd(&out->out_of_range_targets, 1u);
      continue;
    }
    if (nodes[route.target].lineage != node.lineage) ++cross;
  }

  if (active != node.active_route_count) atomicAdd(&out->slice_count_mismatch, 1u);
  atomicAdd(&out->active_routes, static_cast<unsigned long long>(active));
  atomicAdd(&out->cross_lineage_routes, static_cast<unsigned long long>(cross));
  atomicMax(&out->max_out_degree, active);
  if (active >= 2u) atomicAdd(&out->branching_nodes, 1u);
}

bool scan_juvenile(const DirectBrain& brain, std::uint32_t block_size, DeviceScan* out) {
  DeviceScan host{};
  host.min_lineage = 0xffffffffu;
  host.max_lineage = 0u;

  DeviceScan* device = nullptr;
  if (cudaMalloc(&device, sizeof(DeviceScan)) != cudaSuccess) return false;
  bool ok = cudaMemcpy(device, &host, sizeof(DeviceScan), cudaMemcpyHostToDevice) == cudaSuccess;
  if (ok && brain.node_count > 0u) {
    const std::uint32_t block = block_size == 0u ? 256u : block_size;
    const std::uint32_t grid = (brain.node_count + block - 1u) / block;
    scan_juvenile_kernel<<<grid, block>>>(brain.nodes, brain.routes, brain.node_count,
                                          brain.route_capacity, device);
    ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
  }
  if (ok) ok = cudaMemcpy(out, device, sizeof(DeviceScan), cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(device);
  return ok;
}

std::uint32_t popcount32(std::uint32_t value) {
  std::uint32_t count = 0u;
  while (value != 0u) {
    value &= value - 1u;
    ++count;
  }
  return count;
}

std::uint32_t lineage_bucket_count(const DeviceScan& scan) {
  std::uint32_t count = 0u;
  for (std::uint32_t word = 0u; word < kLineageBucketWords; ++word)
    count += popcount32(scan.lineage_bits[word]);
  return count;
}

bool pointer_inside_arena(const void* pointer, const DirectBrain& brain) {
  if (pointer == nullptr) return true;  // an absent pool is not a leak
  if (brain.arena == nullptr) return false;
  const auto base = reinterpret_cast<std::uintptr_t>(brain.arena);
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return address >= base && address < base + brain.arena_bytes;
}

// Matter closure: the organism's matter is the arena it was charged for, and
// every pool it executes out of lives inside that arena. A pool allocated
// beside the arena is matter the ledger never saw.
//
// MEASURED 2026-08-18, and the reason this function now enumerates every pool:
// an earlier version checked four of the thirteen and reported closure. The t2
// basin probe, which clones the arena by pointer offset, found a pool it could
// not reach that way -- `resource_ecology`, the #1178 ledger, is its own
// `cudaMalloc` beside the arena and is freed separately by
// destroy_direct_brain. It is live matter: development kernels take it as
// an argument. A closure check that omits nine pools cannot see that, and a
// spot check that happens to sample only the compliant ones reads exactly like
// a full one.
std::uint32_t pools_outside_arena(const DirectBrain& brain) {
  std::uint32_t outside = 0u;
  const void* pools[] = {brain.nodes,
                         brain.routes,
                         brain.dense_blocks,
                         brain.dense_weight_fp16_bits,
                         brain.boundary_ports,
                         brain.resident_fields,
                         brain.resident_rules,
                         brain.recipe_cells,
                         brain.recipe_edges,
                         brain.recipe_ranges,
                         brain.recipe_indices,
                         brain.development,
                         brain.retention_bank,
                         brain.resource_ecology};
  for (const void* pool : pools)
    if (!pointer_inside_arena(pool, brain)) ++outside;
  return outside;
}

bool matter_closed(const JuvenileReplica& replica) {
  const DirectBrain& brain = *replica.brain;
  if (brain.arena == nullptr || brain.arena_bytes == 0u) return false;
  if (brain.arena_bytes != replica.birth->arena_bytes) return false;
  return pools_outside_arena(brain) == 0u;
}

bool nonzero_root(const Root256& root) {
  const Root256 zero{};
  return root != zero;
}

bool construction_pointers_present(const DirectBrain& brain) {
  return brain.arena != nullptr && brain.arena_bytes != 0u &&
      brain.nodes != nullptr && brain.routes != nullptr &&
      brain.route_incarnations != nullptr && brain.boundary_ports != nullptr &&
      brain.development != nullptr && brain.postbirth_derivations != nullptr &&
      brain.postbirth_constructor != nullptr && brain.resource_ecology != nullptr &&
      brain.node_count != 0u && brain.route_capacity != 0u &&
      brain.boundary_port_count != 0u;
}

bool physical_body_complete(const DirectBodyManifestV1& body) {
  std::uint32_t roles = 0u;
  for (std::uint32_t index = 0u; index < body.binding_count; ++index)
    roles |= body.bindings[index].role_mask;
  const std::uint32_t required =
      static_cast<std::uint32_t>(BoundaryRole::sensor) |
      static_cast<std::uint32_t>(BoundaryRole::motor) |
      static_cast<std::uint32_t>(BoundaryRole::world_return);
  return body.binding_count != 0u && (roles & required) == required;
}

// Predicted-vs-measured front geometry. "Predicted" is what the life function
// reported when it returned; "measured" is what a device scan of the born
// arena finds now. They are separate readings of the same construction and the
// spec requires them EXACT, not close.
bool front_geometry_exact(const JuvenileReplica& replica, const DeviceScan& scan) {
  const DirectBrain& brain = *replica.brain;
  const DirectBirthReceiptV1& birth = *replica.birth;
  return birth.node_count == brain.node_count &&
         birth.active_route_count == brain.active_route_count &&
         birth.route_capacity == brain.route_capacity &&
         birth.territory_count == brain.territory_count &&
         scan.active_routes == static_cast<unsigned long long>(brain.active_route_count);
}

}  // namespace

void score_stage(StageReceipt& stage, std::uint32_t index, std::uint32_t requirement_count,
                 std::uint32_t assayed_mask, std::uint32_t unmet_mask) {
  const std::uint32_t all = requirement_count >= 32u ? 0xffffffffu
                                                     : ((1u << requirement_count) - 1u);
  stage.stage = index;
  stage.requirement_count = requirement_count;
  stage.unassayed_mask = all & ~assayed_mask;
  stage.unmet_mask = unmet_mask & assayed_mask;
  stage.requirement_assayed = popcount32(assayed_mask & all);
  stage.requirement_met = popcount32((assayed_mask & all) & ~stage.unmet_mask);

  // The whole point of the three-way verdict. A stage certifies only when the
  // spec's requirements were ALL assayed and ALL met. Any failure refuses; any
  // hole leaves the stage unevaluated, which is not a pass and is never
  // silently promoted into one.
  if (stage.unmet_mask != 0u)
    stage.verdict = StageVerdict::refused;
  else if (stage.unassayed_mask != 0u)
    stage.verdict = StageVerdict::not_evaluated;
  else
    stage.verdict = StageVerdict::certified;
}

void recompute_certification(NetworkFoundationReceipt& receipt) {
  receipt.stages_certified = 0u;
  receipt.stages_refused = 0u;
  for (std::uint32_t index = 0u; index < kStageCount; ++index) {
    if (receipt.stage[index].verdict == StageVerdict::certified) ++receipt.stages_certified;
    if (receipt.stage[index].verdict == StageVerdict::refused) ++receipt.stages_refused;
  }
  receipt.certified = receipt.stages_certified == kStageCount ? 1u : 0u;
}


NetworkFoundationReceipt certify_direct_juvenile(const JuvenileReplica* replicas,
                                                 std::uint32_t replica_count,
                                                 std::uint32_t block_size) {
  NetworkFoundationReceipt receipt{};
  receipt.replica_count = replica_count;
  for (std::uint32_t index = 0u; index < kStageCount; ++index) receipt.stage[index].stage = index;

  const std::uint32_t spec_counts[kStageCount] = {
      kT0RequirementCount, kT1RequirementCount, kT2RequirementCount, kT3RequirementCount,
      kT4RequirementCount, kT5RequirementCount, kT6RequirementCount};

  // No organism at all: every stage stays unassayed and the receipt refuses to
  // certify. An empty argument list must not be indistinguishable from a
  // perfect one.
  if (replicas == nullptr || replica_count == 0u || replicas[0].brain == nullptr ||
      replicas[0].birth == nullptr) {
    for (std::uint32_t index = 0u; index < kStageCount; ++index)
      score_stage(receipt.stage[index], index, spec_counts[index], 0u, 0u);
    return receipt;
  }

  const JuvenileReplica& primary = replicas[0];
  const DirectBrain& brain = *primary.brain;

  receipt.genome_root = brain.genome_root;
  receipt.body_root = brain.body_root;
  receipt.environment_root = brain.environment_root;
  receipt.birth_root = brain.birth_root;
  receipt.matter_bytes_paid = brain.arena_bytes;
  receipt.juvenile_morphology_root = direct_brain_root(brain, block_size);

  DeviceScan scan{};
  const bool scanned = scan_juvenile(brain, block_size, &scan);

  // Replay evidence. Two organisms grown from the same Gamma must land on the
  // same morphology root; anything else means allocation order, scheduling, or
  // a host write reached the phenotype.
  bool replicas_identical = replica_count >= 2u;
  for (std::uint32_t index = 1u; index < replica_count; ++index) {
    if (replicas[index].brain == nullptr) {
      replicas_identical = false;
      break;
    }
    if (direct_brain_root(*replicas[index].brain, block_size) !=
        receipt.juvenile_morphology_root) {
      replicas_identical = false;
      break;
    }
  }
  const bool replay_assayable = replica_count >= 2u;

  // ---- t0: construction fronts -------------------------------------------
  {
    std::uint32_t assayed = 0u;
    std::uint32_t unmet = 0u;
    if (scanned) {
      assayed |= kT0TwoGenesisLineages | kT0FrontGeometryExact | kT0NoHostWrittenMatureEdges;
      if (lineage_bucket_count(scan) < 2u) unmet |= kT0TwoGenesisLineages;
      if (!front_geometry_exact(primary, scan)) unmet |= kT0FrontGeometryExact;
      // gh #1309/#1290 C0: a route the genesis ring fallback placed consults
      // neither Gamma, geometry, nor the developmental score -- wiring chosen
      // by substrate arithmetic rather than grown, exactly the shape this
      // requirement already refuses a mis-sourced device edge for. #1309
      // deliberately deferred this policy pending a real measurement; a
      // production-config genesis measures fallback_wired_route_count == 0 of
      // 32,896 active routes (see the diary this change ships with), so a
      // zero-tolerance check does not regress the standing baseline and closes
      // the one gap in this requirement a device scan alone cannot see -- the
      // fallback's target is a syntactically valid, correctly-sourced edge.
      if (scan.foreign_source_routes != 0u || scan.out_of_range_targets != 0u ||
          scan.slice_count_mismatch != 0u || primary.birth->fallback_wired_route_count != 0u)
        unmet |= kT0NoHostWrittenMatureEdges;
    }
    assayed |= kT0MatterClosure;
    if (!matter_closed(primary)) unmet |= kT0MatterClosure;
    if (replay_assayable) {
      assayed |= kT0AllocationPermutationInvariance;
      if (!replicas_identical) unmet |= kT0AllocationPermutationInvariance;
    }
    StageReceipt& stage = receipt.stage[0];
    score_stage(stage, 0u, kT0RequirementCount, assayed, unmet);
    stage.measured[0] = lineage_bucket_count(scan);
    stage.measured[1] = brain.node_count;
    stage.measured[2] = brain.active_route_count;
    stage.measured[3] = scan.foreign_source_routes;
    stage.measured[4] = scan.out_of_range_targets;
    stage.measured[5] = scan.slice_count_mismatch;
    stage.measured[6] = pools_outside_arena(brain);
    stage.measured[7] = replica_count;
    // The two halves of matter closure, reported separately. They share one
    // spec requirement, so once either is permanently unmet the bit alone can
    // no longer show the other half still fires -- an obstruction arm would go
    // vacuous with nothing to say so.
    stage.measured[8] = brain.arena_bytes == primary.birth->arena_bytes ? 1u : 0u;
    stage.measured[9] = static_cast<std::uint32_t>(brain.arena_bytes >> 10);
    // gh #1309/#1290 C0: routes the genesis ring fallback placed rather than
    // Gamma, the same count kT0NoHostWrittenMatureEdges now checks above.
    // Reported separately here for the same reason measured[8]/[9] are: once
    // the bit is unmet from any cause, the bit alone cannot show which of its
    // now-three causes actually fired, and an obstruction arm targeting one
    // cause needs to see the other two stay at zero.
    stage.measured[10] = primary.birth->fallback_wired_route_count;
  }

  // ---- t1: branch / fuse / retract ---------------------------------------
  {
    std::uint32_t assayed = 0u;
    std::uint32_t unmet = 0u;
    // kT1RouteExtension and kT1RetractionUnderPressure are EVENTS during
    // development. A born arena is a snapshot: a route that was extended and a
    // route that was authored look identical in it, and a retracted route is
    // simply absent. Scoring them from this evidence would be reading a
    // distinction the procedure never wrote, so they stay unassayed.
    if (scanned) {
      assayed |= kT1BranchEvents | kT1CrossLineageFusion | kT1NoDenseCollapse |
                 kT1NoSilentOverflow;
      if (scan.branching_nodes == 0u) unmet |= kT1BranchEvents;
      if (scan.cross_lineage_routes == 0ull) unmet |= kT1CrossLineageFusion;
      if (scan.max_out_degree > kMaxSparseDegree) unmet |= kT1NoDenseCollapse;
      if (scan.overflow_nodes != 0u ||
          scan.active_routes != static_cast<unsigned long long>(brain.active_route_count))
        unmet |= kT1NoSilentOverflow;
    }
    if (replay_assayable) {
      assayed |= kT1ReplayIdentical;
      if (!replicas_identical) unmet |= kT1ReplayIdentical;
    }
    StageReceipt& stage = receipt.stage[1];
    score_stage(stage, 1u, kT1RequirementCount, assayed, unmet);
    stage.measured[0] = scan.branching_nodes;
    stage.measured[1] = static_cast<std::uint32_t>(scan.cross_lineage_routes);
    stage.measured[2] = scan.max_out_degree;
    stage.measured[3] = scan.overflow_nodes;
    stage.measured[4] = kMaxSparseDegree;
  }

  // ---- t2..t5: no assay exists in this build -----------------------------
  //
  // Probe masks over live morphology, post-hoc basin discovery, connector
  // communities, dose-matched device lesions and the two-formation joint
  // behaviour are all real work that has not been built. The receipt reports
  // the SIZE of the hole -- requirement_count with requirement_assayed == 0 --
  // rather than omitting these stages, so nothing downstream can mistake
  // silence for success.
  for (std::uint32_t index = 2u; index <= 5u; ++index)
    score_stage(receipt.stage[index], index, spec_counts[index], 0u, 0u);

  // ---- t6: certified language-naive juvenile ------------------------------
  {
    std::uint32_t assayed = kT6ExactMatterClosure;
    std::uint32_t unmet = 0u;
    if (!matter_closed(primary)) unmet |= kT6ExactMatterClosure;
    if (replay_assayable) {
      assayed |= kT6DeterministicRegrowth;
      if (!replicas_identical) unmet |= kT6DeterministicRegrowth;
    }
    StageReceipt& stage = receipt.stage[6];
    score_stage(stage, 6u, kT6RequirementCount, assayed, unmet);
    stage.measured[0] = replica_count;
    stage.measured[1] = replicas_identical ? 1u : 0u;
  }

  recompute_certification(receipt);
  return receipt;
}

LawfulConstructionReceipt certify_lawful_direct_construction(
    const DirectGenomeV1& genome, const DirectBodyManifestV1& body,
    const DirectDevelopmentEnvironmentV1& environment,
    const JuvenileReplica& juvenile, std::uint32_t block_size) {
  LawfulConstructionReceipt receipt{};
  receipt.assayed_mask = kC0AllRequirements;
  if (juvenile.brain == nullptr || juvenile.birth == nullptr) {
    receipt.unmet_mask = kC0AllRequirements;
    return receipt;
  }

  const DirectBrain& brain = *juvenile.brain;
  const DirectBirthReceiptV1& birth = *juvenile.birth;
  receipt.genome_root = brain.genome_root;
  receipt.territory_layout_root = brain.territory_layout_root;
  receipt.body_root = brain.body_root;
  receipt.environment_root = brain.environment_root;
  receipt.birth_root = brain.birth_root;
  receipt.node_count = brain.node_count;
  receipt.active_route_count = brain.active_route_count;
  receipt.route_capacity = brain.route_capacity;
  receipt.territory_count = brain.territory_count;
  receipt.boundary_port_count = brain.boundary_port_count;

  const Root256 expected_genome = canonical_direct_genome_root_v1(genome);
  const DirectGenomeLoweringV1 lowered =
      lower_direct_genome_v1(genome, body, environment);
  const Root256 expected_body = canonical_direct_body_root_v1(body);
  const Root256 expected_environment =
      canonical_direct_environment_root_v1(environment);
  const Root256 canonical_species =
      canonical_direct_genome_root_v1(seed_atlas_canonical_species());
  if (expected_genome != canonical_species || birth.genome_root != canonical_species)
    receipt.unmet_mask |= kC0CanonicalSpecies;
  if (brain.genome_root != expected_genome ||
      brain.territory_layout_root != lowered.territory_layout_root ||
      brain.body_root != expected_body ||
      brain.environment_root != expected_environment)
    receipt.unmet_mask |= kC0ExactConstructionInputs;

  if (!construction_pointers_present(brain)) {
    receipt.unmet_mask |= kC0ExactBirthAuthority | kC0GrownMorphology |
        kC0DetachedResidentHandoff | kC0SemanticallyBlank |
        kC0FiniteOwnedMatter | kC0PhysicalBody;
    return receipt;
  }

  DeviceScan scan{};
  const bool scanned = scan_juvenile(brain, block_size, &scan);
  receipt.lineage_count_lower_bound = lineage_bucket_count(scan);
  receipt.lived_node_count = scan.lived_nodes;
  receipt.credited_route_count = scan.credited_routes;

  ResidentDevelopmentState development{};
  ResidentPostbirthConstructorState constructor{};
  substrate::direct_adult::DirectResourceEcologyState ecology{};
  const bool read_resident =
      cudaMemcpy(&development, brain.development, sizeof(development),
                 cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(&constructor, brain.postbirth_constructor, sizeof(constructor),
                 cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(&ecology, brain.resource_ecology, sizeof(ecology),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
  if (read_resident) {
    receipt.exact_history_records = development.exact_history.committed_slots;
    receipt.raw_contact_binding_count = constructor.raw_contact_binding_count;
    receipt.recipe_incidence_count = constructor.recipe_incidence_count;
    receipt.current_charged_bytes = ecology.global_charged_bytes;
  }
  receipt.born_owned_bytes = brain.arena_bytes + sizeof(ecology);

  receipt.observed_state_root = direct_brain_root(brain, block_size);
  if (!nonzero_root(brain.birth_root) || brain.birth_root != birth.birth_root ||
      receipt.observed_state_root != brain.birth_root ||
      birth.genome_root != brain.genome_root ||
      birth.territory_layout_root != brain.territory_layout_root ||
      birth.body_root != brain.body_root ||
      birth.environment_root != brain.environment_root)
    receipt.unmet_mask |= kC0ExactBirthAuthority;
  if (!scanned || !front_geometry_exact(juvenile, scan) ||
      receipt.lineage_count_lower_bound < 2u ||
      scan.foreign_source_routes != 0u || scan.out_of_range_targets != 0u ||
      scan.slice_count_mismatch != 0u || scan.overflow_nodes != 0u ||
      birth.fallback_wired_route_count != 0u)
    receipt.unmet_mask |= kC0GrownMorphology;
  if (birth.final_connectome_loaded || !birth.external_life_function_detached ||
      !birth.resident_development_present)
    receipt.unmet_mask |= kC0DetachedResidentHandoff;
  if (!read_resident || scan.lived_nodes != 0u || scan.credited_routes != 0u ||
      development.exact_history.committed_slots != 0u ||
      development.exact_history.archived_record_count != 0u ||
      development.exact_history.next_sequence != 0u ||
      development.exact_history_tiers.entry_count != 0u ||
      constructor.raw_contact_binding_count != 0u ||
      constructor.recipe_incidence_count != 0u)
    receipt.unmet_mask |= kC0SemanticallyBlank;
  if (!read_resident || ecology.global_capacity_bytes == 0u ||
      ecology.global_charged_bytes < receipt.born_owned_bytes ||
      ecology.global_high_water_bytes < receipt.born_owned_bytes ||
      ecology.global_charged_bytes > ecology.global_capacity_bytes)
    receipt.unmet_mask |= kC0FiniteOwnedMatter;
  if (!physical_body_complete(body) || birth.body_root != expected_body ||
      brain.boundary_port_count != body.binding_count)
    receipt.unmet_mask |= kC0PhysicalBody;

  receipt.lawful = receipt.unmet_mask == 0u ? 1u : 0u;
  return receipt;
}

bool adult_test_eligible(const NetworkFoundationReceipt& receipt) {
  if (receipt.certified != 1u) return false;
  for (std::uint32_t index = 0u; index < kStageCount; ++index)
    if (receipt.stage[index].verdict != StageVerdict::certified) return false;
  return true;
}

}  // namespace substrate::direct_network::certification

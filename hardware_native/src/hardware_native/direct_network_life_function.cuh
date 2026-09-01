#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_LIFE_FUNCTION_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_LIFE_FUNCTION_CUH

#include "hardware_native/direct_network_genome.cuh"

namespace substrate::direct_network {

enum class DirectCompileError : std::uint32_t {
  none = 0,
  invalid_genome,
  invalid_body,
  no_territories,
  budget_exceeded,
  cuda_failure,
};

// AOT compilation: one planning barrier then GPU materialization.
DirectBirthReceiptV1 compile_direct_brain(const DirectGenomeV1& genome,
                                          const DirectBodyManifestV1& body,
                                          const DirectDevelopmentEnvironmentV1& environment,
                                          DirectBrain* out_brain,
                                          DirectCompileOptions options = {});
bool apply_observer_prose_bytes_to_direct_compile_inputs(
    DirectGenomeV1*, DirectBodyManifestV1*, DirectDevelopmentEnvironmentV1*,
    const void*, std::uint64_t);
bool apply_gamma_g1_executable_seed_bytes_to_direct_compile_inputs(
    DirectGenomeV1*, DirectBodyManifestV1*, DirectDevelopmentEnvironmentV1*,
    const void*, std::uint64_t);
bool apply_gamma_g2_arm_bytes_to_direct_compile_inputs(
    DirectGenomeV1*, DirectBodyManifestV1*, DirectDevelopmentEnvironmentV1*,
    const void*, std::uint64_t, std::uint32_t);

// Exact non-Gamma folds used by compile_direct_brain; observers reuse them
// rather than authoring a second identity convention.
// They expose identity only: none constructs, mutates, or executes a brain,
// and none replaces the canonical DirectGenome authority.
// The returned roots are content identities, never semantic labels.
Root256 canonical_direct_body_root_v1(const DirectBodyManifestV1& body);
Root256 canonical_direct_environment_root_v1(const DirectDevelopmentEnvironmentV1& environment);

void destroy_direct_brain(DirectBrain* brain);

// Deterministic root over the born arena + logical metadata.  The heavy arena
// hash is computed on device; only a Root256-sized digest crosses to the host.
Root256 direct_brain_root(const DirectBrain& brain,
                              std::uint32_t block_size = 256u);

// Host-side read of the SAME territory planner plan_territories_kernel runs, so
// a contract can falsify what a Gamma rule's authored flags do without a device.
// Not a production path: nothing in genesis calls this.
// Every field derive_territory_plan itself derives. The layout offsets
// (node_offset, route_offset, ...) are deliberately absent: a later pass assigns
// them, so exposing them here would invite a claim the planner cannot support.
struct DirectTerritoryPlanProbe {
  std::uint32_t active;
  std::uint32_t flags;
  std::uint32_t node_count;
  std::uint32_t sparse_degree;
  std::uint32_t radius;
  std::uint32_t dense_width;
  std::uint32_t long_tract_count;
  std::uint32_t route_capacity_per_node;
  std::uint32_t active_route_estimate;
  std::uint32_t lineage;
  std::uint32_t chemotype;
  std::uint32_t attract_field;
  std::uint32_t repel_field;
  std::uint32_t resource_field;
  std::uint32_t maturation_field;
  std::uint32_t inhibition_field;
  std::uint32_t repair_field;
  std::uint32_t bound_field_count;
};
DirectTerritoryPlanProbe direct_probe_territory_plan(const GammaV1& gamma,
                                                     std::uint32_t seed_index,
                                                     std::uint32_t route_reserve_per_node =
                                                         kDefaultRouteReserve);

struct DirectFieldEvaluationProbe {
  std::int32_t potential_q16;
  std::int32_t gradient_tilt_q16;
  std::uint32_t bound_field_count;
};
DirectFieldEvaluationProbe direct_probe_field_evaluation(
    const GammaV1& gamma, std::uint32_t seed_index,
    const std::int32_t growth_site[3], const std::int32_t cone_origin[3],
    std::uint32_t logical_tick,
    std::uint32_t route_reserve_per_node = kDefaultRouteReserve);

// Host-side read of select_node_site, the four-candidate argmax that decides
// where one node is placed -- the same code materialize_nodes_kernel runs.
// `chosen_hard_excluded` and `candidates_hard_excluded` are computed HERE by
// calling environment_hard_excludes; construction itself never asks that
// question, which is the whole of github #1319.
struct DirectNodeSiteProbe {
  std::uint32_t valid;
  std::int32_t coordinate[3];
  std::int32_t best_score_q16;
  std::uint32_t chosen_hard_excluded;
  std::uint32_t candidates_hard_excluded;
  std::uint32_t candidate_count;
  // gh #1348: 0 when one of the first `candidate_count` candidates was lawful.
  // Otherwise the 1-based index into the bounded extended draw that supplied the
  // site, so a reader can tell an ordinary placement from a rescued one. Still 0
  // when the extension found nothing either -- read it with
  // `chosen_hard_excluded`, which then stays 1.
  std::uint32_t extended_draws;
};
DirectNodeSiteProbe direct_probe_node_site(const GammaV1& gamma,
                                           const DirectDevelopmentEnvironmentV1& environment,
                                           std::uint32_t seed_index, std::uint32_t local,
                                           std::uint32_t route_reserve_per_node =
                                               kDefaultRouteReserve);

// Host-side read of attach_boundary_port, the membrane attachment
// attach_boundary_ports_kernel runs. `node_flags` is what the kernel would OR
// into the node for this role mask. `refused` is the binding the compiler
// declines and counts as invalid_boundary_bindings.
struct DirectBoundaryPortProbe {
  std::uint32_t attached;
  std::uint32_t refused;
  std::uint32_t node;
  std::uint32_t channel;
  std::uint32_t role_mask;
  std::uint32_t node_flags;
  std::uint32_t physical_route;
  std::uint32_t parent_route;
};
DirectBoundaryPortProbe direct_probe_boundary_port(const GammaV1& gamma,
                                                   const BoundaryPortBinding& binding,
                                                   std::uint32_t route_reserve_per_node =
                                                       kDefaultRouteReserve);

// Host-side read of compile_resident_rules, the admission decision that fills
// brain.resident_rules -- the set select_recipe_cell reads after birth. It has
// internal linkage, so this is the only way to observe it without a device
// compile. Deliberately NOT a re-derivation of the planner's own
// resident_after_birth test: the whole of github #1344 is that the two disagree,
// and a probe that restated one of them could not show that.
struct DirectResidentRuleProbe {
  std::uint32_t admitted_count;
  std::uint32_t admitted_retract_count;
  std::uint32_t admitted_without_resident_flag;
  std::uint32_t first_admitted_source_rule_index;
};
DirectResidentRuleProbe direct_probe_resident_rules(const GammaV1& gamma);

// github #1194/#1290's gap: attach_boundary_port copies `channel` opaquely and
// never compares it against any other binding, so two sensor-role bindings can
// claim the same channel and both attach. direct_adult_core.cu then resolves a
// channel-addressed sensory event by the FIRST port whose channel matches, so
// which node receives it is decided by binding order -- silently, with no
// error. True for exactly the sensor role: ingress addresses a contact by
// channel only when routing to a sensor; other roles are looked up by port
// index, not channel, so a channel shared across roles is not this ambiguity.
bool direct_boundary_bindings_share_a_sensor_channel(const DirectBodyManifestV1& body);

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_LIFE_FUNCTION_CUH

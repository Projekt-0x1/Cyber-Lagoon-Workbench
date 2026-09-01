// Included inside direct_adult_legacy_oracle.cu's anonymous namespace ahead of
// every other stage. Owns V01 birth: the lean runtime TerritoryPlan, Gamma-seed
// planning, offset installation, territory materialization, and boundary
// attachment. Launch order stays with compile_direct_brain_v01 below.
// gh #1295: direct_network_life_function.cu (a different translation unit,
// namespace substrate::direct_network) names its own TU-local TerritoryPlan
// with 23 fields, a genome-compilation territory descriptor -- not this
// file's lean 11-field runtime topology plan. Six field names overlap
// (active, seed_index, node_count, node_offset, route_offset, lineage) at
// different struct offsets; the rest diverge entirely. Both copies are
// TU-local (anonymous namespace), so there is no ODR conflict -- only a
// naming trap for anyone grepping across files or copy-pasting between them.
struct TerritoryPlan {
  std::uint32_t active;
  std::uint32_t seed_index;
  std::uint32_t node_count;
  std::uint32_t route_count;
  std::uint32_t logical_route_count;
  std::uint32_t virtual_route_count;
  std::uint32_t local_degree;
  std::uint32_t chord_stride;
  std::uint32_t node_offset;
  std::uint32_t route_offset;
  std::uint32_t lineage;
};

__global__ void plan_territories_kernel(const Gamma* gamma, TerritoryPlan* plans,
                                        std::uint32_t plan_count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= plan_count)
    return;
  TerritoryPlan plan{};
  if (i < gamma->header.seed_count) {
    const auto seed = gamma->seeds[i];
    plan.active = 1u;
    plan.seed_index = i;
    plan.node_count = 1024u;
    plan.local_degree = 4u;
    plan.chord_stride = 13u;
    plan.lineage = seed.lineage;
    for (std::uint32_t r = 0; r < gamma->header.rule_count; ++r) {
      const auto rule = gamma->rules[r];
      if (rule.opcode != substrate::direct_network::recipe::RuleOpcode::branch)
        continue;
      if ((seed.chemistry & rule.require_mask) != rule.require_value)
        continue;
      plan.node_count = max(64u, min(65536u, rule.branch_count));
      plan.local_degree = min(kMaximumDirectFanout, max(2u, rule.child_slot));
      plan.chord_stride = max(2u, rule.extent);
      plan.logical_route_count = plan.node_count * plan.local_degree;
      const std::uint32_t explicit_degree = min(2u, plan.local_degree);
      const std::uint32_t explicit_extra = plan.local_degree > explicit_degree ? 1u : 0u;
      plan.route_count = plan.node_count * explicit_degree + explicit_extra;
      plan.virtual_route_count = plan.logical_route_count - plan.route_count;
      break;
    }
  }
  plans[i] = plan;
}

__global__ void install_offsets_kernel(TerritoryPlan* plans, const std::uint32_t* node_offsets,
                                       const std::uint32_t* route_offsets,
                                       std::uint32_t plan_count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= plan_count)
    return;
  plans[i].node_offset = node_offsets[i];
  plans[i].route_offset = route_offsets[i];
}

__device__ std::uint32_t next_active_plan(const TerritoryPlan* plans, std::uint32_t plan_count,
                                          std::uint32_t current) {
  for (std::uint32_t step = 1; step < plan_count; ++step) {
    const std::uint32_t candidate = (current + step) % plan_count;
    if (plans[candidate].active != 0u)
      return candidate;
  }
  return current;
}

__global__ void materialize_territories_kernel(const TerritoryPlan* plans, std::uint32_t plan_count,
                                               DirectNode* nodes, DirectRoute* routes) {
  const std::uint32_t plan_index = blockIdx.x;
  if (plan_index >= plan_count || plans[plan_index].active == 0u)
    return;
  const TerritoryPlan plan = plans[plan_index];
  const std::uint32_t next_plan = next_active_plan(plans, plan_count, plan_index);
  const std::uint32_t explicit_degree = min(2u, plan.local_degree);
  const bool has_extra = plan.local_degree > explicit_degree;

  for (std::uint32_t local = threadIdx.x; local < plan.node_count; local += blockDim.x) {
    const std::uint32_t node_index = plan.node_offset + local;
    const std::uint32_t shift = has_extra && local != 0u ? 1u : 0u;
    DirectNode node{};
    node.first_route = plan.route_offset + local * explicit_degree + shift;
    node.route_count = explicit_degree + (has_extra && local == 0u ? 1u : 0u);
    node.lineage = plan.lineage;
    node.maintenance_q16 = 1u << 16;
    node.output_channel = kInvalidIndex;
    node.output_word = 0u;
    node.implicit_family = plan_index;
    nodes[node_index] = node;

    for (std::uint32_t slot = 0; slot < explicit_degree; ++slot) {
      DirectRoute route{};
      route.source = node_index;
      const std::uint32_t target_local = slot == 0u
                                             ? (local + 1u) % plan.node_count
                                             : (local + plan.node_count - 1u) % plan.node_count;
      route.target = plan.node_offset + target_local;
      route.delay = 1u + ((local + slot + plan.lineage) & 3u);
      route.conductance_q16 = kConductanceOneQ16;
      route.eligibility_context = kInvalidIndex;
      route.predicted_context = kInvalidIndex;
      route.implicit_family = kInvalidIndex;
      route.implicit_slot = kInvalidIndex;
      route.next_route =
          slot + 1u < node.route_count ? node.first_route + slot + 1u : kInvalidIndex;
      routes[node.first_route + slot] = route;
    }

    if (has_extra && local == 0u) {
      const std::uint32_t physical_slot = node.first_route + explicit_degree;
      const std::uint32_t logical_slot = plan.local_degree - 1u;
      DirectRoute route{};
      route.source = node_index;
      if (next_plan != plan_index) {
        route.target = plans[next_plan].node_offset;
        route.flags |= kRouteFlagLongTract;
        route.delay = 2u + ((plan.lineage ^ plans[next_plan].lineage) & 7u);
      } else {
        const std::uint32_t target_local =
            (plan.chord_stride + logical_slot * (1u + (plan.lineage & 3u))) % plan.node_count;
        route.target = plan.node_offset + target_local;
        route.delay = 1u + ((logical_slot + plan.lineage) & 3u);
      }
      route.conductance_q16 = kConductanceOneQ16;
      route.eligibility_context = kInvalidIndex;
      route.predicted_context = kInvalidIndex;
      route.implicit_family = kInvalidIndex;
      route.implicit_slot = kInvalidIndex;
      route.next_route = kInvalidIndex;
      routes[physical_slot] = route;
    }
  }
}

__global__ void attach_boundaries_kernel(DirectNode* nodes, const TerritoryPlan* plans,
                                         std::uint32_t plan_count, BodyManifestV0 body) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= body.binding_count)
    return;
  const BoundaryBinding binding = body.bindings[i];
  if (binding.seed_index >= plan_count || plans[binding.seed_index].active == 0u)
    return;
  const TerritoryPlan plan = plans[binding.seed_index];
  if (binding.local_node >= plan.node_count)
    return;
  const std::uint32_t node_index = plan.node_offset + binding.local_node;
  DirectNode& node = nodes[node_index];
  if ((binding.role_mask & static_cast<std::uint32_t>(BoundaryRole::sensor)) != 0u) {
    atomicOr(&node.flags, kNodeFlagSensor);
    if (node.output_channel == kInvalidIndex)
      node.output_channel = binding.channel;
  }
  if ((binding.role_mask & static_cast<std::uint32_t>(BoundaryRole::motor)) != 0u) {
    atomicOr(&node.flags, kNodeFlagMotor);
    node.output_channel = binding.channel;
    node.output_word = binding.word;
  }
  if ((binding.role_mask & static_cast<std::uint32_t>(BoundaryRole::world_consequence)) != 0u)
    atomicOr(&node.flags, kNodeFlagWorldConsequence);
}

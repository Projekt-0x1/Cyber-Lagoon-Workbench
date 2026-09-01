#include "hardware_native/direct_network_life_function.cuh"
#include "hardware_native/direct_adult_core_constants.cuh"
#include "hardware_native/direct_gamma_evidence_ladder.cuh"
#include "hardware_native/direct_network_genome_lowering.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "hardware_native/direct_network_recipe_abi.cuh"

#define DIRECT_NETWORK_HD __host__ __device__

namespace substrate::direct_network {
namespace {

using substrate::direct_network::recipe::content_root;
using substrate::direct_network::recipe::GenomeValidationError;
using substrate::direct_network::recipe::validate_genome;

constexpr std::uint32_t kRootChunkBytes = 4096u;

// The genesis compiler is extracted whole into the stage units below; this
// translation unit keeps composition: stage contracts, developmental laws,
// planning, materialization, resident AOT -- and every kernel launch site.
#include "direct_network_genesis_stage_abi.cuh"
#include "direct_network_developmental_placement_law.cuh"
#include "direct_network_territory_planning.cuh"
#include "direct_network_corridor_law.inl"
#include "direct_network_genesis_materialization.cuh"

__device__ bool construction_front_growth_opcode(RuleOpcode opcode) {
  return opcode == RuleOpcode::branch || opcode == RuleOpcode::extend ||
         opcode == RuleOpcode::repair || opcode == RuleOpcode::long_tract ||
         opcode == RuleOpcode::fuse;
}

// The only static-tissue seeding pass. It runs after Gamma materialization and
// before the birth root, so every later front is resident arena state.
__global__ void seed_construction_fronts_at_birth_kernel(DirectBrain brain) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t count = 0u;
  const std::uint64_t front_budget = brain.resource_ecology == nullptr
      ? 0u : (brain.resource_ecology->maintenance_scan_budget + 1u) / 2u;
  const std::uint32_t front_limit = front_budget < brain.construction_front_capacity
      ? static_cast<std::uint32_t>(front_budget) : brain.construction_front_capacity;
  for (std::uint32_t node_index = 0u; node_index < brain.node_count; ++node_index) {
    if (count >= front_limit) break;
    const DirectNode& node = brain.nodes[node_index];
    if ((node.flags & kNodeFlagConstructor) == 0u ||
        node.territory_index >= brain.recipe_range_count) continue;
    const ResidentRecipeRange range = brain.recipe_ranges[node.territory_index];
    for (std::uint32_t local = 0u; local < range.index_count; ++local) {
      const std::uint32_t index = range.index_offset + local;
      if (index >= brain.recipe_index_count) break;
      const std::uint32_t cell_index = brain.recipe_indices[index];
      if (cell_index >= brain.recipe_cell_count) continue;
      const ResidentRecipeCell& cell = brain.recipe_cells[cell_index];
      if (cell.rule_index >= brain.resident_rule_count) continue;
      const ResidentConstructorRule& rule = brain.resident_rules[cell.rule_index];
      if ((rule.flags & kRuleFlagPostBirthResident) == 0u ||
          !construction_front_growth_opcode(rule.opcode) || rule.begin_age > 0u ||
          (rule.end_age != 0u && rule.end_age < rule.begin_age) ||
          (node.chemotype & rule.require_mask) != rule.require_value) continue;
      const std::uint64_t generation = next_construction_front_generation(
          brain.construction_front_generation_by_node[node_index]);
      if (generation == 0u) continue;
      brain.construction_fronts[count++] = ResidentConstructionFront{
          node_index, cell_index, cell.rule_index, kInvalidIndex, generation,
          0u, 0u, node.territory_index, kConstructionFrontLive, 0u};
      brain.construction_front_generation_by_node[node_index] = generation;
      break;
    }
  }
  *brain.construction_front_count = count;
}

struct BrainRootHeader {
  Root256 arena_root;
  Root256 genome_root;
  Root256 territory_layout_root;
  Root256 body_root;
  Root256 environment_root;
  std::uint32_t node_count;
  std::uint32_t active_route_count;
  std::uint32_t route_capacity;
  std::uint32_t dense_block_count;
  std::uint32_t dense_weight_count;
  std::uint32_t boundary_port_count;
  std::uint32_t resident_field_count;
  std::uint32_t resident_field_range_count;
  std::uint32_t resident_field_index_count;
  std::uint32_t resident_rule_count;
  std::uint32_t resident_tract_delay_law_count;
  std::uint32_t recipe_cell_count;
  std::uint32_t recipe_edge_count;
  std::uint32_t recipe_range_count;
  std::uint32_t recipe_index_count;
  std::uint32_t territory_count;
  std::uint32_t territory_ancestry_count;
  std::uint32_t long_tract_count;
};

struct Scratch {
  GammaV1* gamma = nullptr;
  DirectBodyManifestV1* body = nullptr;
  DirectDevelopmentEnvironmentV1* environment = nullptr;
  TerritoryPlan* plans = nullptr;
  std::uint32_t* node_counts = nullptr;
  std::uint32_t* route_capacity_counts = nullptr;
  std::uint32_t* active_route_counts = nullptr;
  std::uint32_t* dense_block_counts = nullptr;
  std::uint32_t* dense_weight_counts = nullptr;
  std::uint32_t* node_offsets = nullptr;
  std::uint32_t* route_offsets = nullptr;
  std::uint32_t* active_route_offsets = nullptr;
  std::uint32_t* dense_block_offsets = nullptr;
  std::uint32_t* dense_weight_offsets = nullptr;
  std::uint32_t* in_degree = nullptr;
  CompileTotals* totals = nullptr;
  CompileDiagnostics* diagnostics = nullptr;
  void* scan_storage = nullptr;
  std::size_t scan_storage_bytes = 0u;
  Root256* hash_a = nullptr;
  Root256* hash_b = nullptr;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void release_scratch(Scratch& s) {
  cudaFree(s.hash_b);
  cudaFree(s.hash_a);
  cudaFree(s.scan_storage);
  cudaFree(s.diagnostics);
  cudaFree(s.totals);
  cudaFree(s.in_degree);
  cudaFree(s.dense_weight_offsets);
  cudaFree(s.dense_block_offsets);
  cudaFree(s.active_route_offsets);
  cudaFree(s.route_offsets);
  cudaFree(s.node_offsets);
  cudaFree(s.dense_weight_counts);
  cudaFree(s.dense_block_counts);
  cudaFree(s.active_route_counts);
  cudaFree(s.route_capacity_counts);
  cudaFree(s.node_counts);
  cudaFree(s.plans);
  cudaFree(s.environment);
  cudaFree(s.body);
  cudaFree(s.gamma);
  s = Scratch{};
}

std::uint32_t grid_for(std::uint32_t count, std::uint32_t block_size) {
  return std::max(1u, (count + block_size - 1u) / block_size);
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

Root256 canonical_direct_body_root_impl(const DirectBodyManifestV1& body) {
  const std::size_t bytes = offsetof(DirectBodyManifestV1, bindings) +
                            sizeof(BoundaryPortBinding) * body.binding_count;
  return content_root(&body, bytes);
}

Root256 canonical_direct_environment_root_impl(
    const DirectDevelopmentEnvironmentV1& environment) {
  const std::size_t bytes = offsetof(DirectDevelopmentEnvironmentV1, constraints) +
                            sizeof(DevelopmentEnvironmentConstraint) * environment.constraint_count;
  return content_root(&environment, bytes);
}

void validate_direct_inputs(const GammaV1& gamma, const DirectBodyManifestV1& body,
                            const DirectDevelopmentEnvironmentV1& environment) {
  if (gamma.header.development_end_tick == 0u) {
    throw std::invalid_argument("direct Gamma birth-handoff tick must be nonzero");
  }
  for (std::uint32_t i = 0; i < gamma.header.field_count; ++i) {
    const FieldBlock& field = gamma.fields[i];
    // polarity packs DevelopmentFieldKind in the low digit, the
    // independent_decay_timescale class (#1276/NET12) above it (field_decay_class),
    // the gain-participation bit (#1276/NET12 rung 3) above that
    // (field_gain_participates), and the spatial-gradient bit (#1289
    // b.local_fields) above that (field_gradient_participates), so the valid
    // range is the product of all four arities.
    if (field.polarity >= kDevelopmentFieldKindCount * kFieldDecayClassCount *
                               kFieldGainParticipationCount *
                               kFieldGradientParticipationCount ||
        field.end_tick < field.begin_tick) {
      throw std::invalid_argument("direct Gamma contains an invalid developmental field");
    }
  }
  const std::uint32_t allowed_roles = static_cast<std::uint32_t>(BoundaryRole::sensor) |
                                      static_cast<std::uint32_t>(BoundaryRole::motor) |
                                      static_cast<std::uint32_t>(BoundaryRole::world_return);
  if (environment.abi_version != kDirectDevelopmentEnvironmentAbiV1 ||
      environment.constraint_count > kMaxDevelopmentConstraints) {
    throw std::invalid_argument("development environment ABI/bounds invalid");
  }
  for (std::uint32_t i = 0; i < environment.constraint_count; ++i) {
    const DevelopmentEnvironmentConstraint& constraint = environment.constraints[i];
    const std::uint32_t allowed = kEnvironmentConstraintHardExclude |
                                  kEnvironmentConstraintSoftPenalty;
    if ((constraint.flags & ~allowed) != 0u) {
      throw std::invalid_argument("development environment contains invalid constraint flags");
    }
  }
  for (std::uint32_t i = 0; i < body.binding_count; ++i) {
    const BoundaryPortBinding& binding = body.bindings[i];
    if (binding.seed_index >= gamma.header.seed_count || binding.role_mask == 0u ||
        (binding.role_mask & ~allowed_roles) != 0u) {
      throw std::invalid_argument("body manifest contains an invalid physical boundary binding");
    }
  }
  if (direct_boundary_bindings_share_a_sensor_channel(body)) {
    throw std::invalid_argument(
        "body manifest contains two sensor boundary ports sharing one channel");
  }
}

DIRECT_NETWORK_HD inline Root256 hash_bytes_chunk(const unsigned char* bytes,
                                                  std::size_t size,
                                                  std::uint32_t chunk_index) {
  std::uint32_t lanes[8];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int lane = 0; lane < 8; ++lane) {
    lanes[lane] = 0x811c9dc5u ^ (static_cast<std::uint32_t>(lane) * 0x01000193u) ^
                  (chunk_index * 0x9e3779b9u);
  }
  for (std::size_t i = 0; i < size; ++i) {
    std::uint32_t& lane = lanes[i & 7u];
    lane ^= bytes[i];
    lane *= 0x01000193u;
  }
  Root256 out{};
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int lane = 0; lane < 8; ++lane) out.word[lane] = lanes[lane];
  return out;
}

__global__ void hash_arena_chunks_kernel(const unsigned char* arena, std::size_t arena_bytes,
                                         Root256* roots, std::uint32_t chunk_count) {
  const std::uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= chunk_count) return;
  const std::size_t begin = static_cast<std::size_t>(chunk) * kRootChunkBytes;
  const std::size_t remaining = arena_bytes - begin;
  const std::size_t size = remaining < kRootChunkBytes ? remaining : kRootChunkBytes;
  roots[chunk] = hash_bytes_chunk(arena + begin, size, chunk);
}

__global__ void reduce_roots_kernel(const Root256* input, Root256* output,
                                    std::uint32_t input_count) {
  const std::uint32_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t left_index = out_index * 2u;
  if (left_index >= input_count) return;
  const Root256 left = input[left_index];
  const Root256 right = left_index + 1u < input_count ? input[left_index + 1u] : Root256{};
  Root256 merged{};
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int lane = 0; lane < 8; ++lane) {
    std::uint32_t x = left.word[lane] ^ (right.word[(lane + 3) & 7] + 0x9e3779b9u);
    x ^= out_index * (0x85ebca6bu + static_cast<std::uint32_t>(lane) * 0x27d4eb2du);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    merged.word[lane] = x;
  }
  output[out_index] = merged;
}

Root256 hash_arena_device(const DirectBrain& brain, Scratch& scratch,
                          std::uint32_t block_size) {
  const std::uint32_t chunk_count = static_cast<std::uint32_t>(
      (brain.arena_bytes + kRootChunkBytes - 1u) / kRootChunkBytes);
  if (chunk_count == 0u) return Root256{};
  check_cuda(cudaMalloc(&scratch.hash_a, sizeof(Root256) * chunk_count), "allocate arena hash A");
  check_cuda(cudaMalloc(&scratch.hash_b, sizeof(Root256) * chunk_count), "allocate arena hash B");
  hash_arena_chunks_kernel<<<grid_for(chunk_count, block_size), block_size>>>(
      static_cast<const unsigned char*>(brain.arena), brain.arena_bytes, scratch.hash_a, chunk_count);
  check_cuda(cudaGetLastError(), "hash arena chunks");
  Root256* input = scratch.hash_a;
  Root256* output = scratch.hash_b;
  std::uint32_t count = chunk_count;
  while (count > 1u) {
    const std::uint32_t next_count = (count + 1u) / 2u;
    reduce_roots_kernel<<<grid_for(next_count, block_size), block_size>>>(input, output, count);
    check_cuda(cudaGetLastError(), "reduce arena roots");
    Root256* swap = input;
    input = output;
    output = swap;
    count = next_count;
  }
  Root256 root{};
  check_cuda(cudaMemcpy(&root, input, sizeof(root), cudaMemcpyDeviceToHost), "read arena root");
  return root;
}

#include "direct_network_resident_recipe_compilation.cuh"
#include "direct_network_arena_layout.inl"

}  // namespace

Root256 canonical_direct_body_root_v1(const DirectBodyManifestV1& body) {
  return canonical_direct_body_root_impl(body);
}

Root256 canonical_direct_environment_root_v1(
    const DirectDevelopmentEnvironmentV1& environment) {
  return canonical_direct_environment_root_impl(environment);
}

// github #1194/#1290's gap, closed. attach_boundary_port (above) copies
// `channel` opaquely and never compares it against any other binding; the
// sensory ingress kernel then resolves a channel-addressed contact by the
// FIRST port whose channel matches, so two sensor ports sharing a channel
// would be resolved by binding order rather than by any rule construction
// states. Scoped to the sensor role specifically: that is the only role the
// ingress kernel looks up by channel at all -- motor/world_return ports are
// addressed by port index, so a channel shared across roles is not this
// ambiguity.
bool direct_boundary_bindings_share_a_sensor_channel(const DirectBodyManifestV1& body) {
  constexpr std::uint32_t kSensorRole = static_cast<std::uint32_t>(BoundaryRole::sensor);
  for (std::uint32_t i = 0; i < body.binding_count; ++i) {
    if ((body.bindings[i].role_mask & kSensorRole) == 0u) continue;
    for (std::uint32_t j = i + 1u; j < body.binding_count; ++j) {
      if ((body.bindings[j].role_mask & kSensorRole) != 0u &&
          body.bindings[j].channel == body.bindings[i].channel) {
        return true;
      }
    }
  }
  return false;
}

// Gamma is intentionally private to this lowered implementation. The public
// Direct entry below owns the genome authority and creates this POD only after
// Direct territory planning has derived world-relative origins.
DirectBirthReceiptV1 compile_lowered_direct_brain(
    const GammaV1& gamma, const DirectBodyManifestV1& body,
    const DirectDevelopmentEnvironmentV1& environment,
    const Root256& direct_genome_root, const Root256& territory_layout_root,
    const DirectTractDelayLawV1* rule_delay_laws,
    std::uint32_t rule_delay_law_count,
    const DirectTerritoryIdentityV1* territory_identities,
    std::uint32_t territory_identity_count, DirectBrain* out_brain,
    DirectCompileOptions options) {
  if (out_brain == nullptr) throw std::invalid_argument("out_brain must not be null");
  // i.life_function_versioning: external evolution selects birth recipes, but
  // its authority ends at birth -- a second birth aimed at an already-born
  // continuing subject is refused instead of silently destroying it.
  if (out_brain->arena != nullptr)
    throw std::invalid_argument("refusing rebirth over a continuing Direct adult subject");
  *out_brain = DirectBrain{};
  if (validate_genome(gamma) != GenomeValidationError::kNone) {
    throw std::invalid_argument("Gamma is not a valid canonical network recipe");
  }
  if (body.abi_version != kDirectBodyAbiV1 || body.binding_count > kMaxBoundaryPorts) {
    throw std::invalid_argument("body manifest ABI/bounds invalid");
  }
  if (gamma.header.seed_count == 0u) throw std::invalid_argument("Gamma has no developmental seeds");
  if (rule_delay_law_count != 0u &&
      (rule_delay_laws == nullptr || rule_delay_law_count != gamma.header.rule_count))
    throw std::invalid_argument("Direct tract delay-law sidecar does not match rule ABI");
  if ((territory_identity_count == 0u) != (territory_identities == nullptr) ||
      (territory_identity_count != 0u && territory_identity_count != gamma.header.seed_count)) {
    throw std::invalid_argument("Direct territory identity sidecar does not match seed ABI");
  }
  validate_direct_inputs(gamma, body, environment);
  if (options.block_size < 64u || options.block_size > 1024u ||
      (options.block_size & (options.block_size - 1u)) != 0u) {
    throw std::invalid_argument("block_size must be a power of two in [64,1024]");
  }
  if (options.silicon_byte_budget == 0u || options.route_pool_capacity == 0u ||
      options.dense_tile_pool_capacity == 0u ||
      options.boundary_port_pool_capacity == 0u ||
      options.recipe_cell_pool_capacity == 0u ||
      options.recipe_edge_pool_capacity == 0u) {
    throw std::invalid_argument("Direct requires finite nonzero matter capacities");
  }
  options.candidate_targets = clamp_u32(options.candidate_targets, 2u, 32u);
  options.overload_refinement_passes = clamp_u32(options.overload_refinement_passes, 1u, 4u);
  options.prenatal_stabilization_passes = clamp_u32(options.prenatal_stabilization_passes, 1u, 8u);
  options.route_reserve_per_node = clamp_u32(options.route_reserve_per_node, 1u, 12u);
  preflight_direct_capacity(gamma, options.route_reserve_per_node);

  // Pay CUDA's one-time cost BEFORE the clock starts.
  //
  // Measured 2026-08-18 by cuda_direct_gestation_cost_attribution_contract:
  // compiling the identical genome twice inside one process gave cold
  // planning=190.492 ms and warm planning=0.999 ms, a ratio of 0.0052. Without
  // this line the first compile in a process charges lazy context creation and
  // module load -- work that has nothing to do with Gamma -- to `planning_ms`,
  // which is why the 65k receipt read `gestation_ms=855.557
  // planning_ms=853.299 materialization_ms=1.455`. Read literally that says
  // planning four territory plans costs 587x more than materializing 65,536
  // nodes and 524,544 routes, and it sends a reader to optimise the wrong
  // phase. Warm, that same brain gestates in 2.956 ms.
  //
  // cudaFree(nullptr) is the documented no-op that forces context
  // initialisation. It is idempotent and free once the context exists.
  (void)cudaFree(nullptr);

  const auto started = std::chrono::steady_clock::now();
  Scratch scratch{};
  DirectBrain brain{};
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create Life Function stream");

  try {
    const std::uint32_t seed_count = gamma.header.seed_count;
    const std::size_t seed_bytes = sizeof(std::uint32_t) * seed_count;
    check_cuda(cudaMalloc(&scratch.gamma, sizeof(GammaV1)), "allocate Gamma scratch");
    check_cuda(cudaMalloc(&scratch.body, sizeof(DirectBodyManifestV1)), "allocate body scratch");
    check_cuda(cudaMalloc(&scratch.environment, sizeof(DirectDevelopmentEnvironmentV1)),
               "allocate development environment scratch");
    check_cuda(cudaMalloc(&scratch.plans, sizeof(TerritoryPlan) * seed_count), "allocate plans");
    check_cuda(cudaMalloc(&scratch.node_counts, seed_bytes), "allocate node counts");
    check_cuda(cudaMalloc(&scratch.route_capacity_counts, seed_bytes), "allocate route counts");
    check_cuda(cudaMalloc(&scratch.active_route_counts, seed_bytes), "allocate active route counts");
    check_cuda(cudaMalloc(&scratch.dense_block_counts, seed_bytes), "allocate dense block counts");
    check_cuda(cudaMalloc(&scratch.dense_weight_counts, seed_bytes), "allocate dense weight counts");
    check_cuda(cudaMalloc(&scratch.node_offsets, seed_bytes), "allocate node offsets");
    check_cuda(cudaMalloc(&scratch.route_offsets, seed_bytes), "allocate route offsets");
    check_cuda(cudaMalloc(&scratch.active_route_offsets, seed_bytes), "allocate active offsets");
    check_cuda(cudaMalloc(&scratch.dense_block_offsets, seed_bytes), "allocate dense block offsets");
    check_cuda(cudaMalloc(&scratch.dense_weight_offsets, seed_bytes), "allocate dense weight offsets");
    check_cuda(cudaMalloc(&scratch.totals, sizeof(CompileTotals)), "allocate compile totals");
    check_cuda(cudaMalloc(&scratch.diagnostics, sizeof(CompileDiagnostics)), "allocate diagnostics");
    clear_compile_scratch_kernel<<<grid_for(seed_count, options.block_size), options.block_size, 0,
                                   stream>>>(scratch.diagnostics, scratch.dense_block_counts,
                                             scratch.dense_weight_counts, seed_count);
    check_cuda(cudaGetLastError(), "clear compile scratch state");
    check_cuda(cudaMemcpyAsync(scratch.gamma, &gamma, sizeof(GammaV1), cudaMemcpyHostToDevice, stream),
               "copy Gamma");
    check_cuda(cudaMemcpyAsync(scratch.body, &body, sizeof(DirectBodyManifestV1), cudaMemcpyHostToDevice,
                               stream),
               "copy body manifest");
    check_cuda(cudaMemcpyAsync(scratch.environment, &environment,
                               sizeof(DirectDevelopmentEnvironmentV1), cudaMemcpyHostToDevice, stream),
               "copy development environment");

    plan_territories_kernel<<<grid_for(seed_count, options.block_size), options.block_size, 0, stream>>>(
        scratch.gamma, scratch.plans, options.route_reserve_per_node, scratch.node_counts,
        scratch.route_capacity_counts, scratch.active_route_counts, scratch.dense_block_counts,
        scratch.dense_weight_counts);
    check_cuda(cudaGetLastError(), "plan territories");

    std::size_t scan_bytes = 0u;
    std::size_t bytes = 0u;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, scratch.node_counts, scratch.node_offsets,
                                  seed_count, stream);
    scan_bytes = std::max(scan_bytes, bytes);
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, scratch.route_capacity_counts, scratch.route_offsets,
                                  seed_count, stream);
    scan_bytes = std::max(scan_bytes, bytes);
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, scratch.active_route_counts,
                                  scratch.active_route_offsets, seed_count, stream);
    scan_bytes = std::max(scan_bytes, bytes);
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, scratch.dense_block_counts,
                                  scratch.dense_block_offsets, seed_count, stream);
    scan_bytes = std::max(scan_bytes, bytes);
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, scratch.dense_weight_counts,
                                  scratch.dense_weight_offsets, seed_count, stream);
    scan_bytes = std::max(scan_bytes, bytes);
    scratch.scan_storage_bytes = scan_bytes;
    check_cuda(cudaMalloc(&scratch.scan_storage, scan_bytes), "allocate compiler scan storage");

    cub::DeviceScan::ExclusiveSum(scratch.scan_storage, scan_bytes, scratch.node_counts,
                                  scratch.node_offsets, seed_count, stream);
    cub::DeviceScan::ExclusiveSum(scratch.scan_storage, scan_bytes, scratch.route_capacity_counts,
                                  scratch.route_offsets, seed_count, stream);
    cub::DeviceScan::ExclusiveSum(scratch.scan_storage, scan_bytes, scratch.active_route_counts,
                                  scratch.active_route_offsets, seed_count, stream);
    cub::DeviceScan::ExclusiveSum(scratch.scan_storage, scan_bytes, scratch.dense_block_counts,
                                  scratch.dense_block_offsets, seed_count, stream);
    cub::DeviceScan::ExclusiveSum(scratch.scan_storage, scan_bytes, scratch.dense_weight_counts,
                                  scratch.dense_weight_offsets, seed_count, stream);

    install_plan_offsets_kernel<<<grid_for(seed_count, options.block_size), options.block_size, 0, stream>>>(
        scratch.plans, scratch.node_offsets, scratch.route_offsets, scratch.active_route_offsets,
        scratch.dense_block_offsets, scratch.dense_weight_offsets, seed_count);
    write_compile_totals_kernel<<<1, options.block_size, 0, stream>>>(
        scratch.plans, scratch.node_offsets, scratch.node_counts, scratch.route_offsets,
        scratch.route_capacity_counts, scratch.active_route_offsets, scratch.active_route_counts,
        scratch.dense_block_offsets, scratch.dense_block_counts, scratch.dense_weight_offsets,
        scratch.dense_weight_counts, seed_count, scratch.totals);
    check_cuda(cudaGetLastError(), "write compiler totals");

    CompileTotals totals{};
    check_cuda(cudaMemcpyAsync(&totals, scratch.totals, sizeof(totals), cudaMemcpyDeviceToHost, stream),
               "read compiler totals");
    check_cuda(cudaStreamSynchronize(stream), "planning barrier");
    const auto planned = std::chrono::steady_clock::now();

    std::vector<ResidentTerritoryAncestry> territory_ancestry;
    if (territory_identity_count != 0u) {
      std::vector<TerritoryPlan> host_plans(seed_count);
      check_cuda(cudaMemcpy(host_plans.data(), scratch.plans,
                            sizeof(TerritoryPlan) * seed_count, cudaMemcpyDeviceToHost),
                 "read territory ancestry plans");
      territory_ancestry.resize(seed_count);
      for (std::uint32_t i = 0u; i < seed_count; ++i) {
        if (territory_identities[i].lineage != gamma.seeds[i].lineage) {
          throw std::invalid_argument("Direct territory identity lineage does not match lowered seed");
        }
        ResidentTerritoryAncestry ancestry{};
        ancestry.lineage = territory_identities[i].lineage;
        ancestry.axis = territory_identities[i].axis;
        ancestry.ordinal = territory_identities[i].ordinal;
        ancestry.founder_origin[0] = static_cast<std::int32_t>(gamma.seeds[i].coordinate[0]);
        ancestry.founder_origin[1] = static_cast<std::int32_t>(gamma.seeds[i].coordinate[1]);
        ancestry.founder_origin[2] = static_cast<std::int32_t>(gamma.seeds[i].coordinate[2]);
        ancestry.node_begin = host_plans[i].node_offset;
        ancestry.node_count = host_plans[i].node_count;
        ancestry.prenatal_begin_tick = gamma.seeds[i].begin_tick;
        territory_ancestry[i] = ancestry;
      }
    }

    if (totals.territory_count == 0u || totals.node_count == 0u || totals.route_capacity == 0u) {
      throw std::runtime_error("Gamma produced no recurrent mathematical territory");
    }
    const std::uint64_t logical_matter = static_cast<std::uint64_t>(totals.node_count) +
                                         totals.active_route_estimate + totals.dense_weight_count;
    if (logical_matter > gamma.header.matter_budget) {
      throw std::runtime_error("Gamma mathematical network exceeds finite matter budget");
    }

    const std::vector<ResidentDevelopmentField> resident_fields = compile_resident_fields(gamma);
    const std::vector<ResidentConstructorRule> resident_rules = compile_resident_rules(gamma);
    const ResidentRecipeBuild recipe_network = compile_resident_recipe_network(gamma, resident_rules);
    const ResidentFieldBindingBuild field_bindings =
        compile_resident_field_bindings(gamma, resident_rules, recipe_network);
    const ArenaLayout layout = build_arena_layout(
        totals, body.binding_count, static_cast<std::uint32_t>(territory_ancestry.size()),
        static_cast<std::uint32_t>(resident_fields.size()),
        static_cast<std::uint32_t>(resident_rules.size()),
        static_cast<std::uint32_t>(field_bindings.ranges.size()),
        static_cast<std::uint32_t>(field_bindings.indices.size()),
        rule_delay_law_count,
        static_cast<std::uint32_t>(recipe_network.cells.size()),
        static_cast<std::uint32_t>(recipe_network.edges.size()),
        static_cast<std::uint32_t>(recipe_network.ranges.size()),
        static_cast<std::uint32_t>(recipe_network.indices.size()));
    const std::uint64_t born_bytes =
        static_cast<std::uint64_t>(layout.total) +
        sizeof(substrate::direct_adult::DirectResourceEcologyState);
    using substrate::direct_adult::DirectResourcePoolKind;
    using substrate::direct_adult::DirectResourcePoolState;
    substrate::direct_adult::DirectResourceEcologyState grown_ecology{};
    grown_ecology.maintenance_scan_budget = 64u;
    grown_ecology.eligibility_scan_bound = 256u;
    grown_ecology.churn.mutation_budget = 512u;
    grown_ecology.policy_revision = 1u;

    if (born_bytes > options.silicon_byte_budget) {
      throw std::runtime_error(
          "born Direct exceeds the declared silicon budget: needs " +
          std::to_string(born_bytes) + " bytes, budget is " +
          std::to_string(options.silicon_byte_budget) + " bytes");
    }
    if (static_cast<std::uint64_t>(totals.route_capacity) > options.route_pool_capacity) {
      throw std::runtime_error(
          "grown brain route pool exceeds its declared capacity: plan needs " +
          std::to_string(totals.route_capacity) + " slots, capacity is " +
          std::to_string(options.route_pool_capacity) + " slots");
    }
    if (static_cast<std::uint64_t>(totals.dense_block_count) >
        options.dense_tile_pool_capacity) {
      throw std::runtime_error(
          "grown brain dense tile pool exceeds its declared capacity: plan needs " +
          std::to_string(totals.dense_block_count) + " tiles, capacity is " +
          std::to_string(options.dense_tile_pool_capacity) + " tiles");
    }
    if (static_cast<std::uint64_t>(body.binding_count) >
        options.boundary_port_pool_capacity) {
      throw std::runtime_error(
          "grown brain boundary port pool exceeds its declared capacity: body asks for " +
          std::to_string(body.binding_count) + " ports, capacity is " +
          std::to_string(options.boundary_port_pool_capacity) + " ports");
    }
    if (recipe_network.cells.size() > options.recipe_cell_pool_capacity) {
      throw std::runtime_error(
          "grown brain recipe cell pool exceeds its declared capacity: plan needs " +
          std::to_string(recipe_network.cells.size()) + " cells, capacity is " +
          std::to_string(options.recipe_cell_pool_capacity) + " cells");
    }
    if (recipe_network.edges.size() > options.recipe_edge_pool_capacity) {
      throw std::runtime_error(
          "grown brain recipe edge pool exceeds its declared capacity: plan needs " +
          std::to_string(recipe_network.edges.size()) + " edges, capacity is " +
          std::to_string(options.recipe_edge_pool_capacity) + " edges");
    }

    // Book the reservations for matter the plan is about to create. Note what
    // reserved-versus-live means here, because the grown brain makes the
    // distinction physically: every node pre-pays `route_capacity` route slots as
    // dormant reserve, and only `sparse_degree` of them are activated at birth.
    // So the arena's route slots are RESERVED at plan time and each activated
    // slot is COMMITTED by the materialization kernel that writes its target.
    // The remainder stays reserved -- that is not slack in the accounting, it is
    // the developmental reserve ResidentDevelopment later activates.
    DirectResourcePoolState& grown_nodes =
        grown_ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::node_state)];
    grown_nodes.capacity_units = totals.node_count;
    grown_nodes.live_units = totals.node_count;
    grown_nodes.charged_units = totals.node_count;
    grown_nodes.high_water_units = totals.node_count;
    grown_nodes.bytes_per_unit = sizeof(DirectNode);

    DirectResourcePoolState& grown_routes =
        grown_ecology
            .pools[static_cast<std::uint32_t>(DirectResourcePoolKind::explicit_interaction)];
    grown_routes.capacity_units = options.route_pool_capacity;
    grown_routes.reserved_units = totals.route_capacity;
    grown_routes.charged_units = totals.route_capacity;
    grown_routes.high_water_units = totals.route_capacity;
    grown_routes.bytes_per_unit = sizeof(DirectRoute) + sizeof(std::uint64_t) * 2u +
        (rule_delay_law_count == 0u
             ? 0u
             : sizeof(std::uint32_t) * 2u + sizeof(std::uint64_t));

    DirectResourcePoolState& grown_dense =
        grown_ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::dense_tile)];
    grown_dense.capacity_units = options.dense_tile_pool_capacity;
    grown_dense.reserved_units = totals.dense_block_count;
    grown_dense.charged_units = totals.dense_block_count;
    grown_dense.high_water_units = totals.dense_block_count;
    grown_dense.bytes_per_unit = sizeof(DirectDenseBlock);

    // Boundary ports: reserve one per binding the body asks for. Only the
    // bindings that actually attach are committed by the attachment kernel, so
    // the reserved remainder ends up equal to the diagnostics' invalid-binding
    // count -- the ledger and the compiler's own error counter have to agree.
    DirectResourcePoolState& grown_ports =
        grown_ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::boundary_port)];
    grown_ports.capacity_units = options.boundary_port_pool_capacity;
    grown_ports.reserved_units = body.binding_count;
    grown_ports.charged_units = body.binding_count;
    grown_ports.high_water_units = body.binding_count;
    grown_ports.bytes_per_unit = sizeof(DirectBoundaryPort);

    // The resident recipe network is the grown brain's constructor/derivation
    // record: cells and the edges between them. Unlike routes and ports it is
    // compiled on the host and uploaded whole, so reserve and commit collapse
    // into one step here -- but the contract still censuses the uploaded arrays
    // on device, so a truncated upload shows up as live > census rather than
    // passing because the counter agrees with itself.
    DirectResourcePoolState& grown_cells =
        grown_ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::derivation_record)];
    const std::uint64_t cell_headroom = options.recipe_cell_pool_capacity > recipe_network.cells.size()
        ? options.recipe_cell_pool_capacity - recipe_network.cells.size() : 0u;
    const std::uint64_t edge_headroom = options.recipe_edge_pool_capacity > recipe_network.edges.size()
        ? options.recipe_edge_pool_capacity - recipe_network.edges.size() : 0u;
    const std::uint64_t postbirth_reserve = std::min<std::uint64_t>(
        kResidentPostbirthRecipeReserve, std::min(cell_headroom, edge_headroom));
    const std::uint64_t resident_recipe_capacity = recipe_network.cells.size() + postbirth_reserve;
    grown_cells.capacity_units = resident_recipe_capacity;
    grown_cells.live_units = recipe_network.cells.size();
    grown_cells.reserved_units = postbirth_reserve;
    grown_cells.charged_units = resident_recipe_capacity;
    grown_cells.high_water_units = resident_recipe_capacity;
    grown_cells.bytes_per_unit = sizeof(ResidentRecipeCell) + sizeof(ResidentRecipeDerivation);

    DirectResourcePoolState& grown_edges =
        grown_ecology
            .pools[static_cast<std::uint32_t>(DirectResourcePoolKind::derivation_parent_edge)];
    grown_edges.capacity_units = recipe_network.edges.size() + postbirth_reserve;
    grown_edges.live_units = recipe_network.edges.size();
    grown_edges.reserved_units = postbirth_reserve;
    grown_edges.charged_units = grown_edges.capacity_units;
    grown_edges.high_water_units = grown_edges.capacity_units;
    grown_edges.bytes_per_unit = sizeof(ResidentRecipeEdge);

    // Adult runtime working-set pools share their capacity and byte ABI with
    // the runtime. A capacity change cannot leave birth on stale literals.
    using namespace substrate::direct_adult_core;
    DirectResourcePoolState& grown_eligibility =
        grown_ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::eligibility_record)];
    grown_eligibility.capacity_units = kAdultEligibilityResourceCapacity;
    grown_eligibility.live_units = 0u;
    grown_eligibility.reserved_units = 0u;
    grown_eligibility.charged_units = 0u;
    grown_eligibility.high_water_units = 0u;
    grown_eligibility.bytes_per_unit = kAdultEligibilityResourceBytesPerUnit;

    DirectResourcePoolState& grown_packets =
        grown_ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::delayed_packet)];
    grown_packets.capacity_units = kAdultPacketResourceCapacity;
    grown_packets.live_units = 0u;
    grown_packets.reserved_units = 0u;
    grown_packets.charged_units = 0u;
    grown_packets.high_water_units = 0u;
    grown_packets.bytes_per_unit = kAdultPacketResourceBytesPerUnit;

    DirectResourcePoolState& grown_tickets =
        grown_ecology.pools[static_cast<std::uint32_t>(DirectResourcePoolKind::pending_consequence_ticket)];
    grown_tickets.capacity_units = kAdultTicketResourceCapacity;
    grown_tickets.live_units = 0u;
    grown_tickets.reserved_units = 0u;
    grown_tickets.charged_units = 0u;
    grown_tickets.high_water_units = 0u;
    grown_tickets.bytes_per_unit = kAdultTicketResourceBytesPerUnit;

    grown_ecology.global_capacity_bytes = options.silicon_byte_budget;
    grown_ecology.global_charged_bytes = born_bytes;
    grown_ecology.global_high_water_bytes = born_bytes;

    check_cuda(cudaMalloc(&brain.arena, layout.total), "allocate born brain arena");
    brain.arena_bytes = layout.total;

    // The grown brain carries its own ledger. The adult's cannot reach here --
    // the two stacks share no type and no bridge -- so wiring the grown
    // allocators to the adult's ledger was never an option; giving the grown
    // organism one of its own is.
    check_cuda(cudaMalloc(&brain.resource_ecology,
                          sizeof(substrate::direct_adult::DirectResourceEcologyState)),
               "allocate grown brain resource ecology");
    check_cuda(cudaMemcpy(brain.resource_ecology, &grown_ecology, sizeof(grown_ecology),
                          cudaMemcpyHostToDevice),
               "initialize grown brain resource ecology");

    assign_arena_views(brain, layout);
    check_cuda(cudaMemsetAsync(brain.arena, 0, layout.total, stream), "zero born brain arena");

    brain.node_count = totals.node_count;
    brain.route_capacity = totals.route_capacity;
    brain.dense_block_count = totals.dense_block_count;
    brain.dense_weight_count = totals.dense_weight_count;
    brain.boundary_port_count = body.binding_count;
    brain.resident_field_count = static_cast<std::uint32_t>(resident_fields.size());
    brain.resident_field_range_count =
        static_cast<std::uint32_t>(field_bindings.ranges.size());
    brain.resident_field_index_count =
        static_cast<std::uint32_t>(field_bindings.indices.size());
    brain.resident_rule_count = static_cast<std::uint32_t>(resident_rules.size());
    brain.resident_tract_delay_law_count = rule_delay_law_count;
    brain.recipe_cell_count = static_cast<std::uint32_t>(recipe_network.cells.size());
    brain.recipe_cell_capacity = brain.recipe_cell_count + static_cast<std::uint32_t>(postbirth_reserve);
    brain.recipe_edge_count = static_cast<std::uint32_t>(recipe_network.edges.size());
    brain.recipe_range_count = static_cast<std::uint32_t>(recipe_network.ranges.size());
    brain.recipe_index_count = static_cast<std::uint32_t>(recipe_network.indices.size());
    brain.territory_count = totals.territory_count;
    brain.territory_ancestry_count = static_cast<std::uint32_t>(territory_ancestry.size());
    brain.construction_front_capacity = totals.node_count;
    brain.genome_root = direct_genome_root;
    brain.territory_layout_root = territory_layout_root;
    brain.body_root = canonical_direct_body_root_v1(body);
    brain.environment_root = canonical_direct_environment_root_v1(environment);

    if (!territory_ancestry.empty()) {
      check_cuda(cudaMemcpyAsync(brain.territory_ancestry, territory_ancestry.data(),
                                 territory_ancestry.size() * sizeof(ResidentTerritoryAncestry),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident territory ancestry");
    }
    if (!resident_fields.empty()) {
      check_cuda(cudaMemcpyAsync(brain.resident_fields, resident_fields.data(),
                                 resident_fields.size() * sizeof(ResidentDevelopmentField),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident developmental fields");
    }
    if (!field_bindings.ranges.empty()) {
      check_cuda(cudaMemcpyAsync(brain.resident_field_ranges, field_bindings.ranges.data(),
                                 field_bindings.ranges.size() * sizeof(ResidentFieldRange),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident field ranges");
    }
    if (!field_bindings.indices.empty()) {
      check_cuda(cudaMemcpyAsync(brain.resident_field_indices, field_bindings.indices.data(),
                                 field_bindings.indices.size() * sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident field indices");
    }
    if (!resident_rules.empty()) {
      check_cuda(cudaMemcpyAsync(brain.resident_rules, resident_rules.data(),
                                 resident_rules.size() * sizeof(ResidentConstructorRule),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident constructor rules");
    }
    if (rule_delay_law_count != 0u) {
      check_cuda(cudaMemcpyAsync(brain.resident_tract_delay_laws, rule_delay_laws,
                                 sizeof(DirectTractDelayLawV1) * rule_delay_law_count,
                                 cudaMemcpyHostToDevice, stream),
                 "install resident tract delay laws");
    }
    if (!recipe_network.cells.empty()) {
      check_cuda(cudaMemcpyAsync(brain.recipe_cells, recipe_network.cells.data(),
                                 recipe_network.cells.size() * sizeof(ResidentRecipeCell),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident recipe cells");
    }
    if (!recipe_network.edges.empty()) {
      check_cuda(cudaMemcpyAsync(brain.recipe_edges, recipe_network.edges.data(),
                                 recipe_network.edges.size() * sizeof(ResidentRecipeEdge),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident recipe edges");
    }
    if (!recipe_network.ranges.empty()) {
      check_cuda(cudaMemcpyAsync(brain.recipe_ranges, recipe_network.ranges.data(),
                                 recipe_network.ranges.size() * sizeof(ResidentRecipeRange),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident recipe ranges");
    }
    if (!recipe_network.indices.empty()) {
      check_cuda(cudaMemcpyAsync(brain.recipe_indices, recipe_network.indices.data(),
                                 recipe_network.indices.size() * sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice, stream),
                 "install resident recipe indices");
    }

    const std::uint64_t used_matter = static_cast<std::uint64_t>(totals.node_count) +
                                      totals.active_route_estimate + totals.dense_weight_count;
    ResidentDevelopmentState development{};
    development.age_tick = 0u;
    development.phase = 0u;
    development.plasticity_q16 = kQ16One;
    development.mature_plasticity_floor_q16 = kQ16One / 4u;
    development.critical_period_q16 = kQ16One;
    development.inhibition_gain_q16 = kQ16One / 2u;
    development.constructor_reserve = gamma.header.matter_budget > used_matter
                                          ? gamma.header.matter_budget - used_matter
                                          : 0u;
    development.live_node_matter = totals.node_count;
    development.live_route_matter = totals.active_route_estimate;
    development.field_count = brain.resident_field_count;
    development.constructor_rule_count = brain.resident_rule_count;
    development.recipe_cell_count = brain.recipe_cell_count;
    development.recipe_edge_count = brain.recipe_edge_count;
    development.birth_handoff_tick = gamma.header.development_end_tick;
    check_cuda(cudaMemcpyAsync(brain.development, &development, sizeof(development),
                               cudaMemcpyHostToDevice, stream),
               "install resident development state");
    ResidentPostbirthConstructorState postbirth{};
    postbirth.derivation_capacity = static_cast<std::uint32_t>(postbirth_reserve);
    postbirth.recipe_cell_capacity = brain.recipe_cell_capacity;
    postbirth.port_capacity = static_cast<std::uint32_t>(postbirth_reserve) * 2u;
    postbirth.relation_capacity = static_cast<std::uint32_t>(postbirth_reserve);
    postbirth.parameter_capacity = static_cast<std::uint32_t>(postbirth_reserve);
    check_cuda(cudaMemcpyAsync(brain.postbirth_constructor, &postbirth, sizeof(postbirth),
                               cudaMemcpyHostToDevice, stream),
               "install bounded postbirth constructor ecology");

    materialize_nodes_kernel<<<grid_for(totals.node_count, options.block_size), options.block_size, 0,
                               stream>>>(scratch.gamma, scratch.environment, scratch.plans,
                                         seed_count, brain.nodes, totals.node_count,
                                         scratch.diagnostics);
    materialize_sparse_routes_kernel<<<grid_for(totals.node_count, options.block_size),
                                       options.block_size, 0, stream>>>(
        scratch.gamma, scratch.plans, seed_count, brain.nodes, brain.routes,
        brain.route_incarnations, brain.route_delay_law_indices,
        brain.route_mature_delays, brain.route_delay_law_incarnations,
        reinterpret_cast<const DirectTractDelayLawV1*>(brain.resident_tract_delay_laws),
        brain.resident_tract_delay_law_count,
        totals.node_count, options.candidate_targets, scratch.diagnostics,
        brain.resource_ecology);
    if (totals.dense_block_count != 0u) {
      materialize_dense_blocks_kernel<<<seed_count, options.block_size, 0, stream>>>(
          scratch.plans, seed_count, brain.dense_blocks, brain.dense_weight_fp16_bits,
          gamma.header.development_seed, brain.resource_ecology);
    }
    if (body.binding_count != 0u) {
      attach_boundary_ports_kernel<<<grid_for(body.binding_count, options.block_size),
                                     options.block_size, 0, stream>>>(
          scratch.body, scratch.plans, seed_count, brain.nodes, brain.boundary_ports,
          scratch.diagnostics, brain.resource_ecology);
    }
    check_cuda(cudaGetLastError(), "materialize mathematical brain");

    check_cuda(cudaMalloc(&scratch.in_degree, sizeof(std::uint32_t) * totals.node_count),
               "allocate in-degree census");
    for (std::uint32_t pass = 0; pass < options.overload_refinement_passes; ++pass) {
      clear_u32_kernel<<<grid_for(totals.node_count, options.block_size), options.block_size, 0, stream>>>(
          scratch.in_degree, totals.node_count);
      count_in_degree_kernel<<<grid_for(totals.route_capacity, options.block_size), options.block_size,
                               0, stream>>>(brain.routes, totals.route_capacity, totals.node_count,
                                            scratch.in_degree);
      refine_overloaded_targets_kernel<<<grid_for(totals.node_count, options.block_size),
                                         options.block_size, 0, stream>>>(
          scratch.gamma, scratch.plans, seed_count, brain.nodes, brain.routes, totals.node_count,
          scratch.in_degree, options.maximum_in_degree, options.candidate_targets);
    }

    for (std::uint32_t pass = 0; pass < options.prenatal_stabilization_passes; ++pass) {
      clear_u32_kernel<<<grid_for(totals.node_count, options.block_size), options.block_size, 0, stream>>>(
          scratch.in_degree, totals.node_count);
      count_in_degree_kernel<<<grid_for(totals.route_capacity, options.block_size), options.block_size,
                               0, stream>>>(brain.routes, totals.route_capacity, totals.node_count,
                                            scratch.in_degree);
      prenatal_stabilize_kernel<<<grid_for(totals.node_count, options.block_size), options.block_size,
                                  0, stream>>>(brain.nodes, brain.routes,
                                               brain.route_incarnations, totals.node_count,
                                               scratch.in_degree, options.maximum_in_degree, pass,
                                               brain.resource_ecology, scratch.diagnostics);
    }
    clear_u32_kernel<<<grid_for(totals.node_count, options.block_size), options.block_size, 0, stream>>>(
        scratch.in_degree, totals.node_count);
    count_in_degree_kernel<<<grid_for(totals.route_capacity, options.block_size), options.block_size,
                             0, stream>>>(brain.routes, totals.route_capacity, totals.node_count,
                                          scratch.in_degree);
    install_active_in_degree_kernel<<<grid_for(totals.node_count, options.block_size),
                                      options.block_size, 0, stream>>>(
        brain.nodes, scratch.in_degree, totals.node_count);
    // Last write before the birth root is hashed: everything above is
    // developmental, everything after reads adult state.
    birth_handoff_kernel<<<grid_for(totals.node_count, options.block_size), options.block_size, 0,
                           stream>>>(brain.nodes, totals.node_count);
    seed_construction_fronts_at_birth_kernel<<<1, 1, 0, stream>>>(brain);
    census_kernel<<<grid_for(std::max(totals.node_count, totals.route_capacity), options.block_size),
                    options.block_size, 0, stream>>>(brain.routes, totals.route_capacity,
                                                      scratch.in_degree, totals.node_count,
                                                      scratch.diagnostics);
    check_cuda(cudaGetLastError(), "stabilize/census mathematical brain");
    check_cuda(cudaStreamSynchronize(stream), "finish mathematical gestation");
    const auto materialized = std::chrono::steady_clock::now();

    CompileDiagnostics diagnostics{};
    check_cuda(cudaMemcpy(&diagnostics, scratch.diagnostics, sizeof(diagnostics),
                          cudaMemcpyDeviceToHost),
               "read compile diagnostics");
    if (diagnostics.invalid_boundary_bindings != 0u) {
      throw std::runtime_error("body manifest references a non-existent grown node");
    }
    brain.active_route_count = diagnostics.active_routes;
    brain.long_tract_count = diagnostics.long_tracts;

    // Hash after all born state is final, before any resident post-birth epoch.
    const Root256 arena_root = hash_arena_device(brain, scratch, options.block_size);
    BrainRootHeader root_header{arena_root,
                               brain.genome_root,
                               brain.territory_layout_root,
                               brain.body_root,
                               brain.environment_root,
                               brain.node_count,
                               brain.active_route_count,
                               brain.route_capacity,
                               brain.dense_block_count,
                               brain.dense_weight_count,
                               brain.boundary_port_count,
                               brain.resident_field_count,
                               brain.resident_field_range_count,
                               brain.resident_field_index_count,
                               brain.resident_rule_count,
                               brain.resident_tract_delay_law_count,
                               brain.recipe_cell_count,
                               brain.recipe_edge_count,
                               brain.recipe_range_count,
                               brain.recipe_index_count,
                               brain.territory_count,
                               brain.territory_ancestry_count,
                               brain.long_tract_count};
    brain.birth_root = content_root(&root_header, sizeof(root_header));

    // Scratch is the external Life Function.  Destroy it before publishing
    // the result/receipt; only compiled resident fields/rules remain.
    release_scratch(scratch);
    check_cuda(cudaStreamDestroy(stream), "destroy Life Function stream");
    stream = nullptr;
    *out_brain = brain;

    const auto ended = std::chrono::steady_clock::now();
    const std::uint64_t logical_recipe_bytes = sizeof(gamma.header) +
        sizeof(gamma.seeds[0]) * gamma.header.seed_count +
        sizeof(gamma.fields[0]) * gamma.header.field_count +
        sizeof(gamma.rules[0]) * gamma.header.rule_count +
        sizeof(DirectTerritoryIdentityV1) * territory_identity_count +
        sizeof(DirectTractDelayLawV1) * rule_delay_law_count;
    DirectBirthReceiptV1 receipt{};
    receipt.genome_root = brain.genome_root;
    receipt.territory_layout_root = brain.territory_layout_root;
    receipt.body_root = brain.body_root;
    receipt.environment_root = brain.environment_root;
    receipt.birth_root = brain.birth_root;
    receipt.node_count = brain.node_count;
    receipt.active_route_count = brain.active_route_count;
    receipt.route_capacity = brain.route_capacity;
    receipt.territory_count = brain.territory_count;
    receipt.dense_block_count = brain.dense_block_count;
    receipt.dense_weight_count = brain.dense_weight_count;
    receipt.long_tract_count = brain.long_tract_count;
    receipt.resident_field_count = brain.resident_field_count;
    receipt.resident_rule_count = brain.resident_rule_count;
    receipt.recipe_cell_count = brain.recipe_cell_count;
    receipt.recipe_edge_count = brain.recipe_edge_count;
    receipt.recipe_range_count = brain.recipe_range_count;
    receipt.recipe_index_count = brain.recipe_index_count;
    receipt.maximum_observed_in_degree = diagnostics.maximum_in_degree;
    receipt.fallback_wired_route_count = diagnostics.fallback_wired_routes;
    receipt.environment_violating_node_count = diagnostics.environment_violating_nodes;
    receipt.extended_draw_node_count = diagnostics.extended_draw_nodes;
    receipt.refused_partner_tract_count = diagnostics.refused_partner_tracts;
    receipt.partner_steered_tract_count = diagnostics.partner_steered_tracts;
    receipt.immature_deferred_tract_count = diagnostics.immature_deferred_tracts;
    receipt.arena_bytes = brain.arena_bytes;
    receipt.logical_recipe_bytes = logical_recipe_bytes;
    receipt.planning_ms = std::chrono::duration<float, std::milli>(planned - started).count();
    receipt.materialization_ms =
        std::chrono::duration<float, std::milli>(materialized - planned).count();
    receipt.total_gestation_ms = std::chrono::duration<float, std::milli>(ended - started).count();
    receipt.compact_recipe = logical_recipe_bytes < brain.arena_bytes;
    receipt.final_connectome_loaded = false;
    receipt.external_life_function_detached = true;
    receipt.resident_development_present = brain.development != nullptr;
    return receipt;
  } catch (...) {
    if (brain.resource_ecology != nullptr) cudaFree(brain.resource_ecology);
    if (brain.arena != nullptr) cudaFree(brain.arena);
    release_scratch(scratch);
    if (stream != nullptr) cudaStreamDestroy(stream);
    throw;
  }
}

bool apply_observer_prose_bytes_to_direct_compile_inputs(
    DirectGenomeV1* genome, DirectBodyManifestV1* body,
    DirectDevelopmentEnvironmentV1* environment, const void* bytes,
    std::uint64_t byte_count) {
  (void)genome;
  (void)body;
  (void)environment;
  (void)bytes;
  (void)byte_count;
  return false;
}

bool apply_gamma_g1_executable_seed_bytes_to_direct_compile_inputs(
    DirectGenomeV1* genome, DirectBodyManifestV1* body,
    DirectDevelopmentEnvironmentV1* environment, const void* bytes,
    std::uint64_t byte_count) {
  if (genome == nullptr || body == nullptr || environment == nullptr || bytes == nullptr ||
      byte_count != sizeof(gamma_evidence::GammaFoundryExecutableSeed))
    return false;
  const DirectBodyManifestV1 body_before = *body;
  const DirectDevelopmentEnvironmentV1 environment_before = *environment;
  const auto* seed =
      static_cast<const gamma_evidence::GammaFoundryExecutableSeed*>(bytes);
  gamma_evidence::GammaG1Refuse refuse = gamma_evidence::GammaG1Refuse::none;
  if (!gamma_evidence::apply_gamma_g1_executable_seed_to_direct_genome(
          genome, *seed, nullptr, 0ull, &refuse) ||
      refuse != gamma_evidence::GammaG1Refuse::none)
    return false;
  return std::memcmp(&body_before, body, sizeof(body_before)) == 0 &&
         std::memcmp(&environment_before, environment,
                     sizeof(environment_before)) == 0;
}

bool apply_gamma_g2_arm_bytes_to_direct_compile_inputs(
    DirectGenomeV1* genome, DirectBodyManifestV1* body,
    DirectDevelopmentEnvironmentV1* environment, const void* bytes,
    std::uint64_t byte_count, std::uint32_t arm) {
  if (genome == nullptr || body == nullptr || environment == nullptr || bytes == nullptr ||
      byte_count != sizeof(gamma_evidence::GammaFoundryNeonatalSiblingAssay))
    return false;
  const DirectBodyManifestV1 body_before = *body;
  const DirectDevelopmentEnvironmentV1 environment_before = *environment;
  const auto* assay =
      static_cast<const gamma_evidence::GammaFoundryNeonatalSiblingAssay*>(
          bytes);
  gamma_evidence::GammaG2Refuse refuse = gamma_evidence::GammaG2Refuse::none;
  if (!gamma_evidence::apply_gamma_g2_arm_to_direct_genome(
          genome, *assay, static_cast<gamma_evidence::GammaG2Arm>(arm),
          assay->experiment_identity, &refuse) ||
      refuse != gamma_evidence::GammaG2Refuse::none)
    return false;
  return std::memcmp(&body_before, body, sizeof(body_before)) == 0 &&
         std::memcmp(&environment_before, environment,
                     sizeof(environment_before)) == 0;
}

DirectBirthReceiptV1 compile_direct_brain(const DirectGenomeV1& genome,
                                          const DirectBodyManifestV1& body,
                                          const DirectDevelopmentEnvironmentV1& environment,
                                          DirectBrain* out_brain,
                                          DirectCompileOptions options) {
  const DirectGenomeLoweringV1 lowered = lower_direct_genome_v1(genome, body, environment);
  std::vector<DirectTerritoryIdentityV1> territory_identities(genome.header.territory_count);
  for (std::uint32_t i = 0u; i < genome.header.territory_count; ++i)
    territory_identities[i] = genome.territories[i].identity;
  return compile_lowered_direct_brain(
      lowered.gamma, body, environment, lowered.direct_genome_root,
      lowered.territory_layout_root, lowered.rule_delay_laws, lowered.rule_delay_law_count,
      territory_identities.data(), static_cast<std::uint32_t>(territory_identities.size()),
      out_brain, options);
}

DirectBirthReceiptV1 compile_donor_gamma_ir_direct_brain(
    const GammaV1& lowered_gamma, const DirectBodyManifestV1& body,
    const DirectDevelopmentEnvironmentV1& environment, DirectBrain* out_brain,
    DirectCompileOptions options) {
  // This explicitly names the retained Gamma POD as donor IR. It lets old
  // regression contracts remain reproducible without granting them canonical
  // genome authority or a place in the first-adult frontier.
  if (validate_genome(lowered_gamma) != GenomeValidationError::kNone)
    throw std::invalid_argument("donor Gamma IR is structurally invalid");
  const Root256 lowered_root = substrate::direct_network::recipe::canonical_genome_root(lowered_gamma);
  return compile_lowered_direct_brain(
      lowered_gamma, body, environment, lowered_root, lowered_root, nullptr,
      0u, nullptr, 0u, out_brain, options);
}

void destroy_direct_brain(DirectBrain* brain) {
  if (brain == nullptr) return;
  cudaFree(brain->resource_ecology);
  cudaFree(brain->arena);
  *brain = DirectBrain{};
}

Root256 direct_brain_root(const DirectBrain& brain, std::uint32_t block_size) {
  Scratch scratch{};
  Root256 arena_root{};
  try {
    arena_root = hash_arena_device(brain, scratch, block_size);
    BrainRootHeader header{arena_root,
                           brain.genome_root,
                           brain.territory_layout_root,
                           brain.body_root,
                           brain.environment_root,
                           brain.node_count,
                           brain.active_route_count,
                           brain.route_capacity,
                           brain.dense_block_count,
                           brain.dense_weight_count,
                           brain.boundary_port_count,
                           brain.resident_field_count,
                           brain.resident_field_range_count,
                           brain.resident_field_index_count,
                           brain.resident_rule_count,
                           brain.resident_tract_delay_law_count,
                           brain.recipe_cell_count,
                           brain.recipe_edge_count,
                           brain.recipe_range_count,
                           brain.recipe_index_count,
                           brain.territory_count,
                           brain.territory_ancestry_count,
                           brain.long_tract_count};
    release_scratch(scratch);
    return content_root(&header, sizeof(header));
  } catch (...) {
    release_scratch(scratch);
    throw;
  }
}

DirectTerritoryPlanProbe direct_probe_territory_plan(const GammaV1& gamma,
                                                     std::uint32_t seed_index,
                                                     std::uint32_t route_reserve_per_node) {
  if (seed_index >= gamma.header.seed_count) return DirectTerritoryPlanProbe{};
  const TerritoryPlan plan = derive_territory_plan(gamma, seed_index, route_reserve_per_node);
  return DirectTerritoryPlanProbe{plan.active,
                                  plan.flags,
                                  plan.node_count,
                                  plan.sparse_degree,
                                  plan.radius,
                                  plan.dense_width,
                                  plan.long_tract_count,
                                  plan.route_capacity_per_node,
                                  plan.active_route_estimate,
                                  plan.lineage,
                                  plan.chemotype,
                                  plan.attract_field,
                                  plan.repel_field,
                                  plan.resource_field,
                                  plan.maturation_field,
                                  plan.inhibition_field,
                                  plan.repair_field,
                                  plan.bound_field_count};
}

DirectFieldEvaluationProbe direct_probe_field_evaluation(
    const GammaV1& gamma, std::uint32_t seed_index,
    const std::int32_t growth_site[3], const std::int32_t cone_origin[3],
    std::uint32_t logical_tick, std::uint32_t route_reserve_per_node) {
  if (seed_index >= gamma.header.seed_count) return DirectFieldEvaluationProbe{};
  const TerritoryPlan plan =
      derive_territory_plan(gamma, seed_index, route_reserve_per_node);
  return DirectFieldEvaluationProbe{
      combined_developmental_score_q16(gamma, plan, growth_site, logical_tick),
      combined_gradient_tilt_q16(gamma, plan, growth_site, cone_origin,
                                 logical_tick),
      plan.bound_field_count};
}

DirectNodeSiteProbe direct_probe_node_site(const GammaV1& gamma,
                                           const DirectDevelopmentEnvironmentV1& environment,
                                           std::uint32_t seed_index, std::uint32_t local,
                                           std::uint32_t route_reserve_per_node) {
  DirectNodeSiteProbe out{};
  if (seed_index >= gamma.header.seed_count) return out;
  const TerritoryPlan plan = derive_territory_plan(gamma, seed_index, route_reserve_per_node);
  if (plan.active == 0u || local >= plan.node_count) return out;
  const std::uint32_t logical_tick = logical_node_birth_tick(gamma, plan, local);
  out.best_score_q16 = select_node_site(
      gamma, environment, plan, local, logical_tick, out.coordinate,
      &out.extended_draws);
  out.chosen_hard_excluded = environment_hard_excludes(environment, out.coordinate) ? 1u : 0u;
  out.candidate_count = kCandidateCoordinateCount;
  for (std::uint32_t candidate = 0; candidate < kCandidateCoordinateCount; ++candidate) {
    std::int32_t coord[3];
    candidate_coordinate(gamma, plan, local, candidate, coord);
    if (environment_hard_excludes(environment, coord)) ++out.candidates_hard_excluded;
  }
  out.valid = 1u;
  return out;
}

DirectBoundaryPortProbe direct_probe_boundary_port(const GammaV1& gamma,
                                                   const BoundaryPortBinding& binding,
                                                   std::uint32_t route_reserve_per_node) {
  DirectBoundaryPortProbe out{};
  // The real compiler assigns node_offset in a later layout pass. This probe
  // reproduces the simplest consistent model -- a prefix sum over active
  // territories in seed order -- so ABSOLUTE node indices here are the probe's,
  // while what the arms assert is the DEPENDENCY structure: which binding fields
  // the attachment reads at all. That distinction is stated in the contract too.
  TerritoryPlan plans[substrate::direct_network::recipe::kMaxSeeds];
  const std::uint32_t plan_count =
      min_u32(gamma.header.seed_count, substrate::direct_network::recipe::kMaxSeeds);
  std::uint32_t node_offset = 0u;
  for (std::uint32_t seed = 0; seed < plan_count; ++seed) {
    plans[seed] = derive_territory_plan(gamma, seed, route_reserve_per_node);
    plans[seed].node_offset = node_offset;
    node_offset += plans[seed].active != 0u ? plans[seed].node_count : 0u;
  }
  DirectBoundaryPort port{};
  if (!attach_boundary_port(binding, plans, plan_count, &port)) {
    out.refused = 1u;
    return out;
  }
  out.attached = 1u;
  out.node = port.node;
  out.channel = port.channel;
  out.role_mask = port.role_mask;
  out.physical_route = port.physical_route;
  out.parent_route = port.parent_route;
  out.node_flags = 0u;
  if ((port.role_mask & static_cast<std::uint32_t>(BoundaryRole::sensor)) != 0u)
    out.node_flags |= kNodeFlagSensor;
  if ((port.role_mask & static_cast<std::uint32_t>(BoundaryRole::motor)) != 0u)
    out.node_flags |= kNodeFlagMotor;
  if ((port.role_mask & static_cast<std::uint32_t>(BoundaryRole::world_return)) != 0u)
    out.node_flags |= kNodeFlagWorldReturn;
  return out;
}

DirectResidentRuleProbe direct_probe_resident_rules(const GammaV1& gamma) {
  // compile_resident_rules is what actually fills brain.resident_rules, and it
  // has internal linkage, so nothing outside this translation unit could ever
  // observe which rules survive birth. It is the ONLY admission decision on that
  // path -- this probe calls it rather than restating its condition, because a
  // restatement is a second implementation that agrees with itself (github #1344).
  const std::vector<ResidentConstructorRule> resident = compile_resident_rules(gamma);
  DirectResidentRuleProbe out{};
  out.admitted_count = static_cast<std::uint32_t>(resident.size());
  out.first_admitted_source_rule_index = kInvalidIndex;
  for (const ResidentConstructorRule& rule : resident) {
    if (rule.opcode == RuleOpcode::retract) ++out.admitted_retract_count;
    if ((rule.flags & kRuleFlagPostBirthResident) == 0u) ++out.admitted_without_resident_flag;
    if (out.first_admitted_source_rule_index == kInvalidIndex)
      out.first_admitted_source_rule_index = rule.source_rule_index;
  }
  return out;
}

}  // namespace substrate::direct_network

#undef DIRECT_NETWORK_HD

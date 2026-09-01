#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_BRAIN_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_BRAIN_CUH
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_exact_history.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"
#include "hardware_native/direct_network_recipe.hpp"
#if defined(__CUDACC__)
#define DIRECT_NETWORK_HD __host__ __device__
#else
#define DIRECT_NETWORK_HD
#endif

namespace substrate::direct_network {

// Transitional implementation primitive only: the fixed-width Gamma ABI is
// retained as Direct's internal lowering format. The canonical authored input
// is DirectGenomeV1; the mathematical organism below has no BCC lattice/F/RWR
// dependency. Keeping this POD ABI does not make it genome authority.
using GammaV1 = substrate::direct_network::recipe::Genome;
using Root256 = substrate::direct_network::recipe::Root256;
using SiteWord = substrate::direct_network::recipe::SiteWord;
using RuleOpcode = substrate::direct_network::recipe::RuleOpcode;
using ConstructionRule = substrate::direct_network::recipe::ConstructionRule;
using FieldBlock = substrate::direct_network::recipe::FieldBlock;
using SeedBlock = substrate::direct_network::recipe::SeedBlock;

inline constexpr std::uint32_t kInvalidIndex = 0xffffffffu;
inline constexpr std::uint32_t kDirectBodyAbiV1 = 1u;
inline constexpr std::uint32_t kMaxBoundaryPorts = 256u;
inline constexpr std::uint32_t kDirectDevelopmentEnvironmentAbiV1 = 1u;
inline constexpr std::uint32_t kMaxDevelopmentConstraints = 256u;
inline constexpr std::uint32_t kDefaultRouteReserve = 4u;
inline constexpr std::uint32_t kMaxSparseDegree = 32u;
// A motor action may freeze the union of four bounded terminal ancestry paths,
// each at most four edges deep. This is deliberately separate from the four
// live participation slots carried by one physical node.
inline constexpr std::uint32_t kMaxResidentActionParticipationLinks = 16u;
inline constexpr std::uint32_t kMaxRouteCapacityPerNode = 48u;
inline constexpr std::uint16_t kInvalidFieldIndex16 = 0xffffu;
inline constexpr std::uint32_t kMaxRecipeEdgesPerCell = 8u;
inline constexpr std::uint32_t kDenseWidthLimit = 256u;
inline constexpr std::uint32_t kQ16One = 1u << 16;
inline constexpr std::uint32_t kMaxLocalCoordinateCharts = 4u;
inline constexpr std::uint32_t kMaxLocalChartTransitions = 8u;

// Resident coordinates belong to one frame. The atlas has no slot for a
// canonical global vector; overlap constraints are the only cross-frame seam.
struct DirectLocalCoordinateChartV1 {
  std::uint64_t frame_identity;
  std::int32_t lower_q16;
  std::int32_t upper_q16;
  std::uint32_t generation;
  std::uint32_t reserved;
};
struct DirectLocalChartTransitionV1 {
  std::uint64_t transition_identity;
  std::int32_t source_lower_q16;
  std::int32_t source_upper_q16;
  std::int32_t target_lower_q16;
  std::int32_t target_upper_q16;
  std::int32_t offset_q16;
  std::uint32_t source_chart;
  std::uint32_t target_chart;
  std::int32_t orientation;
  std::uint32_t reserved[2];
};
struct DirectLocalCoordinateAtlasV1 {
  std::uint64_t atlas_identity;
  std::uint32_t chart_count;
  std::uint32_t transition_count;
  std::uint32_t version;
  std::uint32_t reserved;
  DirectLocalCoordinateChartV1 charts[kMaxLocalCoordinateCharts];
  DirectLocalChartTransitionV1 transitions[kMaxLocalChartTransitions];
};
struct DirectLocalChartTraceV1 {
  std::uint64_t frame_identities[kMaxLocalCoordinateCharts];
  std::int32_t coordinates_q16[kMaxLocalCoordinateCharts];
  std::uint32_t count;
  std::int32_t constraint_residual_q16;
  std::uint32_t refused;
  std::uint32_t reserved;
};
static_assert(std::has_unique_object_representations_v<
                  DirectLocalCoordinateChartV1> &&
              std::has_unique_object_representations_v<
                  DirectLocalChartTransitionV1> &&
              std::has_unique_object_representations_v<
                  DirectLocalCoordinateAtlasV1> &&
              std::has_unique_object_representations_v<DirectLocalChartTraceV1>);

// FieldBlock::polarity is interpreted this way only by the direct-network
// compiler.  Historical BCC contracts retain their original interpretation.
enum class DevelopmentFieldKind : std::uint32_t {
  attract = 0,
  repel = 1,
  resource = 2,
  maturation = 3,
  inhibition = 4,
  repair = 5,
};
inline constexpr std::uint32_t kDevelopmentFieldKindCount = 6u;

// ConstructionRule::flags are intentionally generic phenotype/compiler hints,
// never semantic region identifiers.
inline constexpr std::uint32_t kRuleFlagDenseIntegrative = 1u << 16;
inline constexpr std::uint32_t kRuleFlagInhibitoryBias = 1u << 17;
inline constexpr std::uint32_t kRuleFlagConstructorReserve = 1u << 18;
inline constexpr std::uint32_t kRuleFlagPostBirthResident = 1u << 19;
inline constexpr std::uint32_t kRuleFlagSequenceIntegrator = 1u << 20;
inline constexpr std::uint32_t kRuleFlagLongRangePreferred = 1u << 21;
inline constexpr std::uint32_t kRuleFlagCompetitionDensityAuthored = 1u << 22;
inline constexpr std::uint32_t kRuleFlagCompetitionMagnitudeAuthored = 1u << 23;

// Node flags are execution/development phenotypes only.
inline constexpr std::uint32_t kNodeFlagSensor = 1u << 0;
inline constexpr std::uint32_t kNodeFlagMotor = 1u << 1;
inline constexpr std::uint32_t kNodeFlagWorldReturn = 1u << 2;
inline constexpr std::uint32_t kNodeFlagDenseMember = 1u << 3;
inline constexpr std::uint32_t kNodeFlagConstructor = 1u << 4;
inline constexpr std::uint32_t kNodeFlagInhibitory = 1u << 5;
inline constexpr std::uint32_t kNodeFlagImmature = 1u << 6;
inline constexpr std::uint32_t kNodeFlagCompetitive = 1u << 7;
// #1310: raw zero preserves the historical 32768 default; the high bit distinguishes explicit zero.
inline constexpr std::uint32_t kCompetitionStrengthExplicit = 1u << 31, kDefaultCompetitionStrengthQ16 = kQ16One / 2u;
DIRECT_NETWORK_HD inline std::uint32_t encode_competition_strength_q16(std::uint32_t v) { return v == kDefaultCompetitionStrengthQ16 ? 0u : kCompetitionStrengthExplicit | v; }
DIRECT_NETWORK_HD inline std::uint32_t decode_competition_strength_q16(std::uint32_t code) { return code == 0u ? kDefaultCompetitionStrengthQ16 : code & ~kCompetitionStrengthExplicit; }
inline constexpr std::uint32_t kRouteFlagActive = 1u << 0;
// gh #1243: substrate::direct_adult::DirectRoute (direct_adult_legacy_oracle.cuh)
// is a separate, non-interchangeable struct that happens to share this
// name, most field names, and this whole flag vocabulary's names -- but NOT
// the bit positions. This file's kRouteFlagActive above (bit 0) has no
// equivalent in direct_adult, so every flag from here down is offset by one
// bit relative to that file's copy: this kRouteFlagLongTract sits at
// 1u << 1, one bit above direct_adult's. Reinterpreting a raw .flags value
// across the two types would misread this file's unconditionally-set
// kRouteFlagActive as direct_adult's kRouteFlagLongTract, and this file's
// kRouteFlagLongTract as direct_adult's kRouteFlagLearnedOutput -- not
// merely a mislabeled tract. No translation unit currently includes both
// headers (enforced by tools/check_direct_route_namespace_isolation.py) and
// no verified conversion path copies .flags between them. Do not assume the
// bit position or the raw flags value carries across a future
// growth-to-adult lowering; translate the semantic flag, not the integer.
inline constexpr std::uint32_t kRouteFlagLongTract = 1u << 1, kRouteFlagInhibitory = 1u << 2;
inline constexpr std::uint32_t kRouteFlagDevelopmentalReserve = 1u << 3, kRouteFlagRecurrent = 1u << 4;
inline constexpr std::uint32_t kRouteRecipeBuilderShift = 16u, kRouteRecipeBuilderMask = 0xffffu << kRouteRecipeBuilderShift;
DIRECT_NETWORK_HD inline std::uint32_t encode_route_recipe_builder(std::uint32_t flags, std::uint32_t recipe_cell) { const std::uint32_t encoded = recipe_cell < 0xffffu ? recipe_cell + 1u : 0u; return (flags & ~kRouteRecipeBuilderMask) | (encoded << kRouteRecipeBuilderShift); }
DIRECT_NETWORK_HD inline std::uint32_t decode_route_recipe_builder(std::uint32_t flags) { const std::uint32_t encoded = (flags & kRouteRecipeBuilderMask) >> kRouteRecipeBuilderShift; return encoded == 0u ? kInvalidIndex : encoded - 1u; }
inline constexpr std::uint64_t kRouteIncarnationStride = 1ull << 32;
DIRECT_NETWORK_HD inline std::uint64_t initial_route_incarnation(std::uint32_t route_index) {
  return kRouteIncarnationStride | (static_cast<std::uint64_t>(route_index) + 1u);
}

inline constexpr std::uint32_t kDenseBlockFlagTensorEligible = 1u << 0;
inline constexpr std::uint32_t kDenseBlockFlagRecurrent = 1u << 1;
inline constexpr std::uint32_t kDenseBlockFlagSequenceIntegrator = 1u << 2;

// Physical membrane attachment.  Deliberately no word/token/expected-output
// field: the body tells us where a raw channel attaches, never what the adult
// should say on it.
enum class BoundaryRole : std::uint32_t {
  sensor = 1u << 0,
  motor = 1u << 1,
  world_return = 1u << 2,
};

struct BoundaryPortBinding {
  std::uint32_t seed_index;
  std::uint32_t local_node;
  std::uint32_t channel;
  std::uint32_t role_mask;
  std::uint32_t physical_route;
  std::uint32_t parent_route;
};
static_assert(std::is_standard_layout_v<BoundaryPortBinding> &&
              std::is_trivial_v<BoundaryPortBinding>);

struct DirectBodyManifestV1 {
  std::uint32_t abi_version;
  std::uint32_t binding_count;
  BoundaryPortBinding bindings[kMaxBoundaryPorts];
};
static_assert(std::is_standard_layout_v<DirectBodyManifestV1> &&
              std::is_trivial_v<DirectBodyManifestV1>);
static_assert(std::has_unique_object_representations_v<BoundaryPortBinding>);
static_assert(std::has_unique_object_representations_v<DirectBodyManifestV1>);

// External developmental/body geometry visible only during gestation.  These
// constraints are physical, never semantic: they can make a region unavailable
// or expensive and thereby force Gamma to grow a different morphology.
inline constexpr std::uint32_t kEnvironmentConstraintHardExclude = 1u << 0;
inline constexpr std::uint32_t kEnvironmentConstraintSoftPenalty = 1u << 1;

struct DevelopmentEnvironmentConstraint {
  std::int32_t center[3];
  std::uint32_t half_extent[3];
  std::int32_t penalty_q16;
  std::uint32_t flags;
};
static_assert(std::is_standard_layout_v<DevelopmentEnvironmentConstraint> &&
              std::is_trivial_v<DevelopmentEnvironmentConstraint>);

struct DirectDevelopmentEnvironmentV1 {
  std::uint32_t abi_version;
  std::uint32_t constraint_count;
  std::uint64_t environment_seed;
  DevelopmentEnvironmentConstraint constraints[kMaxDevelopmentConstraints];
};
static_assert(std::is_standard_layout_v<DirectDevelopmentEnvironmentV1> &&
              std::is_trivial_v<DirectDevelopmentEnvironmentV1>);
static_assert(std::has_unique_object_representations_v<DevelopmentEnvironmentConstraint>);
static_assert(std::has_unique_object_representations_v<DirectDevelopmentEnvironmentV1>);

inline constexpr std::int32_t kEnvironmentHardExcludeScoreQ16 = -(32 << 16);

// This is the one geometry predicate for every direct developmental consumer.
// It uses widened differences so a coordinate near an int32 boundary cannot
// overflow while deciding whether a physical constraint contains it.
DIRECT_NETWORK_HD inline bool environment_constraint_contains(
    const DevelopmentEnvironmentConstraint& constraint, const std::int32_t coord[3]) {
  for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
    const std::int64_t delta = static_cast<std::int64_t>(coord[axis]) -
                               static_cast<std::int64_t>(constraint.center[axis]);
    const std::uint64_t distance = static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
    if (distance > static_cast<std::uint64_t>(constraint.half_extent[axis])) return false;
  }
  return true;
}

DIRECT_NETWORK_HD inline bool environment_hard_excludes(
    const DirectDevelopmentEnvironmentV1& environment, const std::int32_t coord[3]) {
  for (std::uint32_t index = 0u; index < environment.constraint_count; ++index) {
    const DevelopmentEnvironmentConstraint& constraint = environment.constraints[index];
    if ((constraint.flags & kEnvironmentConstraintHardExclude) != 0u &&
        environment_constraint_contains(constraint, coord))
      return true;
  }
  return false;
}

// The shared direct-life-function environmental score. Hard exclusion refuses
// a coordinate; soft penalties keep it legal but make it less preferred.
DIRECT_NETWORK_HD inline std::int32_t environment_score_q16(
    const DirectDevelopmentEnvironmentV1& environment, const std::int32_t coord[3]) {
  std::int64_t score = 0;
  for (std::uint32_t index = 0u; index < environment.constraint_count; ++index) {
    const DevelopmentEnvironmentConstraint& constraint = environment.constraints[index];
    if (!environment_constraint_contains(constraint, coord)) continue;
    if ((constraint.flags & kEnvironmentConstraintHardExclude) != 0u)
      return kEnvironmentHardExcludeScoreQ16;
    if ((constraint.flags & kEnvironmentConstraintSoftPenalty) != 0u) {
      const std::int64_t penalty = static_cast<std::int64_t>(constraint.penalty_q16);
      score -= penalty < 0 ? -penalty : penalty;
    }
  }
  return score <= kEnvironmentHardExcludeScoreQ16
             ? kEnvironmentHardExcludeScoreQ16
             : static_cast<std::int32_t>(score);
}

struct alignas(16) DirectNode {
  std::int32_t coordinate[3];
  std::uint32_t lineage;
  std::uint32_t chemotype;
  std::uint32_t flags;
  std::uint32_t route_offset;
  std::uint16_t active_route_count;
  std::uint16_t route_capacity;
  std::uint32_t active_in_degree;
  std::uint16_t territory_index;
  std::uint16_t territory_padding;
  std::uint16_t attract_field;
  std::uint16_t repel_field;
  std::uint16_t resource_field;
  std::uint16_t maturation_field;
  std::uint16_t inhibition_field;
  std::uint16_t repair_field;
  std::uint32_t competition_strength_code_q16;  // #1310: 0=historical/default 32768; high bit marks explicit value, including zero.
  std::int32_t activation_q16;
  std::int32_t activity_ema_q16;
  std::int32_t credit_ema_q16;
  std::int32_t attractor_support_q16;
  std::int32_t inhibition_q16;
  std::int32_t maintenance_q16;
  std::uint32_t maturation_q16;
  std::uint32_t refractory_until;
  std::uint32_t last_actual_tick;
  std::uint32_t last_endogenous_tick;
};
static_assert(std::is_standard_layout_v<DirectNode> && std::is_trivial_v<DirectNode>); static_assert(sizeof(DirectNode) == 96u); static_assert(alignof(DirectNode) == 16u);
struct alignas(16) DirectRoute {
  std::uint32_t source;
  std::uint32_t target;
  std::uint32_t flags;
  std::uint32_t delay;
  std::int32_t conductance_q16;
  std::int32_t eligibility_q16;
  std::int32_t last_credit_q16;
  std::int32_t developmental_score_q16;
  std::uint32_t eligibility_context;
  std::uint32_t eligibility_expires;
  // WHICH episode last recarved this route. `last_credit_q16` alone is a
  // last-writer-wins scalar: measured 2026-08-18, every one of the 242
  // eligibility-carrying routes held records under MORE THAN ONE ticket
  // ancestry, so reading that value back as "episode A's credit" returned
  // whatever settled last. Without this stamp there is no readout anywhere for
  // "this route changed because of this episode", which patches 0005/0008 of
  // the network-recipe train assume exists.
  //
  // This costs nothing: it replaces `provenance_slot` (only ever written as
  // kInvalidIndex at birth, never read) and `reserved` (never touched at all),
  // two adjacent dead 32-bit words at an 8-aligned offset. sizeof(DirectRoute)
  // is unchanged at 48 bytes.
  std::uint64_t last_credit_ticket;
};
inline constexpr std::uint64_t kNoCreditTicket = ~0ull;
static_assert(sizeof(DirectRoute) == 48u,
              "DirectRoute must stay 48 bytes: the credit stamp reuses two dead words");
static_assert(std::is_standard_layout_v<DirectRoute> && std::is_trivial_v<DirectRoute>);
// Dense integrative territory descriptor.  Weight storage is IEEE-fp16 bits;
// the adult can lower eligible blocks directly to Tensor Core kernels without
// changing the logical morphology.  Sparse routes remain the causal bridge to
// the rest of the organism.
struct DirectDenseBlock {
  std::uint32_t node_begin;
  std::uint32_t node_count;
  std::uint32_t weight_offset;
  std::uint32_t weight_count;
  std::uint32_t lineage;
  std::uint32_t flags;
  std::uint32_t tile_width;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<DirectDenseBlock> &&
              std::is_trivial_v<DirectDenseBlock>);

struct DirectBoundaryPort {
  std::uint32_t node;
  std::uint32_t channel;
  std::uint32_t role_mask;
  std::uint32_t physical_route;
  std::uint32_t parent_route;
};
static_assert(std::is_standard_layout_v<DirectBoundaryPort> &&
              std::is_trivial_v<DirectBoundaryPort>);
// f.efference (#1470): an internal motor command dispatched to a body
// actuator travels beside an efference copy routed to ONE predictive-shadow
// target -- the far end of the acting node's own active outbound topology.
// The copy carries the same grown word as the command, no semantic label,
// and drives its shadow through the ordinary incoming-excitation path.
struct DirectEfferenceCopy {
  std::uint64_t ticket_id;
  std::uint32_t acting_node;
  std::uint32_t shadow_node;
  std::uint32_t channel;
  std::uint32_t word;
  std::uint32_t timestamp;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<DirectEfferenceCopy> &&
              std::is_trivial_v<DirectEfferenceCopy>);
static_assert(std::has_unique_object_representations_v<DirectEfferenceCopy>);
DIRECT_NETWORK_HD inline std::uint32_t efference_shadow_target(
    const DirectNode* nodes, const DirectRoute* routes,
    const DirectNode& acting) {
  if (nodes == nullptr || routes == nullptr) return kInvalidIndex;
  const std::uint32_t route_end =
      acting.route_offset +
      (acting.active_route_count < acting.route_capacity
           ? acting.active_route_count
           : acting.route_capacity);
  for (std::uint32_t r = acting.route_offset; r < route_end; ++r)
    if ((routes[r].flags & kRouteFlagActive) != 0u) return routes[r].target;
  return kInvalidIndex;
}
struct ResidentTerritoryAncestry { std::uint32_t lineage, axis, ordinal; std::int32_t founder_origin[3]; std::uint32_t node_begin, node_count, prenatal_begin_tick; }; static_assert(std::is_standard_layout_v<ResidentTerritoryAncestry> && std::is_trivial_v<ResidentTerritoryAncestry>);
// A field compiled into the organism.  External Gamma is gone at birth; these
// are ordinary resident developmental state used by juvenile/adult maturation.
struct ResidentDevelopmentField {
  std::int32_t center[3];
  std::uint32_t radius;
  std::int32_t strength_q16;
  std::int32_t decay_q16_per_tick;
  DevelopmentFieldKind kind;
  std::uint32_t require_mask;
  std::uint32_t require_value;
  std::uint32_t begin_tick;
  std::uint32_t end_tick;
};
static_assert(std::is_standard_layout_v<ResidentDevelopmentField> &&
              std::is_trivial_v<ResidentDevelopmentField>);

// Resident fields keep Gamma's absolute developmental clock across birth.
// Decay is monotone toward zero and cannot restart or reverse sign at handoff.
DIRECT_NETWORK_HD inline std::int32_t resident_field_decayed_strength_q16(
    const ResidentDevelopmentField& field, std::uint64_t logical_tick) {
  if (field.decay_q16_per_tick <= 0 || logical_tick <= field.begin_tick)
    return field.strength_q16;
  const std::uint64_t magnitude = field.strength_q16 < 0
      ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(field.strength_q16))
      : static_cast<std::uint64_t>(field.strength_q16);
  const std::uint64_t rate = static_cast<std::uint32_t>(field.decay_q16_per_tick);
  const std::uint64_t elapsed = logical_tick - field.begin_tick;
  if (elapsed >= (magnitude + rate - 1u) / rate) return 0;
  const std::int64_t remaining =
      static_cast<std::int64_t>(magnitude - elapsed * rate);
  return static_cast<std::int32_t>(field.strength_q16 < 0 ? -remaining : remaining);
}

struct ResidentFieldRange {
  std::uint32_t index_offset;
  std::uint16_t index_count;
  std::uint16_t reserved;
};
static_assert(std::is_standard_layout_v<ResidentFieldRange> &&
              std::is_trivial_v<ResidentFieldRange>);

struct ResidentConstructorRule {
  RuleOpcode opcode;
  std::uint32_t flags;
  std::uint32_t field_index;
  std::uint32_t begin_age;
  std::uint32_t end_age;
  std::uint32_t critical_begin_age;
  std::uint32_t critical_end_age;
  std::uint32_t threshold_q16;
  std::uint32_t extent;
  std::uint32_t branch_count;
  std::uint32_t require_mask;
  std::uint32_t require_value;
  std::uint32_t write_mask;
  std::uint32_t write_value;
  std::uint32_t source_rule_index;
};
static_assert(std::is_standard_layout_v<ResidentConstructorRule> &&
              std::is_trivial_v<ResidentConstructorRule>);

enum class ResidentRecipeRelation : std::uint16_t {
  trigger = 0u,
  inhibit = 1u,
  repair = 2u,
  temporal = 3u,
  shared_field = 4u, route_gain = 5u,
};

// A resident recipe is not a host interpreter opcode. It is ordinary born
// developmental state whose support/credit can be changed by the organism.
struct ResidentRecipeReceptorState {
  std::int32_t activation_q16;
  std::int32_t plasticity_q16;
  std::uint64_t causal_identity;
  std::uint64_t revision_identity;
};
static_assert(std::is_standard_layout_v<ResidentRecipeReceptorState> &&
              std::is_trivial_v<ResidentRecipeReceptorState> &&
              std::has_unique_object_representations_v<ResidentRecipeReceptorState>);

inline constexpr std::uint16_t kResidentRecipeLevelCConstructor = 1u << 0;

// Bounded numeric update law carried by the resident Recipe cell.  These are
// data instructions for the fixed device interpreter, never host-generated
// CUDA/PTX and never evidence records.  A program can transform an already
// authenticated exact-credit delta into a parameter delta; it has no opcode
// that can create eligibility, consequence, participation, or authority.
inline constexpr std::uint32_t kResidentRecipeIrAbiV1 = 1u;
inline constexpr std::uint32_t kResidentRecipeIrCapacity = 8u;
enum class ResidentRecipeIrOp : std::uint16_t {
  halt = 0u,
  load_exact_credit = 1u,
  scale_q16 = 2u,
  clamp_symmetric_q16 = 3u,
  emit_parameter_delta = 4u,
};
struct ResidentRecipeIrInstruction {
  ResidentRecipeIrOp op;
  std::uint16_t reserved;
  std::int32_t operand_q16;
};
struct ResidentRecipeIrProgram {
  ResidentRecipeIrInstruction instructions[kResidentRecipeIrCapacity];
  std::uint64_t program_identity;
  std::uint32_t op_count;
  std::uint32_t abi_version;
  std::uint32_t layout_stride;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<ResidentRecipeIrInstruction> &&
              std::is_trivial_v<ResidentRecipeIrInstruction> &&
              std::has_unique_object_representations_v<
                  ResidentRecipeIrInstruction>);
static_assert(std::is_standard_layout_v<ResidentRecipeIrProgram> &&
              std::is_trivial_v<ResidentRecipeIrProgram> &&
              std::has_unique_object_representations_v<ResidentRecipeIrProgram>);

DIRECT_NETWORK_HD inline std::uint64_t resident_recipe_ir_identity(
    const ResidentRecipeIrProgram& program) {
  std::uint64_t identity = exact_history_fold_word(
      0x7265636970697232ull, program.abi_version);
  identity = exact_history_fold_word(identity, program.op_count);
  for (std::uint32_t i = 0u;
       i < program.op_count && i < kResidentRecipeIrCapacity; ++i) {
    identity = exact_history_fold_word(
        identity, static_cast<std::uint16_t>(program.instructions[i].op));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(
                      program.instructions[i].operand_q16));
  }
  return identity == 0u ? 1u : identity | (1ull << 63u);
}

DIRECT_NETWORK_HD inline bool make_resident_recipe_update_ir(
    std::uint32_t layout_stride, std::int32_t scale_q16,
    std::int32_t clamp_q16, ResidentRecipeIrProgram* out) {
  if (out == nullptr || layout_stride == 0u || scale_q16 == 0 ||
      clamp_q16 <= 0)
    return false;
  *out = ResidentRecipeIrProgram{};
  out->instructions[0].op = ResidentRecipeIrOp::load_exact_credit;
  out->instructions[1] = {
      ResidentRecipeIrOp::scale_q16, 0u, scale_q16};
  out->instructions[2] = {
      ResidentRecipeIrOp::clamp_symmetric_q16, 0u, clamp_q16};
  out->instructions[3].op = ResidentRecipeIrOp::emit_parameter_delta;
  out->instructions[4].op = ResidentRecipeIrOp::halt;
  out->op_count = 5u;
  out->abi_version = kResidentRecipeIrAbiV1;
  out->layout_stride = layout_stride;
  out->program_identity = resident_recipe_ir_identity(*out);
  return true;
}

struct ResidentRecipeCell {
  // Logical identity survives every learned revision.  Revision identity is
  // the exact causal lineage head for the current support/credit body.
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::int64_t support_q16;
  std::int64_t credit_q16;
  std::uint64_t revision;
  std::uint32_t rule_index;
  std::uint32_t edge_offset;
  std::uint32_t last_active_age;
  std::uint16_t edge_count;
  std::uint16_t flags;
  ResidentRecipeReceptorState receptor_state;
  std::uint64_t constructor_evidence_identity;
  ResidentRecipeIrProgram update_program;
  // Exact credit remains the immutable-evidence accumulator above.  The
  // update program may change only this separately named learned parameter.
  std::int64_t policy_parameter_q16;
  std::uint64_t ir_subject_identity;
  std::uint64_t ir_bound_revision_identity;
  std::uint64_t ir_binding_identity;
  std::uint64_t ir_last_consequence_identity;
  std::uint64_t ir_last_execution_identity;
  std::uint32_t ir_execution_count;
  std::uint32_t ir_reserved;
};
static_assert(std::is_standard_layout_v<ResidentRecipeCell> &&
              std::is_trivial_v<ResidentRecipeCell> &&
              std::has_unique_object_representations_v<ResidentRecipeCell>);

// Every persistent Recipe creation path receives the same modality-neutral
// resident update operator.  Callers must refuse the creation transaction if
// this bounded program cannot be formed; a zero program is never a legacy
// interpretation mode.
DIRECT_NETWORK_HD inline bool initialize_resident_recipe_update_ir(
    ResidentRecipeCell* cell) {
  return cell != nullptr && make_resident_recipe_update_ir(
      1u, static_cast<std::int32_t>(kQ16One),
      static_cast<std::int32_t>(4u * kQ16One), &cell->update_program);
}

DIRECT_NETWORK_HD inline std::uint64_t resident_recipe_revision_identity(
    std::uint64_t logical_recipe_id, std::uint64_t prior_revision_identity,
    std::uint64_t revision, std::uint64_t contributor_identity,
    std::int64_t support_q16, std::int64_t credit_q16) {
  std::uint64_t identity = exact_history_fold_word(
      0x7265636970657631ull, logical_recipe_id);
  identity = exact_history_fold_word(identity, prior_revision_identity);
  identity = exact_history_fold_word(identity, revision);
  identity = exact_history_fold_word(identity, contributor_identity);
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(support_q16));
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(credit_q16));
  return identity == 0u ? 1u : identity;
}

enum class ResidentRecipeRevisionAuthority : std::uint32_t {
  none = 0u,
  structural = 1u,
  experience = 2u,
  resident_ir = 3u,
};
inline constexpr std::uint16_t kResidentRecipeRevisionAuthorityMask = 0x6u; DIRECT_NETWORK_HD inline ResidentRecipeRevisionAuthority resident_recipe_current_revision_authority(const ResidentRecipeCell& cell) { return static_cast<ResidentRecipeRevisionAuthority>((cell.flags & kResidentRecipeRevisionAuthorityMask) >> 1u); }
DIRECT_NETWORK_HD inline void set_resident_recipe_current_revision_authority(ResidentRecipeCell* cell, ResidentRecipeRevisionAuthority authority) { if (cell != nullptr) cell->flags = static_cast<std::uint16_t>((cell->flags & ~kResidentRecipeRevisionAuthorityMask) | (static_cast<std::uint16_t>(authority) << 1u)); }

DIRECT_NETWORK_HD inline std::uint64_t resident_recipe_revision_contributor(
    const DirectExactHistoryRecord& event) {
  std::uint64_t identity = exact_history_fold_word(
      0x64656c746172686full, event.incarnation_before);
  identity = exact_history_fold_word(identity, event.incarnation_after);
  identity = exact_history_fold_word(identity, event.resident_tick);
  identity = exact_history_fold_word(identity, event.event_tick);
  identity = exact_history_fold_word(identity, event.source);
  identity = exact_history_fold_word(identity, event.subject);
  identity = exact_history_fold_word(identity, event.value);
  identity = exact_history_fold_word(identity, event.context);
  identity = exact_history_fold_word(identity, event.flags);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(event.resource_delta));
  return identity == 0u ? 1u : identity;
}

DIRECT_NETWORK_HD inline bool stage_resident_recipe_revision_event(
    DirectExactHistoryRecord* event, const ResidentRecipeCell& cell,
    std::uint32_t recipe_cell, ResidentRecipeRevisionAuthority authority,
    std::uint64_t occurrence_identity, std::uint64_t evidence_identity,
    std::uint32_t tick, std::uint32_t event_tick, std::uint32_t locator,
    std::uint32_t occurrence_ordinal, std::uint32_t flags,
    std::int64_t delta_q16) {
  if (event == nullptr || occurrence_identity == 0u || evidence_identity == 0u ||
      delta_q16 == 0) return false;
  *event = DirectExactHistoryRecord{};
  event->parent_identity = cell.revision_identity;
  event->resident_tick = tick; event->event_tick = event_tick;
  event->kind = DirectExactHistoryKind::recipe_revision;
  event->source = recipe_cell; event->subject = static_cast<std::uint32_t>(authority);
  event->value = occurrence_ordinal; event->context = locator; event->flags = flags;
  event->incarnation_before = occurrence_identity;
  event->incarnation_after = evidence_identity; event->resource_delta = delta_q16;
  const std::int64_t support = cell.support_q16 +
      (authority == ResidentRecipeRevisionAuthority::structural ? delta_q16 : 0);
  const std::int64_t credit = cell.credit_q16 +
      (authority == ResidentRecipeRevisionAuthority::experience ? delta_q16 : 0);
  if (authority != ResidentRecipeRevisionAuthority::structural &&
      authority != ResidentRecipeRevisionAuthority::experience) return false;
  event->identity = resident_recipe_revision_identity(
      cell.logical_recipe_id, cell.revision_identity, cell.revision + 1u,
      resident_recipe_revision_contributor(*event), support, credit);
  return true;
}

DIRECT_NETWORK_HD inline bool apply_resident_recipe_revision_event(
    ResidentRecipeCell* cell, const DirectExactHistoryRecord& event,
    std::uint32_t recipe_cell) {
  if (cell == nullptr || event.kind != DirectExactHistoryKind::recipe_revision ||
      event.source != recipe_cell || event.parent_identity != cell->revision_identity)
    return false;
  const auto authority = static_cast<ResidentRecipeRevisionAuthority>(event.subject);
  const std::int64_t support = cell->support_q16 +
      (authority == ResidentRecipeRevisionAuthority::structural ? event.resource_delta : 0);
  const std::int64_t credit = cell->credit_q16 +
      (authority == ResidentRecipeRevisionAuthority::experience ? event.resource_delta : 0);
  if ((authority != ResidentRecipeRevisionAuthority::structural &&
       authority != ResidentRecipeRevisionAuthority::experience) ||
      event.identity != resident_recipe_revision_identity(
          cell->logical_recipe_id, cell->revision_identity, cell->revision + 1u,
          resident_recipe_revision_contributor(event), support, credit)) return false;
  cell->revision_identity = event.identity; ++cell->revision;
  cell->support_q16 = support; cell->credit_q16 = credit; set_resident_recipe_current_revision_authority(cell, authority);
  if (cell->receptor_state.causal_identity != 0u)
    cell->receptor_state.revision_identity = event.identity;
  return true;
}

DIRECT_NETWORK_HD inline void advance_resident_recipe_revision(
    ResidentRecipeCell* cell, std::uint64_t contributor_identity,
    std::int64_t support_q16, std::int64_t credit_q16) {
  if (cell == nullptr || contributor_identity == 0u) return;
  const std::uint64_t next_revision = cell->revision + 1u;
  cell->revision_identity = resident_recipe_revision_identity(
      cell->logical_recipe_id, cell->revision_identity, next_revision,
      contributor_identity, support_q16, credit_q16);
  cell->revision = next_revision;
  cell->support_q16 = support_q16;
  cell->credit_q16 = credit_q16; set_resident_recipe_current_revision_authority(cell, ResidentRecipeRevisionAuthority::none);
  if (cell->receptor_state.causal_identity != 0u)
    cell->receptor_state.revision_identity = cell->revision_identity;
}

DIRECT_NETWORK_HD inline void advance_resident_recipe_credit_revision(
    ResidentRecipeCell* cell, const DirectExactHistoryRecord& record,
    std::uint32_t occurrence_ordinal) {
  std::uint64_t contributor = exact_history_fold_word(
      0x7265636970656372ull, record.identity);
  contributor = exact_history_fold_word(contributor, record.parent_identity);
  contributor = exact_history_fold_word(contributor, record.source);
  contributor = exact_history_fold_word(contributor, record.subject);
  contributor = exact_history_fold_word(contributor, record.context);
  contributor = exact_history_fold_word(contributor, record.flags);
  contributor = exact_history_fold_word(contributor, record.resident_tick);
  contributor = exact_history_fold_word(contributor, record.event_tick);
  contributor = exact_history_fold_word(contributor, record.incarnation_before);
  contributor = exact_history_fold_word(contributor, record.incarnation_after);
  contributor = exact_history_fold_word(
      contributor, static_cast<std::uint64_t>(record.resource_delta));
  contributor = exact_history_fold_word(contributor, occurrence_ordinal);
  advance_resident_recipe_revision(
      cell, contributor == 0u ? 1u : contributor, cell->support_q16,
      static_cast<std::int64_t>(record.incarnation_after)); set_resident_recipe_current_revision_authority(cell, ResidentRecipeRevisionAuthority::experience);
}

struct ResidentRecipeEdge {
  std::uint16_t source_cell;
  std::uint16_t target_cell;
  ResidentRecipeRelation relation;
  std::uint16_t flags;
  std::int32_t weight_q16;
  std::uint32_t field_index;
};
static_assert(std::is_standard_layout_v<ResidentRecipeEdge> &&
              std::is_trivial_v<ResidentRecipeEdge>);

struct ResidentRecipeRange {
  std::uint32_t index_offset;
  std::uint16_t index_count;
  std::uint16_t reserved;
};
static_assert(std::is_standard_layout_v<ResidentRecipeRange> &&
              std::is_trivial_v<ResidentRecipeRange>);

inline constexpr std::uint32_t kResidentRematerializedSupportCapacity = 4u;
inline constexpr std::uint32_t kResidentRematerializedSupportMinimum = 2u;
struct ResidentRematerializedRecipeSupport {
  std::uint64_t logical_recipe_id, revision_identity, generation;
  std::uint32_t recipe_cell, relation;
  std::int32_t parameter_q16;
  std::uint32_t input_node, output_node;
  std::uint16_t input_arity, output_arity;
  std::uint8_t input_domain, input_direction;
  std::uint8_t output_domain, output_direction;
  std::uint32_t reserved;
};
struct ResidentRefinementRematerializationState {
  std::uint64_t compacted_logical_recipe_id, compacted_revision_identity;
  std::uint64_t witness_identity, instruction_identity;
  std::uint64_t contradiction_identity, transaction_identity;
  ResidentRematerializedRecipeSupport
      support[kResidentRematerializedSupportCapacity];
  std::uint32_t support_count, generation;
  std::uint32_t active, refusals;
};
static_assert(std::is_standard_layout_v<ResidentRematerializedRecipeSupport> &&
              std::is_trivial_v<ResidentRematerializedRecipeSupport> &&
              std::has_unique_object_representations_v<
                  ResidentRematerializedRecipeSupport>);
static_assert(std::is_standard_layout_v<ResidentRefinementRematerializationState> &&
              std::is_trivial_v<ResidentRefinementRematerializationState> &&
              std::has_unique_object_representations_v<
                  ResidentRefinementRematerializationState>);

DIRECT_NETWORK_HD inline std::uint64_t
resident_refinement_instruction_identity(
    const ResidentRefinementRematerializationState& state) {
  std::uint64_t identity = exact_history_fold_word(
      0x72656d6174696e73ull, state.compacted_logical_recipe_id);
  identity = exact_history_fold_word(identity, state.compacted_revision_identity);
  identity = exact_history_fold_word(identity, state.witness_identity);
  identity = exact_history_fold_word(identity, state.support_count);
  for (std::uint32_t i = 0u; i < state.support_count &&
       i < kResidentRematerializedSupportCapacity; ++i) {
    const auto& source = state.support[i];
    identity = exact_history_fold_word(identity, source.logical_recipe_id);
    identity = exact_history_fold_word(identity, source.revision_identity);
    identity = exact_history_fold_word(identity, source.generation);
    identity = exact_history_fold_word(identity, source.recipe_cell);
    identity = exact_history_fold_word(identity, source.relation);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(source.parameter_q16));
    identity = exact_history_fold_word(identity, source.input_node);
    identity = exact_history_fold_word(identity, source.output_node);
  }
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline std::uint64_t
resident_refinement_transaction_identity(
    const ResidentRefinementRematerializationState& state) {
  std::uint64_t identity = exact_history_fold_word(
      0x72656d617474786eull, state.instruction_identity);
  identity = exact_history_fold_word(identity, state.contradiction_identity);
  identity = exact_history_fold_word(identity, state.generation);
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline bool resident_refinement_rematerialization_valid(
    const ResidentRefinementRematerializationState& state) {
  return state.active == 1u &&
      state.support_count >= kResidentRematerializedSupportMinimum &&
      state.support_count <= kResidentRematerializedSupportCapacity &&
      state.compacted_logical_recipe_id != 0u &&
      state.compacted_revision_identity != 0u && state.witness_identity != 0u &&
      state.contradiction_identity != 0u &&
      state.instruction_identity == resident_refinement_instruction_identity(state) &&
      state.transaction_identity == resident_refinement_transaction_identity(state);
}

inline constexpr std::uint32_t kResidentRecruitedNetworkCapacity = 64u;
inline constexpr std::uint32_t kResidentRecruitedNetworkMaxMembers = 8u;

struct alignas(8) ResidentRecruitedNetworkIncidence {
  // Persistent recruitment is compact morphology/credit mathematics. Exact
  // active Network identity, membership and eligibility are unfolded from the
  // current Occurrence frontier and are never serialized as a latest snapshot.
  std::uint64_t recruitment_identity, activation_count;
  std::int64_t credit_q16;
  std::uint32_t last_active_tick, last_credit_tick;
  std::uint16_t member_count, coupling_count, boundary_count, actual_count;
  std::uint64_t reserved;
};
static_assert(sizeof(ResidentRecruitedNetworkIncidence) == 48u &&
              std::is_standard_layout_v<ResidentRecruitedNetworkIncidence> &&
              std::is_trivial_v<ResidentRecruitedNetworkIncidence> &&
              std::has_unique_object_representations_v<
                  ResidentRecruitedNetworkIncidence>);

struct alignas(8) ResidentRecruitedNetworkState {
  ResidentRecruitedNetworkIncidence incidences[kResidentRecruitedNetworkCapacity];
  std::uint32_t incidence_count, capacity_refusals;
  std::uint64_t nominations, activations;
};
static_assert(sizeof(ResidentRecruitedNetworkState) == 3096u &&
              std::is_standard_layout_v<ResidentRecruitedNetworkState> &&
              std::is_trivial_v<ResidentRecruitedNetworkState> &&
              std::has_unique_object_representations_v<
                  ResidentRecruitedNetworkState>);

// D1 Adult admission is resident developmental state, not an observer label.
// The counters name physical post-birth changes made by the generic resident
// development transaction.  No host or certification API can write them.
struct alignas(8) ResidentAdultAdmissionState {
  std::uint64_t development_epochs;
  std::uint64_t field_update_events;
  std::uint64_t structural_revision_events;
  std::uint32_t admission_tick;
  std::uint32_t earned;
};
static_assert(sizeof(ResidentAdultAdmissionState) == 32u &&
              std::is_standard_layout_v<ResidentAdultAdmissionState> &&
              std::is_trivial_v<ResidentAdultAdmissionState> &&
              std::has_unique_object_representations_v<
                  ResidentAdultAdmissionState>);

struct ResidentDevelopmentState {
  std::uint32_t age_tick;
  std::uint32_t phase;
  std::uint32_t plasticity_q16;
  std::uint32_t mature_plasticity_floor_q16;
  std::uint32_t critical_period_q16;
  std::uint32_t inhibition_gain_q16;
  std::uint64_t constructor_reserve;
  std::uint64_t reclaimed_resource;
  std::uint64_t live_route_matter;
  std::uint64_t live_node_matter;
  std::uint32_t field_count;
  std::uint32_t constructor_rule_count;
  std::uint32_t recipe_cell_count;
  std::uint32_t recipe_edge_count;
  std::uint32_t birth_handoff_tick;
  std::uint32_t last_maturation_event;
  ResidentAdultAdmissionState adult_admission;
  DirectExactHistoryHotPage exact_history;
  DirectExactHistoryTierState exact_history_tiers;
  ResidentRecruitedNetworkState recruited_networks;
  ResidentRefinementRematerializationState refinement_rematerialization;
  // Exact unresolved motor lineage displaced by bounded ticket-slot reuse.
  // This is resident checkpointed state: a late physical return is resolved
  // against the action that actually occurred, never the newer slot occupant.
  struct alignas(8) DelayedActionParticipant {
    std::uint64_t participant_ticket_id, logical_recipe_id, revision_identity;
    std::uint64_t occurrence_identity, participation_identity;
    std::uint64_t occurrence_route_incarnation;
    std::uint32_t source_node, target_node, route_index, context_signature;
    std::uint32_t occurrence_context_signature, composition_depth;
    std::uint32_t expiry_tick, claim_incarnation;
    std::uint64_t route_incarnation;
    std::uint32_t authority_incarnation, authority, contribution_kind;
    std::int32_t frozen_eligibility_q16;
    std::uint32_t eligibility_slot, eligibility_generation;
  };
  struct alignas(8) DelayedActionRecord {
    std::uint64_t ticket_id, upstream_ticket_id;
    std::uint32_t motor_node, motor_channel, motor_word, context_signature;
    std::uint32_t emission_tick, ticket_reserved;
    std::uint64_t action_ticket_id, action_network_identity;
    std::uint64_t action_recruitment_identity;
    std::int64_t action_network_eligibility_signed_q16;
    std::uint64_t action_network_eligibility_l1_q16;
    std::uint32_t participant_offset, participant_count, action_emission_tick;
    std::uint32_t action_context_signature, action_motor_node;
    std::uint32_t action_motor_channel, expiry_tick;
    std::uint32_t occurrence_identity_required, occurrence_identity_complete;
    std::uint32_t action_reserved;
    DelayedActionParticipant participants[kMaxResidentActionParticipationLinks];
    std::uint32_t state, reserved0;
  };
  static constexpr std::uint32_t kDelayedActionCapacity = 8u;
  DelayedActionRecord delayed_actions[kDelayedActionCapacity];
  std::uint64_t delayed_actions_preserved;
  std::uint64_t delayed_actions_settled;
  std::uint64_t delayed_action_backpressure;
};
static_assert(std::is_standard_layout_v<ResidentDevelopmentState> && std::is_trivial_v<ResidentDevelopmentState>);
static_assert(std::is_standard_layout_v<ResidentDevelopmentState::DelayedActionParticipant> &&
              std::is_trivial_v<ResidentDevelopmentState::DelayedActionParticipant> &&
              std::has_unique_object_representations_v<ResidentDevelopmentState::DelayedActionParticipant>);
static_assert(std::is_standard_layout_v<ResidentDevelopmentState::DelayedActionRecord> &&
              std::is_trivial_v<ResidentDevelopmentState::DelayedActionRecord> &&
              std::has_unique_object_representations_v<ResidentDevelopmentState::DelayedActionRecord>);
#include "direct_network_construction_fronts.inl"
// The birth/current-state root hashes the raw arena bytes. Hidden C++ padding
// would make that identity depend on unspecified bytes, so every arena record
// must have a unique object representation. Explicit reserved fields are part
// of the ABI and are zeroed by arena initialization.
static_assert(std::has_unique_object_representations_v<DirectNode>);
static_assert(std::has_unique_object_representations_v<DirectRoute>);
static_assert(std::has_unique_object_representations_v<DirectDenseBlock>);
static_assert(std::has_unique_object_representations_v<DirectBoundaryPort>); static_assert(std::has_unique_object_representations_v<ResidentTerritoryAncestry>);
static_assert(std::has_unique_object_representations_v<ResidentDevelopmentField>);
static_assert(std::has_unique_object_representations_v<ResidentFieldRange>);
static_assert(std::has_unique_object_representations_v<ResidentConstructorRule>);
static_assert(std::has_unique_object_representations_v<ResidentRecipeCell>);
static_assert(std::has_unique_object_representations_v<ResidentRecipeEdge>);
static_assert(std::has_unique_object_representations_v<ResidentRecipeRange>);
static_assert(std::has_unique_object_representations_v<ResidentDevelopmentState>);

// One contiguous arena owns the born brain.  Pointer members are views into
// that allocation; the external Life Function owns no pointer once compilation
// returns.
struct DirectBrain {
  void* arena;
  std::uint64_t arena_bytes;
  DirectNode* nodes;
  DirectRoute* routes;
  std::uint64_t* route_incarnations;
  // #1178: future-affecting reclaim opportunity belongs to the born brain, not
  // disposable adult-runtime scratch. A route is opportunity-tested only when
  // this stamp equals the current structural incarnation for the same slot.
  std::uint64_t* route_opportunity_incarnations; std::uint64_t* route_delay_law_incarnations;
  std::uint32_t* route_delay_law_indices; std::uint32_t* route_mature_delays;
  DirectDenseBlock* dense_blocks;
  std::uint16_t* dense_weight_fp16_bits;
  DirectBoundaryPort* boundary_ports; ResidentTerritoryAncestry* territory_ancestry;
  ResidentDevelopmentField* resident_fields;
  ResidentFieldRange* resident_field_ranges;
  std::uint16_t* resident_field_indices;
  ResidentConstructorRule* resident_rules; std::uint32_t* resident_tract_delay_laws;
  ResidentRecipeCell* recipe_cells;
  ResidentRecipeEdge* recipe_edges;
  ResidentRecipeRange* recipe_ranges;
  std::uint16_t* recipe_indices;
  ResidentDevelopmentState* development; ResidentConstructionFront* construction_fronts; ResidentRecipeDerivation* postbirth_derivations; ResidentPostbirthConstructorState* postbirth_constructor;
  std::uint32_t* construction_front_count; std::uint64_t* construction_front_generation_by_node;
  std::uint32_t node_count;
  std::uint32_t active_route_count;
  std::uint32_t route_capacity;
  std::uint32_t dense_block_count;
  std::uint32_t dense_weight_count;
  std::uint32_t boundary_port_count;
  std::uint32_t resident_field_count;
  std::uint32_t resident_field_range_count;
  std::uint32_t resident_field_index_count;
  std::uint32_t resident_rule_count; std::uint32_t resident_tract_delay_law_count;
  std::uint32_t recipe_cell_count; std::uint32_t recipe_cell_capacity;
  std::uint32_t recipe_edge_count;
  std::uint32_t recipe_range_count;
  std::uint32_t recipe_index_count;
  std::uint32_t territory_count; std::uint32_t territory_ancestry_count; std::uint32_t construction_front_capacity;
  std::uint32_t long_tract_count;
  Root256 genome_root;
  Root256 territory_layout_root;
  Root256 body_root;
  Root256 environment_root;
  Root256 birth_root;
  // Structured, generation-bound causal evidence for each physical route.
  // This is born matter: resident turnover and adult settlement share it, and
  // arena ownership makes cloning/checkpoint replay preserve the same state.
  substrate::direct_adult::DirectRetentionState* retention_bank;
  // #1178: the grown brain's matter is one arena allocation, and until this
  // existed no capacity law governed it. The adult (DirectBrainV01) carries its
  // own ledger, but the two are disjoint stacks with no bridge, so the ledger
  // that governs the adult could never reach the organism gestation produces.
  substrate::direct_adult::DirectResourceEcologyState* resource_ecology;
};
static_assert(std::is_standard_layout_v<DirectBrain> && std::is_trivial_v<DirectBrain>);
#if defined(__CUDACC__)
__device__ bool materialize_postbirth_route_recipe_parent_external(
    DirectBrain brain, DirectRoute& route, std::uint32_t route_index,
    std::uint64_t route_incarnation, std::uint32_t* parent_cell);
#endif

DIRECT_NETWORK_HD inline bool evaluate_resident_rematerialized_support(
    const ResidentRefinementRematerializationState& state,
    std::int32_t input_q16, std::int32_t* output_q16) {
  if (output_q16 == nullptr ||
      !resident_refinement_rematerialization_valid(state))
    return false;
  std::int32_t value = input_q16;
  for (std::uint32_t i = 0u; i < state.support_count; ++i) {
    const ResidentRematerializedRecipeSupport& source = state.support[i];
    if (source.logical_recipe_id == 0u || source.revision_identity == 0u ||
        source.input_direction != static_cast<std::uint8_t>(
            ResidentRecipePortDirection::input) ||
        source.output_direction != static_cast<std::uint8_t>(
            ResidentRecipePortDirection::output) ||
        !evaluate_resident_recipe_boundary_q16(
            source.relation, source.parameter_q16, value, &value))
      return false;
    for (std::uint32_t prior = 0u; prior < i; ++prior)
      if (state.support[prior].recipe_cell == source.recipe_cell) return false;
    if (i != 0u) {
      const auto& previous = state.support[i - 1u];
      if (previous.output_node != source.input_node ||
          previous.output_arity != source.input_arity ||
          previous.output_domain != source.input_domain)
        return false;
    }
  }
  *output_q16 = value;
  return true;
}

#if defined(__CUDACC__)
__global__ void rematerialize_condensed_support_kernel(
    DirectBrain brain, std::uint32_t derivation_index,
    DirectExactHistoryRecord contradiction);
#endif

inline constexpr std::uint64_t kDirectSiliconBudgetBytes = 12ull * 1024ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kDirectPoolUnitLimit = 0xffffffffull;

struct DirectCompileOptions {
  std::uint64_t silicon_byte_budget = kDirectSiliconBudgetBytes;
  std::uint64_t route_pool_capacity = kDirectPoolUnitLimit;
  std::uint64_t dense_tile_pool_capacity = kDirectPoolUnitLimit;
  std::uint64_t boundary_port_pool_capacity = kDirectPoolUnitLimit;
  std::uint64_t recipe_cell_pool_capacity = kDirectPoolUnitLimit;
  std::uint64_t recipe_edge_pool_capacity = kDirectPoolUnitLimit;
  std::uint32_t block_size = 256u;
  std::uint32_t candidate_targets = 8u;
  std::uint32_t overload_refinement_passes = 2u;
  std::uint32_t prenatal_stabilization_passes = 4u;
  std::uint32_t route_reserve_per_node = kDefaultRouteReserve;
  std::uint32_t maximum_in_degree = 64u;
};

struct DirectBirthReceiptV1 {
  Root256 genome_root{};
  Root256 territory_layout_root{};
  Root256 body_root{};
  Root256 environment_root{};
  Root256 birth_root{};
  std::uint32_t node_count = 0u;
  std::uint32_t active_route_count = 0u;
  std::uint32_t route_capacity = 0u;
  std::uint32_t territory_count = 0u;
  std::uint32_t dense_block_count = 0u;
  std::uint32_t dense_weight_count = 0u;
  std::uint32_t long_tract_count = 0u;
  std::uint32_t resident_field_count = 0u;
  std::uint32_t resident_rule_count = 0u;
  std::uint32_t recipe_cell_count = 0u;
  std::uint32_t recipe_edge_count = 0u;
  std::uint32_t recipe_range_count = 0u;
  std::uint32_t recipe_index_count = 0u;
  std::uint32_t maximum_observed_in_degree = 0u;
  // gh #1309: routes the ring fallback placed, not Gamma. See #1290 C0.
  std::uint32_t fallback_wired_route_count = 0u;
  // gh #1332: nodes placed inside a hard-excluded region. See #1319.
  std::uint32_t environment_violating_node_count = 0u;
  // gh #1348: nodes rescued by the bounded extended candidate draw after all
  // four first-draw candidates were hard-excluded.
  std::uint32_t extended_draw_node_count = 0u;
  // gh #1268 / #1267. Long tracts REFUSED because the source territory declares
  // a partner affinity no territory in this organism satisfies. A named corridor
  // with no partner must grow nothing rather than fall through to the geometric
  // argmax, and the refusal is reported so it is a measurement rather than a
  // silent absence.
  std::uint32_t refused_partner_tract_count = 0u;
  // Long tracts whose chosen target carried a nonzero partner-affinity term.
  // This is the dose precondition for any claim that authored affinity steered
  // growth: zero here means the mechanism never fired, and a passing direction
  // test would be measuring geometry.
  std::uint32_t partner_steered_tract_count = 0u;
  // gh #1268 / #1267. Long tracts a node did NOT grow because its own
  // developmental birth tick fell outside the family's authored
  // minimum_age/maximum_age maturation window. Zero here means either no
  // genome authors a window yet or none is currently closed against the
  // sampled nodes -- not that the mechanism does not exist.
  std::uint32_t immature_deferred_tract_count = 0u;
  std::uint64_t arena_bytes = 0u;
  std::uint64_t logical_recipe_bytes = 0u;
  float planning_ms = 0.0f;
  float materialization_ms = 0.0f;
  float total_gestation_ms = 0.0f;
  bool compact_recipe = false;
  bool final_connectome_loaded = false;
  bool external_life_function_detached = false;
  bool resident_development_present = false;
};

DIRECT_NETWORK_HD inline bool route_is_active(const DirectRoute& route) {
  return (route.flags & kRouteFlagActive) != 0u;
}

}  // namespace substrate::direct_network
#include "direct_adult_volitional_veto.cuh"
#undef DIRECT_NETWORK_HD

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_BRAIN_CUH

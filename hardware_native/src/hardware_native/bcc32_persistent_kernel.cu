#include "bcc32_persistent_kernel.hpp"

#include "bcc32_adaptive_ecology.cuh"
#include "bcc32_developmental_adult_bridge.hpp"
#include "bcc32_device_ordinary_f_timeline.cuh"
#include "bcc32_execution_image_identity.hpp"
#include "bcc32_law_identity.hpp"
#include "bcc32_persistent_language_tissue.cuh"
#include "bcc32_sparse_source_joint.cuh"
#include "bcc32_source_joint_field_response.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace substrate::bcc32::persistent_kernel {
namespace {

namespace language = persistent_language_tissue;
namespace adaptive = adaptive_ecology;
namespace grown = developmental_adult;
namespace ordinary_f = device_ordinary_f_timeline;

constexpr std::size_t kMaxCells = 256u;
constexpr std::size_t kMaxContactWords = PersistentKernel::kMaximumRawContactWords;
constexpr std::size_t kIngressSlots = 4u;
constexpr std::size_t kEgressSlots = 4u;
constexpr unsigned kDevicePacingNanoseconds = 1000000u;
constexpr std::uint32_t kPredictionRoutes = adaptive::kRouteCount;
constexpr std::int32_t kTraceMax = 4096;
constexpr std::int32_t kResourceMax = 1024;
constexpr std::int32_t kPredictionMax = 4096;
constexpr std::int32_t kCreditWeightMax = 4096;
constexpr std::uint32_t kFieldSourceCapacity =
    sparse_source_joint::kSourceCapacity;
constexpr std::int32_t kFieldFluxLimit = 8192;
constexpr std::uint32_t kFieldPacketHistoryLifetime = 64u;
constexpr std::int64_t kFieldPacketBraidScale = 4096;
constexpr std::uint32_t kFieldPacketMaturityUpdates = 8u;
constexpr std::uint32_t kFieldPacketHistoryExpired = 0xffffffffu;
constexpr std::uint64_t kRawContactByteCapacity = 1u << 20u;
static_assert(kMaxCells == adaptive::kMaxCells);
static_assert(kAdaptiveInitialStructuralMatterQ8PerCell ==
              adaptive::kInitialStructuralMatterQ8);
static_assert(kAdaptiveEdgeStructuralDoseQ8 ==
              adaptive::kEdgeStructuralDoseQ8);
static_assert(kAdaptiveLocalProposalRadius == adaptive::kLocalProposalRadius);

struct DeviceDigest {
  std::uint8_t bytes[32]{};
  std::uint64_t byte_count = 0u;
};

struct DeviceReceipt {
  std::uint64_t tick = 0u;
  std::uint64_t phase = 0u;
  std::uint64_t contact_sequence = 0u;
  DeviceDigest sealed_execution{};
  DeviceDigest law{};
  DeviceDigest image{};
  DeviceDigest input{};
  DeviceDigest output{};
  DeviceDigest predecessor{};
  DeviceDigest commitment{};
  std::int32_t joint_output = 0;
  std::int32_t joint_residual = 0;
  std::uint32_t joint_output_valid = 0u;
  std::uint64_t joint_route = 0u;
  std::int32_t field_response_l1 = 0;
  std::int32_t field_residual_l1 = 0;
  std::int32_t field_peak_response = 0;
  std::int32_t field_peak_residual = 0;
  std::uint32_t field_active_routes = 0u;
  std::uint32_t field_contact_routes = 0u;
  std::uint32_t field_responsive_routes = 0u;
  std::uint64_t field_owner_mix = 0u;
  std::int32_t developmental_pressure_l1 = 0;
  std::int32_t functional_horizon = 0;
  std::int32_t functional_integration = 0;
  std::int32_t functional_delay = 0;
  std::int32_t functional_spatial_tv_l1 = 0;
  DeviceDigest structure{};
  DeviceDigest intervention{};
  std::uint64_t intervention_sequence = 0u;
  std::uint32_t structural_focus_cell = 0u;
  std::uint32_t removed_matter_q8_sum = 0u;
  std::uint32_t cut_coupling_q8_sum = 0u;
  std::uint32_t structural_support_q8 = 0u;
  std::int32_t effective_field_response_l1 = 0;
  std::uint32_t intervention_active = 0u;
  std::uint32_t field_source_count = 0u;
  std::uint32_t field_supported_source_count = 0u;
  std::uint32_t field_receptor_count = 0u;
  std::int32_t field_packet_flux_l1 = 0;
  std::int32_t field_polarity_l1 = 0;
  std::uint64_t field_return_packets_q8 = 0u;
  std::uint32_t field_near_profile = 0u;
  std::uint32_t field_middle_profile = 0u;
  std::uint32_t field_far_profile = 0u;
  std::uint64_t field_anchor_mix = 0u;
  std::uint32_t field_anchor_moment = 0u;
  std::uint32_t field_profile_tv_l1 = 0u;
  std::uint64_t field_relation_mix = 0u;
  std::uint32_t field_relation_l1 = 0u;
  std::int32_t field_relation_sum = 0;
  std::int32_t field_polarity_sum = 0;
  std::uint32_t field_receptor_gain_q8 = 256u;
  std::uint32_t field_diffusion_l1 = 0u;
  std::uint32_t field_damage_q8 = 0u;
  std::uint32_t field_recruited_sources = 0u;
  std::uint32_t field_repaired_sources = 0u;
  std::uint32_t field_growth_front_count = 0u;
  std::uint32_t field_component_count = 0u;
  std::int32_t field_component0_response_l1 = 0;
  std::int32_t field_component1_response_l1 = 0;
  std::int32_t field_component0_residual_l1 = 0;
  std::int32_t field_component1_residual_l1 = 0;
  std::uint32_t field_component0_support_q8 = 0u;
  std::uint32_t field_component1_support_q8 = 0u;
  std::uint32_t field_withdrawn_sources = 0u;
  std::uint32_t active_joint_locus = 0u;
  std::uint32_t tissue_matter_q8 = 256u;
  std::uint32_t tissue_coupling_q8 = 256u;
  std::int32_t field_packet_braid_sum = 0;
  std::uint32_t field_packet_braid_l1 = 0u;
  std::uint32_t field_packet_braid_sources = 0u;
  std::uint32_t adaptive_context_l1 = 0u;
  std::uint32_t adaptive_context_anchor = 0u;
  std::uint32_t adaptive_context_matter_q8 = 0u;
  std::uint32_t adaptive_plasticity_q8 = 0u;
  std::uint32_t adaptive_consolidation_q8 = 0u;
  std::uint32_t adaptive_growth_q8 = 0u;
  std::uint32_t adaptive_turnover_q8 = 0u;
  std::uint32_t adaptive_repair_q8 = 0u;
  std::uint32_t adaptive_replay_q8 = 0u;
  std::uint32_t adaptive_live_edges = 0u;
  std::uint32_t adaptive_consolidated_routes = 0u;
  std::uint64_t adaptive_structural_revision = 0u;
  std::uint64_t adaptive_recruited_edges = 0u;
  std::uint64_t adaptive_pruned_edges = 0u;
  std::uint64_t adaptive_repaired_cells = 0u;
  std::uint64_t adaptive_free_structural_matter_q8 = 0u;
  std::uint64_t adaptive_edge_structural_matter_q8 = 0u;
  std::uint64_t adaptive_cumulative_structural_debit_q8 = 0u;
  std::uint64_t adaptive_cumulative_structural_refund_q8 = 0u;
  std::uint64_t adaptive_local_proposal_activity = 0u;
  std::uint32_t alternate_expression_endpoint = 0u;
  std::uint32_t joint_expression_endpoints = 0u;
  std::uint32_t primary_expression_gain_q8 = 0u;
  std::uint32_t alternate_expression_gain_q8 = 0u;
  std::int32_t field_packet_alignment_sum = 0;
  std::uint32_t field_packet_alignment_l1 = 0u;
  std::uint32_t field_packet_alignment_sources = 0u;
  std::uint32_t field_mature_packet_sources = 0u;
  std::int32_t field_effective_alignment_sum = 0;
  std::int32_t field_effective_braid_sum = 0;
  std::uint32_t field_effective_alignment_l1 = 0u;
  std::uint32_t field_effective_braid_l1 = 0u;
  std::uint32_t field_alignment_locus = 0u;
  std::uint32_t field_braid_locus = 0u;
  std::int32_t recurrent_context_l1_q16 = 0;
  std::uint64_t recurrent_context_mix = 0u;
  std::uint64_t recurrent_context_revision = 0u;
  DeviceDigest genesis_manifest{};
  DeviceDigest f_world{};
  std::uint64_t completed_f_ticks = 0u;
  std::uint64_t f_generation = 0u;
  std::uint32_t f_fault = 0u;
  std::uint32_t f_owned_clock = 0u;
  std::uint32_t legacy_action_authority = 0u;
  std::uint32_t f_motor_zero = 0u;
  std::uint32_t f_motor_one = 0u;
};

struct FieldSourceState {
  std::uint64_t owner = 0u;
  std::uint32_t anchor = 0u;
  std::int32_t flux_q8 = 0;
  std::int32_t receptor_q8 = 0;
  std::int32_t polarity_q8 = 0;
  std::int32_t relation_q8 = 256;
  std::uint32_t support_q8 = 0u;
  std::uint32_t receptor_gain_q8 = 256u;
  std::uint32_t damage_q8 = 0u;
  std::uint32_t repair_reserve_q8 = 0u;
  std::uint32_t neighbor_mask = 0u;
  std::uint32_t return_age = 0u;
  std::uint32_t support_age = 0u;
  // The prior packet and its bounded causal age are resident state. The
  // signed braid is the antisymmetric temporal composition of successive
  // packets at this source, not a host-selected relation label.
  std::int32_t packet_history_flux_q8 = 0;
  std::int32_t packet_history_polarity_q8 = 0;
  std::uint32_t packet_history_age = kFieldPacketHistoryExpired;
  std::int32_t packet_braid_q8 = 0;
  std::int32_t packet_alignment_q8 = 0;
  std::uint32_t packet_history_updates = 0u;
  std::uint8_t live = 0u;
  std::uint8_t reserved[3]{};
};

struct IngressSlot {
  std::uint64_t sequence = 0u;
  std::uint32_t count = 0u;
  std::uint32_t raw_byte_count = 0u;
  BoundaryWord words[kMaxContactWords]{};
  // Byte view of the same admitted raw event and sequence as the body, read by
  // the ordinary-F boundary gate. This is transport representation only; it is
  // not a second ingress queue, clock, corpus path, or semantic channel.
  std::uint8_t raw_bytes[kRawContactByteCapacity]{};
};

struct IngressRing {
  std::uint64_t published = 0u;
  std::uint64_t consumed = 0u;
  IngressSlot slots[kIngressSlots]{};
};

struct PhysicalIngress {
  std::uint64_t published = 0u;
  std::uint64_t consumed = 0u;
  RawPhysicalIntervention event{};
};

enum : std::uint32_t {
  kEgressReady = 0u,
  kEgressWriting = 1u,
  kEgressReading = 2u,
};

struct EgressSlot {
  std::uint64_t generation = 0u;
  std::uint32_t state = kEgressReady;
  std::uint32_t reserved = 0u;
  std::uint64_t energy = 0u;
  std::uint64_t host_bootstrap_launches = 0u;
  std::uint64_t device_epochs = 0u;
  DeviceReceipt receipt{};
  BoundaryWord actions[kMaxCells]{};
  language::RawTrajectoryFrame language_frame{};
  std::uint64_t language_contacts = 0u;
  std::uint64_t language_recruited_cells = 0u;
  std::uint64_t language_reused_cells = 0u;
  std::uint64_t language_strengthened_edges = 0u;
};

struct EgressRing {
  std::uint64_t published = 0u;
  EgressSlot slots[kEgressSlots]{};
};

struct Lifecycle {
  std::uint32_t shutdown = 0u;
  std::uint32_t stopped = 0u;
  std::uint32_t continuation_fault = 0u;
};

enum class ContinuationFault : std::uint32_t {
  none = 0u,
  ordinary_f_handle = 1u,
  ordinary_f_publish = 2u,
  ordinary_f_receipt_tick = 3u,
  ordinary_f_receipt_clock = 4u,
  ordinary_f_boundary = 5u,
  ordinary_f_launch_base = 0x10000u,
  root_tail_launch_base = 0x20000u,
  ordinary_f_fault_base = 0x30000u,
};

__device__ __forceinline__ std::uint32_t continuation_cuda_fault(
    ContinuationFault base, cudaError_t error) {
  return static_cast<std::uint32_t>(base) |
         static_cast<std::uint32_t>(error);
}

// This allocation is never returned by the public class. The host may copy
// only the declared ingress, passive action projection, energy counter, and
// receipt fields; the resident state and its work buffer remain device-owned.
struct DeviceState {
  std::uint64_t founder = 0u;
  std::uint64_t tick = 0u;
  std::uint64_t phase = 0u;
  std::uint32_t cell_count = 0u;
  std::uint32_t contact_count = 0u;
  std::int32_t state[kMaxCells]{};
  std::int32_t next[kMaxCells]{};
  std::int32_t trace[kMaxCells]{};
  std::int32_t trace_next[kMaxCells]{};
  std::int32_t resource[kMaxCells]{};
  std::int32_t prediction[kPredictionRoutes][kMaxCells]{};
  std::int32_t eligibility[kPredictionRoutes][kMaxCells]{};
  std::uint32_t eligibility_age[kPredictionRoutes][kMaxCells]{};
  std::int32_t credit_weight[kPredictionRoutes][kMaxCells]{};
  BoundaryWord contact[kMaxContactWords]{};
  BoundaryWord actions[kMaxCells]{};
  std::int32_t reafference[kMaxCells]{};
  sparse_source_joint::State source_joint{};
  language::Tissue language{};
  adaptive::State adaptive{};
  std::uint64_t contact_revision = 0u;
  std::uint64_t last_external_revision = 0u;
  std::uint64_t sparse_last_revision = 0u;
  std::int32_t joint_output = 0;
  std::int32_t joint_residual = 0;
  std::uint32_t joint_output_valid = 0u;
  std::uint64_t joint_route = 0u;
  std::int32_t field_response[kMaxCells]{};
  std::int32_t field_residual[kMaxCells]{};
  std::int32_t field_braid_drive[kMaxCells]{};
  std::int32_t field_alignment_drive[kMaxCells]{};
  source_joint_field_response::Metrics field_metrics{};
  std::uint32_t reafference_pending = 0u;
  std::uint64_t energy = 0u;
  std::uint64_t host_bootstrap_launches = 0u;
  std::uint64_t device_epochs = 0u;
  std::uint64_t intervention_sequence = 0u;
  RawPhysicalIntervention intervention{};
  std::uint32_t intervention_remaining = 0u;
  std::uint32_t intervention_active = 0u;
  std::uint32_t structural_focus_cell = 0u;
  std::uint32_t removed_matter_q8_sum = 0u;
  std::uint32_t cut_coupling_q8_sum = 0u;
  std::uint32_t structural_support_q8 = 0u;
  std::int32_t effective_field_response_l1 = 0;
  FieldSourceState field_sources[kFieldSourceCapacity]{};
  std::uint32_t field_source_count = 0u;
  std::uint32_t field_supported_source_count = 0u;
  std::uint64_t field_return_packets_q8 = 0u;
  std::uint32_t field_near_profile = 0u;
  std::uint32_t field_middle_profile = 0u;
  std::uint32_t field_far_profile = 0u;
  std::uint64_t field_anchor_mix = 0u;
  std::uint32_t field_anchor_moment = 0u;
  std::uint32_t field_profile_tv_l1 = 0u;
  std::uint64_t field_relation_mix = 0u;
  std::uint32_t field_relation_l1 = 0u;
  std::int32_t field_relation_sum = 0;
  std::int32_t field_polarity_sum = 0;
  std::uint32_t field_receptor_gain_q8 = 256u;
  std::uint32_t field_diffusion_l1 = 0u;
  std::uint32_t field_damage_q8 = 0u;
  std::uint32_t field_recruited_sources = 0u;
  std::uint32_t field_repaired_sources = 0u;
  std::uint32_t field_growth_front_count = 0u;
  std::uint32_t field_component_count = 0u;
  std::int32_t field_component0_response_l1 = 0;
  std::int32_t field_component1_response_l1 = 0;
  std::int32_t field_component0_residual_l1 = 0;
  std::int32_t field_component1_residual_l1 = 0;
  std::uint32_t field_component0_support_q8 = 0u;
  std::uint32_t field_component1_support_q8 = 0u;
  std::uint32_t field_withdrawn_sources = 0u;
  std::int32_t field_packet_braid_sum = 0;
  std::uint32_t field_packet_braid_l1 = 0u;
  std::uint32_t field_packet_braid_sources = 0u;
  std::int32_t field_packet_alignment_sum = 0;
  std::uint32_t field_packet_alignment_l1 = 0u;
  std::uint32_t field_packet_alignment_sources = 0u;
  std::uint32_t field_mature_packet_sources = 0u;
  std::int32_t field_effective_alignment_sum = 0;
  std::int32_t field_effective_braid_sum = 0;
  std::uint32_t field_effective_alignment_l1 = 0u;
  std::uint32_t field_effective_braid_l1 = 0u;
  std::uint32_t field_alignment_locus = 0u;
  std::uint32_t field_braid_locus = 0u;
  // Device-owned tissue gates expression and transport.  These are physical
  // matter/coupling fields, never host-selected semantic slots.
  std::uint32_t tissue_matter_q8[kMaxCells]{};
  std::uint32_t tissue_coupling_q8[kMaxCells]{};
  std::uint32_t active_joint_index = sparse_source_joint::kNoIndex;
  std::uint32_t active_joint_locus = 0u;
  std::uint32_t alternate_expression_endpoint = 0u;
  std::uint32_t joint_expression_endpoints = 0u;
  std::uint32_t primary_expression_gain_q8 = 0u;
  std::uint32_t alternate_expression_gain_q8 = 0u;
  std::int32_t recurrent_context_l1_q16 = 0;
  std::uint64_t recurrent_context_mix = 0u;
  std::uint64_t recurrent_context_revision = 0u;
  DeviceDigest structure{};
  DeviceDigest intervention_digest{};
  DeviceDigest sealed_execution{};
  DeviceDigest law{};
  DeviceDigest image{};
  DeviceReceipt receipt{};
  ordinary_f::DeviceLaunchHandle ordinary_f{};
  DeviceDigest genesis_manifest{};
  std::uint64_t expected_f_tick = 0u;
  std::uint32_t f_owned_clock = 0u;
};

static_assert(std::is_trivially_copyable_v<DeviceState>);
static_assert(std::is_trivially_copyable_v<IngressRing>);
static_assert(std::is_trivially_copyable_v<PhysicalIngress>);
static_assert(std::is_trivially_copyable_v<EgressRing>);
static_assert(std::is_trivially_copyable_v<Lifecycle>);

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t h,
                                                std::uint64_t value) {
  h ^= value + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 27u;
  return h;
}

__device__ __forceinline__ void digest_finish(DeviceDigest* result,
                                                std::uint64_t lanes[4],
                                                std::uint64_t byte_count) {
  result->byte_count = byte_count;
  for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
    const std::uint64_t value = mix64(lanes[lane], byte_count + lane);
    for (std::uint32_t byte = 0u; byte < 8u; ++byte)
      result->bytes[lane * 8u + byte] =
          static_cast<std::uint8_t>(value >> (byte * 8u));
  }
}

__device__ void digest_words(const BoundaryWord* words, std::uint32_t count,
                             DeviceDigest* result,
                             std::uint64_t seed = 0x243f6a8885a308d3ull) {
  std::uint64_t lanes[4]{seed, seed ^ 0x13198a2e03707344ull,
                         seed ^ 0xa4093822299f31d0ull,
                         seed ^ 0x082efa98ec4e6c89ull};
  for (std::uint32_t index = 0u; index < count; ++index) {
    const std::uint64_t word = words[index];
    lanes[index & 3u] = mix64(lanes[index & 3u], word);
  }
  digest_finish(result, lanes,
                static_cast<std::uint64_t>(count) * sizeof(BoundaryWord));
}

__device__ void absorb_resident_bytes(std::uint64_t lanes[4],
                                      const void* data, std::size_t bytes) {
  const auto* raw = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0u; index < bytes; ++index)
    lanes[index & 3u] = mix64(lanes[index & 3u], raw[index]);
  lanes[0u] = mix64(lanes[0u], bytes);
}

__device__ void digest_state(const DeviceState* state, DeviceDigest* result) {
  std::uint64_t lanes[4]{0x6a09e667f3bcc909ull, 0xbb67ae8584caa73bull,
                         0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull};
  for (std::uint32_t index = 0u; index < state->cell_count; ++index)
    lanes[index & 3u] = mix64(
        lanes[index & 3u],
        static_cast<std::uint32_t>(state->state[index]));
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    lanes[index & 3u] = mix64(
        lanes[index & 3u], static_cast<std::uint32_t>(state->trace[index]));
    lanes[(index + 1u) & 3u] = mix64(
        lanes[(index + 1u) & 3u],
        static_cast<std::uint32_t>(state->resource[index]));
  }
  for (std::uint32_t route = 0u; route < kPredictionRoutes; ++route) {
    for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
      lanes[(route + index) & 3u] = mix64(
          lanes[(route + index) & 3u],
          static_cast<std::uint32_t>(state->prediction[route][index]));
      lanes[(route + index + 1u) & 3u] = mix64(
          lanes[(route + index + 1u) & 3u],
          static_cast<std::uint32_t>(state->eligibility[route][index]));
      lanes[(route + index + 2u) & 3u] = mix64(
          lanes[(route + index + 2u) & 3u],
          state->eligibility_age[route][index]);
      lanes[(route + index + 3u) & 3u] = mix64(
          lanes[(route + index + 3u) & 3u],
          static_cast<std::uint32_t>(state->credit_weight[route][index]));
    }
  }
  lanes[0u] = mix64(lanes[0u], state->contact_revision);
  lanes[1u] = mix64(lanes[1u], state->last_external_revision);
  lanes[2u] = mix64(lanes[2u], state->reafference_pending);
  lanes[0u] = mix64(lanes[0u],
                    static_cast<std::uint32_t>(state->source_joint.rule_gain_q));
  lanes[1u] = mix64(lanes[1u],
                    static_cast<std::uint32_t>(state->source_joint.rule_bias_q));
  lanes[2u] = mix64(lanes[2u],
                    static_cast<std::uint32_t>(state->source_joint.rule_credit));
  lanes[3u] = mix64(lanes[3u], state->source_joint.source_count);
  lanes[0u] = mix64(lanes[0u], state->source_joint.route_count);
  lanes[1u] = mix64(lanes[1u], state->source_joint.joint_count);
  lanes[2u] = mix64(lanes[2u], state->source_joint.live_matter);
  lanes[3u] = mix64(lanes[3u], state->source_joint.free_matter);
  for (std::uint32_t index = 0u; index <
       sparse_source_joint::kSourceCapacity; ++index) {
    const sparse_source_joint::SourceTrace& source =
        state->source_joint.sources[index];
    if (source.live == 0u) continue;
    lanes[index & 3u] = mix64(lanes[index & 3u], source.owner);
    lanes[(index + 1u) & 3u] = mix64(lanes[(index + 1u) & 3u], source.route);
    lanes[(index + 2u) & 3u] = mix64(
        lanes[(index + 2u) & 3u], static_cast<std::uint32_t>(source.operand));
  }
  for (std::uint32_t index = 0u; index <
       sparse_source_joint::kJointCapacity; ++index) {
    const sparse_source_joint::SparseJoint& joint =
        state->source_joint.joints[index];
    if (joint.live == 0u) continue;
    lanes[index & 3u] = mix64(lanes[index & 3u], joint.first_owner);
    lanes[(index + 1u) & 3u] = mix64(lanes[(index + 1u) & 3u], joint.second_owner);
    lanes[(index + 2u) & 3u] = mix64(lanes[(index + 2u) & 3u], joint.route);
    lanes[(index + 3u) & 3u] = mix64(
        lanes[(index + 3u) & 3u], static_cast<std::uint32_t>(joint.gain_q));
    lanes[index & 3u] = mix64(lanes[index & 3u],
                               static_cast<std::uint32_t>(joint.prediction));
    lanes[(index + 1u) & 3u] =
        mix64(lanes[(index + 1u) & 3u], joint.alternate_endpoint);
    lanes[(index + 2u) & 3u] = mix64(
        lanes[(index + 2u) & 3u], joint.alternate_recruitment_updates);
    lanes[(index + 3u) & 3u] = mix64(
        lanes[(index + 3u) & 3u], joint.alternate_endpoint_committed);
  }
  for (std::uint32_t index = 0u; index <
       sparse_source_joint::kRouteCapacity; ++index) {
    const sparse_source_joint::SparseRoute& route =
        state->source_joint.routes[index];
    if (route.live == 0u) continue;
    lanes[index & 3u] = mix64(lanes[index & 3u], route.owner);
    lanes[(index + 1u) & 3u] = mix64(
        lanes[(index + 1u) & 3u], static_cast<std::uint32_t>(route.credit));
  }
  for (std::uint32_t index = 0u; index < state->cell_count; ++index)
    lanes[index & 3u] = mix64(
        lanes[index & 3u], static_cast<std::uint32_t>(state->reafference[index]));
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    lanes[index & 3u] = mix64(lanes[index & 3u], state->tissue_matter_q8[index]);
    lanes[(index + 1u) & 3u] =
        mix64(lanes[(index + 1u) & 3u], state->tissue_coupling_q8[index]);
  }
  lanes[2u] = mix64(lanes[2u], state->active_joint_index);
  lanes[3u] = mix64(lanes[3u], state->active_joint_locus);
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    lanes[index & 3u] = mix64(
        lanes[index & 3u], static_cast<std::uint32_t>(state->field_response[index]));
    lanes[(index + 1u) & 3u] = mix64(
        lanes[(index + 1u) & 3u], static_cast<std::uint32_t>(state->field_residual[index]));
    lanes[(index + 2u) & 3u] = mix64(
        lanes[(index + 2u) & 3u],
        static_cast<std::uint32_t>(state->field_braid_drive[index]));
    lanes[(index + 3u) & 3u] = mix64(
        lanes[(index + 3u) & 3u],
        static_cast<std::uint32_t>(state->field_alignment_drive[index]));
  }
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    const FieldSourceState& field = state->field_sources[index];
    if (field.live == 0u) continue;
    lanes[index & 3u] = mix64(lanes[index & 3u], field.owner);
    lanes[(index + 1u) & 3u] = mix64(lanes[(index + 1u) & 3u], field.anchor);
    lanes[(index + 2u) & 3u] = mix64(
        lanes[(index + 2u) & 3u], static_cast<std::uint32_t>(field.flux_q8));
    lanes[(index + 3u) & 3u] = mix64(
        lanes[(index + 3u) & 3u], static_cast<std::uint32_t>(field.receptor_q8));
    lanes[index & 3u] = mix64(
        lanes[index & 3u], static_cast<std::uint32_t>(field.polarity_q8));
    lanes[(index + 1u) & 3u] = mix64(
        lanes[(index + 1u) & 3u], static_cast<std::uint32_t>(field.relation_q8));
    lanes[(index + 2u) & 3u] = mix64(lanes[(index + 2u) & 3u],
                                     field.support_q8);
    lanes[(index + 3u) & 3u] = mix64(lanes[(index + 3u) & 3u],
                                     field.receptor_gain_q8);
    lanes[(index + 2u) & 3u] = mix64(lanes[(index + 2u) & 3u],
                                     field.damage_q8);
    lanes[(index + 3u) & 3u] = mix64(lanes[(index + 3u) & 3u],
                                     field.neighbor_mask);
    lanes[index & 3u] = mix64(lanes[index & 3u], field.repair_reserve_q8);
    lanes[(index + 1u) & 3u] = mix64(lanes[(index + 1u) & 3u],
                                     field.support_age);
    lanes[(index + 2u) & 3u] = mix64(
        lanes[(index + 2u) & 3u],
        static_cast<std::uint32_t>(field.packet_history_flux_q8));
    lanes[(index + 3u) & 3u] = mix64(
        lanes[(index + 3u) & 3u],
        static_cast<std::uint32_t>(field.packet_history_polarity_q8));
    lanes[index & 3u] = mix64(lanes[index & 3u], field.packet_history_age);
    lanes[(index + 1u) & 3u] = mix64(
        lanes[(index + 1u) & 3u],
        static_cast<std::uint32_t>(field.packet_braid_q8));
    lanes[index & 3u] = mix64(
        lanes[index & 3u], static_cast<std::uint32_t>(field.packet_alignment_q8));
    lanes[(index + 1u) & 3u] =
        mix64(lanes[(index + 1u) & 3u], field.packet_history_updates);
  }
  lanes[0u] = mix64(lanes[0u], state->field_source_count);
  lanes[1u] = mix64(lanes[1u], state->field_supported_source_count);
  lanes[1u] = mix64(lanes[1u], state->field_return_packets_q8);
  lanes[2u] = mix64(lanes[2u], state->field_anchor_mix);
  lanes[3u] = mix64(lanes[3u], state->field_anchor_moment);
  lanes[0u] = mix64(lanes[0u], state->field_profile_tv_l1);
  lanes[3u] = mix64(lanes[3u], state->field_relation_mix);
  lanes[0u] = mix64(lanes[0u], state->field_relation_l1);
  lanes[1u] = mix64(lanes[1u], static_cast<std::uint32_t>(state->field_relation_sum));
  lanes[0u] = mix64(lanes[0u],
                    static_cast<std::uint32_t>(state->field_polarity_sum));
  lanes[1u] = mix64(lanes[1u], state->field_receptor_gain_q8);
  lanes[2u] = mix64(lanes[2u], state->field_diffusion_l1);
  lanes[3u] = mix64(lanes[3u], state->field_damage_q8);
  lanes[0u] = mix64(lanes[0u], state->field_recruited_sources);
  lanes[1u] = mix64(lanes[1u], state->field_repaired_sources);
  lanes[2u] = mix64(lanes[2u], state->field_growth_front_count);
  lanes[3u] = mix64(lanes[3u], state->field_component_count);
  lanes[0u] = mix64(
      lanes[0u], static_cast<std::uint32_t>(state->field_component0_response_l1));
  lanes[1u] = mix64(
      lanes[1u], static_cast<std::uint32_t>(state->field_component1_response_l1));
  lanes[2u] = mix64(
      lanes[2u], static_cast<std::uint32_t>(state->field_component0_residual_l1));
  lanes[3u] = mix64(
      lanes[3u], static_cast<std::uint32_t>(state->field_component1_residual_l1));
  lanes[0u] = mix64(lanes[0u], state->field_component0_support_q8);
  lanes[1u] = mix64(lanes[1u], state->field_component1_support_q8);
  lanes[2u] = mix64(lanes[2u], state->field_withdrawn_sources);
  lanes[3u] = mix64(
      lanes[3u], static_cast<std::uint32_t>(state->field_packet_braid_sum));
  lanes[0u] = mix64(lanes[0u], state->field_packet_braid_l1);
  lanes[1u] = mix64(lanes[1u], state->field_packet_braid_sources);
  lanes[2u] = mix64(lanes[2u], state->alternate_expression_endpoint);
  lanes[3u] = mix64(lanes[3u], state->joint_expression_endpoints);
  lanes[0u] = mix64(lanes[0u], state->primary_expression_gain_q8);
  lanes[1u] = mix64(lanes[1u], state->alternate_expression_gain_q8);
  lanes[2u] = mix64(
      lanes[2u], static_cast<std::uint32_t>(state->field_packet_alignment_sum));
  lanes[3u] = mix64(lanes[3u], state->field_packet_alignment_l1);
  lanes[0u] = mix64(lanes[0u], state->field_packet_alignment_sources);
  lanes[1u] = mix64(lanes[1u], state->field_mature_packet_sources);
  lanes[2u] = mix64(
      lanes[2u], static_cast<std::uint32_t>(state->field_effective_alignment_sum));
  lanes[3u] = mix64(
      lanes[3u], static_cast<std::uint32_t>(state->field_effective_braid_sum));
  lanes[0u] = mix64(lanes[0u], state->field_effective_alignment_l1);
  lanes[1u] = mix64(lanes[1u], state->field_effective_braid_l1);
  lanes[2u] = mix64(lanes[2u], state->field_alignment_locus);
  lanes[3u] = mix64(lanes[3u], state->field_braid_locus);
  lanes[0u] = mix64(
      lanes[0u], static_cast<std::uint32_t>(state->recurrent_context_l1_q16));
  lanes[1u] = mix64(lanes[1u], state->recurrent_context_mix);
  lanes[2u] = mix64(lanes[2u], state->recurrent_context_revision);
  lanes[0u] = mix64(lanes[0u],
                    static_cast<std::uint32_t>(state->field_metrics.response_l1));
  lanes[1u] = mix64(lanes[1u],
                    static_cast<std::uint32_t>(state->field_metrics.residual_l1));
  lanes[2u] = mix64(lanes[2u], state->field_metrics.active_routes);
  lanes[3u] = mix64(lanes[3u], state->field_metrics.contact_routes);
  lanes[0u] = mix64(lanes[0u], state->field_metrics.responsive_routes);
  lanes[0u] = mix64(lanes[0u], state->field_metrics.owner_mix);
  lanes[1u] = mix64(lanes[1u], state->field_metrics.developmental_pressure_l1);
  lanes[2u] = mix64(lanes[2u], state->field_metrics.functional_horizon);
  lanes[3u] = mix64(lanes[3u], state->field_metrics.functional_integration);
  lanes[0u] = mix64(lanes[0u], state->field_metrics.functional_delay);
  lanes[1u] = mix64(lanes[1u], state->field_metrics.functional_spatial_tv_l1);
  lanes[1u] = mix64(lanes[1u], state->intervention_sequence);
  lanes[2u] = mix64(lanes[2u], state->structural_focus_cell);
  lanes[3u] = mix64(lanes[3u], state->removed_matter_q8_sum);
  lanes[0u] = mix64(lanes[0u], state->cut_coupling_q8_sum);
  lanes[1u] = mix64(lanes[1u], state->structural_support_q8);
  lanes[2u] = mix64(lanes[2u], static_cast<std::uint32_t>(
                                      state->effective_field_response_l1));
  lanes[3u] = mix64(lanes[3u], state->intervention_active);
  lanes[0u] = mix64(lanes[0u], state->language.learned_contacts);
  lanes[1u] = mix64(lanes[1u], state->language.recruited_cells);
  lanes[2u] = mix64(lanes[2u], state->language.strengthened_edges);
  for (std::uint32_t index = 0u; index < language::kCellCount; ++index) {
    const language::FormCell& cell = state->language.cells[index];
    lanes[index & 3u] = mix64(lanes[index & 3u], cell.claimed);
    lanes[(index + 1u) & 3u] = mix64(lanes[(index + 1u) & 3u], cell.use_count);
    for (std::uint32_t bit = 0u; bit < language::kMotorBits; ++bit)
      lanes[(index + bit) & 3u] = mix64(
          lanes[(index + bit) & 3u],
          static_cast<std::uint16_t>(cell.motor[bit]));
    for (std::uint32_t edge = 0u; edge < language::kEdgeCapacity; ++edge) {
      lanes[(index + edge + 2u) & 3u] =
          mix64(lanes[(index + edge + 2u) & 3u], cell.targets[edge]);
      lanes[(index + edge + 3u) & 3u] =
          mix64(lanes[(index + edge + 3u) & 3u], cell.strengths[edge]);
    }
  }
  // This commits the per-source reserve, per-edge escrow, cumulative
  // debit/refund receipts, and resident-local proposal activity with the
  // adjacent adaptive ecology state.
  absorb_resident_bytes(lanes, &state->adaptive, sizeof(state->adaptive));
  digest_finish(result, lanes,
                static_cast<std::uint64_t>(state->cell_count) *
                    sizeof(std::int32_t) *
                    (3u + kPredictionRoutes * 4u + 1u) +
                    sizeof(state->source_joint) + sizeof(state->language) +
                    sizeof(state->adaptive) +
                    sizeof(state->intervention));
}

__device__ void absorb_digest(std::uint64_t lanes[4],
                              const DeviceDigest& digest) {
  for (std::uint32_t index = 0u; index < 32u; ++index)
    lanes[index & 3u] = mix64(lanes[index & 3u], digest.bytes[index]);
  lanes[0u] = mix64(lanes[0u], digest.byte_count);
}

__device__ DeviceDigest ordinary_f_world_digest(
    const ordinary_f::WorldDigest& world) {
  DeviceDigest result{};
  static_assert(offsetof(ordinary_f::WorldDigest, lane0) == 0u);
  const auto* source = reinterpret_cast<const std::uint64_t*>(&world);
  std::uint64_t lanes[4]{source[0u], source[1u], source[2u], source[3u]};
  digest_finish(&result, lanes, sizeof(lanes));
  return result;
}

__device__ void digest_commitment(const DeviceState* state,
                                  const DeviceDigest& input,
                                  const DeviceDigest& output,
                                  const DeviceDigest& predecessor,
                                  DeviceDigest* result) {
  std::uint64_t lanes[4]{0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                         0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  absorb_digest(lanes, state->sealed_execution);
  absorb_digest(lanes, state->law);
  absorb_digest(lanes, state->image);
  absorb_digest(lanes, input);
  absorb_digest(lanes, output);
  absorb_digest(lanes, predecessor);
  if (state->f_owned_clock != 0u && state->ordinary_f.publication != nullptr) {
    absorb_digest(lanes, state->genesis_manifest);
    absorb_digest(lanes,
                  ordinary_f_world_digest(
                      state->ordinary_f.publication->world));
    lanes[0u] = mix64(lanes[0u],
                      state->ordinary_f.publication->motor.zero);
    lanes[1u] = mix64(lanes[1u],
                      state->ordinary_f.publication->motor.one);
    lanes[2u] = mix64(lanes[2u],
                      state->ordinary_f.publication->completed_ticks);
    lanes[3u] = mix64(lanes[3u],
                      state->ordinary_f.publication->generation);
  }
  lanes[0u] = mix64(lanes[0u], state->founder);
  lanes[1u] = mix64(lanes[1u], state->tick);
  lanes[2u] = mix64(lanes[2u],
                    static_cast<std::uint32_t>(state->joint_output));
  lanes[3u] = mix64(lanes[3u], state->joint_output_valid);
  lanes[0u] = mix64(lanes[0u], state->joint_route);
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    lanes[index & 3u] = mix64(lanes[index & 3u], state->tissue_matter_q8[index]);
    lanes[(index + 1u) & 3u] =
        mix64(lanes[(index + 1u) & 3u], state->tissue_coupling_q8[index]);
  }
  lanes[2u] = mix64(lanes[2u], state->active_joint_index);
  lanes[3u] = mix64(lanes[3u], state->active_joint_locus);
  lanes[1u] = mix64(lanes[1u],
                    static_cast<std::uint32_t>(state->field_metrics.response_l1));
  lanes[2u] = mix64(lanes[2u],
                    static_cast<std::uint32_t>(state->field_metrics.residual_l1));
  lanes[3u] = mix64(lanes[3u], state->field_metrics.active_routes);
  lanes[0u] = mix64(lanes[0u], state->field_metrics.contact_routes);
  lanes[1u] = mix64(lanes[1u], state->field_metrics.responsive_routes);
  lanes[1u] = mix64(lanes[1u], state->field_metrics.owner_mix);
  lanes[2u] = mix64(lanes[2u], state->field_metrics.developmental_pressure_l1);
  lanes[3u] = mix64(lanes[3u], state->field_metrics.functional_horizon);
  lanes[0u] = mix64(lanes[0u], state->field_metrics.functional_integration);
  lanes[1u] = mix64(lanes[1u], state->field_metrics.functional_delay);
  lanes[2u] = mix64(lanes[2u], state->field_metrics.functional_spatial_tv_l1);
  lanes[2u] = mix64(lanes[2u], state->intervention_sequence);
  lanes[3u] = mix64(lanes[3u], state->structural_focus_cell);
  lanes[0u] = mix64(lanes[0u], state->removed_matter_q8_sum);
  lanes[1u] = mix64(lanes[1u], state->cut_coupling_q8_sum);
  lanes[2u] = mix64(lanes[2u], state->structural_support_q8);
  lanes[3u] = mix64(lanes[3u], state->intervention_active);
  lanes[0u] = mix64(lanes[0u], state->field_source_count);
  lanes[1u] = mix64(lanes[1u], state->field_supported_source_count);
  lanes[1u] = mix64(lanes[1u], state->field_return_packets_q8);
  lanes[2u] = mix64(lanes[2u], state->field_near_profile);
  lanes[3u] = mix64(lanes[3u], state->field_far_profile);
  lanes[0u] = mix64(lanes[0u], state->field_anchor_mix);
  lanes[1u] = mix64(lanes[1u], state->field_anchor_moment);
  lanes[2u] = mix64(lanes[2u], state->field_profile_tv_l1);
  lanes[1u] = mix64(lanes[1u], state->field_relation_mix);
  lanes[2u] = mix64(lanes[2u], state->field_relation_l1);
  lanes[3u] = mix64(lanes[3u], static_cast<std::uint32_t>(state->field_relation_sum));
  lanes[2u] = mix64(lanes[2u],
                    static_cast<std::uint32_t>(state->field_polarity_sum));
  lanes[3u] = mix64(lanes[3u], state->field_receptor_gain_q8);
  lanes[0u] = mix64(lanes[0u], state->field_diffusion_l1);
  lanes[1u] = mix64(lanes[1u], state->field_damage_q8);
  lanes[2u] = mix64(lanes[2u], state->field_recruited_sources);
  lanes[3u] = mix64(lanes[3u], state->field_repaired_sources);
  lanes[0u] = mix64(lanes[0u], state->field_growth_front_count);
  lanes[1u] = mix64(lanes[1u], state->field_component_count);
  lanes[2u] = mix64(
      lanes[2u], static_cast<std::uint32_t>(state->field_component0_response_l1));
  lanes[3u] = mix64(
      lanes[3u], static_cast<std::uint32_t>(state->field_component1_response_l1));
  lanes[0u] = mix64(
      lanes[0u], static_cast<std::uint32_t>(state->field_component0_residual_l1));
  lanes[1u] = mix64(
      lanes[1u], static_cast<std::uint32_t>(state->field_component1_residual_l1));
  lanes[2u] = mix64(lanes[2u], state->field_component0_support_q8);
  lanes[3u] = mix64(lanes[3u], state->field_component1_support_q8);
  lanes[0u] = mix64(lanes[0u], state->field_withdrawn_sources);
  lanes[1u] = mix64(
      lanes[1u], static_cast<std::uint32_t>(state->field_packet_braid_sum));
  lanes[2u] = mix64(lanes[2u], state->field_packet_braid_l1);
  lanes[3u] = mix64(lanes[3u], state->field_packet_braid_sources);
  lanes[0u] = mix64(lanes[0u], state->alternate_expression_endpoint);
  lanes[1u] = mix64(lanes[1u], state->joint_expression_endpoints);
  lanes[2u] = mix64(lanes[2u], state->primary_expression_gain_q8);
  lanes[3u] = mix64(lanes[3u], state->alternate_expression_gain_q8);
  lanes[0u] = mix64(
      lanes[0u], static_cast<std::uint32_t>(state->field_packet_alignment_sum));
  lanes[1u] = mix64(lanes[1u], state->field_packet_alignment_l1);
  lanes[2u] = mix64(lanes[2u], state->field_mature_packet_sources);
  lanes[3u] = mix64(lanes[3u], state->field_effective_alignment_l1);
  lanes[0u] = mix64(lanes[0u], state->field_effective_braid_l1);
  lanes[1u] = mix64(
      lanes[1u], static_cast<std::uint32_t>(state->recurrent_context_l1_q16));
  lanes[2u] = mix64(lanes[2u], state->recurrent_context_mix);
  lanes[3u] = mix64(lanes[3u], state->recurrent_context_revision);
  for (std::uint32_t index = 0u;
       index < sparse_source_joint::kJointCapacity; ++index) {
    const sparse_source_joint::SparseJoint& joint =
        state->source_joint.joints[index];
    if (joint.live == 0u) continue;
    lanes[index & 3u] = mix64(lanes[index & 3u], joint.alternate_endpoint);
    lanes[(index + 1u) & 3u] = mix64(
        lanes[(index + 1u) & 3u], joint.alternate_recruitment_updates);
    lanes[(index + 2u) & 3u] = mix64(
        lanes[(index + 2u) & 3u], joint.alternate_endpoint_committed);
  }
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    const FieldSourceState& field = state->field_sources[index];
    if (field.live == 0u) continue;
    lanes[index & 3u] = mix64(
        lanes[index & 3u], static_cast<std::uint32_t>(field.packet_history_flux_q8));
    lanes[(index + 1u) & 3u] = mix64(
        lanes[(index + 1u) & 3u],
        static_cast<std::uint32_t>(field.packet_history_polarity_q8));
    lanes[(index + 2u) & 3u] = mix64(lanes[(index + 2u) & 3u],
                                     field.packet_history_age);
    lanes[(index + 3u) & 3u] = mix64(
        lanes[(index + 3u) & 3u], static_cast<std::uint32_t>(field.packet_braid_q8));
    lanes[index & 3u] = mix64(
        lanes[index & 3u], static_cast<std::uint32_t>(field.packet_alignment_q8));
    lanes[(index + 1u) & 3u] =
        mix64(lanes[(index + 1u) & 3u], field.packet_history_updates);
  }
  absorb_resident_bytes(lanes, &state->adaptive, sizeof(state->adaptive));
  digest_finish(result, lanes, 32u + 32u + 32u + input.byte_count +
                                output.byte_count + predecessor.byte_count);
}

#include "bcc32_persistent_kernel_device_ecology.inl"

ContentAddress host_address(const DeviceDigest& digest) {
  ContentAddress result;
  std::copy(std::begin(digest.bytes), std::end(digest.bytes),
            result.digest.begin());
  result.byte_count = digest.byte_count;
  return result;
}

DeviceDigest device_digest(const ContentAddress& address) {
  DeviceDigest result;
  std::copy(address.digest.begin(), address.digest.end(), result.bytes);
  result.byte_count = address.byte_count;
  return result;
}

void release_host_boundary(void*& pointer) noexcept {
  if (pointer != nullptr) {
    cudaFree(pointer);
    pointer = nullptr;
  }
}

std::uint64_t load_host_u64(const std::uint64_t* address) {
  return std::atomic_ref<std::uint64_t>(*const_cast<std::uint64_t*>(address))
      .load(std::memory_order_acquire);
}

void store_host_u64(std::uint64_t* address, std::uint64_t value) {
  std::atomic_ref<std::uint64_t>(*address).store(value,
                                                  std::memory_order_release);
}

void store_host_u32(std::uint32_t* address, std::uint32_t value) {
  std::atomic_ref<std::uint32_t>(*address).store(value,
                                                  std::memory_order_release);
}

std::uint32_t load_host_u32(const std::uint32_t* address) {
  return std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(address))
      .load(std::memory_order_acquire);
}

bool claim_host_egress(EgressSlot* slot) {
  std::uint32_t expected = kEgressReady;
  return std::atomic_ref<std::uint32_t>(slot->state)
      .compare_exchange_strong(expected, kEgressReading,
                               std::memory_order_acq_rel,
                               std::memory_order_acquire);
}

void release_host_egress(EgressSlot* slot) {
  std::atomic_ref<std::uint32_t>(slot->state).store(
      kEgressReady, std::memory_order_release);
}

void copy_snapshot(const EgressRing* egress, const Lifecycle* lifecycle,
                   std::size_t cell_count, PassiveSnapshot* result) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(30);
  for (;;) {
    const std::uint32_t continuation_fault =
        load_host_u32(&lifecycle->continuation_fault);
    if (continuation_fault != 0u)
      throw std::runtime_error(
          "persistent kernel continuation fault=" +
          std::to_string(continuation_fault));
    const std::uint64_t sequence = load_host_u64(&egress->published);
    if (sequence == 0u) {
      if (std::chrono::steady_clock::now() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    EgressSlot* slot = const_cast<EgressSlot*>(&egress->slots[sequence % kEgressSlots]);
    if (!claim_host_egress(slot)) {
      if (std::chrono::steady_clock::now() >= deadline) break;
      std::this_thread::yield();
      continue;
    }
    if (slot->generation != sequence) {
      release_host_egress(slot);
      continue;
    }
    PassiveSnapshot candidate;
    candidate.actions.assign(slot->actions, slot->actions + cell_count);
    const std::uint32_t language_length =
        slot->language_frame.length <= language::kOutputCapacity
            ? slot->language_frame.length
            : 0u;
    candidate.language_bytes.assign(
        slot->language_frame.bytes,
        slot->language_frame.bytes + language_length);
    candidate.energy = slot->energy;
    candidate.host_bootstrap_launches = slot->host_bootstrap_launches;
    candidate.device_epochs = slot->device_epochs;
    candidate.language_generation = slot->language_frame.generation;
    candidate.language_contacts = slot->language_contacts;
    candidate.language_recruited_cells = slot->language_recruited_cells;
    candidate.language_reused_cells = slot->language_reused_cells;
    candidate.language_strengthened_edges = slot->language_strengthened_edges;
    const DeviceReceipt receipt = slot->receipt;
    candidate.receipt = {receipt.tick,
                        receipt.phase,
                        receipt.contact_sequence,
                        host_address(receipt.sealed_execution),
                        host_address(receipt.law),
                        host_address(receipt.image),
                        host_address(receipt.input),
                        host_address(receipt.output),
                        host_address(receipt.predecessor),
                        host_address(receipt.commitment),
                        receipt.joint_output,
                        receipt.joint_residual,
                        receipt.joint_output_valid,
                        receipt.joint_route,
                        receipt.field_response_l1,
                        receipt.field_residual_l1,
                        receipt.field_peak_response,
                        receipt.field_peak_residual,
                        receipt.field_active_routes,
                        receipt.field_contact_routes,
                        receipt.field_responsive_routes,
                        receipt.field_owner_mix,
                        receipt.developmental_pressure_l1,
                        receipt.functional_horizon,
                        receipt.functional_integration,
                        receipt.functional_delay,
                        receipt.functional_spatial_tv_l1,
                        host_address(receipt.structure),
                        host_address(receipt.intervention),
                        receipt.intervention_sequence,
                        receipt.structural_focus_cell,
                        receipt.removed_matter_q8_sum,
                        receipt.cut_coupling_q8_sum,
                        receipt.structural_support_q8,
                        receipt.effective_field_response_l1,
                        receipt.intervention_active,
                        receipt.field_source_count,
                        receipt.field_supported_source_count,
                        receipt.field_receptor_count,
                        receipt.field_packet_flux_l1,
                        receipt.field_polarity_l1,
                        receipt.field_return_packets_q8,
                        receipt.field_near_profile,
                        receipt.field_middle_profile,
                        receipt.field_far_profile,
                        receipt.field_anchor_mix,
                        receipt.field_anchor_moment,
                        receipt.field_profile_tv_l1,
                        receipt.field_relation_mix,
                        receipt.field_relation_l1,
                        receipt.field_relation_sum,
                        receipt.field_polarity_sum,
                        receipt.field_receptor_gain_q8,
                        receipt.field_diffusion_l1,
                        receipt.field_damage_q8,
                        receipt.field_recruited_sources,
                        receipt.field_repaired_sources,
                        receipt.field_growth_front_count,
                        receipt.field_component_count,
                        receipt.field_component0_response_l1,
                        receipt.field_component1_response_l1,
                        receipt.field_component0_residual_l1,
                        receipt.field_component1_residual_l1,
                        receipt.field_component0_support_q8,
                        receipt.field_component1_support_q8,
                        receipt.field_withdrawn_sources,
                        receipt.active_joint_locus,
                        receipt.tissue_matter_q8,
                        receipt.tissue_coupling_q8,
                        receipt.field_packet_braid_sum,
                        receipt.field_packet_braid_l1,
                        receipt.field_packet_braid_sources,
                        receipt.adaptive_context_l1,
                        receipt.adaptive_context_anchor,
                        receipt.adaptive_context_matter_q8,
                        receipt.adaptive_plasticity_q8,
                        receipt.adaptive_consolidation_q8,
                        receipt.adaptive_growth_q8,
                        receipt.adaptive_turnover_q8,
                        receipt.adaptive_repair_q8,
                        receipt.adaptive_replay_q8,
                        receipt.adaptive_live_edges,
                        receipt.adaptive_consolidated_routes,
                        receipt.adaptive_structural_revision,
                        receipt.adaptive_recruited_edges,
                        receipt.adaptive_pruned_edges,
                        receipt.adaptive_repaired_cells,
                        receipt.adaptive_free_structural_matter_q8,
                        receipt.adaptive_edge_structural_matter_q8,
                        receipt.adaptive_cumulative_structural_debit_q8,
                        receipt.adaptive_cumulative_structural_refund_q8,
                        receipt.adaptive_local_proposal_activity,
                        receipt.alternate_expression_endpoint,
                        receipt.joint_expression_endpoints,
                        receipt.primary_expression_gain_q8,
                        receipt.alternate_expression_gain_q8,
                        receipt.field_packet_alignment_sum,
                        receipt.field_packet_alignment_l1,
                        receipt.field_packet_alignment_sources,
                        receipt.field_mature_packet_sources,
                        receipt.field_effective_alignment_sum,
                        receipt.field_effective_braid_sum,
                        receipt.field_effective_alignment_l1,
                        receipt.field_effective_braid_l1,
                        receipt.field_alignment_locus,
                        receipt.field_braid_locus,
                        receipt.recurrent_context_l1_q16,
                        receipt.recurrent_context_mix,
                         receipt.recurrent_context_revision};
    candidate.receipt.genesis_manifest =
        host_address(receipt.genesis_manifest);
    candidate.receipt.f_world = host_address(receipt.f_world);
    candidate.receipt.completed_f_ticks = receipt.completed_f_ticks;
    candidate.receipt.f_generation = receipt.f_generation;
    candidate.receipt.f_fault = receipt.f_fault;
    candidate.receipt.f_owned_clock = receipt.f_owned_clock;
    candidate.receipt.legacy_action_authority =
        receipt.legacy_action_authority;
    candidate.receipt.f_motor_zero = receipt.f_motor_zero;
    candidate.receipt.f_motor_one = receipt.f_motor_one;
    release_host_egress(slot);
    *result = std::move(candidate);
    return;
  }
  throw std::runtime_error("persistent kernel passive snapshot timed out");
}

void require_managed_boundary_support() {
  int device = 0;
  require_cuda(cudaGetDevice(&device), "query current CUDA device");
  int unified_addressing = 0;
  int concurrent_managed_access = 0;
  require_cuda(cudaDeviceGetAttribute(&unified_addressing,
                                      cudaDevAttrUnifiedAddressing, device),
               "query unified addressing");
  require_cuda(cudaDeviceGetAttribute(&concurrent_managed_access,
                                      cudaDevAttrConcurrentManagedAccess,
                                      device),
               "query concurrent managed access");
  if (unified_addressing == 0 || concurrent_managed_access == 0)
    throw std::runtime_error(
        "autonomous CUDA graph requires unified addressing and concurrent "
        "managed access; unified_addressing=" +
        std::to_string(unified_addressing) +
        " concurrent_managed_access=" +
        std::to_string(concurrent_managed_access) +
        ". Each device-owned epoch is bounded for watchdog-safe execution.");
}

bool wait_for_stopped(cudaStream_t stream, Lifecycle* lifecycle) noexcept {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(5);
  for (;;) {
    const std::uint32_t stopped = load_host_u32(&lifecycle->stopped);
    const cudaError_t status = cudaStreamQuery(stream);
    if (stopped != 0u && status == cudaSuccess) {
      if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
      return true;
    }
    if (status == cudaSuccess) return false;
    if (status != cudaSuccess && status != cudaErrorNotReady) return false;
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void require_stream_healthy(cudaStream_t stream, const Lifecycle* lifecycle) {
  const std::uint32_t continuation_fault =
      load_host_u32(&lifecycle->continuation_fault);
  if (continuation_fault != 0u)
    throw std::runtime_error("persistent kernel continuation fault=" +
                             std::to_string(continuation_fault));
  const cudaError_t status = cudaStreamQuery(stream);
  if (status == cudaErrorNotReady) return;
  if (status == cudaSuccess)
    throw std::runtime_error("persistent kernel continuation stopped unexpectedly");
  throw std::runtime_error(std::string("persistent kernel stream failed: ") +
                           cudaGetErrorString(status));
}

std::uint64_t founder_word() {
  const DevelopmentalHash hash = grown::founder_hash();
  return hash.lane0 ^ hash.lane1 ^ hash.lane2 ^ hash.lane3;
}

void destroy_graph_handles(void*& graph_pointer,
                           void*& graph_exec_pointer) noexcept {
  if (graph_exec_pointer != nullptr) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_pointer));
    graph_exec_pointer = nullptr;
  }
  if (graph_pointer != nullptr) {
    cudaGraphDestroy(static_cast<cudaGraph_t>(graph_pointer));
    graph_pointer = nullptr;
  }
}

}  // namespace

PersistentKernel::PersistentKernel()
    : PersistentKernel(founder_word(), 64u, true) {}

PersistentKernel::PersistentKernel(std::uint64_t founder,
                                   std::size_t cell_count)
    : PersistentKernel(founder, cell_count, false) {}

PersistentKernel::PersistentKernel(std::uint64_t founder,
                                   std::size_t cell_count,
                                   bool f_owned_clock)
    : founder_(founder),
      sealed_execution_(sealed_execution_identity()),
      law_(canonical_law_identity()),
      image_(running_image_identity()),
      cell_count_(std::max<std::size_t>(cell_count, 4u)),
      f_owned_clock_(f_owned_clock) {
  if (cell_count_ > kMaxCells)
    throw std::invalid_argument("persistent kernel cell count exceeds device aperture");
  require_managed_boundary_support();
  IngressRing* ingress_host = nullptr;
  PhysicalIngress* physical_host = nullptr;
  EgressRing* egress_host = nullptr;
  Lifecycle* lifecycle_host = nullptr;
  DeviceState* allocated = nullptr;
  bool device_work_started = false;
  require_cuda(cudaMalloc(&allocated, sizeof(DeviceState)),
               "allocate persistent kernel device state");
  device_state_ = allocated;
  try {
    ordinary_f::DeviceLaunchHandle ordinary_f_handle{};
    if (f_owned_clock_) {
      auto* adult = grown::make_founder_grown_adult();
      grown_adult_ = adult;
      genesis_manifest_ = grown::genesis_manifest_identity(adult);
      ordinary_f_timeline_ = grown::claim_ordinary_f_timeline(adult);
      ordinary_f_handle =
          static_cast<ordinary_f::DeviceOrdinaryFTimeline*>(
              ordinary_f_timeline_)
              ->device_leaf_launch_handle();
    }
    require_cuda(cudaMemset(allocated, 0, sizeof(DeviceState)),
                 "clear persistent kernel device state once at startup");
    require_cuda(cudaMallocManaged(&ingress_host, sizeof(IngressRing)),
                 "allocate managed persistent ingress ring");
    ingress_ = ingress_host;
    require_cuda(cudaMallocManaged(&physical_host, sizeof(PhysicalIngress)),
                 "allocate managed persistent physical ingress");
    physical_ = physical_host;
    require_cuda(cudaMallocManaged(&egress_host, sizeof(EgressRing)),
                 "allocate managed persistent egress ring");
    egress_ = egress_host;
    require_cuda(cudaMallocManaged(&lifecycle_host, sizeof(Lifecycle)),
                 "allocate managed persistent lifecycle control");
    lifecycle_ = lifecycle_host;
    std::memset(ingress_host, 0, sizeof(*ingress_host));
    std::memset(physical_host, 0, sizeof(*physical_host));
    std::memset(egress_host, 0, sizeof(*egress_host));
    std::memset(lifecycle_host, 0, sizeof(*lifecycle_host));
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "create persistent private stream");
    private_stream_ = stream;
    cudaGraph_t graph = nullptr;
    require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "begin autonomous epoch graph capture");
    std::uint64_t founder = founder_;
    std::uint32_t cell_count = static_cast<std::uint32_t>(cell_count_);
    DeviceDigest sealed = device_digest(sealed_execution_);
    DeviceDigest law = device_digest(law_);
    DeviceDigest image = device_digest(image_);
    void* autonomous_epoch_arguments[] = {
        &allocated,      &ingress_host, &physical_host, &egress_host,
        &lifecycle_host, &founder,     &cell_count,     &sealed,
        &law,            &image};
    require_cuda(cudaLaunchKernel(
                     reinterpret_cast<const void*>(autonomous_epoch_kernel),
                     dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                     autonomous_epoch_arguments, 0u, stream),
                 "capture autonomous epoch kernel");
    require_cuda(cudaGetLastError(), "capture autonomous epoch kernel");
    require_cuda(cudaStreamEndCapture(stream, &graph),
                 "end autonomous epoch graph capture");
    graph_ = graph;
    cudaGraphExec_t graph_exec = nullptr;
    require_cuda(cudaGraphInstantiateWithFlags(
                     &graph_exec, graph, cudaGraphInstantiateFlagDeviceLaunch),
                 "instantiate device-launchable autonomous graph");
    graph_exec_ = graph_exec;
    // Upload before the first host launch. In production F-owned mode the
    // ordinary-F graph is a tail predecessor of this root's self-tail; it
    // deliberately has no ancestor return edge into the active root exec.
    require_cuda(cudaGraphUpload(graph_exec, stream),
                 "upload autonomous device graph");
    if (f_owned_clock_) {
      require_cuda(
          cudaMemcpy(reinterpret_cast<std::uint8_t*>(allocated) +
                         offsetof(DeviceState, ordinary_f),
                     &ordinary_f_handle, sizeof(ordinary_f_handle),
                     cudaMemcpyHostToDevice),
          "attach ordinary-F production graph");
      const DeviceDigest manifest = device_digest(genesis_manifest_);
      require_cuda(
          cudaMemcpy(reinterpret_cast<std::uint8_t*>(allocated) +
                         offsetof(DeviceState, genesis_manifest),
                     &manifest, sizeof(manifest), cudaMemcpyHostToDevice),
          "attach ordinary-F Genesis manifest identity");
      const std::uint32_t owned = 1u;
      require_cuda(
          cudaMemcpy(reinterpret_cast<std::uint8_t*>(allocated) +
                         offsetof(DeviceState, f_owned_clock),
                     &owned, sizeof(owned), cudaMemcpyHostToDevice),
          "attach ordinary-F clock authority");
    }
    require_cuda(cudaGraphLaunch(graph_exec, stream),
                 "host-launch autonomous device graph");
    device_work_started = true;
  } catch (...) {
    if (private_stream_ != nullptr) {
      if (lifecycle_host != nullptr) {
        if (device_work_started)
          store_host_u32(&lifecycle_host->shutdown, 1u);
        if (!device_work_started ||
            wait_for_stopped(static_cast<cudaStream_t>(private_stream_),
                             lifecycle_host)) {
          cudaStreamDestroy(static_cast<cudaStream_t>(private_stream_));
          private_stream_ = nullptr;
          destroy_graph_handles(graph_, graph_exec_);
        }
      }
    }
    if (private_stream_ == nullptr) {
      delete static_cast<ordinary_f::DeviceOrdinaryFTimeline*>(
          ordinary_f_timeline_);
      ordinary_f_timeline_ = nullptr;
      grown::destroy_grown_adult(
          static_cast<grown::GrownAdult*>(grown_adult_));
      grown_adult_ = nullptr;
      release_host_boundary(ingress_);
      release_host_boundary(physical_);
      release_host_boundary(egress_);
      release_host_boundary(lifecycle_);
      cudaFree(allocated);
      device_state_ = nullptr;
    }
    throw;
  }
}

PersistentKernel::~PersistentKernel() {
  shutdown();
  if (shutdown_complete_) {
    delete static_cast<ordinary_f::DeviceOrdinaryFTimeline*>(
        ordinary_f_timeline_);
    ordinary_f_timeline_ = nullptr;
    grown::destroy_grown_adult(static_cast<grown::GrownAdult*>(grown_adult_));
    grown_adult_ = nullptr;
    destroy_graph_handles(graph_, graph_exec_);
    cudaFree(static_cast<DeviceState*>(device_state_));
    device_state_ = nullptr;
    release_host_boundary(ingress_);
    release_host_boundary(physical_);
    release_host_boundary(egress_);
    release_host_boundary(lifecycle_);
  }
}

void PersistentKernel::shutdown() noexcept {
  if (shutdown_requested_) return;
  shutdown_requested_ = true;
  if (lifecycle_ != nullptr)
    store_host_u32(&static_cast<Lifecycle*>(lifecycle_)->shutdown, 1u);
  if (private_stream_ != nullptr) {
    if (wait_for_stopped(static_cast<cudaStream_t>(private_stream_),
                         static_cast<Lifecycle*>(lifecycle_))) {
      cudaStreamDestroy(static_cast<cudaStream_t>(private_stream_));
      private_stream_ = nullptr;
      shutdown_complete_ = true;
    }
  } else {
    shutdown_complete_ = true;
  }
}

void PersistentKernel::present_raw(std::span<const BoundaryWord> contact) {
  if (shutdown_requested_) throw std::runtime_error("persistent kernel is stopped");
  require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                         static_cast<const Lifecycle*>(lifecycle_));
  auto* ingress = static_cast<IngressRing*>(ingress_);
  // Preserve the historical empty raw contact as one event. Every nonempty
  // oversized contact is streamed in fixed-width transport packets; no suffix
  // is silently truncated and packet boundaries carry no semantic meaning.
  std::size_t offset = 0u;
  do {
    const std::size_t body_count =
        std::min(contact.size() - offset, kMaxContactWords);
    const std::uint64_t sequence = next_contact_sequence_;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    for (;;) {
      const std::uint64_t consumed = load_host_u64(&ingress->consumed);
      if (sequence - consumed < kIngressSlots) break;
      if (load_host_u32(&static_cast<Lifecycle*>(lifecycle_)->shutdown) != 0u)
        throw std::runtime_error("persistent kernel stopped while enqueueing contact");
      const cudaError_t status =
          cudaStreamQuery(static_cast<cudaStream_t>(private_stream_));
      const std::uint32_t continuation_fault = load_host_u32(
          &static_cast<Lifecycle*>(lifecycle_)->continuation_fault);
      if (continuation_fault != 0u)
        throw std::runtime_error("persistent kernel continuation fault=" +
                                 std::to_string(continuation_fault));
      if (status == cudaSuccess)
        throw std::runtime_error("persistent kernel continuation stopped unexpectedly");
      if (status != cudaSuccess && status != cudaErrorNotReady)
        throw std::runtime_error(std::string("persistent kernel stream failed while "
                                             "enqueueing contact: ") +
                                 cudaGetErrorString(status));
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("persistent kernel ingress ring remained full");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    IngressSlot& slot = ingress->slots[sequence % kIngressSlots];
    slot.count = static_cast<std::uint32_t>(body_count);
    if (body_count != 0u)
      std::memcpy(slot.words, contact.data() + offset,
                  body_count * sizeof(BoundaryWord));
    slot.raw_byte_count = static_cast<std::uint32_t>(body_count);
    for (std::size_t index = 0u; index < body_count; ++index)
      slot.raw_bytes[index] =
          static_cast<std::uint8_t>(contact[offset + index] & 0xffu);
    std::atomic_thread_fence(std::memory_order_release);
    store_host_u64(&slot.sequence, sequence);
    store_host_u64(&ingress->published, sequence);
    ++next_contact_sequence_;
    offset += body_count;
  } while (offset < contact.size());
}

void PersistentKernel::present_physical(const RawPhysicalIntervention& event) {
  if (shutdown_requested_) throw std::runtime_error("persistent kernel is stopped");
  require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                         static_cast<const Lifecycle*>(lifecycle_));
  auto* physical = static_cast<PhysicalIngress*>(physical_);
  const std::uint64_t sequence = next_intervention_sequence_;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(5);
  while (load_host_u64(&physical->consumed) + 1u < sequence) {
    if (load_host_u32(&static_cast<Lifecycle*>(lifecycle_)->shutdown) != 0u)
      throw std::runtime_error("persistent kernel stopped while enqueueing physical event");
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("persistent kernel physical ingress remained full");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  physical->event = event;
  std::atomic_thread_fence(std::memory_order_release);
  store_host_u64(&physical->published, sequence);
  ++next_intervention_sequence_;
}

PassiveSnapshot PersistentKernel::read_snapshot() const {
  PassiveSnapshot result;
  if (egress_ == nullptr) return result;
  if (!shutdown_requested_)
    require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                           static_cast<const Lifecycle*>(lifecycle_));
  copy_snapshot(static_cast<const EgressRing*>(egress_),
                static_cast<const Lifecycle*>(lifecycle_), cell_count_,
                &result);
  result.continuation_fault =
      load_host_u32(&static_cast<const Lifecycle*>(lifecycle_)->continuation_fault);
  return result;
}

std::vector<BoundaryWord> PersistentKernel::read_actions() const {
  return read_snapshot().actions;
}

std::vector<std::uint8_t> PersistentKernel::read_language_bytes() const {
  return read_snapshot().language_bytes;
}

TickReceipt PersistentKernel::read_receipt() const {
  return read_snapshot().receipt;
}

}  // namespace substrate::bcc32::persistent_kernel

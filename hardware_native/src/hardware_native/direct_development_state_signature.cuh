#ifndef HARDWARE_NATIVE_DIRECT_DEVELOPMENT_STATE_SIGNATURE_CUH
#define HARDWARE_NATIVE_DIRECT_DEVELOPMENT_STATE_SIGNATURE_CUH

#include <cstddef>
#include <cstdint>

#include "hardware_native/direct_content_address.cuh"
#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network {

#if defined(__CUDACC__)
#define DIRECT_DEVELOPMENT_SIGNATURE_HD __host__ __device__
#else
#define DIRECT_DEVELOPMENT_SIGNATURE_HD
#endif

// Identity-only seal over the bounded developmental checkpoint status.  The
// encoding is explicit and endian-stable: no pointer, padding byte, or unrelated
// adult state enters the digest.
using DirectDevelopmentStateSignatureV2 = DirectSha256Address;

namespace development_signature_detail {

DIRECT_DEVELOPMENT_SIGNATURE_HD inline void append_u32(
    detail::DirectSha256State* state, std::uint32_t value) {
  const std::uint8_t bytes[4] = {
      static_cast<std::uint8_t>(value >> 24u),
      static_cast<std::uint8_t>(value >> 16u),
      static_cast<std::uint8_t>(value >> 8u),
      static_cast<std::uint8_t>(value)};
  state->update(bytes, sizeof(bytes));
}

DIRECT_DEVELOPMENT_SIGNATURE_HD inline void append_u64(
    detail::DirectSha256State* state, std::uint64_t value) {
  append_u32(state, static_cast<std::uint32_t>(value >> 32u));
  append_u32(state, static_cast<std::uint32_t>(value));
}

DIRECT_DEVELOPMENT_SIGNATURE_HD inline void append_root(
    detail::DirectSha256State* state, const recipe::Root256& root) {
  for (std::uint32_t i = 0u; i < 8u; ++i) append_u32(state, root.word[i]);
}

}  // namespace development_signature_detail

DIRECT_DEVELOPMENT_SIGNATURE_HD inline bool direct_development_state_signature_v2(
    const DirectBrain* brain, DirectDevelopmentStateSignatureV2* out) {
  if (brain == nullptr || out == nullptr || brain->development == nullptr ||
      brain->nodes == nullptr || brain->node_count == 0u)
    return false;

  using namespace development_signature_detail;
  detail::DirectSha256State digest{};
  constexpr std::uint8_t domain[] = {
      '0', 'x', '1', '/', 'd', 'e', 'v', '/', 's', 't', 'a', 't', 'e', '/',
      'v', '2'};
  digest.update(domain, sizeof(domain));

  append_root(&digest, brain->genome_root);
  append_root(&digest, brain->territory_layout_root);
  append_root(&digest, brain->body_root);
  append_root(&digest, brain->environment_root);
  append_root(&digest, brain->birth_root);
  append_u32(&digest, brain->node_count);

  const ResidentDevelopmentState& development = *brain->development;
  append_u32(&digest, development.age_tick);
  append_u32(&digest, development.phase);
  append_u32(&digest, development.plasticity_q16);
  append_u32(&digest, development.mature_plasticity_floor_q16);
  append_u32(&digest, development.critical_period_q16);
  append_u32(&digest, development.inhibition_gain_q16);
  append_u64(&digest, development.constructor_reserve);
  append_u64(&digest, development.reclaimed_resource);
  append_u64(&digest, development.live_route_matter);
  append_u64(&digest, development.live_node_matter);
  append_u32(&digest, development.field_count);
  append_u32(&digest, development.constructor_rule_count);
  append_u32(&digest, development.recipe_cell_count);
  append_u32(&digest, development.recipe_edge_count);
  append_u32(&digest, development.birth_handoff_tick);
  append_u32(&digest, development.last_maturation_event);

  const DirectExactHistoryHotPage& history = development.exact_history;
  append_u64(&digest, history.prefix_root);
  append_u64(&digest, history.page_prefix_root);
  append_u64(&digest, history.next_sequence);
  append_u64(&digest, history.archived_record_count);
  append_u64(&digest, history.archived_bytes);
  append_u64(&digest, history.archive_capacity_bytes);
  append_root(&digest, history.archive_chain_head);
  append_u32(&digest, history.committed_slots);
  append_u32(&digest, history.archived_pages);
  append_u32(&digest, history.sealed);
  append_u32(&digest, history.overflow_refusals);
  append_u32(&digest, history.phase_base);
  append_u32(&digest, history.phase_width);
  append_u32(&digest, static_cast<std::uint32_t>(history.phase_kind));
  append_u32(&digest, history.phase_admitted);
  append_u32(&digest, history.phase_tick);
  append_u32(&digest, history.last_phase_records);

  for (std::uint32_t i = 0u; i < brain->node_count; ++i)
    append_u32(&digest, brain->nodes[i].maturation_q16);

  *out = digest.finish();
  return true;
}

}  // namespace substrate::direct_network

#undef DIRECT_DEVELOPMENT_SIGNATURE_HD

#endif  // HARDWARE_NATIVE_DIRECT_DEVELOPMENT_STATE_SIGNATURE_CUH

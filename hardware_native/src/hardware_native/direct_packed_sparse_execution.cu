#include <vector>

#include "hardware_native/direct_canonical_transition_device.cuh"
#include "hardware_native/direct_execution_fabric.cuh"

namespace substrate::direct_adult {

__device__ bool evaluate_packed_sparse_source(
    DirectBrainV01 brain,
    const DirectExecutionFabricDeviceView& fabric,
    std::uint32_t panel_index,
    const ActivityEvent& event,
    DirectCanonicalTransitionResult* out_result,
    AdultCounters* counters) {
  (void)panel_index;
  if (out_result == nullptr || event.node >= brain.node_count)
    return false;

  const DirectPackedSourceMeta meta = fabric.packed_source_meta[event.node];
  const DirectNode node = brain.nodes[event.node];
  if (meta.source_revision != node.source_revision || meta.entry_count == 0u) {
    if (counters != nullptr)
      atomicAdd(&counters->guard_fallbacks, 1u);
    return false;
  }

  // #1236: packed storage discovers the same explicit candidate set without a
  // dependent next_route chase. It does not own admission/ranking or commit.
  // Every ranking field is re-read from the live route at consult time.
  const std::uint64_t event_signature =
      direct_canonical_event_context_signature(event.history_signature,
                                               event.word);
  DirectCanonicalWinner canonical = direct_canonical_winner_empty();
  bool any_stale_entry = false;

  for (std::uint32_t e = 0u; e < meta.entry_count; ++e) {
    const DirectPackedEntry entry = fabric.packed_entries[meta.entry_offset + e];
    if (entry.route_slot >= brain.route_capacity) {
      any_stale_entry = true;
      break;
    }
    const DirectRouteSlotMeta slot_meta =
        brain.topology.slot_meta[entry.route_slot];
    if (slot_meta.live == 0u ||
        slot_meta.generation != entry.route_generation) {
      any_stale_entry = true;
      break;
    }

    const DirectRoute route = brain.routes[entry.route_slot];
    if (!direct_canonical_route_admissible(route))
      continue;
    const std::uint32_t rank =
        direct_canonical_context_rank(route, event_signature);
    if (rank != 0u && counters != nullptr)
      atomicAdd(&counters->context_index_hits, 1u);
    direct_canonical_consider(&canonical, entry.route_slot, rank,
                              static_cast<std::int64_t>(
                                  route.conductance_q16));
  }

  if (any_stale_entry) {
    if (counters != nullptr)
      atomicAdd(&counters->guard_fallbacks, 1u);
    return false;
  }

  *out_result = direct_canonical_finalize_transition(
      brain, event, canonical, meta.entry_count, event_signature);
  if (counters != nullptr)
    atomicAdd(&counters->packed_hits, 1u);
  return true;
}

bool refresh_direct_packed_sparse_source(
    const DirectBrainV01& brain, DirectExecutionFabricRuntime* fabric,
    std::uint32_t source) {
  if (fabric == nullptr || source >= brain.node_count)
    return false;

  DirectPackedSourceMeta meta{};
  cudaMemcpy(&meta, &fabric->view.packed_source_meta[source],
             sizeof(DirectPackedSourceMeta), cudaMemcpyDeviceToHost);

  DirectNode node{};
  cudaMemcpy(&node, &brain.nodes[source], sizeof(DirectNode),
             cudaMemcpyDeviceToHost);

  if (node.route_count != meta.entry_count || meta.entry_count == 0u)
    return false;

  std::vector<DirectPackedEntry> entries(meta.entry_count);
  std::uint32_t slot = node.first_route;
  for (std::uint32_t i = 0; i < meta.entry_count && slot != kInvalidIndex; ++i) {
    DirectRoute route{};
    cudaMemcpy(&route, &brain.routes[slot], sizeof(DirectRoute),
               cudaMemcpyDeviceToHost);
    DirectRouteSlotMeta slot_meta{};
    cudaMemcpy(&slot_meta, &brain.topology.slot_meta[slot],
               sizeof(DirectRouteSlotMeta), cudaMemcpyDeviceToHost);

    entries[i].target = route.target;
    entries[i].conductance_q16 = route.conductance_q16;
    entries[i].delay = route.delay;
    entries[i].route_flags = route.flags;
    entries[i].context_signature = route.context_signature;
    entries[i].route_slot = slot;
    entries[i].route_generation = slot_meta.generation;
    entries[i].learned_output_word = route.learned_output_word;
    entries[i].reserved16 = 0u;

    slot = route.next_route;
  }

  cudaMemcpy(&fabric->view.packed_entries[meta.entry_offset], entries.data(),
             sizeof(DirectPackedEntry) * meta.entry_count,
             cudaMemcpyHostToDevice);

  meta.source_revision = node.source_revision;
  cudaMemcpy(&fabric->view.packed_source_meta[source], &meta,
             sizeof(DirectPackedSourceMeta), cudaMemcpyHostToDevice);

  return true;
}

}  // namespace substrate::direct_adult

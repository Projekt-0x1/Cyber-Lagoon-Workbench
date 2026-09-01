// Patch 0005 of the FULL CUDA network-recipe patch program: actual/shadow
// recurrent activity and causal credit. Actual and shadow activity use
// distinct physical frontier rails and distinct provenance -- a shadow
// (endogenous, speculative) event may activate routes, compete, converge,
// and stage a motor candidate, but it may never increment external witness
// count, close its own prediction as world evidence, or satisfy
// returned-contact provenance. bcc32_network_credit.cu (companion file)
// owns the delayed-credit consequence of an actual_world_return event
// recontacting a node that carried an unconfirmed shadow prediction.

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_ACTIVITY_CUH
#define HARDWARE_NATIVE_BCC32_NETWORK_ACTIVITY_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/bcc32_network_matter.cuh"
#include "hardware_native/bcc32_network_recipe.hpp"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define BCC32_NETWORK_ACTIVITY_HD __host__ __device__
#else
#define BCC32_NETWORK_ACTIVITY_HD
#endif

namespace substrate::bcc32::network_recipe {

enum class NetworkEventOrigin : std::uint32_t {
  actual_external = 0,
  actual_world_return = 1,
  endogenous_shadow = 2,
  generated_motor = 3,
};

BCC32_NETWORK_ACTIVITY_HD inline bool is_actual_origin(NetworkEventOrigin origin) {
  return origin == NetworkEventOrigin::actual_external ||
         origin == NetworkEventOrigin::actual_world_return;
}

struct NetworkActivityEvent {
  std::uint32_t node;
  SiteWord word;
  NetworkEventOrigin origin;
  std::uint32_t producer;
  std::uint32_t source_route;
  std::uint32_t parent_route;
  std::uint32_t horizon;
};
static_assert(std::is_standard_layout_v<NetworkActivityEvent> &&
                  std::is_trivial_v<NetworkActivityEvent>,
              "NetworkActivityEvent must be a fixed-width POD for device residency");

inline constexpr SiteWord kNoPrediction = 0xffffffffu;

// One rail (actual or shadow) of bounded double-buffered frontier state.
// Kept as a distinct type -- rather than reusing patch 0004's construction
// FrontierEntry -- because activity propagation and construction fanout are
// physically different processes over the same NetworkNode graph, and
// "distinct physical frontier rails" is a constitutional requirement for
// actual vs. shadow specifically, not an implementation convenience.
struct ActivityRail {
  NetworkActivityEvent* frontier;
  std::uint32_t* frontier_count;
  NetworkActivityEvent* next_frontier;
  std::uint32_t* next_frontier_count;
  std::uint32_t capacity;
};

struct NetworkActivityDeviceState {
  NetworkNode* nodes;
  std::uint32_t node_capacity;
  ActivityRail actual;
  ActivityRail shadow;
  // Per-node running shadow prediction, consumed (reset to kNoPrediction)
  // the first time an actual_world_return event recontacts that node --
  // "renewed raw contact must pay the resident cost again" means a
  // confirmed/violated prediction does not linger to be reused.
  SiteWord* predicted_word;
  std::uint32_t tick;
};

// Node flags bits this patch writes (bits 8-10; bits 0-7 reserved for
// earlier patches/callers). Exactly one of these three is set by
// classify_and_credit_kernel per recontacted node per tick -- match,
// violation, and omission are three distinct physical aftermath states,
// not one collapsed "prediction wrong/right" bit.
inline constexpr std::uint32_t kNodeFlagMatch = 1u << 8;
inline constexpr std::uint32_t kNodeFlagViolation = 1u << 9;
inline constexpr std::uint32_t kNodeFlagOmission = 1u << 10;

// Per-tick outcome tally for the activity+credit pass.
struct ActivityTickReport {
  std::uint32_t actual_events_processed;
  std::uint32_t shadow_events_processed;
  std::uint32_t matches;
  std::uint32_t violations;
  std::uint32_t omissions;
  std::uint32_t new_actual_frontier_size;
  std::uint32_t new_shadow_frontier_size;
};

// Runs one life tick: fans out both rails through live child matter
// (preserving each event's rail/origin -- shadow stays shadow, actual
// stays actual), accumulates real per-node actual_traffic/shadow_traffic
// counters (NetworkNode fields already defined by patch 0003), refreshes
// predicted_word at every node a shadow event newly reaches, classifies
// every actual_world_return event against that node's outstanding
// prediction (match/violation/omission), and awards delayed causal credit
// (NetworkNode.revision) on a match. Swaps each rail's frontier buffers and
// advances state.tick by one.
//
// This is a host-orchestrated sequence of kernel launches on the default
// stream, matching patch 0004's run_one_tick() -- not yet the CUDA-Graph
// device-tail-launched single graph the patch program specifies.
ActivityTickReport run_one_activity_tick(NetworkActivityDeviceState& state,
                                          std::uint32_t block_size = 128);

}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_ACTIVITY_CUH

// Patch 0003 of the FULL CUDA network-recipe patch program: paid
// network-matter representation. Every live site and every installed link
// consumes matter from a finite, per-development budget
// (Genome.header.matter_budget, patch 0002); nothing in the network grows
// for free. bcc32_network_page_directory.cuh (companion file) owns *where*
// a node lives (page-bucketed spatial addressing); this file owns *what* a
// node is (NetworkNode) and *how much it cost* (NetworkMatterAccount).

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_MATTER_CUH
#define HARDWARE_NATIVE_BCC32_NETWORK_MATTER_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/bcc32_network_recipe.hpp"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define BCC32_NETWORK_MATTER_HD __host__ __device__
#else
#define BCC32_NETWORK_MATTER_HD
#endif

namespace substrate::bcc32::network_recipe {

// Fixed fan-in/fan-out. Arbitrary multiway fan-in remains expressible
// through ordinary binary connector tissue and high fanout through trees of
// branch nodes -- every extra connector still pays matter, so this bound is
// storage/device-work discipline, not an expressiveness ceiling (patch
// program section "Patch 0003 -- Why two parents and eight children").
inline constexpr std::uint32_t kMaximumParents = 2;
inline constexpr std::uint32_t kMaximumChildren = 8;

inline constexpr std::uint32_t kInvalidNodeIndex = 0xffffffffu;

struct NetworkNode {
  SiteWord chemistry;
  std::uint32_t lineage;
  std::uint32_t flags;
  std::uint32_t birth_tick;
  std::uint32_t last_actual_tick;
  std::uint32_t actual_traffic;
  std::uint32_t shadow_traffic;
  std::uint32_t revision;
  std::uint32_t parent[kMaximumParents];
  std::uint32_t child[kMaximumChildren];
  SiteWord edge_chemistry[kMaximumChildren];
};
static_assert(std::is_standard_layout_v<NetworkNode> && std::is_trivial_v<NetworkNode>,
              "NetworkNode must be a fixed-width POD for device residency");

// A node with no parents/children yet, occupying no matter beyond its own
// slot. Every parent/child reference starts at kInvalidNodeIndex so an
// uninitialized slot can never be mistaken for a real edge.
BCC32_NETWORK_MATTER_HD inline NetworkNode empty_network_node(SiteWord chemistry,
                                                                std::uint32_t lineage,
                                                                std::uint32_t birth_tick) {
  NetworkNode node{};
  node.chemistry = chemistry;
  node.lineage = lineage;
  node.flags = 0;
  node.birth_tick = birth_tick;
  node.last_actual_tick = birth_tick;
  node.actual_traffic = 0;
  node.shadow_traffic = 0;
  node.revision = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (std::uint32_t i = 0; i < kMaximumParents; ++i) node.parent[i] = kInvalidNodeIndex;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (std::uint32_t i = 0; i < kMaximumChildren; ++i) {
    node.child[i] = kInvalidNodeIndex;
    node.edge_chemistry[i] = 0;
  }
  return node;
}

// Finite construction economy. `initial` is fixed at development start
// (Genome.header.matter_budget) and never grows. `live_nodes`/`live_edges`/
// `dormant`/`lesioned` are matter currently allocated to something;
// `reclaimed` is a monotonic cumulative audit counter of matter that has
// been returned to the available pool (turnover, patch 0004's
// retract/repair rules); `repair_pressure` is an audit signal (unmet repair
// demand), not part of the balance equation. The conservation invariant is
// `initial >= live_nodes + live_edges + dormant + lesioned` at every
// instant -- a committed transaction may never make this false.
struct NetworkMatterAccount {
  std::uint64_t initial;
  std::uint64_t live_nodes;
  std::uint64_t live_edges;
  std::uint64_t dormant;
  std::uint64_t lesioned;
  std::uint64_t reclaimed;
  std::uint64_t repair_pressure;
};
static_assert(std::is_standard_layout_v<NetworkMatterAccount> &&
                  std::is_trivial_v<NetworkMatterAccount>,
              "NetworkMatterAccount must be a fixed-width POD for device residency");

BCC32_NETWORK_MATTER_HD inline std::uint64_t committed_matter(const NetworkMatterAccount& account) {
  return account.live_nodes + account.live_edges + account.dormant + account.lesioned;
}

enum class MatterCommitError : std::uint32_t {
  kNone = 0,
  kInsufficientMatter,
};

// Which bucket a unit of matter is charged to. There is no "free" bucket --
// every debit names exactly one.
enum class MatterBucket : std::uint32_t {
  kLiveNode = 0,
  kLiveEdge = 1,
  kDormant = 2,
  kLesioned = 3,
};

// Atomically (device-safe) debits `amount` units of matter into `bucket`.
// Fails closed with kInsufficientMatter and leaves the account byte-for-byte
// unchanged when `committed_matter(account) + amount > account.initial` --
// this function may never clamp the request down to what's affordable or
// silently discard part of it; the caller (a Life Function construction
// rule, patch 0004) must reject the whole candidate on failure.
//
// Known limitation: this per-call atomicAdd-then-verify-then-refund pattern
// is exact for concurrent debits into the *same* bucket, but has a genuine
// TOCTOU race across *different* buckets debited by different threads in
// the same tick (each thread's overshoot check reads all four bucket totals
// without a fence, so two threads debiting different buckets can both pass
// their individual checks in an interleaving that nets the account over
// `initial`). Patch 0004's Life Function graph is specified to resolve
// conflicting proposals via a deterministic commutative reduction *before*
// any commit is issued for a tick (patch program section "Patch 0004 --
// Deterministic conflict law"), which sidesteps this race by construction
// rather than by locking; this function is safe as used sequentially (this
// patch's own contract) and safe under same-bucket contention, but must not
// be called from multiple threads targeting different buckets without that
// upstream serialization until patch 0004 lands it.
BCC32_NETWORK_MATTER_HD inline MatterCommitError debit_matter(NetworkMatterAccount& account,
                                                                 MatterBucket bucket,
                                                                 std::uint64_t amount) {
  std::uint64_t* target = nullptr;
  switch (bucket) {
    case MatterBucket::kLiveNode: target = &account.live_nodes; break;
    case MatterBucket::kLiveEdge: target = &account.live_edges; break;
    case MatterBucket::kDormant: target = &account.dormant; break;
    case MatterBucket::kLesioned: target = &account.lesioned; break;
  }
#if defined(__CUDA_ARCH__)
  // Reserve first (atomicAdd), then verify; refund on overshoot. This is
  // the same claim-then-verify pattern as reserve_node_slot() in
  // bcc32_network_page_directory.cuh, and for the same reason: CUDA has no
  // atomic compare-and-add-if-under-limit primitive, so a plain compare
  // followed by a separate add would race between threads.
  atomicAdd(reinterpret_cast<unsigned long long*>(target), static_cast<unsigned long long>(amount));
  std::uint64_t committed_after = account.live_nodes + account.live_edges + account.dormant +
                                   account.lesioned;
  if (committed_after > account.initial) {
    atomicAdd(reinterpret_cast<unsigned long long*>(target),
              static_cast<unsigned long long>(-static_cast<long long>(amount)));
    return MatterCommitError::kInsufficientMatter;
  }
  return MatterCommitError::kNone;
#else
  const std::uint64_t committed_before = committed_matter(account);
  if (committed_before + amount > account.initial) {
    return MatterCommitError::kInsufficientMatter;
  }
  *target += amount;
  return MatterCommitError::kNone;
#endif
}

// Moves `amount` units from `bucket` back to the available pool (turnover):
// decrements the named bucket and increments the cumulative `reclaimed`
// audit counter. Never invents matter -- callers must not reclaim more than
// is currently committed to that bucket; this function does not itself
// enforce that (the caller already knows how much it lesioned/retired), but
// the conservation contract still holds because it can only ever remove
// matter from a bucket, never add.
BCC32_NETWORK_MATTER_HD inline void reclaim_matter(NetworkMatterAccount& account,
                                                     MatterBucket bucket, std::uint64_t amount) {
  switch (bucket) {
    case MatterBucket::kLiveNode: account.live_nodes -= amount; break;
    case MatterBucket::kLiveEdge: account.live_edges -= amount; break;
    case MatterBucket::kDormant: account.dormant -= amount; break;
    case MatterBucket::kLesioned: account.lesioned -= amount; break;
  }
  account.reclaimed += amount;
}

}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_MATTER_CUH

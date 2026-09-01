#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

#include <cuda/atomic>

#include "bcc32_persistent_kernel.hpp"
#include "bcc32_resident_developmental_adult_bridge.hpp"
#include "bcc32_device_ordinary_f_timeline.cuh"
#include "bcc32_resident_causal_constraint_reafferent.cuh"
#include "bcc32_resident_close_work.cuh"
#include "bcc32_resident_cross_contact_context.cuh"
#include "bcc32_resident_causal_constraint_participation_end.cuh"
#include "bcc32_resident_egress_history.cuh"
#include "bcc32_resident_episodic_completion.cuh"
#include "bcc32_resident_learned_cost_search.cuh"
#include "bcc32_resident_means_end_inversion.cuh"
#include "bcc32_resident_mixed_provenance_evidence.cuh"
#include "bcc32_resident_mouth_compartment_lease.cuh"
#include "bcc32_resident_open_inquiry.cuh"
#include "bcc32_resident_open_inquiry_return_gate.cuh"
#include "bcc32_resident_pending_means.cuh"
#include "bcc32_resident_predictive_shadow_assay.cuh"
#include "bcc32_resident_recurrent_carrier.cuh"
#include "causal_rewrite_universe.cuh"

namespace substrate::bcc32::causal_rewrite {

// A generated execution prefix is valid context only while the following
// external suffix actually engages the resident relation ecology. If PAUSE
// proves that no relation is applicable, preserve the exact external suffix
// as a fresh trajectory and retire only the exhausted zero-authority prefix.
// Relation-linked and ambiguous paths never call this boundary.
__device__ __noinline__ bool restart_external_suffix_after_unresolved_context(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  const std::uint32_t current_slot = find_current_trajectory(state);
  if (current_slot == kInvalid) return false;
  const Record& current = state->records[current_slot];
  if ((current.lane[7] & kTrajectoryHasGenerated) == 0u ||
      (current.lane[7] &
       ~(kTrajectoryHasGenerated | kTrajectoryWasYielded |
         kTrajectoryHasCarry)) != 0u ||
      current.lane[2] == 0u)
    return false;

  std::uint32_t suffix_begin = kInvalid;
  if (!causal_relation_distributed_context_suffix(
          state, current, nullptr, &suffix_begin) ||
      suffix_begin >= current.lane[2])
    return false;
  const std::uint32_t suffix_count = current.lane[2] - suffix_begin;
  if (suffix_count > kMaximumTrajectoryEvents) return false;

  std::uint32_t suffix[kMaximumTrajectoryEvents]{};
  for (std::uint32_t index = 0u; index < suffix_count; ++index) {
    const std::uint32_t source_index = suffix_begin + index;
    if (!trajectory_word_at(state, current.lane[1], source_index,
                            &suffix[index]))
      return false;
    mixed_provenance::Origin origin = mixed_provenance::Origin::external;
    std::uint32_t producer = kInvalid;
    if (!mixed_provenance::origin_at(state, current, source_index, &origin,
                                     &producer) ||
        origin != mixed_provenance::Origin::external ||
        producer != kInvalid)
      return false;
  }

  const std::uint32_t owner = current.lane[1];
  mixed_provenance::clear_provenance(state, owner);
  clear_trajectory(state, current_slot);
  clear_generated_word(state);
  ResidentRewriteEngine engine(state);
  for (std::uint32_t index = 0u; index < suffix_count; ++index) {
    if (!mixed_provenance::consume_external_event(
            engine, RawRewriteEvent{suffix[index], 1u, kEventFrameNone},
            false))
      return false;
  }
  refresh_receipt(state);
  const std::uint32_t fresh = find_current_trajectory(state);
  return fresh != kInvalid && state->records[fresh].lane[2] == suffix_count &&
         (state->records[fresh].lane[7] &
          (kTrajectoryHasGenerated | kTrajectoryHasCarry)) == 0u;
}

}  // namespace substrate::bcc32::causal_rewrite

namespace substrate::bcc32::persistent_kernel {
namespace {

namespace rewrite = substrate::bcc32::causal_rewrite;
namespace resident_recurrent_carrier = substrate::bcc32::resident_recurrent_carrier;
namespace resident_predictive_shadow_assay =
    substrate::bcc32::resident_predictive_shadow_assay;
namespace constraint_reafferent =
    substrate::bcc32::resident_causal_constraint_participation::reafferent;
namespace participation_end =
    substrate::bcc32::resident_causal_constraint_participation::physical_end;
namespace inquiry_return =
    substrate::bcc32::causal_rewrite::open_inquiry_return_gate;
namespace ordinary_f = substrate::bcc32::device_ordinary_f_timeline;
namespace grown = substrate::bcc32::developmental_adult;

// 1 MiB canonical byte corpus => 1,048,576 BoundaryWords => 4096 packets.
// A power-of-two reservoir keeps modulo cheap and removes producer/device
// lockstep for the existing adult-facing workload.
inline constexpr std::uint32_t kIngressSlots = 4096u;
inline constexpr std::uint32_t kIngressPacketWords =
    static_cast<std::uint32_t>(PersistentKernel::kMaximumRawContactWords);
// Ordered scheduling slice only; never a semantic chunk.
inline constexpr std::uint32_t kIngressWordsPerEpoch = 64u;
static_assert(kIngressWordsPerEpoch != 0u &&
              kIngressPacketWords % kIngressWordsPerEpoch == 0u &&
              kIngressWordsPerEpoch < kIngressPacketWords);
inline constexpr std::uint32_t kMaxContactWords = 64u;
// One ticket may span transport slots and continuation pages. The per-slot
// ingress count is a transport bound only; logical return extent is consumed
// incrementally into the private staging adult below.
inline constexpr std::uint32_t kActionWords = 256u;
// This is a passive raw motor surface, not a language model context window.
// Four KiB allows a resident to retain and expose paragraph-scale output while
// egress history still determines its actual lifetime and provenance.
inline constexpr std::uint32_t kLanguageBytes = 4096u;
inline constexpr std::size_t kEgressHistoryBytes =
    sizeof(egress_history::Event) * egress_history::capacity;
inline constexpr std::uint32_t kIdlePacingNanoseconds = 250000u;

// RUNG-1 circuit tracing is deliberately small. It observes only public
// boundary publications and is not a resident-history substitute.
inline constexpr std::uint32_t kBitBusL0Capacity = 1024u;

// Physical aperture identifier. This is not a semantic region/organ label.
// Resident Record loci occupy ordinary uint32_t loci; this reserved aperture
// value denotes the process boundary itself.
inline constexpr std::uint32_t kBitBusBoundaryLocus = 0xfffffffeu;
inline constexpr std::uint32_t kBitBusTransferBoundary = 1u;

inline constexpr std::uint32_t kResidentLineageReceiptVersion = 1u;
std::atomic<std::uint64_t> next_resident_instance_nonce{1u};
// The linked sm_89 root currently exceeds the device-graph launch aperture
// when learned-cost search is inlined into it. Keep the process-wide reserve
// explicit and move that bounded search behind a real device call below; both
// the linked frame and the graph launch remain runtime-gated.
//
// This limit is a device-wide reservation charged per *potential* resident
// thread slot (80 SMs x 1536 slots on this sm_89 card = 122,880 slots), not
// per launched thread -- so each 1 KiB added here costs exactly
// 122,880 x 1,024 bytes = 125,829,120 bytes = 120.0 MiB of device-wide
// reservation before any organism work runs (see GitHub #1159/#1160/#1060
// and docs/diary/2026-08-16/2026-08-16T20-55-00+02-00-...). At the
// historical 120 KiB this reserved 14.1 of 16 GiB up front.
//
// GitHub #1060/#1162 (docs/diary/2026-08-17/2026-08-17T01-21-37+02-00-
// github-1060-revision-lineage-stack-bound.md) broke the eight-function
// mutual-recursion cycle in the causal-revision-transfer path that used to
// make `nvlink` report resident_rewrite_epoch_kernel's stack as
// EIATTR_MIN_STACK_SIZE=0xffffffff (`cuobjdump -res-usage`: `STACK:UNKNOWN`).
// The whole reachable closure is now a bounded, non-recursive worklist, and
// a fresh relink of the registered
// bcc32_cuda_resident_action_return_mixed_provenance_production_contract.real
// gives a real, concrete, nvlink-computed bound for both kernels that
// previously could not be bounded at all:
//   resident_rewrite_epoch_kernel      STACK:103912 bytes (101.48 KiB)
//   resident_epoch_post_return_kernel  STACK:20480 bytes  (20.00 KiB)
// This is now a *provable* per-kernel worst case, not an empirical failure
// point the way the prior 48 KiB crash / 80 KiB floor were -- so it does not
// need the same ~67% empirical-uncertainty margin the unprovable
// STACK:UNKNOWN situation warranted. It still needs real margin for
// (a) compiler/toolchain variance across future builds, (b) other kernels
// or contracts not yet measured that could link a slightly different call
// graph shape onto the same context-wide limit, and (c) code changes to
// this call graph before this comment is next revisited.
//
// 112 KiB (114,688 bytes) was chosen against that 103,912-byte measurement:
// +10,776 bytes (+10.4%) of margin, reserving 13.125 GiB of the 16 GiB card.
//
// The 103,912 bytes were then reduced rather than budgeted for. Two of the
// three largest frames on that chain were storage-placement accidents, not
// work the kernel needs stack for:
//
//   * learned_cost_search::select_resident_learned_cost_route held its whole
//     Search working set as a stack local -- Label labels[1024] (16,384 B),
//     ProgramEdge edges[1024] (24,576 B), and reachable_cycle's
//     color/stack/next_slot[1024] (9,216 B) inlined into it. It is now a
//     single __device__ working set and the frame is 96 bytes.
//   * The cross_contact dormant-compaction helpers were plain `inline`, so
//     nvcc folded several into one caller frame and their multi-KiB scratch
//     arrays summed instead of overlapping. Outlined, they no longer do.
//
// A fresh relink of the same registered contract measured the result:
//   resident_rewrite_epoch_kernel      STACK:69080 bytes (67.46 KiB)
//   resident_epoch_post_return_kernel  STACK:20480 bytes  (unchanged)
// -- 34,832 bytes off the worst case, from nvlink itself. 80 KiB (81,920
// bytes) was chosen against that 69,080-byte bound: +12,840 bytes (+18.6%).
//
// 8824fd92b3 then took a third bite out of the same chain: one span-match
// scratch block was being charged twice on a single call path, so its bytes
// were summed into the caller frame instead of overlapping with themselves.
// A relink of the same registered contract on 6978232956 measures:
//   resident_rewrite_epoch_kernel      STACK:47184 bytes (46.08 KiB)  <- max
//   resident_epoch_post_return_kernel  STACK:20480 bytes  (unchanged)
// and 47,184 bytes is also the maximum over ALL 426 functions in that linked
// image, not just this closure -- `hardware_native/scripts/audit_device_stack_
// bound.py` maxes over the whole binary because cudaLimitStackSize is a
// context-wide reservation. Cross-validated by two independent readings of
// the same cubin that agree to the byte: EIATTR_MIN_STACK_SIZE as reported by
// `cuobjdump -res-usage`, and an EIATTR_FRAME_SIZE + `.rel.text` call-graph
// reconstruction of the deepest chain.
//
// 56 KiB (57,344 bytes) is chosen against that: +10,160 bytes (+21.5%) over
// the measured requirement, i.e. strictly MORE relative margin than the
// 80 KiB value carried over the 69,080-byte bound it was set against
// (+18.6%), and more than the 112 KiB value carried (+10.4%). The reasons
// margin is needed are unchanged (toolchain variance, unmeasured contracts
// sharing this context-wide limit, future call-graph change); this is a
// wider cushion bought at a lower price, not a tightening. It reserves
// 56 x 120.0 MiB = 6,720 MiB = 6.5625 GiB, returning a further 2.8125 GiB
// of the card relative to 80 KiB (2,880 MiB predicted; measured below).
//
// Note that neither 80 KiB nor this 56 KiB is the old 80 KiB. That value was
// an empirical bisection artefact and was genuinely short of the then-true
// 103,912-byte worst case, passing only because the tested paths never
// reached the deepest one. Every value since sits above a bound nvlink
// computed, with the margin computed rather than guessed. Derive from the
// measured bound; never from "the tests pass".
//
// Do not lower this below 47,184 bytes without a fresh nvlink measurement of
// the current call graph -- and prefer re-deriving the number over trusting
// this comment, because these bounds have moved three times in one night.
// Two ctests now hold the line and did not exist for the earlier round trips:
//   bcc32_resident_device_stack_budget_contract   (the ~9 RWR0 kernels)
//   bcc32_device_stack_bound_ratchet_contract     (every function in the
//       image, plus growth of the STACK:UNKNOWN set)
// Verified at 56 KiB by relink (cuobjdump -res-usage, both ctests GREEN)
// plus a real GPU run of the mixed-provenance production contract under
// tools/gpu_runtime_lock.sh: GREEN-BOUNDED, output byte-identical to the
// 80 KiB run. Measured cost on an idle card (488 MiB): peak 10,462 MiB at
// 80 KiB, 7,582 MiB at 56 KiB -- 2,880 MiB freed, exactly the 120.0 MiB/KiB
// the 122,880 thread slots predict.
inline constexpr std::size_t kResidentDeviceStackDefaultBytes = 56u * 1024u;
inline constexpr char kResidentDeviceStackBytesEnvironment[] =
    "BCC32_RESIDENT_DEVICE_STACK_BYTES";
// Grown resident RecordPages are allocated by the same deterministic device
// allocator used by the substrate's HD growth law.  The host-side page
// contract exercises that law without CUDA's device heap; the adult runtime
// must provision the heap before its persistent graph can cross page zero.
inline constexpr std::size_t kResidentDeviceHeapBytes = 16u * 1024u * 1024u;
// The canonical adult and its ordinary-F child share the production GPU.
// Keep the first integrated shell to one physical 1M-site chunk until lawful
// reclamation/compaction is available; this is a capacity bound, not a
// semantic limit.
inline constexpr std::uint32_t kResidentFHistoryCapacity = 1u << 20u;
inline constexpr std::uint32_t kResidentFHistoryDepth = 1u;
inline constexpr std::int64_t kResidentFApertureEdgeChunks = 1;

// The ordered epoch remains lane 0's sequencer. At the action-return close
// boundary the other lanes classify the bounded resident extent into stable
// slot lists; lane 0 then applies the exact historical mutation order. This
// is a real GPU frontier, not a host-side candidate selector.
inline constexpr std::uint32_t kEpochCleanupThreads = 256u;
inline constexpr std::uint32_t kEpochCleanupMaxItems =
    rewrite::kMaxResidentPages * rewrite::kRecordsPerPage;
inline constexpr std::uint32_t kEpochCleanupBlocks =
    (kEpochCleanupMaxItems + kEpochCleanupThreads - 1u) /
    kEpochCleanupThreads;
inline constexpr std::uint32_t kEpochCleanupIdle = 0u;
inline constexpr std::uint32_t kEpochCleanupStaged = 1u;
inline constexpr std::uint32_t kEpochCleanupCommitted = 2u;
inline constexpr std::uint32_t kEpochCleanupFault = 3u;

struct alignas(8) ResidentEpochCleanupScratch {
  std::uint8_t distributed_flags[kEpochCleanupMaxItems]{};
  std::uint8_t qualified_mixed_flags[kEpochCleanupMaxItems]{};
  std::uint32_t phase = kEpochCleanupIdle;
  std::uint32_t scanned_records = 0u;
  std::uint32_t committed_records = 0u;
  std::uint32_t fault = 0u;
  std::uint32_t receipt_bits = 0u;
  std::uint64_t epoch_generation = 0u;
  std::uint64_t packet_generation = 0u;
  std::uint64_t continuation_next = 0u;
  ActionReturnTicket continuation_ticket{};
};

static_assert(std::is_trivially_copyable_v<ResidentEpochCleanupScratch>);

struct ResidentEpochGraphControl {
  cudaGraphExec_t root_graph_exec = nullptr;
  cudaGraphExec_t cleanup_graph_exec = nullptr;
  cudaGraphExec_t post_graph_exec = nullptr;
  std::uint32_t publish_due = 0u;
  std::uint32_t history_changed = 0u;
  std::uint32_t generated = 0u;
  std::uint32_t consumed_contact = 0u;
  std::uint32_t accepted_action_return = 0u;
  std::uint32_t device_body_return = 0u;
  std::uint32_t device_body_attached = 0u;
};

static_assert(std::is_trivially_copyable_v<ResidentEpochGraphControl>);

// The disclosed RWR0 Genesis hash reuses the project's first public 256-bit
// heredity value, but the new law identity is explicit: this is not an
// unchanged-BCC-F claim.
inline constexpr std::uint64_t kGenesisHash[4]{0x6d3a91c5c71b2e49ull, 0xa24baed4963ee407ull,
                                               0x9fb21c651e98df25ull, 0xd1b54a32d192ed03ull};

struct IngressSlot {
  std::uint64_t sequence = 0u;
  std::uint32_t count = 0u;
  std::uint32_t offset = 0u;
  // Transport envelope only: marks the first packet of one present_raw call.
  // It carries no word class, relation, answer, or resident authority.
  std::uint32_t contact_start = 0u;
  BoundaryWord words[kIngressPacketWords]{};
};

struct IngressRing {
  IngressSlot slots[kIngressSlots]{};
  std::uint64_t published = 0u;
  std::uint64_t consumed = 0u;
};

struct ActionReturnIngress {
  BoundaryWord words[kMaxContactWords]{};
  std::uint32_t count = 0u;
  std::uint32_t final_chunk = 0u;
  ActionReturnTicket ticket{};
  std::uint64_t chunk_sequence = 0u;
  std::uint64_t producer_instance = 0u;
  std::uint64_t source_epoch = 0u;
  std::uint64_t route_sequence = 0u;
  // Device-produced before release publication. This is a declared route
  // integrity commitment, not a physical-source or body-reafference proof.
  ContentAddress producer_commitment{};
  std::uint32_t device_body_produced = 0u;
  std::uint64_t published = 0u;
  std::uint64_t consumed = 0u;
};

struct DeviceBodyControl {
  std::uint32_t attachment = static_cast<std::uint32_t>(DeviceBodyAttachment::detached);
  // Opaque body geometry: the same resident actuator word can encounter a
  // different lawful body mapping without the host supplying a target sensor
  // word. This is a body-side control, not a semantic decoder or learner.
  std::uint32_t actuator_permutation = 0u;
  // Test-only corruption gate. It is never an ingress writer: when set, the
  // device producer flips one raw word after release publication so the next
  // consumer must reject the still-pending ticketed return.
  std::uint32_t test_corrupt_post_publication_word = 0u;
};

DeviceBodyControl* test_device_body_control = nullptr;

#include "bcc32_resident_census.inl"

// Defined in the action-return extraction below. Keeping the PAUSE relation
// probe with the action-return boundary prevents the oversized runtime parent
// from accumulating another resident scheduling responsibility.
__device__ __noinline__ std::uint32_t
pause_relation_disposition(DeviceState* state);


// Egress derivation, receipt assembly, and publication for this epoch's
// DeviceState are implemented in bcc32_resident_rewrite_runtime_egress.inl,
// included later in this file once EgressState is fully defined too. These
// forward declarations let the epoch kernel and its device wrappers call
// them ahead of that later definition point in the same translation unit.
__device__ void refresh_egress_history_digest(DeviceState* state);
__device__ __noinline__ void derive_output(const DeviceState* state,
                                           bool generated_this_epoch,
                                           DerivedOutput* output);
__device__ void update_output_projection(DeviceState* state);
__device__ bool derive_resident_world_lineage(
    const DeviceState* state, std::uint64_t* organization_digest);
__device__ bool derive_active_open_inquiry_identity(
    const DeviceState* state, std::uint32_t* owner,
    std::uint32_t* identity, std::uint32_t* generation);
__device__ void fill_receipt(DeviceState* state, bool consumed_contact,
                             bool accepted_action_return, bool device_body_return,
                             bool device_body_attached,
                             const DerivedOutput& derived);
__device__ void publish_egress(const DeviceState* state, const DerivedOutput& derived,
                               bool history_changed, EgressState* egress);

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::size_t resident_device_stack_bytes() {
  const char* text = std::getenv(kResidentDeviceStackBytesEnvironment);
  if (text == nullptr) return kResidentDeviceStackDefaultBytes;
  if (*text == '\0')
    throw std::invalid_argument(
        "BCC32_RESIDENT_DEVICE_STACK_BYTES must be a nonzero decimal byte count");

  std::uint64_t parsed = 0u;
  const char* const end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, parsed, 10);
  if (result.ec != std::errc{} || result.ptr != end || parsed == 0u ||
      parsed > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(
        "BCC32_RESIDENT_DEVICE_STACK_BYTES must be a nonzero decimal byte count");
  return static_cast<std::size_t>(parsed);
}

void configure_resident_device_stack_once() {
  static std::once_flag configured;
  std::call_once(configured, [] {
    // The host may vary only this physical CUDA reservation between otherwise
    // byte-identical executions. It cannot select resident routes, evidence,
    // actions, or outputs. Parse before changing any context-owned limit so a
    // malformed experiment fails without partially configuring the runtime.
    const std::size_t stack_bytes = resident_device_stack_bytes();
    // Device malloc/free uses a context-owned fixed heap. Establish it before
    // any other module-sensitive resident resource configuration can make the
    // device allocator active; CUDA rejects a later heap-limit change.
    require_cuda(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                                    kResidentDeviceHeapBytes),
                 "set resident rewrite device heap");

    std::size_t resident_heap_bytes = 0u;
    require_cuda(cudaDeviceGetLimit(&resident_heap_bytes,
                                    cudaLimitMallocHeapSize),
                 "query resident rewrite device heap");
    if (resident_heap_bytes < kResidentDeviceHeapBytes)
      throw std::runtime_error(
          "resident rewrite device heap smaller than required grown-page aperture");

    require_cuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_bytes),
                 "set resident rewrite device stack");
    std::size_t applied_stack_bytes = 0u;
    require_cuda(cudaDeviceGetLimit(&applied_stack_bytes, cudaLimitStackSize),
                 "query resident rewrite device stack");
    if (applied_stack_bytes < stack_bytes)
      throw std::runtime_error(
          "resident rewrite device stack smaller than requested aperture");
  });
}

ContentAddress literal_identity(const std::array<std::uint64_t, 4u>& lanes,
                                std::uint64_t byte_count) {
  ContentAddress result{};
  result.byte_count = byte_count;
  for (std::uint32_t lane = 0u; lane < 4u; ++lane)
    for (std::uint32_t byte = 0u; byte < 8u; ++byte)
      result.digest[lane * 8u + byte] = static_cast<std::uint8_t>(lanes[lane] >> (byte * 8u));
  return result;
}

ContentAddress rewrite_law_identity() {
  return literal_identity(
      {0x525752305f4c4157ull, 0x4556454e545f3030ull, 0x5455504c455f3030ull, 0x5245575249544552ull},
      32u);
}

ContentAddress rewrite_genesis_identity() {
  return literal_identity({kGenesisHash[0], kGenesisHash[1], kGenesisHash[2], kGenesisHash[3]},
                          32u);
}

ContentAddress rewrite_image_identity() {
  return literal_identity(
      {0x525752305f534d38ull, 0x554e49464f524d52ull, 0x45434f52445f5630ull, 0x3030303030303031ull},
      32u);
}

ContentAddress rewrite_sealed_identity() {
  return literal_identity(
      {0x525752305f534541ull, 0x45445f5241575f49ull, 0x4f5f434f4d4d4954ull, 0x5f4f4e455f434c4bull},
      32u);
}

template <typename T>
T load_host(const T* address) {
  std::atomic_ref<T> atomic(*const_cast<T*>(address));
  return atomic.load(std::memory_order_acquire);
}

template <typename T>
void store_host(T* address, T value) {
  std::atomic_ref<T> atomic(*address);
  atomic.store(value, std::memory_order_release);
}

// Focused-contract hook only. It cannot publish a return, choose raw words, or
// alter physical proof rails; it merely enables the device-local mutation arm.
extern "C" void bcc32_test_set_device_body_post_publication_corruption(
    std::uint32_t enabled) {
  if (test_device_body_control == nullptr)
    return;
  store_host(&test_device_body_control->test_corrupt_post_publication_word,
             enabled == 0u ? 0u : 1u);
}

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t h, std::uint64_t value) {
  h ^= value + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
  h *= 0xbf58476d1ce4e5b9ull;
  return h ^ (h >> 27u);
}

__device__ void digest_bytes(const void* data, std::size_t bytes, ContentAddress* result,
                             std::uint64_t seed) {
  const auto* raw = static_cast<const std::uint8_t*>(data);
  std::uint64_t lanes[4]{seed, seed ^ 0x13198a2e03707344ull, seed ^ 0xa4093822299f31d0ull,
                         seed ^ 0x082efa98ec4e6c89ull};
  for (std::size_t index = 0u; index < bytes; ++index)
    lanes[index & 3u] = mix64(lanes[index & 3u], raw[index]);
  result->byte_count = bytes;
  auto* digest = reinterpret_cast<std::uint8_t*>(&result->digest);
  for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
    const std::uint64_t value = mix64(lanes[lane], bytes + lane);
    for (std::uint32_t byte = 0u; byte < 8u; ++byte)
      digest[lane * 8u + byte] = static_cast<std::uint8_t>(value >> (byte * 8u));
  }
}

__device__ void begin_action_return_digest(DeviceState* state) {
  constexpr std::uint64_t kSeed = 0x4152544e5f524157ull;
  state->action_return_stream_lanes[0] = kSeed;
  state->action_return_stream_lanes[1] = kSeed ^ 0x13198a2e03707344ull;
  state->action_return_stream_lanes[2] = kSeed ^ 0xa4093822299f31d0ull;
  state->action_return_stream_lanes[3] = kSeed ^ 0x082efa98ec4e6c89ull;
  state->action_return_stream_bytes = 0u;
  state->action_return_stream_words = 0u;
}

__device__ void extend_action_return_digest(DeviceState* state, const BoundaryWord* words,
                                            std::uint32_t count) {
  const auto* raw = reinterpret_cast<const std::uint8_t*>(words);
  const std::uint64_t bytes = static_cast<std::uint64_t>(count) * sizeof(BoundaryWord);
  for (std::uint64_t index = 0u; index < bytes; ++index) {
    const std::uint64_t ordinal = state->action_return_stream_bytes + index;
    const std::uint32_t lane = static_cast<std::uint32_t>(ordinal & 3u);
    state->action_return_stream_lanes[lane] =
        mix64(state->action_return_stream_lanes[lane], raw[index] + ordinal);
  }
  state->action_return_stream_bytes += bytes;
  state->action_return_stream_words += count;
}

__device__ ContentAddress finish_action_return_digest(const DeviceState* state) {
  ContentAddress result{};
  result.byte_count = state->action_return_stream_bytes;
  auto* digest = reinterpret_cast<std::uint8_t*>(&result.digest);
  for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
    const std::uint64_t value =
        mix64(state->action_return_stream_lanes[lane], state->action_return_stream_bytes + lane);
    for (std::uint32_t byte = 0u; byte < 8u; ++byte)
      digest[lane * 8u + byte] = static_cast<std::uint8_t>(value >> (byte * 8u));
  }
  return result;
}

__device__ void release_owned_rewrite_pages(
    rewrite::ResidentRewriteState* state) {
  if (state == nullptr) return;
  for (std::uint32_t page = 0u;
       page + 1u < rewrite::kMaxResidentPages; ++page) {
    if (state->directory.pages[page] != nullptr) {
      free(state->directory.pages[page]);
      state->directory.pages[page] = nullptr;
    }
  }
  state->directory.live_page_count = 1u;
}

// RWR0 §3.4 page-level copy-on-write: staging owns a deep clone of every
// grown RecordPage. A raw struct copy is still used for page zero and scalar
// receipts, but page pointers are immediately cleared before page records are
// cloned, so a rejected transaction can never mutate canonical matter.
__device__ bool clone_rewrite_state(
    rewrite::ResidentRewriteState* destination,
    const rewrite::ResidentRewriteState* source) {
  if (destination == nullptr || source == nullptr || destination == source)
    return false;
  const std::uint32_t source_pages = source->directory.live_page_count;
  if (source_pages == 0u || source_pages > rewrite::kMaxResidentPages) {
    destination->fault = 1u;
    return false;
  }
  release_owned_rewrite_pages(destination);
  auto* out = reinterpret_cast<std::uint8_t*>(destination);
  const auto* in = reinterpret_cast<const std::uint8_t*>(source);
  for (std::size_t index = 0u; index < sizeof(rewrite::ResidentRewriteState); ++index)
    out[index] = in[index];
  destination->directory.live_page_count = 1u;
  for (std::uint32_t page = 0u;
       page + 1u < rewrite::kMaxResidentPages; ++page)
    destination->directory.pages[page] = nullptr;
  for (std::uint32_t page = 1u; page < source_pages; ++page) {
    auto* clone = static_cast<rewrite::RecordPage*>(
        malloc(sizeof(rewrite::RecordPage)));
    if (clone == nullptr) {
      release_owned_rewrite_pages(destination);
      destination->fault = 1u;
      return false;
    }
    const rewrite::RecordPage* original = source->directory.pages[page - 1u];
    if (original == nullptr) {
      free(clone);
      release_owned_rewrite_pages(destination);
      destination->fault = 1u;
      return false;
    }
    for (std::uint32_t slot = 0u; slot < rewrite::kRecordsPerPage; ++slot)
      clone->slots[slot] = original->slots[slot];
    destination->directory.pages[page - 1u] = clone;
    destination->directory.live_page_count = page + 1u;
  }
  return true;
}

// Commit transfers page ownership from a completed staging world to the
// canonical world. The source staging directory is nulled after the raw copy,
// preventing a later rejected transaction from freeing canonical pages.
__device__ bool move_rewrite_state(
    rewrite::ResidentRewriteState* destination,
    rewrite::ResidentRewriteState* source) {
  if (destination == nullptr || source == nullptr || destination == source)
    return false;
  release_owned_rewrite_pages(destination);
  auto* out = reinterpret_cast<std::uint8_t*>(destination);
  const auto* in = reinterpret_cast<const std::uint8_t*>(source);
  for (std::size_t index = 0u; index < sizeof(rewrite::ResidentRewriteState); ++index)
    out[index] = in[index];
  for (std::uint32_t page = 0u;
       page + 1u < rewrite::kMaxResidentPages; ++page)
    source->directory.pages[page] = nullptr;
  source->directory.live_page_count = 1u;
  return true;
}

__device__ void reset_action_return_stream(DeviceState* state) {
  release_owned_rewrite_pages(&state->action_return_staging_world);
  state->action_return_stream_ticket = ActionReturnTicket{};
  state->action_return_stream_next_chunk = 1u;
  state->action_return_stream_bytes = 0u;
  state->action_return_stream_words = 0u;
  state->action_return_stream_active = 0u;
  state->action_return_stream_saw_physical_end = 0u;
  state->action_return_stream_inquiry_reply_required = 0u;
  state->action_return_stream_inquiry_reply_settled = 0u;
  state->action_return_stream_from_device_body = 0u;
  state->action_return_stream_device_body_consequence_word = 0u;
  state->action_return_stream_constraint_delta = ActionReturnConstraintDelta{};
  state->action_return_stream_distributed_trajectory_owner = rewrite::kInvalid;
}

__device__ void clear_action_return_distributed_trajectory(
    DeviceState* state, ResidentEpochCleanupScratch* scratch) {
  if (state == nullptr || scratch == nullptr ||
      state->action_return_stream_distributed_trajectory_owner ==
          rewrite::kInvalid)
    return;
  const std::uint32_t owner =
      state->action_return_stream_distributed_trajectory_owner;
  const std::uint32_t extent = scratch->scanned_records;
  rewrite::mixed_provenance::clear_provenance(
      &state->action_return_staging_world, owner);
  std::uint32_t committed = 0u;
  for (std::uint32_t slot = 0u; slot < extent; ++slot) {
    if (scratch->distributed_flags[slot] == 0u)
      continue;
    rewrite::clear_trajectory(&state->action_return_staging_world, slot);
    ++committed;
  }

  // The second pass deliberately rechecks the post-first-pass record image.
  // This preserves the former scalar order when a slot could satisfy both
  // classifications before the first trajectory clear.
  for (std::uint32_t slot = 0u; slot < extent; ++slot) {
    if (scratch->qualified_mixed_flags[slot] == 0u)
      continue;
    rewrite::Record& record =
        rewrite::record_at(&state->action_return_staging_world, slot);
    if (record.matter_q8 == 0u ||
        record.lane[0] != rewrite::kFormTrajectory || record.lane[3] == 0u ||
        record.lane[7] !=
            rewrite::mixed_provenance::kTrajectoryQualifiedMixed)
      continue;
    const std::uint32_t mixed_owner = record.lane[1];
    rewrite::mixed_provenance::clear_provenance(
        &state->action_return_staging_world, mixed_owner);
    rewrite::clear_trajectory(&state->action_return_staging_world, slot);
    ++committed;
  }
  scratch->committed_records = committed;
}

__global__ void resident_epoch_cleanup_classify_kernel(
    DeviceState* state, ResidentEpochCleanupScratch* scratch) {
  if (state == nullptr || scratch == nullptr)
    return;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kEpochCleanupMaxItems)
    return;
  if (scratch->phase != kEpochCleanupStaged) {
    scratch->distributed_flags[index] = 0u;
    scratch->qualified_mixed_flags[index] = 0u;
    if (index == 0u)
      scratch->fault = rewrite::kCloseWorkTransactionFault;
    return;
  }
  const std::uint32_t capacity =
      rewrite::live_record_capacity(&state->action_return_staging_world);
  const std::uint32_t extent =
      capacity < kEpochCleanupMaxItems ? capacity : kEpochCleanupMaxItems;
  if (index == 0u)
    scratch->scanned_records = extent;
  if (index >= extent) {
    scratch->distributed_flags[index] = 0u;
    scratch->qualified_mixed_flags[index] = 0u;
    return;
  }
  const rewrite::Record& record =
      rewrite::record_at(&state->action_return_staging_world, index);
  scratch->distributed_flags[index] =
      record.matter_q8 != 0u &&
              record.lane[0] == rewrite::kFormTrajectory &&
              record.lane[1] ==
                  state->action_return_stream_distributed_trajectory_owner
          ? 1u
          : 0u;
  scratch->qualified_mixed_flags[index] =
      record.matter_q8 != 0u &&
              record.lane[0] == rewrite::kFormTrajectory &&
              record.lane[3] != 0u &&
              record.lane[7] ==
                  rewrite::mixed_provenance::kTrajectoryQualifiedMixed
          ? 1u
          : 0u;
}

__global__ void resident_epoch_cleanup_commit_kernel(
    DeviceState* state, ResidentEpochCleanupScratch* scratch) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  if (state == nullptr || scratch == nullptr ||
      scratch->phase != kEpochCleanupStaged || scratch->fault != 0u) {
    if (scratch != nullptr) {
      scratch->fault = rewrite::kCloseWorkTransactionFault;
      scratch->phase = kEpochCleanupFault;
    }
    return;
  }
  clear_action_return_distributed_trajectory(state, scratch);
  scratch->phase = kEpochCleanupCommitted;
  __threadfence();
}

__global__ void resident_epoch_cleanup_return_kernel(
    DeviceState* state, Lifecycle* lifecycle,
    ResidentEpochCleanupScratch* scratch,
    ResidentEpochGraphControl* control) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  if (state == nullptr || lifecycle == nullptr || scratch == nullptr ||
      control == nullptr || scratch->phase != kEpochCleanupCommitted ||
      control->root_graph_exec == nullptr) {
    if (scratch != nullptr) {
      scratch->fault = rewrite::kCloseWorkTransactionFault;
      scratch->phase = kEpochCleanupFault;
    }
    if (lifecycle != nullptr) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(0x52571101u, cuda::memory_order_release);
    }
    return;
  }
  __threadfence();
  const cudaError_t status =
      cudaGraphLaunch(control->root_graph_exec, cudaStreamGraphTailLaunch);
  if (status != cudaSuccess) {
    scratch->fault = static_cast<std::uint32_t>(status);
    scratch->phase = kEpochCleanupFault;
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
        lifecycle->continuation_fault);
    fault.store(0x52571100u | static_cast<std::uint32_t>(status),
                cuda::memory_order_release);
  }
}

__global__ void resident_epoch_post_return_kernel(
    DeviceState* state, EgressState* egress, Lifecycle* lifecycle,
    ResidentEpochGraphControl* control) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  if (state == nullptr || egress == nullptr || lifecycle == nullptr ||
      control == nullptr ||
      control->root_graph_exec == nullptr) {
    if (lifecycle != nullptr) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(0x52571104u, cuda::memory_order_release);
    }
    return;
  }
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> shutdown(
      lifecycle->shutdown);
  if (shutdown.load(cuda::memory_order_acquire) != 0u) {
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> stopped(
        lifecycle->stopped);
    stopped.store(1u, cuda::memory_order_release);
    return;
  }
  if (control->publish_due != 0u) {
    DerivedOutput derived{};
    derive_output(state, control->generated != 0u, &derived);
    // Edge-triggered history must not make a valid resident surface
    // unobservable between the generating epoch and the next passive poll.
    // Keep the last public surface on a quiet/heartbeat publication while
    // still advancing the receipt and egress generation below. This copies no
    // resident authority and creates no new egress-history event.
    //
    // The `egress->generation != 0u` guard is the whole correction here, and
    // it is what "the last public surface" always meant: an echo is only
    // defined once there IS a prior publication to echo. publish_egress uses
    // an even/odd seqlock starting at 0, so generation==0 means this resident
    // has never published anything and egress->actions/action_count are still
    // the allocation's zero-fill. Without the guard the first publish_due
    // epoch that is also quiet (the periodic `(device_epochs & 63u) == 0u`
    // heartbeat can be that very epoch, before any contact has landed) echoed
    // that zero-fill over derive_output's correct result and published
    // action_count==0 -- measured live as actions.size()==0 with
    // rewrite_motor_valid==1, which is the failure
    // bcc32_cuda_resident_rewrite_production_contract reports as "canonical
    // action surface is not the three raw rails". de6857fa57 instead deleted
    // the action half of this echo outright; that fixed the same symptom but
    // stranded every action-return consumer, because the payload surface
    // derive_output publishes on a generating epoch then survived only for
    // that single epoch and was gone before any host poll could observe it.
    if (control->generated == 0u && control->history_changed == 0u &&
        egress->generation != 0u) {
      derived.action_count = egress->action_count;
      for (std::uint32_t index = 0u; index < kActionWords; ++index)
        derived.actions[index] = egress->actions[index];
      derived.language_count = egress->language_count;
      for (std::uint32_t index = 0u; index < kLanguageBytes; ++index)
        derived.language[index] = egress->language[index];
    }
    fill_receipt(state, control->consumed_contact != 0u,
                 control->accepted_action_return != 0u,
                 control->device_body_return != 0u,
                 control->device_body_attached != 0u, derived);
    publish_egress(state, derived, control->history_changed != 0u, egress);
  }
  control->publish_due = 0u;
  control->history_changed = 0u;
  control->generated = 0u;
  control->consumed_contact = 0u;
  control->accepted_action_return = 0u;
  control->device_body_return = 0u;
  control->device_body_attached = 0u;
  __threadfence();
  const cudaError_t status =
      cudaGraphLaunch(control->root_graph_exec, cudaStreamGraphTailLaunch);
  if (status != cudaSuccess) {
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
        lifecycle->continuation_fault);
    fault.store(0x52571100u | static_cast<std::uint32_t>(status),
                cuda::memory_order_release);
  }
}

__device__ void commit_action_return_constraint_delta(DeviceState* state) {
  const ActionReturnConstraintDelta& delta =
      state->action_return_stream_constraint_delta;
  state->action_return_constraint_reafferent_attempted += delta.attempted;
  state->action_return_constraint_reafferent_accepted += delta.accepted;
  state->action_return_constraint_reafferent_rejected += delta.rejected;
  state->action_return_constraint_countered_records += delta.countered_records;
  state->action_return_constraint_admitted_records += delta.admitted_records;
  if (delta.attempted != 0u) {
    state->action_return_constraint_resident_revision = delta.resident_revision;
    state->action_return_constraint_component_ready = delta.component_ready;
    state->action_return_constraint_component_ambiguous = delta.component_ambiguous;
    state->action_return_constraint_component_records = delta.component_records;
    state->action_return_constraint_component_sources = delta.component_sources;
    state->action_return_constraint_rederived_event = delta.rederived_event;
  }
}

// Keep this as an explicit scalar sequence rather than hashing a padded C++
// struct: producer and consumer share one zero-padding-free domain.
__device__ ContentAddress device_body_return_commitment(
    std::uint64_t producer_instance, std::uint64_t source_epoch,
    std::uint64_t route_sequence, const ActionReturnTicket& ticket,
    const egress_history::Event& action, std::uint32_t count,
    const BoundaryWord* words) {
  constexpr std::uint64_t kDomain = 0x444256525f434f4dull;  // DBVR_COM
  std::uint64_t lanes[4]{kDomain, kDomain ^ 0x13198a2e03707344ull,
                         kDomain ^ 0xa4093822299f31d0ull,
                         kDomain ^ 0x082efa98ec4e6c89ull};
  std::uint64_t ordinal = 0u;
#define BCC32_ADD_DEVICE_BODY_COMMITMENT(value) \
  do { \
    const std::uint32_t lane = static_cast<std::uint32_t>(ordinal & 3u); \
    lanes[lane] = mix64(lanes[lane], ordinal); \
    lanes[lane] = mix64(lanes[lane], static_cast<std::uint64_t>(value)); \
    ++ordinal; \
  } while (false)
  BCC32_ADD_DEVICE_BODY_COMMITMENT(producer_instance);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(source_epoch);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(route_sequence);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(ticket.issuer_instance);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(ticket.action_sequence);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(ticket.nonce);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(action.sequence);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(action.raw_word);
  BCC32_ADD_DEVICE_BODY_COMMITMENT(count);
  for (std::uint32_t index = 0u; index < count; ++index)
    BCC32_ADD_DEVICE_BODY_COMMITMENT(words[index]);
#undef BCC32_ADD_DEVICE_BODY_COMMITMENT
  ContentAddress result{};
  result.byte_count = ordinal * sizeof(std::uint64_t);
  auto* digest = reinterpret_cast<std::uint8_t*>(&result.digest);
  for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
    const std::uint64_t value = mix64(lanes[lane], ordinal + lane);
    for (std::uint32_t byte = 0u; byte < 8u; ++byte)
      digest[lane * 8u + byte] = static_cast<std::uint8_t>(value >> (byte * 8u));
  }
  return result;
}

__device__ bool same_device_content_address(const ContentAddress& left,
                                            const ContentAddress& right) {
  if (left.byte_count != right.byte_count)
    return false;
  const auto* left_bytes = reinterpret_cast<const std::uint8_t*>(&left.digest);
  const auto* right_bytes = reinterpret_cast<const std::uint8_t*>(&right.digest);
  for (std::size_t index = 0u; index < sizeof(left.digest); ++index) {
    if (left_bytes[index] != right_bytes[index])
      return false;
  }
  return true;
}

__device__ void issue_action_return_ticket(DeviceState* state) {
  if (state->action_return_ticket.nonce != 0u)
    return;
  const std::uint64_t sequence = egress_history::newest_sequence(&state->egress_history);
  egress_history::Event event{};
  if (!egress_history::lookup(&state->egress_history, sequence, &event))
    return;

  std::uint64_t nonce = mix64(sequence, state->tick);
  const auto* predecessor_bytes = reinterpret_cast<const std::uint8_t*>(&state->predecessor.digest);
  for (std::size_t index = 0u; index < sizeof(state->predecessor.digest); ++index)
    nonce = mix64(nonce, predecessor_bytes[index]);
  // Every public action may receive a transport ticket. Revision authority is
  // recorded only when resident inquiry matter proves an exact unresolved
  // intervention lineage; a generic action remains revision-inert.
  (void)rewrite::mark_current_revision_action_issuance(&state->world);
  ActionReturnTicket candidate{};
  candidate.issuer_instance = state->action_return_instance_nonce;
  candidate.action_sequence = sequence;
  candidate.nonce = nonce == 0u ? 1u : nonce;
  const std::uint32_t channel =
      (event.raw_word & rewrite::kRawChannelMask) >> 24u;
  const inquiry_return::Publication inquiry_publication =
      inquiry_return::bind_completed_public_inquiry(
          &state->world, &state->egress_history, candidate);
  if (channel != 1u &&
      !inquiry_return::publication_self_consistent(&state->world,
                                                    inquiry_publication))
    return;
  state->action_return_ticket = candidate;
  // A ticket opens a resident-owned membrane transaction. Keep quiet-turnover
  // and mouth-lease expiry from retiring the active return context while the
  // membrane is unresolved, but do not freeze autonomous generation: the adult
  // continues to calculate and may publish other resident-earned activity.
  // This is scheduling/continuity state only: the ticket supplies no answer,
  // relation, source, or semantic authority.
  state->action_return_autonomy_barrier = 1u;
  ++state->action_return_issued;
}

__device__ bool device_body_enabled(const DeviceBodyControl* body) {
  if (body == nullptr)
    return false;
  cuda::atomic_ref<const std::uint32_t, cuda::thread_scope_system> attachment(body->attachment);
  return attachment.load(cuda::memory_order_acquire) ==
         static_cast<std::uint32_t>(DeviceBodyAttachment::simulated_device);
}

// Derives the stable world-cell owner token for this device body's own
// consequence stream, from the body's producer instance and source epoch
// (never from a per-tick value). The token is only meaningful once the body
// is initialized; a fresh, ambiguity-free owner is minted on first use and
// reused (found via find_world_cell) on every later call.
__device__ std::uint32_t device_body_world_owner(
    const DeviceState* state,
    const rewrite::ResidentRewriteState* world) {
  if (state == nullptr || world == nullptr ||
      state->device_body_initialized == 0u)
    return rewrite::kInvalid;
  const std::uint32_t producer_fold =
      static_cast<std::uint32_t>(state->device_body_producer_instance) ^
      static_cast<std::uint32_t>(state->device_body_producer_instance >> 32u);
  const std::uint32_t epoch_fold =
      static_cast<std::uint32_t>(state->device_body_source_epoch) ^
      static_cast<std::uint32_t>(state->device_body_source_epoch >> 32u);
  std::uint32_t owner =
      rewrite::rewrite_mix(producer_fold, epoch_fold, 0x57434c31u);
  for (std::uint32_t attempt = 0u;
       attempt < rewrite::kRecordCapacity; ++attempt) {
    bool ambiguous = false;
    const std::uint32_t existing =
        rewrite::turn_world_consequence_binding::find_world_cell(
            world, owner, &ambiguous);
    if (ambiguous) return rewrite::kInvalid;
    if (existing != rewrite::kInvalid)
      return owner;
    if (owner != 0u && owner != rewrite::kInvalid &&
        rewrite::turn_world_consequence_binding::owner_free(world, owner))
      return owner;
    owner = rewrite::rewrite_mix(owner, 0x57434c31u, attempt + 1u);
  }
  return rewrite::kInvalid;
}

// Stages the physically-grounded world write for this tick's device-body
// consequence (turn 1), and, when a reafferent constraint was just accepted
// on the same chunk, stages a source-anchored claim (reusing the existing
// subject-scoped claim mechanism, keyed by the accepted source revision)
// bound to that same world cell (turn 1 half of the read side). Both writes
// land in the staging world so they either commit or roll back atomically
// with the rest of the close transaction; neither ever mutates the canonical
// world directly.
__device__ bool stage_device_body_world_consequence(DeviceState* state) {
  if (state == nullptr ||
      state->action_return_stream_from_device_body == 0u)
    return true;
  rewrite::ResidentRewriteState* world =
      &state->action_return_staging_world;
  const std::uint32_t world_owner = device_body_world_owner(state, world);
  if (world_owner == rewrite::kInvalid)
    return false;
  std::uint32_t world_slot = rewrite::kInvalid;
  if (!rewrite::turn_world_consequence_binding::apply_grounded_world_write(
          world, world_owner,
          state->action_return_stream_device_body_consequence_word,
          &world_slot))
    return false;
  state->action_return_world_cell_slot = world_slot;
  state->action_return_world_write_count =
      world->records[world_slot].lane[3];
  state->action_return_world_claim_slot = rewrite::kInvalid;
  const ActionReturnConstraintDelta& delta =
      state->action_return_stream_constraint_delta;
  if (delta.accepted == 0u)
    return true;
  if (delta.source_revision == 0u ||
      delta.source_revision == rewrite::kInvalid)
    return false;
  std::uint32_t claim_owner = rewrite::kInvalid;
  if (!rewrite::turn_world_consequence_binding::ensure_subject_claim(
          world, delta.source_revision,
          rewrite::rewrite_mix(delta.source_revision, world_owner,
                               0x434c4d31u),
          &claim_owner) ||
      !rewrite::turn_world_consequence_binding::bind_claim_to_world_cell(
          world, claim_owner, world_owner))
    return false;
  const std::uint32_t claim_slot =
      rewrite::turn_world_consequence_binding::find_claim(world, claim_owner);
  state->action_return_world_claim_slot = claim_slot;
  return true;
}

__device__ void initialize_device_body(DeviceState* state) {
  if (state->device_body_initialized != 0u)
    return;
  state->device_body_producer_instance =
      mix64(state->action_return_instance_nonce, 0x4445564943455f42ull);
  if (state->device_body_producer_instance == 0u)
    state->device_body_producer_instance = 1u;
  state->device_body_source_epoch =
      mix64(state->device_body_producer_instance, 0x53494d5f424f4459ull);
  if (state->device_body_source_epoch == 0u)
    state->device_body_source_epoch = 1u;
  state->device_body_next_route_sequence = 1u;
  state->device_body_state = mix64(state->device_body_source_epoch, 0x424f44595f535441ull);
  if (state->device_body_state == 0u)
    state->device_body_state = 1u;
  state->device_body_transition_count = 0u;
  state->device_body_initialized = 1u;
}

// The body observes the committed channel-1 raw action, advances persistent
// device-local state, and emits a raw sensor consequence on channel 2 inside
// the device graph. This is a deterministic simulated body transition,
// deliberately not an exterior or source-identity proof.
__device__ __noinline__ bool produce_device_body_return(
    DeviceState* state, DeviceBodyControl* body, ActionReturnIngress* returned) {
  if (!device_body_enabled(body) || state->action_return_ticket.nonce == 0u)
    return false;
  initialize_device_body(state);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(returned->published);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(returned->consumed);
  const std::uint64_t next_ingress = consumed.load(cuda::memory_order_relaxed) + 1u;
  if (published.load(cuda::memory_order_acquire) >= next_ingress)
    return false;
  const std::uint64_t action_sequence = state->action_return_ticket.action_sequence;
  egress_history::Event action{};
  if (!egress_history::lookup(&state->egress_history, action_sequence, &action) ||
      ((action.raw_word & rewrite::kRawChannelMask) >> 24u) != 1u)
    return false;

  const std::uint64_t route_sequence = state->device_body_next_route_sequence;
  if (route_sequence == 0u)
    return false;
  state->device_body_state = mix64(
      state->device_body_state,
      action.raw_word ^ mix64(action.sequence, route_sequence));
  if (state->device_body_state == 0u)
    state->device_body_state = 1u;
  ++state->device_body_transition_count;
  cuda::atomic_ref<const std::uint32_t, cuda::thread_scope_system> permutation(
      body->actuator_permutation);
  const std::uint32_t sensor_payload =
      (action.raw_word & rewrite::kRawPayloadMask) ^
      (permutation.load(cuda::memory_order_acquire) & rewrite::kRawPayloadMask);
  returned->words[0] = (2u << 24u) | sensor_payload;
  returned->words[1] = rewrite::kBoundaryEnd;
  returned->count = 2u;
  returned->final_chunk = 1u;
  returned->ticket = state->action_return_ticket;
  returned->chunk_sequence = 1u;
  returned->producer_instance = state->device_body_producer_instance;
  returned->source_epoch = state->device_body_source_epoch;
  returned->route_sequence = route_sequence;
  returned->producer_commitment = device_body_return_commitment(
      returned->producer_instance, returned->source_epoch, route_sequence,
      returned->ticket, action, returned->count, returned->words);
  returned->device_body_produced = 1u;
  state->device_body_last_route_sequence = route_sequence;
  ++state->device_body_next_route_sequence;
  __threadfence_system();
  published.store(next_ingress, cuda::memory_order_release);
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> corrupt(
      body->test_corrupt_post_publication_word);
  if (corrupt.load(cuda::memory_order_acquire) != 0u) {
    // This is deliberately after the release store. The single producer still
    // owns this unconsumed slot in this epoch; the consumer must reject the
    // commitment mismatch rather than assimilate altered raw matter.
    returned->words[0] ^= 1u;
    __threadfence_system();
  }
  return true;
}

#include "bcc32_resident_rewrite_epoch_phases.inl"


void destroy_graph_handles(void*& graph_pointer, void*& exec_pointer) noexcept {
  if (exec_pointer != nullptr) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_pointer));
    exec_pointer = nullptr;
  }
  if (graph_pointer != nullptr) {
    cudaGraphDestroy(static_cast<cudaGraph_t>(graph_pointer));
    graph_pointer = nullptr;
  }
}

void release_managed(void*& pointer) noexcept {
  if (pointer != nullptr) {
    cudaFree(pointer);
    pointer = nullptr;
  }
}

void require_stream_healthy(cudaStream_t stream, const Lifecycle* lifecycle) {
  const std::uint32_t continuation_fault = load_host(&lifecycle->continuation_fault);
  if (continuation_fault != 0u)
    throw std::runtime_error("resident rewrite continuation fault=" +
                             std::to_string(continuation_fault));
  const cudaError_t status = cudaStreamQuery(stream);
  if (status == cudaErrorNotReady)
    return;
  if (status == cudaSuccess)
    throw std::runtime_error("resident rewrite continuation stopped");
  throw std::runtime_error(std::string("resident rewrite stream failed: ") +
                           cudaGetErrorString(status));
}

bool wait_for_stopped(cudaStream_t stream, const Lifecycle* lifecycle) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (load_host(&lifecycle->stopped) != 0u) {
      cudaStreamSynchronize(stream);
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

#include "bcc32_resident_rewrite_runtime_egress.inl"

}  // namespace

void bootstrap_resident_device_runtime() {
  configure_resident_device_stack_once();
}

PersistentKernel::PersistentKernel()
    : PersistentKernel(BitBusCircuitLevel::off, false) {}

PersistentKernel::PersistentKernel(BitBusCircuitLevel bitbus_circuit_level)
    : PersistentKernel(bitbus_circuit_level, false) {}

PersistentKernel::PersistentKernel(BitBusCircuitLevel bitbus_circuit_level,
                                   bool enable_ordinary_f)
    : sealed_execution_(rewrite_sealed_identity()),
      law_(rewrite_law_identity()),
      image_(rewrite_image_identity()),
      genesis_manifest_(rewrite_genesis_identity()),
      cell_count_(kActionWords),
      bitbus_circuit_level_(bitbus_circuit_level),
      f_owned_clock_(enable_ordinary_f) {
  if (bitbus_circuit_level_ != BitBusCircuitLevel::off &&
      bitbus_circuit_level_ != BitBusCircuitLevel::l0)
    throw std::invalid_argument(
        "resident rewrite runtime implements BitBus circuit L0 only");

  int device_count = 0;
  require_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
  if (device_count == 0)
    throw std::runtime_error("resident rewrite runtime requires a CUDA device");
  const std::uint64_t action_return_instance_nonce =
      next_resident_instance_nonce.fetch_add(1u, std::memory_order_relaxed);
  if (action_return_instance_nonce == 0u)
    throw std::runtime_error("resident action-return instance nonce exhausted");

  DeviceState* state = nullptr;
  IngressRing* ingress = nullptr;
  ActionReturnIngress* action_return = nullptr;
  DeviceBodyControl* device_body = nullptr;
  PhysicalIngress* physical = nullptr;
  EgressState* egress = nullptr;
  Lifecycle* lifecycle = nullptr;
  BitBusCircuitRing* bitbus = nullptr;
  ResidentCensusScratch* census = nullptr;
  ResidentEpochCleanupScratch* epoch_cleanup = nullptr;
  ResidentEpochGraphControl* epoch_control = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraph_t cleanup_graph = nullptr;
  cudaGraph_t post_graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaGraphExec_t cleanup_graph_exec = nullptr;
  cudaGraphExec_t post_graph_exec = nullptr;
  try {
    bootstrap_resident_device_runtime();
    if (f_owned_clock_) {
      auto* adult =
          grown::make_resident_founder_grown_adult(
              kResidentFApertureEdgeChunks);
      grown_adult_ = adult;
      genesis_manifest_ = grown::resident_genesis_manifest_identity(adult);
      ordinary_f_timeline_ = grown::claim_resident_ordinary_f_timeline(
          adult, kResidentFHistoryCapacity, kResidentFHistoryDepth);
    }
    require_cuda(cudaMalloc(&state, sizeof(*state)), "allocate resident rewrite world");
    require_cuda(cudaMemset(state, 0, sizeof(*state)), "clear resident rewrite world");
    require_cuda(cudaMallocManaged(&ingress, sizeof(*ingress)), "allocate resident ingress");
    require_cuda(cudaMallocManaged(&action_return, sizeof(*action_return)),
                 "allocate resident action return ingress");
    require_cuda(cudaMallocManaged(&device_body, sizeof(*device_body)),
                 "allocate resident device body control");
    require_cuda(cudaMallocManaged(&physical, sizeof(*physical)),
                 "allocate resident physical ingress");
    require_cuda(cudaMallocManaged(&egress, sizeof(*egress)), "allocate resident egress");
    require_cuda(cudaMallocManaged(&lifecycle, sizeof(*lifecycle)), "allocate resident lifecycle");
    const std::size_t census_allocation_bytes =
        sizeof(*census) + sizeof(ResidentEpochCleanupScratch) +
        sizeof(ResidentEpochGraphControl);
    require_cuda(cudaMalloc(&census, census_allocation_bytes),
                 "allocate resident parallel census and epoch scratch");
    epoch_cleanup = reinterpret_cast<ResidentEpochCleanupScratch*>(
        reinterpret_cast<std::uint8_t*>(census) + sizeof(*census));
    epoch_control = reinterpret_cast<ResidentEpochGraphControl*>(
        reinterpret_cast<std::uint8_t*>(epoch_cleanup) +
        sizeof(*epoch_cleanup));
    require_cuda(cudaMemset(census, 0, census_allocation_bytes),
                 "clear resident parallel census and epoch scratch");
    new (ingress) IngressRing{};
    new (action_return) ActionReturnIngress{};
    new (device_body) DeviceBodyControl{};
    new (physical) PhysicalIngress{};
    new (egress) EgressState{};
    new (lifecycle) Lifecycle{};
    if (f_owned_clock_) {
      auto* timeline = static_cast<ordinary_f::DeviceOrdinaryFTimeline*>(
          ordinary_f_timeline_);
      const auto handle = timeline->device_leaf_launch_handle();
      require_cuda(cudaMemcpy(
                       reinterpret_cast<std::uint8_t*>(state) +
                           offsetof(DeviceState, ordinary_f),
                       &handle, sizeof(handle), cudaMemcpyHostToDevice),
                   "attach resident ordinary-F graph");
      require_cuda(cudaMemcpy(
                       reinterpret_cast<std::uint8_t*>(state) +
                           offsetof(DeviceState, f_genesis_manifest),
                       &genesis_manifest_, sizeof(genesis_manifest_),
                       cudaMemcpyHostToDevice),
                   "attach resident ordinary-F Genesis manifest");
      const std::uint32_t owned = 1u;
      require_cuda(cudaMemcpy(
                       reinterpret_cast<std::uint8_t*>(state) +
                           offsetof(DeviceState, f_owned_clock),
                       &owned, sizeof(owned), cudaMemcpyHostToDevice),
                   "attach resident ordinary-F clock authority");
    }
    if (bitbus_circuit_level_ == BitBusCircuitLevel::l0) {
      // Separate allocation, separate object: resident_rewrite_epoch_kernel
      // below is never given this pointer, so it has no address through
      // which to read or write it.
      require_cuda(cudaMallocManaged(&bitbus, sizeof(*bitbus)),
                   "allocate BitBus circuit L0 ring");
      new (bitbus) BitBusCircuitRing{};
    }
    test_device_body_control = device_body;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "create resident rewrite stream");

    require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "begin resident rewrite graph capture");
    auto sealed = sealed_execution_;
    auto law = law_;
    auto image = image_;
    auto genesis = genesis_manifest_;
    std::uint64_t instance_nonce = action_return_instance_nonce;
    void* epoch_arguments[] = {
        &state,          &ingress,       &action_return, &device_body,
        &physical,       &egress,       &lifecycle,     &sealed,
        &law,            &image,         &genesis,       &instance_nonce,
        &epoch_cleanup,  &epoch_control};
    require_cuda(cudaLaunchKernel(
                     reinterpret_cast<const void*>(resident_rewrite_epoch_kernel),
                     dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u}, epoch_arguments, 0u,
                     stream),
                 "capture resident rewrite epoch");
    require_cuda(cudaGetLastError(), "capture resident rewrite epoch");
    require_cuda(cudaStreamEndCapture(stream, &graph), "end resident rewrite graph capture");
    require_cuda(
        cudaGraphInstantiateWithFlags(&graph_exec, graph, cudaGraphInstantiateFlagDeviceLaunch),
        "instantiate resident rewrite device graph");
    require_cuda(cudaGraphUpload(graph_exec, stream), "upload resident rewrite device graph");

    require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "begin resident action-return cleanup graph capture");
    resident_epoch_cleanup_classify_kernel<<<
        kEpochCleanupBlocks, kEpochCleanupThreads, 0, stream>>>(state,
                                                                  epoch_cleanup);
    require_cuda(cudaGetLastError(),
                 "capture resident action-return cleanup classification");
    void* cleanup_commit_arguments[] = {&state, &epoch_cleanup};
    require_cuda(cudaLaunchKernel(
                     reinterpret_cast<const void*>(
                         resident_epoch_cleanup_commit_kernel),
                     dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                     cleanup_commit_arguments, 0u, stream),
                 "capture resident action-return cleanup commit");
    require_cuda(cudaGetLastError(),
                 "capture resident action-return cleanup commit");
    void* cleanup_return_arguments[] = {&state, &lifecycle, &epoch_cleanup,
                                        &epoch_control};
    require_cuda(cudaLaunchKernel(
                     reinterpret_cast<const void*>(
                         resident_epoch_cleanup_return_kernel),
                     dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                     cleanup_return_arguments, 0u, stream),
                 "capture resident action-return cleanup return");
    require_cuda(cudaGetLastError(),
                 "capture resident action-return cleanup return");
    require_cuda(cudaStreamEndCapture(stream, &cleanup_graph),
                 "end resident action-return cleanup graph capture");
    require_cuda(cudaGraphInstantiateWithFlags(
                     &cleanup_graph_exec, cleanup_graph,
                     cudaGraphInstantiateFlagDeviceLaunch),
                 "instantiate resident action-return cleanup graph");
    require_cuda(cudaGraphUpload(cleanup_graph_exec, stream),
                 "upload resident action-return cleanup graph");

    require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                 "begin resident post-epoch graph capture");
    resident_census_scan_kernel<<<
        (kResidentCensusMaxItems + kResidentCensusThreads - 1u) /
            kResidentCensusThreads,
        kResidentCensusThreads, 0, stream>>>(&state->world, census);
    require_cuda(cudaGetLastError(), "capture resident parallel census scan");
    resident_census_reduce_kernel<<<kResidentCensusBlocks,
                                    kResidentCensusThreads, 0, stream>>>(
        &state->world, census);
    require_cuda(cudaGetLastError(), "capture resident parallel census reduce");
    auto* census_world = &state->world;
    auto* census_resident_receipt = &state->receipt;
    decltype(census_resident_receipt) census_egress_receipt = nullptr;
    std::uint64_t* census_egress_generation = nullptr;
    void* census_commit_arguments[] = {
        &census_world,          &census, &census_resident_receipt,
        &census_egress_receipt, &census_egress_generation};
    require_cuda(cudaLaunchKernel(
                     reinterpret_cast<const void*>(resident_census_commit_kernel),
                     dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                     census_commit_arguments, 0u, stream),
                 "capture resident parallel census commit");
    require_cuda(cudaGetLastError(), "capture resident parallel census commit");
    // The census is a production observer-maintenance node, but it does not
    // enter the ordered epoch or choose resident behavior.  The optional L0
    // BitBus observer remains separate and is captured only at its requested
    // level; no observer pointer is threaded through the epoch kernel.
    if (bitbus_circuit_level_ == BitBusCircuitLevel::l0) {
      void* observer_arguments[] = {&egress, &bitbus};
      require_cuda(cudaLaunchKernel(
                       reinterpret_cast<const void*>(
                           bitbus_circuit_l0_observer_kernel),
                       dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                       observer_arguments, 0u, stream),
                   "capture BitBus circuit L0 observer");
      require_cuda(cudaGetLastError(), "capture BitBus circuit L0 observer");
    }
    void* post_return_arguments[] = {&state, &egress, &lifecycle,
                                     &epoch_control};
    require_cuda(cudaLaunchKernel(
                     reinterpret_cast<const void*>(
                         resident_epoch_post_return_kernel),
                     dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                     post_return_arguments, 0u, stream),
                 "capture resident post-epoch return");
    require_cuda(cudaGetLastError(), "capture resident post-epoch return");
    require_cuda(cudaStreamEndCapture(stream, &post_graph),
                 "end resident post-epoch graph capture");
    require_cuda(cudaGraphInstantiateWithFlags(
                     &post_graph_exec, post_graph,
                     cudaGraphInstantiateFlagDeviceLaunch),
                 "instantiate resident post-epoch graph");
    require_cuda(cudaGraphUpload(post_graph_exec, stream),
                 "upload resident post-epoch graph");
    const ResidentEpochGraphControl host_epoch_control{
        graph_exec, cleanup_graph_exec, post_graph_exec};
    require_cuda(cudaMemcpy(epoch_control, &host_epoch_control,
                            sizeof(host_epoch_control), cudaMemcpyHostToDevice),
                 "publish resident epoch graph handles");
    device_state_ = state;
    ingress_ = ingress;
    action_return_ = action_return;
    device_body_ = device_body;
    physical_ = physical;
    egress_ = egress;
    lifecycle_ = lifecycle;
    bitbus_circuit_ = bitbus;
    resident_census_ = census;
    private_stream_ = stream;
    graph_ = graph;
    graph_exec_ = graph_exec;
    cleanup_graph_ = cleanup_graph;
    cleanup_graph_exec_ = cleanup_graph_exec;
    post_graph_ = post_graph;
    post_graph_exec_ = post_graph_exec;
    require_cuda(cudaGraphLaunch(graph_exec, stream), "host-launch resident rewrite graph");
  } catch (...) {
    if (post_graph_exec != nullptr)
      cudaGraphExecDestroy(post_graph_exec);
    if (post_graph != nullptr)
      cudaGraphDestroy(post_graph);
    if (cleanup_graph_exec != nullptr)
      cudaGraphExecDestroy(cleanup_graph_exec);
    if (cleanup_graph != nullptr)
      cudaGraphDestroy(cleanup_graph);
    if (graph_exec != nullptr)
      cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr)
      cudaGraphDestroy(graph);
    if (stream != nullptr)
      cudaStreamDestroy(stream);
    if (test_device_body_control == device_body)
      test_device_body_control = nullptr;
    if (state != nullptr)
      cudaFree(state);
    if (ingress != nullptr)
      cudaFree(ingress);
    if (action_return != nullptr)
      cudaFree(action_return);
    if (device_body != nullptr)
      cudaFree(device_body);
    if (physical != nullptr)
      cudaFree(physical);
    if (egress != nullptr)
      cudaFree(egress);
    if (lifecycle != nullptr)
      cudaFree(lifecycle);
    if (bitbus != nullptr)
      cudaFree(bitbus);
    if (census != nullptr)
      cudaFree(census);
    delete static_cast<ordinary_f::DeviceOrdinaryFTimeline*>(
        ordinary_f_timeline_);
    ordinary_f_timeline_ = nullptr;
    grown::destroy_resident_grown_adult(
        static_cast<grown::GrownAdult*>(grown_adult_));
    grown_adult_ = nullptr;
    throw;
  }
}

PersistentKernel::~PersistentKernel() {
  shutdown();
  if (!shutdown_complete_)
    return;
  destroy_graph_handles(post_graph_, post_graph_exec_);
  destroy_graph_handles(cleanup_graph_, cleanup_graph_exec_);
  destroy_graph_handles(graph_, graph_exec_);
  if (device_state_ != nullptr) {
    cudaFree(device_state_);
    device_state_ = nullptr;
  }
  release_managed(ingress_);
  release_managed(action_return_);
  if (test_device_body_control == device_body_)
    test_device_body_control = nullptr;
  release_managed(device_body_);
  release_managed(physical_);
  release_managed(egress_);
  release_managed(lifecycle_);
  release_managed(bitbus_circuit_);
  release_managed(resident_census_);
  delete static_cast<ordinary_f::DeviceOrdinaryFTimeline*>(
      ordinary_f_timeline_);
  ordinary_f_timeline_ = nullptr;
  grown::destroy_resident_grown_adult(
      static_cast<grown::GrownAdult*>(grown_adult_));
  grown_adult_ = nullptr;
}

void PersistentKernel::shutdown() noexcept {
  if (shutdown_requested_)
    return;
  shutdown_requested_ = true;
  if (lifecycle_ != nullptr)
    store_host(&static_cast<Lifecycle*>(lifecycle_)->shutdown, 1u);
  if (private_stream_ != nullptr && lifecycle_ != nullptr) {
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
  if (shutdown_requested_)
    throw std::runtime_error("resident rewrite runtime is stopped");
  require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                         static_cast<Lifecycle*>(lifecycle_));
  auto* ingress = static_cast<IngressRing*>(ingress_);
  // Preserve cross-membrane publication order without giving the host
  // semantic authority. If an action-return packet was published first, let
  // the device consume/reject/commit that exact packet before publishing a
  // later raw contact. A raw contact already published first still wins
  // naturally because the epoch consumes raw ingress before action-return
  // ingress.
  auto* returned = static_cast<ActionReturnIngress*>(action_return_);
  if (returned != nullptr) {
    const auto ordering_deadline = std::chrono::steady_clock::now() +
                                   std::chrono::seconds(5);
    while (load_host(&returned->published) >
           load_host(&returned->consumed)) {
      if (std::chrono::steady_clock::now() >= ordering_deadline)
        throw std::runtime_error(
            "resident cross-membrane ingress ordering remained blocked");
      require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                             static_cast<Lifecycle*>(lifecycle_));
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  // The rewrite resident consumes a bounded amount per epoch, but its public
  // raw boundary must not discard prose beyond one packet. Packet boundaries
  // are transport-only and introduce no semantic segmentation.
  std::size_t offset = 0u;
  // Monotonic watermark: stale-low can only cause conservative waiting.
  std::uint64_t observed_consumed = load_host(&ingress->consumed);
  do {
    const std::uint64_t sequence = next_contact_sequence_;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(5);
    // Exactly kIngressSlots outstanding sequences are collision-free.
    while (sequence - observed_consumed > kIngressSlots) {
      observed_consumed = load_host(&ingress->consumed);
      if (sequence - observed_consumed <= kIngressSlots)
        break;
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("resident rewrite ingress remained full");
      require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                             static_cast<Lifecycle*>(lifecycle_));
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    IngressSlot& slot = ingress->slots[sequence % kIngressSlots];
    slot.offset = 0u;
    slot.contact_start = offset == 0u ? 1u : 0u;
    slot.count = static_cast<std::uint32_t>(std::min(
        contact.size() - offset, static_cast<std::size_t>(kIngressPacketWords)));
    if (slot.count != 0u)
      std::memcpy(slot.words, contact.data() + offset,
                  slot.count * sizeof(BoundaryWord));
    std::atomic_thread_fence(std::memory_order_release);
    store_host(&slot.sequence, sequence);
    store_host(&ingress->published, sequence);
    ++next_contact_sequence_;
    offset += slot.count;
  } while (offset < contact.size());
}

void PersistentKernel::present_action_return(ActionReturnTicket ticket,
                                             std::span<const BoundaryWord> contact) {
  if (contact.size() > kMaxContactWords)
    throw std::invalid_argument("one-shot action return exceeds transport capacity");
  present_action_return_chunk(ticket, 1u, contact, true);
}

void PersistentKernel::present_action_return_chunk(ActionReturnTicket ticket,
                                                   std::uint64_t chunk_sequence,
                                                   std::span<const BoundaryWord> contact,
                                                   bool final_chunk) {
  if (contact.size() > kMaxContactWords)
    throw std::invalid_argument("action-return chunk exceeds transport capacity");
  if (!final_chunk && contact.empty())
    throw std::invalid_argument("nonterminal action-return chunk is empty");
  if (final_chunk && (contact.empty() || contact.back() != rewrite::kBoundaryEnd))
    throw std::invalid_argument("terminal action-return chunk lacks physical END");
  if (shutdown_requested_)
    throw std::runtime_error("resident rewrite runtime is stopped");
  require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                         static_cast<Lifecycle*>(lifecycle_));
  const auto* device_body = static_cast<const DeviceBodyControl*>(device_body_);
  if (device_body != nullptr &&
      load_host(&device_body->attachment) ==
          static_cast<std::uint32_t>(DeviceBodyAttachment::simulated_device))
    throw std::runtime_error("device body attachment owns action returns");
  auto* returned = static_cast<ActionReturnIngress*>(action_return_);
  const std::uint64_t sequence = next_action_return_sequence_;
  if (sequence == ~std::uint64_t{0})
    throw std::overflow_error("action-return ingress sequence is exhausted");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (sequence != load_host(&returned->consumed) + 1u) {
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("resident action return ingress remained full");
    require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                           static_cast<Lifecycle*>(lifecycle_));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  returned->count = static_cast<std::uint32_t>(contact.size());
  returned->final_chunk = final_chunk ? 1u : 0u;
  returned->ticket = ticket;
  returned->chunk_sequence = chunk_sequence;
  if (returned->count != 0u)
    std::memcpy(returned->words, contact.data(), returned->count * sizeof(BoundaryWord));
  std::atomic_thread_fence(std::memory_order_release);
  store_host(&returned->published, sequence);
  ++next_action_return_sequence_;
}

void PersistentKernel::attach_device_body(DeviceBodyAttachment attachment) {
  if (shutdown_requested_)
    throw std::runtime_error("resident rewrite runtime is stopped");
  if (attachment != DeviceBodyAttachment::simulated_device)
    throw std::invalid_argument("only the simulated device body is available");
  require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                         static_cast<Lifecycle*>(lifecycle_));
  auto* device_body = static_cast<DeviceBodyControl*>(device_body_);
  if (device_body == nullptr)
    throw std::runtime_error("resident device body control is unavailable");
  store_host(&device_body->attachment, static_cast<std::uint32_t>(attachment));
}

void PersistentKernel::configure_device_body_actuator_permutation(
    std::uint32_t permutation) {
  if (shutdown_requested_)
    throw std::runtime_error("resident rewrite runtime is stopped");
  require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                         static_cast<Lifecycle*>(lifecycle_));
  auto* device_body = static_cast<DeviceBodyControl*>(device_body_);
  if (device_body == nullptr)
    throw std::runtime_error("resident device body control is unavailable");
  store_host(&device_body->actuator_permutation,
             permutation & rewrite::kRawPayloadMask);
}

void PersistentKernel::present_physical(const RawPhysicalIntervention& event) {
  if (shutdown_requested_)
    throw std::runtime_error("resident rewrite runtime is stopped");
  require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                         static_cast<Lifecycle*>(lifecycle_));
  auto* physical = static_cast<PhysicalIngress*>(physical_);
  const std::uint64_t sequence = next_intervention_sequence_;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (load_host(&physical->consumed) + 1u < sequence) {
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("resident physical ingress remained full");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  physical->event = event;
  std::atomic_thread_fence(std::memory_order_release);
  store_host(&physical->published, sequence);
  ++next_intervention_sequence_;
}

PassiveSnapshot PersistentKernel::read_snapshot() const {
  PassiveSnapshot result;
  if (egress_ == nullptr)
    return result;
  if (!shutdown_requested_)
    require_stream_healthy(static_cast<cudaStream_t>(private_stream_),
                           static_cast<Lifecycle*>(lifecycle_));
  copy_snapshot(static_cast<const EgressState*>(egress_), static_cast<const Lifecycle*>(lifecycle_),
                &egress_history_cache_, &egress_history_cache_next_sequence_,
                &egress_history_cache_oldest_sequence_, &egress_history_cache_overwrite_count_,
                &egress_history_cache_fault_, &result);
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

BitBusCircuitSnapshot PersistentKernel::read_bitbus_circuit() const {
  BitBusCircuitSnapshot result;
  if (bitbus_circuit_ == nullptr)
    return result;

  const auto* ring = static_cast<const BitBusCircuitRing*>(bitbus_circuit_);
  const std::uint64_t published = load_host(&ring->published);
  const std::uint64_t oldest =
      published > kBitBusL0Capacity ? published - kBitBusL0Capacity : 0u;

  result.events.reserve(static_cast<std::size_t>(published - oldest));
  for (std::uint64_t sequence = oldest; sequence < published; ++sequence)
    result.events.push_back(ring->events[sequence % kBitBusL0Capacity]);

  result.next_sequence = published;
  result.overwritten = load_host(&ring->overwritten);
  return result;
}

}  // namespace substrate::bcc32::persistent_kernel

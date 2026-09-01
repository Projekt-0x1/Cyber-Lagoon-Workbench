#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "bcc32_cuda_executor.cuh"
#include "bcc32_device_topology.cuh"
#include "bcc32_raw_byte_tape.cuh"

namespace substrate::bcc32::device_ordinary_f_timeline {

#ifndef BCC32_ORDINARY_F_TIMELINE_CAPACITY
inline constexpr std::uint32_t kCapacity = 1u << 21u;
#else
inline constexpr std::uint32_t kCapacity = BCC32_ORDINARY_F_TIMELINE_CAPACITY;
#endif
inline constexpr std::uint32_t kBoundaryCount = 6u;

// ⛔ github #1209. This is a HARD CEILING and is deliberately NOT overridable,
// unlike kCapacity one line above it. Until 2026-08-18 the build set
// `BCC32_ORDINARY_F_HISTORY_CAPACITY=8` on bcc32_cuda_law_aperture_rwr1 and
// bcc32_resident_rewrite_runtime; no source file has ever read that name, so
// the capacity has always been 65 and the flag was inert. It was deleted
// rather than wired, because wiring it would have silently *shrunk* two live
// targets from 65 to 8 -- a behaviour change disguised as a cleanup, in the
// exact quantity that overflows (`capacity_overflow`) when the aperture runs
// at edge-5 geometry.
//
// The real knob is per-instance and already exists: `history_capacity_`, set
// through the `history_capacity` parameter on begin/reset, which this constant
// bounds (see the `history_capacity_ > kHistoryCapacity` rejection in
// bcc32_device_ordinary_f_timeline.cu). Tune there, not with a macro.
inline constexpr std::uint32_t kHistoryCapacity = 65u;
inline constexpr std::uint32_t kGraphTickCapacity = kHistoryCapacity;
inline constexpr std::uint64_t kUnboundSlot =
    std::numeric_limits<std::uint64_t>::max();

static_assert(kCapacity != 0u && (kCapacity & (kCapacity - 1u)) == 0u,
              "ordinary-F timeline capacity must be a power of two");
enum class Fault : std::uint32_t {
  none = 0u,
  invalid_input = 1u,
  closure = 2u,
  capacity_overflow = 3u,
};

struct WorldDigest {
  std::uint64_t lane0 = 0u;
  std::uint64_t lane1 = 0u;
  std::uint64_t lane2 = 0u;
  std::uint64_t lane3 = 0u;

  friend bool operator==(const WorldDigest&, const WorldDigest&) = default;
};

// Device-authored publication of the complete logical ordinary-F world. The
// digest excludes observer time so exact F then F^-1 returns to the same value.
struct OrdinaryFPublication {
  std::uint64_t completed_ticks = 0u;
  std::uint64_t generation = 0u;
  std::uint32_t fault = 0u;
  std::uint32_t active_count = 0u;
  // Mechanical CUDA continuation receipt only. These fields are observer
  // diagnostics; they are not organism state, a score, or a host-selected
  // route. The phases distinguish the one root/leaf hand-off without adding
  // a second scheduler or semantic authority.
  //   0 root has not armed a child
  //   1 root captured its exact graph and armed the child
  //   2 child entered begin_forward
  //   3 child reached end_forward without a commit
  //   4 child committed completed_ticks
  //   5 child queued the exact root graph as its tail
  std::uint32_t continuation_phase = 0u;
  std::uint32_t return_launch_status = 0u;
  WorldDigest world{};
  // Physical outward matter, copied only after the ordinary-F graph has
  // returned. It is deliberately not a digest or a host-selected action.
  RawByteRails motor{};

  friend bool operator==(const OrdinaryFPublication&,
                         const OrdinaryFPublication&) = default;
};

// The only production boundary binding. Port indices and resident slots are
// supplied by GrownAdult's fixed reciprocal topology; no caller chooses a
// route, coordinate, opcode, or action target.
struct DeviceBoundaryBinding {
  SiteWord* boundary_words = nullptr;
  std::uint32_t sensory_zero_boundary = 0u;
  std::uint32_t sensory_one_boundary = 0u;
  std::uint64_t sensory_zero_slot = kUnboundSlot;
  std::uint64_t sensory_one_slot = kUnboundSlot;
  std::uint64_t motor_zero_slot = kUnboundSlot;
  std::uint64_t motor_one_slot = kUnboundSlot;
};

struct HostSnapshot {
  std::uint32_t capacity = 0u;
  std::uint64_t completed_ticks = 0u;
  std::uint32_t inverse_head = 0u;
  std::uint32_t inverse_depth = 0u;
  Fault fault = Fault::none;
  std::uint64_t generation = 0u;
  OrdinaryFPublication publication{};
  std::vector<SiteWord> world_words;
  std::vector<std::uint64_t> active_slots;
  std::vector<std::uint32_t> boundary_counts;
  std::vector<std::vector<std::uint64_t>> boundary_slots;
};

// The state object is a device control/timeline object only.  It contains no
// words, semantic labels, resident factor layouts, or answer buffers.
struct DeviceState {
  const SiteWord* words = nullptr;
  std::uint64_t* slots = nullptr;
  std::uint32_t* count = nullptr;
  std::uint32_t* dispatch_count = nullptr;
  std::uint64_t* work = nullptr;
  std::uint64_t* next = nullptr;
  std::uint64_t* hash = nullptr;
  std::uint32_t* work_count = nullptr;
  // The active aperture is explicitly limited to UINT32_MAX sites. Compact
  // local indices keep 65 x 6 exact inverse boundaries within the same
  // physical budget as the prior smaller 64-bit journal.
  std::uint32_t* history_slots = nullptr;
  std::uint32_t* history_counts = nullptr;
  std::uint32_t* inverse_head = nullptr;
  std::uint32_t* inverse_depth = nullptr;
  std::uint64_t* completed_ticks = nullptr;
  std::uint64_t* requested_target = nullptr;
  std::uint32_t* current_frame = nullptr;
  std::uint32_t* executing = nullptr;
  std::uint32_t* fault = nullptr;
  std::uint64_t* generation = nullptr;
  OrdinaryFPublication* publication = nullptr;
  cudaGraphExec_t return_graph = nullptr;
  DeviceBoundaryBinding boundary{};
  std::uint32_t capacity = 0u;
  std::uint32_t closure_workspace_capacity = 0u;
  std::uint32_t history_capacity = 0u;
  std::uint32_t boundary_count = 0u;
  device_topology::View topology{};
};

// Opaque constitutional execution seam. It exposes only the graph and control
// scalars required for another device-resident timeline to yield one ordinary
// F tick. No support coordinates, words, factors, or phenotype state cross it.
struct DeviceLaunchHandle {
  cudaGraphExec_t forward_graph = nullptr;
  // Device address of the leaf timeline's return graph slot. The resident
  // root writes cudaGetCurrentGraphExec() here immediately before launching
  // the leaf, so the leaf returns to the exact live root rather than to a
  // host-guessed or sibling graph.
  cudaGraphExec_t* return_graph_slot = nullptr;
  std::uint64_t* requested_target = nullptr;
  const std::uint64_t* completed_ticks = nullptr;
  const std::uint32_t* fault = nullptr;
  OrdinaryFPublication* publication = nullptr;
  const SiteWord* words = nullptr;
  SiteWord* mutable_words = nullptr;
  const std::uint64_t* active_slots = nullptr;
  const std::uint32_t* active_count = nullptr;
  std::uint32_t capacity = 0u;
  std::uint32_t chunk_count = 0u;
  DeviceBoundaryBinding boundary{};
};

__device__ __forceinline__ cudaError_t admit_raw_sensory_byte(
    const DeviceLaunchHandle& handle, std::uint8_t value) {
  const DeviceBoundaryBinding& boundary = handle.boundary;
  if (handle.mutable_words == nullptr || boundary.boundary_words == nullptr ||
      boundary.sensory_zero_slot == kUnboundSlot ||
      boundary.sensory_one_slot == kUnboundSlot ||
      boundary.sensory_zero_boundary == boundary.sensory_one_boundary ||
      boundary.sensory_zero_boundary >= kBoundaryCount ||
      boundary.sensory_one_boundary >= kBoundaryCount)
    return cudaErrorInvalidValue;
  const std::uint64_t site_count =
      static_cast<std::uint64_t>(handle.chunk_count) * kChunkSites;
  if (boundary.sensory_zero_slot >= site_count ||
      boundary.sensory_one_slot >= site_count)
    return cudaErrorInvalidValue;
  RawByteRails world{handle.mutable_words[boundary.sensory_zero_slot],
                      handle.mutable_words[boundary.sensory_one_slot]};
  RawByteRails tape{
      boundary.boundary_words[boundary.sensory_zero_boundary],
      boundary.boundary_words[boundary.sensory_one_boundary]};
  tape = with_raw_byte_faces(tape, value);
  reciprocal_raw_byte_exchange(world, tape);
  handle.mutable_words[boundary.sensory_zero_slot] = world.zero;
  handle.mutable_words[boundary.sensory_one_slot] = world.one;
  boundary.boundary_words[boundary.sensory_zero_boundary] = tape.zero;
  boundary.boundary_words[boundary.sensory_one_boundary] = tape.one;
  __threadfence_system();
  return cudaSuccess;
}

[[nodiscard]] __host__ __device__ inline RawByteDecode
decode_committed_outward_motor(const OrdinaryFPublication& publication,
                               std::uint64_t predecessor_tick) {
  if (publication.fault != static_cast<std::uint32_t>(Fault::none) ||
      publication.completed_ticks <= predecessor_tick)
    return {};
  return decode_raw_byte_carriers(publication.motor);
}

__device__ __forceinline__ std::uint64_t publication_mix(
    std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

__device__ __forceinline__ std::uint64_t publication_fold(
    std::uint64_t lane, std::uint64_t value) {
  return publication_mix(
      lane ^ publication_mix(value + 0x9e3779b97f4a7c15ull));
}

// Called only by the graph that regains control after an ordinary-F graph has
// completed. Keeping this observer outside the F graph prevents publication
// work from changing the scheduler shape of the law being committed.
__device__ __forceinline__ cudaError_t publish_current_world(
    const DeviceLaunchHandle& handle) {
  if (handle.publication == nullptr || handle.completed_ticks == nullptr ||
      handle.fault == nullptr || handle.words == nullptr ||
      handle.active_slots == nullptr || handle.active_count == nullptr ||
      handle.capacity == 0u || handle.chunk_count == 0u)
    return cudaErrorInvalidValue;
  const std::uint32_t count = *handle.active_count;
  if (count > handle.capacity) return cudaErrorInvalidValue;
  OrdinaryFPublication publication{};
  publication.completed_ticks = *handle.completed_ticks;
  publication.generation = publication.completed_ticks;
  publication.fault = *handle.fault;
  publication.active_count = count;
  std::uint64_t lane0 = 0x243f6a8885a308d3ull;
  std::uint64_t lane1 = 0x13198a2e03707344ull;
  std::uint64_t lane2 = 0xa4093822299f31d0ull;
  std::uint64_t lane3 = 0x082efa98ec4e6c89ull;
  lane0 = publication_fold(lane0, handle.chunk_count);
  lane1 = publication_fold(lane1, count);
  lane2 = publication_fold(
      lane2, static_cast<std::uint64_t>(handle.chunk_count) * kChunkSites);
  lane3 = publication_fold(lane3, handle.capacity);
  const std::uint64_t site_count =
      static_cast<std::uint64_t>(handle.chunk_count) * kChunkSites;
  const bool motor_surface_bound =
      handle.boundary.motor_zero_slot != kUnboundSlot &&
      handle.boundary.motor_one_slot != kUnboundSlot;
  if (motor_surface_bound &&
      (handle.boundary.motor_zero_slot >= site_count ||
       handle.boundary.motor_one_slot >= site_count))
    return cudaErrorInvalidValue;
  for (std::uint32_t index = 0u; index < count; ++index) {
    const std::uint64_t slot = handle.active_slots[index];
    if (slot >= site_count) return cudaErrorInvalidValue;
    const std::uint64_t word = handle.words[slot];
    lane0 = publication_fold(lane0, slot);
    lane1 = publication_fold(lane1, word);
    lane2 = publication_fold(lane2, slot ^ (word << 32u));
    lane3 = publication_fold(
        lane3, static_cast<std::uint64_t>(index) ^ (slot << 1u) ^ word);
  }
  publication.world = {lane0, lane1, lane2, lane3};
  if (motor_surface_bound) {
    publication.motor = {
        handle.words[handle.boundary.motor_zero_slot],
        handle.words[handle.boundary.motor_one_slot]};
  }
  *handle.publication = publication;
  __threadfence();
  return cudaSuccess;
}

__device__ __forceinline__ std::uint64_t completed_ticks_acquire(
    const DeviceLaunchHandle& handle) {
  if (handle.completed_ticks == nullptr)
    return ~std::uint64_t{0};
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device> completed(
      *const_cast<std::uint64_t*>(handle.completed_ticks));
  return completed.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ cudaError_t arm_forward_tick(
    const DeviceLaunchHandle& handle) {
  if (handle.forward_graph == nullptr || handle.requested_target == nullptr ||
      handle.completed_ticks == nullptr || handle.fault == nullptr ||
      handle.publication == nullptr)
    return cudaErrorInvalidValue;
  const std::uint64_t completed = completed_ticks_acquire(handle);
  if (*handle.fault != static_cast<std::uint32_t>(Fault::none) ||
      completed == ~std::uint64_t{0})
    return cudaErrorInvalidValue;
  *handle.requested_target = completed + 1u;
  return cudaSuccess;
}

__device__ __forceinline__ cudaError_t launch_armed_forward_tick(
    const DeviceLaunchHandle& handle) {
  if (handle.forward_graph == nullptr || handle.requested_target == nullptr ||
      handle.completed_ticks == nullptr || handle.fault == nullptr ||
      handle.publication == nullptr ||
      *handle.requested_target != completed_ticks_acquire(handle) + 1u)
    return cudaErrorInvalidValue;
  return cudaGraphLaunch(handle.forward_graph, cudaStreamGraphTailLaunch);
}

__device__ __forceinline__ cudaError_t request_forward_tick(
    const DeviceLaunchHandle& handle) {
  const cudaError_t armed = arm_forward_tick(handle);
  return armed == cudaSuccess ? launch_armed_forward_tick(handle) : armed;
}

class DeviceOrdinaryFTimeline {
 public:
  DeviceOrdinaryFTimeline(CudaBcc32Executor& executor, DeviceChunkMap chunks,
                          std::uint32_t capacity = kCapacity,
                          std::uint32_t history_capacity = kHistoryCapacity);
  DeviceOrdinaryFTimeline(CudaBcc32Executor& executor, DeviceChunkMap chunks,
                          DeviceBoundaryBinding boundary,
                          std::uint32_t capacity = kCapacity,
                          std::uint32_t history_capacity = kHistoryCapacity);
  DeviceOrdinaryFTimeline(const DeviceOrdinaryFTimeline&) = delete;
  DeviceOrdinaryFTimeline& operator=(const DeviceOrdinaryFTimeline&) = delete;
  DeviceOrdinaryFTimeline(DeviceOrdinaryFTimeline&&) = delete;
  DeviceOrdinaryFTimeline& operator=(DeviceOrdinaryFTimeline&&) = delete;
  ~DeviceOrdinaryFTimeline();

  // Bootstrap is the only point at which the graph is assembled.  Runtime
  // forward/reverse calls submit one graph and exchange only control/receipt
  // scalars; support authority remains device-resident.
  void bootstrap();
  void forward(std::uint64_t ticks);
  void inverse(std::uint64_t ticks);

  // Integration-only graph exchange. The enclosing persistent timeline may
  // request ordinary F from device code and receives control back through the
  // attached graph. Bootstrap remains the only host graph construction phase.
  [[nodiscard]] DeviceLaunchHandle device_launch_handle();
  // Production root-owned continuation handle. Its one-tick F executable is
  // a bounded leaf that returns only to the exact root graph captured by the
  // currently executing resident epoch.
  [[nodiscard]] DeviceLaunchHandle device_leaf_launch_handle();
  void attach_return_graph(cudaGraphExec_t parent_graph);

  // Reacquire complete support after an executor-owned lawful world boundary.
  // No coordinate list crosses the API, and pre-boundary inverse frames are
  // discarded because they cannot invert the external world transaction.
  void reacquire_world_support();

  [[nodiscard]] HostSnapshot snapshot() const;
  void restore(const HostSnapshot& snapshot);
  [[nodiscard]] std::uint64_t completed_ticks() const { return host_completed_ticks_; }
  [[nodiscard]] std::size_t reversible_supersteps() const { return host_inverse_depth_; }
  [[nodiscard]] const std::string& last_status() const { return last_status_; }

 private:
  void allocate_state();
  void rebuild_support_from_world(bool reset_timeline);
  [[nodiscard]] cudaGraphExec_t capture_forward_graph(
      std::uint32_t tick_capacity, bool append_return);
  void capture_inverse_graph();
  void destroy_graphs() noexcept;
  void refresh_host_receipt();
  void submit(cudaGraphExec_t graph, std::uint64_t target, const char* operation);

  CudaBcc32Executor& executor_;
  DeviceChunkMap chunks_{};
  // Host shadow of the device control object.  It contains only device
  // addresses and fixed scheduler constants; no support contents are copied
  // here.  Runtime support/count authority remains behind these pointers.
  DeviceState host_state_{};
  DeviceState* device_state_ = nullptr;
  std::uint64_t* device_window_ = nullptr;
  std::uint8_t* device_resolution_ = nullptr;
  std::uint32_t launch_capacity_ = kCapacity;
  std::uint32_t history_capacity_ = kHistoryCapacity;
  DeviceBoundaryBinding boundary_{};
  cudaStream_t stream_ = nullptr;
  cudaGraphExec_t forward_graph_ = nullptr;
  cudaGraphExec_t device_forward_graph_ = nullptr;
  cudaGraphExec_t device_leaf_forward_graph_ = nullptr;
  cudaGraphExec_t inverse_graph_ = nullptr;
  std::uint64_t host_completed_ticks_ = 0u;
  std::uint32_t host_inverse_head_ = 0u;
  std::uint32_t host_inverse_depth_ = 0u;
  Fault host_fault_ = Fault::none;
  std::uint64_t host_generation_ = 0u;
  std::string last_status_ = "ordinary_f_timeline=UNBOOTSTRAPPED";
};

}  // namespace substrate::bcc32::device_ordinary_f_timeline

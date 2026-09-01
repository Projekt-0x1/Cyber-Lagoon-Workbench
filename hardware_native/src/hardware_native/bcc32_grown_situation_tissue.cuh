#pragma once

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "hardware_native/bcc32_developmental_adult.cuh"

namespace substrate::bcc32::grown_situation_tissue {

using substrate::bcc32::InterventionReason;
using substrate::bcc32::ResidentStageReason;

namespace adult = developmental_adult;
using adult::GrownAdult;
using adult::StateEntry;
using substrate::bcc32::SiteWord;

// This is physical lattice capacity, not a requested cluster count. Every site
// begins equivalent apart from its founder-grown bias; experience decides which
// sites become occupied and which occupied sites form predictive populations.
inline constexpr std::uint32_t kLatticeEdge = 8u;
inline constexpr std::uint32_t kRegionCount = kLatticeEdge * kLatticeEdge;
inline constexpr std::uint32_t kFieldsPerRegion = 6u;
inline constexpr std::uint32_t kActiveWordCount = (kRegionCount + 31u) / 32u;
inline constexpr std::uint32_t kRailCount =
    kRegionCount * kFieldsPerRegion * 2u + kActiveWordCount * 2u;
inline constexpr std::uint32_t kMaxContactBytes = 192u;
inline constexpr std::uint32_t kContextCoalitionWidth = 4u;
inline constexpr std::uint32_t kRecruitCueSimilarity = 28u;
inline constexpr std::uint32_t kPredictionCueSimilarity = 18u;

enum class ContactChannel : std::uint32_t {
  context = 0u,
  background = 1u,
  outcome = 2u,
};

enum Field : std::uint32_t {
  kBias = 0u,
  kCue = 1u,
  kOutcome = 2u,
  kSupport = 3u,
  kOutcomeEscrow = 4u,
  kSupportEscrow = 5u,
};

struct DeviceLayout {
  std::uint64_t rails[kRailCount]{};
};

struct PredictionWitness {
  std::uint32_t valid = 0u;
  std::uint32_t region = 0xffffffffu;
  std::uint64_t region_slot = 0u;
  SiteWord predicted_signature = 0u;
  std::uint32_t cue_similarity = 0u;
  std::uint32_t support = 0u;
  std::uint32_t parallel_regions_eligible = 0u;
  std::uint32_t parallel_regions_completed = 0u;
  std::uint64_t parallel_region_mask = 0u;
  std::uint64_t parallel_launches = 0u;
};

inline constexpr std::uint32_t kPredictThreads = kRegionCount;

struct ContactReceipt {
  std::uint64_t state_before_hash = 0u;
  std::uint64_t state_after_hash = 0u;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
  std::uint32_t active_region = 0xffffffffu;
  std::uint32_t recruited_region = 0xffffffffu;
  std::uint32_t occupied_regions = 0u;
  std::uint64_t active_mask = 0u;
  std::uint64_t recruited_mask = 0u;
};

struct LesionReceipt {
  std::uint64_t state_before_hash = 0u;
  std::uint64_t state_after_hash = 0u;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
  std::uint32_t region = 0xffffffffu;
  std::uint32_t moved_bits = 0u;
  std::uint32_t applied = 0u;
};

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

__host__ __device__ inline std::uint32_t rail_index(std::uint32_t region, std::uint32_t field,
                                                    std::uint32_t polarity) {
  return (region * kFieldsPerRegion + field) * 2u + polarity;
}

__host__ __device__ inline std::uint32_t active_rail_index(std::uint32_t word,
                                                           std::uint32_t polarity) {
  return kRegionCount * kFieldsPerRegion * 2u + word * 2u + polarity;
}

inline DeviceLayout make_layout(const GrownAdult& grown) {
  DeviceLayout layout{};
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    const std::int32_t x = -28 + static_cast<std::int32_t>(region % kLatticeEdge) * 8;
    // y0=-36 (previously -28) is a pure translation: same shape, stride,
    // region count, field count, and z formula. It keeps every situation
    // y-value in {-36,...,20} strictly below grown_form_credit_factor's
    // physical_offset() y-range [24,223], so the two producers' (x,y,z)
    // footprints no longer intersect (they previously shared exactly the
    // single cell x=-28,y=28, colliding at z=20..25 where both occupy that
    // range -- see attach_founder_matter's duplicate-slot rejection).
    const std::int32_t y = -36 + static_cast<std::int32_t>(region / kLatticeEdge) * 8;
    for (std::uint32_t field = 0u; field < kFieldsPerRegion; ++field) {
      for (std::uint32_t polarity = 0u; polarity < 2u; ++polarity) {
        layout.rails[rail_index(region, field, polarity)] =
            grown.physical_slot({x, y, 14 + static_cast<std::int32_t>(field * 2u + polarity)});
      }
    }
  }
  for (std::uint32_t word = 0u; word < kActiveWordCount; ++word) {
    layout.rails[active_rail_index(word, 0u)] =
        grown.physical_slot({0, static_cast<std::int32_t>(word) * 4, 30});
    layout.rails[active_rail_index(word, 1u)] =
        grown.physical_slot({0, static_cast<std::int32_t>(word) * 4, 31});
  }
  return layout;
}

inline SiteWord founder_bias(std::uint32_t region) {
  SiteWord value = 0x9e3779b9u * (region + 1u);
  value ^= std::rotl(value, static_cast<int>(region * 5u + 3u));
  value ^= 0xa5c31f27u;
  return value;
}

inline std::vector<StateEntry> founder_entries(const GrownAdult& grown) {
  const DeviceLayout layout = make_layout(grown);
  std::vector<StateEntry> entries;
  entries.reserve(kRailCount);
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    for (std::uint32_t field = 0u; field < kFieldsPerRegion; ++field) {
      const SiteWord value = field == kBias ? founder_bias(region) : 0u;
      entries.push_back({layout.rails[rail_index(region, field, 0u)], value});
      entries.push_back({layout.rails[rail_index(region, field, 1u)], ~value});
    }
  }
  for (std::uint32_t word = 0u; word < kActiveWordCount; ++word) {
    entries.push_back({layout.rails[active_rail_index(word, 0u)], 0u});
    entries.push_back({layout.rails[active_rail_index(word, 1u)], 0xffffffffu});
  }
  return entries;
}

__device__ inline SiteWord read_value(const SiteWord* words, const DeviceLayout& layout,
                                      std::uint32_t region, std::uint32_t field) {
  return words[layout.rails[rail_index(region, field, 0u)]];
}

__device__ inline void write_value(SiteWord* words, const DeviceLayout& layout,
                                   std::uint32_t region, std::uint32_t field, SiteWord value) {
  words[layout.rails[rail_index(region, field, 0u)]] = value;
  words[layout.rails[rail_index(region, field, 1u)]] = ~value;
}

__device__ inline std::uint64_t read_active(const SiteWord* words, const DeviceLayout& layout) {
  std::uint64_t value = 0u;
  for (std::uint32_t word = 0u; word < kActiveWordCount; ++word) {
    value |= static_cast<std::uint64_t>(words[layout.rails[active_rail_index(word, 0u)]])
             << (word * 32u);
  }
  return value;
}

__device__ inline void write_active(SiteWord* words, const DeviceLayout& layout,
                                    std::uint64_t value) {
  for (std::uint32_t word = 0u; word < kActiveWordCount; ++word) {
    const SiteWord part = static_cast<SiteWord>(value >> (word * 32u));
    words[layout.rails[active_rail_index(word, 0u)]] = part;
    words[layout.rails[active_rail_index(word, 1u)]] = ~part;
  }
}

__device__ inline SiteWord contact_signature(const std::uint8_t* bytes, std::uint32_t count) {
  std::int32_t votes[32]{};
  SiteWord prior = 0x6du;
  SiteWord prior2 = 0xb7u;
  for (std::uint32_t index = 0u; index < count; ++index) {
    const SiteWord value = bytes[index];
    SiteWord atom = value * 0x45d9f3bu + 0x27d4eb2du;
    atom ^= atom >> 16u;
    const SiteWord pair = prior * 257u + value * 17u;
    const SiteWord pair_atom = (pair ^ (pair >> 5u)) * 0x9e3779b9u + 0x7f4a7c15u;
    SiteWord triple_atom = (prior2 * 65537u + prior * 257u + value) * 0x85ebca6bu;
    triple_atom ^= triple_atom >> 13u;
    for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
      votes[bit] += ((atom >> bit) & 1u) != 0u ? 1 : -1;
      votes[bit] += ((pair_atom >> bit) & 1u) != 0u ? 2 : -2;
      votes[bit] += ((triple_atom >> bit) & 1u) != 0u ? 3 : -3;
    }
    prior2 = prior;
    prior = value;
  }
  SiteWord signature = 0u;
  for (std::uint32_t bit = 0u; bit < 32u; ++bit)
    if (votes[bit] >= 0)
      signature |= 1u << bit;
  return signature == 0u ? 1u : signature;
}

__device__ inline std::uint32_t support_count(SiteWord support) {
  return __popc(support);
}

__device__ inline SiteWord increment_support(SiteWord support) {
  if (support == 0xffffffffu)
    return support;
  const SiteWord available = ~support;
  return support | (available & (0u - available));
}

__device__ inline std::uint64_t mix(std::uint64_t hash, SiteWord word, std::uint32_t salt) {
  hash ^=
      static_cast<std::uint64_t>(word) + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u) + salt;
  return hash;
}

__device__ inline std::uint32_t state_matter_bits(const SiteWord* words,
                                                  const DeviceLayout& layout) {
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < kRailCount; ++index)
    total += __popc(words[layout.rails[index]]);
  return total;
}

__device__ inline std::uint64_t state_hash(const SiteWord* words, const DeviceLayout& layout) {
  std::uint64_t hash = 0xcbf29ce484222325ull;
  for (std::uint32_t index = 0u; index < kRailCount; ++index)
    hash = mix(hash, words[layout.rails[index]], index);
  return hash;
}

__device__ inline std::uint32_t occupied_region_count(const SiteWord* words,
                                                      const DeviceLayout& layout) {
  std::uint32_t occupied = 0u;
  for (std::uint32_t region = 0u; region < kRegionCount; ++region)
    occupied += read_value(words, layout, region, kCue) != 0u;
  return occupied;
}

__device__ inline bool region_occupied(const SiteWord* words,
                                       const DeviceLayout& layout,
                                       std::uint32_t region) {
  return read_value(words, layout, region, kCue) != 0u;
}

__device__ inline std::uint32_t best_region(const SiteWord* words, const DeviceLayout& layout,
                                            SiteWord signature, bool allow_empty,
                                            std::uint32_t* best_score) {
  std::uint32_t selected = 0xffffffffu;
  std::uint32_t score = 0u;
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    const SiteWord support = read_value(words, layout, region, kSupport);
    if (!allow_empty && !region_occupied(words, layout, region))
      continue;
    const SiteWord reference = region_occupied(words, layout, region)
                                   ? read_value(words, layout, region, kCue)
                                   : read_value(words, layout, region, kBias);
    const std::uint32_t candidate = 32u - __popc(reference ^ signature);
    if (selected == 0xffffffffu || candidate > score) {
      selected = region;
      score = candidate;
    }
  }
  if (best_score != nullptr)
    *best_score = score;
  return selected;
}

__device__ inline std::uint32_t best_empty_region(const SiteWord* words, const DeviceLayout& layout,
                                                  SiteWord signature) {
  std::uint32_t selected = 0xffffffffu;
  std::uint32_t score = 0u;
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    if (region_occupied(words, layout, region))
      continue;
    const SiteWord bias = read_value(words, layout, region, kBias);
    const std::uint32_t candidate = 32u - __popc(bias ^ signature);
    if (selected == 0xffffffffu || candidate > score) {
      selected = region;
      score = candidate;
    }
  }
  return selected;
}

__device__ inline std::uint64_t strongest_matching_regions(
    const SiteWord* words, const DeviceLayout& layout, SiteWord signature) {
  std::uint32_t selected[kContextCoalitionWidth]{0xffffffffu, 0xffffffffu,
                                                 0xffffffffu, 0xffffffffu};
  std::uint32_t scores[kContextCoalitionWidth]{};
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    if (!region_occupied(words, layout, region))
      continue;
    const std::uint32_t score =
        32u - __popc(read_value(words, layout, region, kCue) ^ signature);
    if (score < kRecruitCueSimilarity)
      continue;
    for (std::uint32_t rank = 0u; rank < kContextCoalitionWidth; ++rank) {
      if (selected[rank] != 0xffffffffu && score <= scores[rank])
        continue;
      for (std::uint32_t shifted = kContextCoalitionWidth - 1u;
           shifted > rank; --shifted) {
        selected[shifted] = selected[shifted - 1u];
        scores[shifted] = scores[shifted - 1u];
      }
      selected[rank] = region;
      scores[rank] = score;
      break;
    }
  }
  std::uint64_t mask = 0u;
  for (std::uint32_t rank = 0u; rank < kContextCoalitionWidth; ++rank)
    if (selected[rank] != 0xffffffffu)
      mask |= 1ull << selected[rank];
  return mask;
}

static __global__ void contact_kernel(SiteWord* words, DeviceLayout layout,
                                      const std::uint8_t* bytes, std::uint32_t count,
                                      ContactChannel channel, ContactReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  ContactReceipt local{};
  local.state_before_hash = state_hash(words, layout);
  local.matter_before_bits = state_matter_bits(words, layout);

  if (channel == ContactChannel::background) {
    local.active_mask = read_active(words, layout);
    local.active_region =
        local.active_mask == 0u
            ? 0xffffffffu
            : static_cast<std::uint32_t>(
                  __ffsll(static_cast<long long>(local.active_mask)) - 1);
  } else if (channel == ContactChannel::context) {
    const SiteWord signature = contact_signature(bytes, count);
    std::uint64_t active = strongest_matching_regions(words, layout, signature);
    while (__popcll(active) < kContextCoalitionWidth) {
      const std::uint32_t empty = best_empty_region(words, layout, signature);
      if (empty == 0xffffffffu)
        break;
      write_value(words, layout, empty, kCue, signature);
      active |= 1ull << empty;
      local.recruited_mask |= 1ull << empty;
      if (local.recruited_region == 0xffffffffu)
        local.recruited_region = empty;
    }
    if (active != 0u) {
      write_active(words, layout, active);
      local.active_mask = active;
      local.active_region = static_cast<std::uint32_t>(
          __ffsll(static_cast<long long>(active)) - 1);
    }
  } else {
    const std::uint64_t active = read_active(words, layout);
    if (active != 0u) {
      const SiteWord outcome = contact_signature(bytes, count);
      for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
        if ((active & (1ull << region)) == 0u)
          continue;
        write_value(words, layout, region, kOutcome, outcome);
        write_value(words, layout, region, kSupport,
                    increment_support(read_value(words, layout, region, kSupport)));
      }
      local.active_mask = active;
      local.active_region = static_cast<std::uint32_t>(
          __ffsll(static_cast<long long>(active)) - 1);
      write_active(words, layout, 0u);
    }
  }

  local.occupied_regions = occupied_region_count(words, layout);
  local.state_after_hash = state_hash(words, layout);
  local.matter_after_bits = state_matter_bits(words, layout);
  if (receipt != nullptr)
    *receipt = local;
}

static __global__ void predict_kernel(const SiteWord* words, DeviceLayout layout,
                                      const std::uint8_t* bytes, std::uint32_t count,
                                      PredictionWitness* witness) {
  if (blockIdx.x != 0u || threadIdx.x >= kPredictThreads)
    return;
  __shared__ SiteWord signature_shared;
  __shared__ std::uint32_t candidate_scores_shared[kRegionCount];
  __shared__ std::uint8_t candidate_valid_shared[kRegionCount];

  if (threadIdx.x == 0u)
    signature_shared = contact_signature(bytes, count);
  __syncthreads();

  const std::uint32_t region = threadIdx.x;
  const bool occupied = region_occupied(words, layout, region);
  candidate_valid_shared[region] = occupied ? 1u : 0u;
  candidate_scores_shared[region] =
      occupied ? 32u - __popc(read_value(words, layout, region, kCue) ^
                               signature_shared)
               : 0u;
  __syncthreads();

  if (threadIdx.x == 0u) {
    PredictionWitness local{};
    local.parallel_regions_completed = kRegionCount;
    local.parallel_launches = 1u;
    std::uint32_t selected = 0xffffffffu;
    std::uint32_t score = 0u;
    for (std::uint32_t candidate = 0u; candidate < kRegionCount; ++candidate) {
      if (candidate_valid_shared[candidate] == 0u) continue;
      ++local.parallel_regions_eligible;
      local.parallel_region_mask |= 1ull << candidate;
      const std::uint32_t candidate_score = candidate_scores_shared[candidate];
      if (selected == 0xffffffffu || candidate_score > score) {
        selected = candidate;
        score = candidate_score;
      }
    }
    if (selected != 0xffffffffu && score >= kPredictionCueSimilarity) {
      const SiteWord support = read_value(words, layout, selected, kSupport);
      if (support != 0u) {
        local.valid = 1u;
        local.region = selected;
        local.region_slot = layout.rails[rail_index(selected, kCue, 0u)];
        local.predicted_signature = read_value(words, layout, selected, kOutcome);
        local.cue_similarity = score;
        local.support = support_count(support);
      }
    }
    if (witness != nullptr)
      *witness = local;
  }
}

static __global__ void lesion_kernel(SiteWord* words, DeviceLayout layout, std::uint32_t region,
                                     LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  LesionReceipt local{};
  local.state_before_hash = state_hash(words, layout);
  local.matter_before_bits = state_matter_bits(words, layout);
  local.region = region;
  if (region < kRegionCount) {
    const SiteWord support = read_value(words, layout, region, kSupport);
    const SiteWord escrow_support = read_value(words, layout, region, kSupportEscrow);
    const SiteWord outcome = read_value(words, layout, region, kOutcome);
    const SiteWord escrow_outcome = read_value(words, layout, region, kOutcomeEscrow);
    write_value(words, layout, region, kSupport, escrow_support);
    write_value(words, layout, region, kSupportEscrow, support);
    write_value(words, layout, region, kOutcome, escrow_outcome);
    write_value(words, layout, region, kOutcomeEscrow, outcome);
    local.moved_bits = 64u;
    local.applied = 1u;
  }
  local.state_after_hash = state_hash(words, layout);
  local.matter_after_bits = state_matter_bits(words, layout);
  if (receipt != nullptr)
    *receipt = local;
}

class Tissue {
 public:
  explicit Tissue(GrownAdult& grown) : grown_(grown), layout_(make_layout(grown)) {
    require_cuda(cudaMalloc(&contact_bytes_, kMaxContactBytes), "allocate situation contact");
    require_cuda(cudaMalloc(&contact_receipt_, sizeof(ContactReceipt)),
                 "allocate situation contact receipt");
    require_cuda(cudaMalloc(&prediction_, sizeof(PredictionWitness)),
                 "allocate situation prediction");
    require_cuda(cudaMalloc(&lesion_receipt_, sizeof(LesionReceipt)),
                 "allocate situation lesion receipt");
  }

  Tissue(const Tissue&) = delete;
  Tissue& operator=(const Tissue&) = delete;

  ~Tissue() {
    destroy_graphs();
    cudaFree(lesion_receipt_);
    cudaFree(prediction_);
    cudaFree(contact_receipt_);
    cudaFree(contact_bytes_);
  }

  ContactReceipt contact(std::span<const std::uint8_t> bytes, ContactChannel channel) {
    if (bytes.size() > kMaxContactBytes)
      throw std::invalid_argument("situation contact exceeds resident cap");
    if (!bytes.empty()) {
      require_cuda(cudaMemcpy(contact_bytes_, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
                   "copy situation contact");
    }
    grown_.resident_stage(
        ResidentStageReason::situation_contact, [&](SiteWord* words) {
          const std::uint32_t count = static_cast<std::uint32_t>(bytes.size());
          void* arguments[] = {
              kernel_argument(words), kernel_argument(layout_),
              kernel_argument(contact_bytes_), kernel_argument(count),
              kernel_argument(channel), kernel_argument(contact_receipt_)};
          launch_graph(contact_graph_, "launch situation contact graph",
                       reinterpret_cast<void*>(contact_kernel), dim3(1u),
                       dim3(1u), arguments);
        });
    synchronize("run situation contact");
    return copy(contact_receipt_);
  }

  PredictionWitness predict(std::span<const std::uint8_t> bytes) {
    if (bytes.size() > kMaxContactBytes)
      throw std::invalid_argument("situation probe exceeds resident cap");
    if (!bytes.empty()) {
      require_cuda(cudaMemcpy(contact_bytes_, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
                   "copy situation probe");
    }
    const std::uint32_t count = static_cast<std::uint32_t>(bytes.size());
    const SiteWord* words = grown_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                         kernel_argument(contact_bytes_), kernel_argument(count),
                         kernel_argument(prediction_)};
    launch_graph(predict_graph_, "launch situation prediction graph",
                 reinterpret_cast<void*>(predict_kernel), dim3(1u),
                 dim3(kPredictThreads), arguments);
    synchronize("run situation prediction");
    return copy(prediction_);
  }

  LesionReceipt lesion(std::uint32_t region) {
    grown_.intervene(InterventionReason::situation_lesion, [&](SiteWord* words) {
      void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                           kernel_argument(region), kernel_argument(lesion_receipt_)};
      launch_graph(lesion_graph_, "launch situation lesion graph",
                   reinterpret_cast<void*>(lesion_kernel), dim3(1u), dim3(1u),
                   arguments);
    });
    synchronize("run situation lesion");
    return copy(lesion_receipt_);
  }

  LesionReceipt restore_lesion(std::uint32_t region) { return lesion(region); }

  [[nodiscard]] const DeviceLayout& layout() const { return layout_; }

 private:
  struct KernelGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraphNode_t node = nullptr;

    void destroy() {
      if (executable != nullptr) cudaGraphExecDestroy(executable);
      if (graph != nullptr) cudaGraphDestroy(graph);
      executable = nullptr;
      graph = nullptr;
      node = nullptr;
    }

    ~KernelGraph() { destroy(); }
  };

  template <typename T>
  T copy(const T* device) const {
    T host{};
    require_cuda(cudaMemcpy(&host, device, sizeof(T), cudaMemcpyDeviceToHost),
                 "copy situation receipt");
    return host;
  }

  static void synchronize(const char* operation) {
    require_cuda(cudaGetLastError(), operation);
    require_cuda(cudaDeviceSynchronize(), operation);
  }

  template <typename T>
  static void* kernel_argument(T& value) {
    return const_cast<void*>(static_cast<const void*>(&value));
  }

  static void launch_graph(KernelGraph& graph, const char* operation,
                           void* function, dim3 grid, dim3 block,
                           void** arguments) {
    cudaKernelNodeParams params{};
    params.func = function;
    params.gridDim = grid;
    params.blockDim = block;
    params.kernelParams = arguments;
    if (graph.executable == nullptr) {
      require_cuda(cudaGraphCreate(&graph.graph, 0u), operation);
      require_cuda(cudaGraphAddKernelNode(&graph.node, graph.graph, nullptr,
                                          0u, &params),
                   operation);
      require_cuda(cudaGraphInstantiate(&graph.executable, graph.graph,
                                        nullptr, nullptr, 0u),
                   operation);
    } else {
      require_cuda(cudaGraphExecKernelNodeSetParams(
                       graph.executable, graph.node, &params),
                   operation);
    }
    require_cuda(cudaGraphLaunch(graph.executable, 0), operation);
  }

  void destroy_graphs() {
    contact_graph_.destroy();
    predict_graph_.destroy();
    lesion_graph_.destroy();
  }

  GrownAdult& grown_;
  DeviceLayout layout_{};
  std::uint8_t* contact_bytes_ = nullptr;
  ContactReceipt* contact_receipt_ = nullptr;
  PredictionWitness* prediction_ = nullptr;
  LesionReceipt* lesion_receipt_ = nullptr;
  KernelGraph contact_graph_;
  KernelGraph predict_graph_;
  KernelGraph lesion_graph_;
};

}  // namespace substrate::bcc32::grown_situation_tissue

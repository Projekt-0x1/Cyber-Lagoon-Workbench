#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/bcc32_developmental_adult.cuh"
#include "hardware_native/bcc32_grown_instance_basin.cuh"

namespace substrate::bcc32::grown_instance_basin_tissue {

namespace factor = grown_instance_basin_factor;
using substrate::bcc32::SiteWord;
using developmental_adult::GrownAdult;
using substrate::bcc32::InterventionReason;
using developmental_adult::StateEntry;

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

inline factor::DeviceLayout make_layout(const GrownAdult& grown) {
  factor::DeviceLayout layout{};
  for (std::uint32_t index = 0u; index < factor::kPhysicalRailCount;
       ++index) {
    const factor::PhysicalOffset offset = factor::physical_offset(index);
    layout.rails[index] =
        grown.physical_slot({offset.x, offset.y, offset.z});
  }
  return layout;
}

inline std::vector<StateEntry> founder_entries(const GrownAdult& grown) {
  const factor::DeviceLayout layout = make_layout(grown);
  std::vector<StateEntry> entries;
  entries.reserve(factor::kPhysicalRailCount);
  for (std::uint32_t index = 0u;
       index < factor::kResidentPhysicalRailCount; index += 2u) {
    const std::uint32_t field = index / 2u;
    SiteWord value = 0u;
    // kQ is 0xff in the adult carrier; keep the founder word distinct while
    // preserving the low-bit free-matter mask read by the device law.
    if (field == factor::kFreeMask) value = factor::kFullMask | 0x100u;
    if (field == factor::kFactorMarker) value = factor::kFactorMarkerValue;
    if (field == factor::kLayoutVersion) value = factor::kLayoutVersionValue;
    entries.push_back({layout.rails[index], value});
    entries.push_back({layout.rails[index + 1u], ~value});
  }
  for (std::uint32_t index = 0u;
       index < factor::kJournalPhysicalRailCount; index += 2u) {
    const std::uint32_t physical =
        factor::kResidentPhysicalRailCount + index;
    entries.push_back({layout.rails[physical], 0u});
    entries.push_back({layout.rails[physical + 1u], ~0u});
  }
  return entries;
}

class InstanceBasin {
 public:
  explicit InstanceBasin(GrownAdult& grown)
      : grown_(grown), layout_(make_layout(grown)) {
    try {
      require_cuda(
          cudaMalloc(&descriptors_, factor::kMaxPatches *
                                      sizeof(factor::PatchDescriptor)),
          "allocate instance descriptors");
      require_cuda(cudaMalloc(&bytes_, factor::kMaxPatchBytes),
                           "allocate instance bytes");
      require_cuda(cudaMalloc(&inputs_, sizeof(*inputs_)),
                           "allocate instance inputs");
      require_cuda(cudaMalloc(&receipt_, sizeof(*receipt_)),
                           "allocate instance receipt");
      require_cuda(cudaMalloc(&prediction_, sizeof(*prediction_)),
                           "allocate instance prediction");
      grown_.configure_instance_basin_factor(layout_, inputs_, receipt_);
      clear_inputs();
    } catch (...) {
      release();
      throw;
    }
  }

  InstanceBasin(const InstanceBasin&) = delete;
  InstanceBasin& operator=(const InstanceBasin&) = delete;

  ~InstanceBasin() {
    destroy_graphs();
    grown_.detach_instance_basin_buffers();
    release();
  }

  factor::PredictionReceipt predict() {
    const SiteWord* words = grown_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                         kernel_argument(prediction_)};
    launch_graph(predict_graph_, reinterpret_cast<void*>(factor::predict_kernel),
                 arguments, "launch instance basin prediction");
    synchronize("predict instance basin");
    return copy(prediction_);
  }

  factor::StepReceipt observe(std::span<const factor::PatchDescriptor> patches,
                             std::span<const std::uint8_t> bytes) {
    if (patches.size() > factor::kMaxPatches ||
        bytes.size() > factor::kMaxPatchBytes) {
      throw std::invalid_argument("instance contact exceeds resident cap");
    }
    for (const factor::PatchDescriptor& patch : patches) {
      if (patch.sensor_site >= factor::kSiteCount ||
          patch.byte_offset > bytes.size() ||
          patch.byte_count > bytes.size() - patch.byte_offset) {
        throw std::invalid_argument("instance descriptor is outside byte rail");
      }
    }
    if (!patches.empty()) {
      require_cuda(
          cudaMemcpy(descriptors_, patches.data(),
                     patches.size() * sizeof(factor::PatchDescriptor),
                     cudaMemcpyHostToDevice),
          "copy instance descriptors");
    }
    if (!bytes.empty()) {
      require_cuda(cudaMemcpy(bytes_, bytes.data(), bytes.size(),
                                      cudaMemcpyHostToDevice),
                             "copy instance byte rail");
    }
    DeviceInputsHost staged{};
    staged.descriptors = descriptors_;
    staged.bytes = bytes_;
    staged.descriptor_count = static_cast<std::uint32_t>(patches.size());
    staged.byte_count = static_cast<std::uint32_t>(bytes.size());
    staged.staged = 1u;
    require_cuda(cudaMemcpy(inputs_, &staged, sizeof(staged),
                                    cudaMemcpyHostToDevice),
                         "stage instance contact");
    grown_.develop(1u);
    clear_inputs();
    return copy(receipt_);
  }

  void lesion(std::uint32_t basin) {
    if (basin >= factor::kBasinCount) {
      throw std::out_of_range("instance basin lesion is outside resident count");
    }
    grown_.intervene(InterventionReason::lesion_relation,
                     [&](SiteWord* words) {
                       void* arguments[] = {kernel_argument(words),
                                            kernel_argument(layout_),
                                            kernel_argument(basin)};
                       launch_graph(
                           lesion_graph_,
                           reinterpret_cast<void*>(factor::lesion_kernel),
                           arguments, "launch instance basin lesion");
                     });
    synchronize("lesion instance basin");
  }

  void reverse_one() { grown_.reverse(1u); }

 private:
  using DeviceInputsHost = factor::DeviceInputs;

  struct KernelGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraphNode_t node = nullptr;

    void destroy() noexcept {
      if (executable != nullptr) {
        (void)cudaGraphExecDestroy(executable);
        executable = nullptr;
      }
      if (graph != nullptr) {
        (void)cudaGraphDestroy(graph);
        graph = nullptr;
      }
      node = nullptr;
    }

    ~KernelGraph() { destroy(); }
  };

  template <typename T>
  static void* kernel_argument(T& value) {
    return const_cast<void*>(static_cast<const void*>(&value));
  }

  static void launch_graph(KernelGraph& graph, void* function,
                           void** arguments, const char* operation) {
    cudaKernelNodeParams params{};
    params.func = function;
    params.gridDim = dim3{1u, 1u, 1u};
    params.blockDim = dim3{1u, 1u, 1u};
    params.sharedMemBytes = 0u;
    params.kernelParams = arguments;
    params.extra = nullptr;
    if (graph.executable == nullptr) {
      require_cuda(cudaGraphCreate(&graph.graph, 0u), operation);
      require_cuda(cudaGraphAddKernelNode(
                       &graph.node, graph.graph, nullptr, 0u, &params),
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

  void destroy_graphs() noexcept {
    predict_graph_.destroy();
    lesion_graph_.destroy();
  }

  template <typename T>
  static T copy(const T* device) {
    T host{};
    require_cuda(cudaMemcpy(&host, device, sizeof(host),
                                    cudaMemcpyDeviceToHost),
                         "copy instance receipt");
    return host;
  }

  void clear_inputs() {
    DeviceInputsHost empty{};
    require_cuda(cudaMemcpy(inputs_, &empty, sizeof(empty),
                                    cudaMemcpyHostToDevice),
                         "clear instance inputs");
  }

  static void synchronize(const char* operation) {
    require_cuda(cudaGetLastError(), operation);
    require_cuda(cudaDeviceSynchronize(), operation);
  }

  void release() noexcept {
    (void)cudaFree(prediction_);
    (void)cudaFree(receipt_);
    (void)cudaFree(inputs_);
    (void)cudaFree(bytes_);
    (void)cudaFree(descriptors_);
    prediction_ = nullptr;
    receipt_ = nullptr;
    inputs_ = nullptr;
    bytes_ = nullptr;
    descriptors_ = nullptr;
  }

  GrownAdult& grown_;
  factor::DeviceLayout layout_{};
  KernelGraph predict_graph_;
  KernelGraph lesion_graph_;
  factor::PatchDescriptor* descriptors_ = nullptr;
  std::uint8_t* bytes_ = nullptr;
  factor::DeviceInputs* inputs_ = nullptr;
  factor::StepReceipt* receipt_ = nullptr;
  factor::PredictionReceipt* prediction_ = nullptr;
};

}  // namespace substrate::bcc32::grown_instance_basin_tissue

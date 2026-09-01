#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "hardware_native/bcc32_grown_cloud_factor.cuh"

namespace substrate::bcc32::grown_cloud_tissue {

namespace factor = grown_cloud_factor;
using developmental_adult::GrownAdult;
using substrate::bcc32::InterventionReason;
using substrate::bcc32::ResidentStageReason;
using developmental_adult::StateEntry;
using substrate::bcc32::SiteWord;

inline factor::DeviceLayout make_layout(const GrownAdult& grown) {
  factor::DeviceLayout layout{};
  for (std::uint32_t index = 0u; index < factor::kResidentRailCount;
       ++index) {
    const factor::PhysicalOffset offset = factor::physical_offset(index);
    layout.rails[index] =
        grown.physical_slot({offset.x, offset.y, offset.z});
  }
  return layout;
}

inline std::vector<StateEntry> founder_entries(
    const GrownAdult& grown, SiteWord resource_mask = 0xffffffffu) {
  const factor::DeviceLayout layout = make_layout(grown);
  std::vector<StateEntry> entries;
  entries.reserve(factor::kRailCount);
  for (std::uint32_t index = 0u; index < factor::kRailCount; index += 2u) {
    const bool marker =
        index == factor::global_rail(factor::kCloudMarker, 0u);
    const bool layout_version =
        index == factor::global_rail(factor::kCloudLayoutVersion, 0u);
    const bool free_matter =
        index < factor::kCellRailCount &&
        ((index / 2u) % factor::kCellFieldCount) ==
            factor::kCellFreeMatter;
    const SiteWord value =
        marker ? factor::kCloudMarkerValue
               : (layout_version ? factor::kCloudLayoutVersionValue
                                 : (free_matter ? resource_mask : 0u));
    const std::uint64_t slot =
        index < factor::kResidentRailCount
            ? layout.rails[index]
            : [&] {
                const factor::PhysicalOffset offset =
                    factor::physical_offset(index);
                return grown.physical_slot(
                    {offset.x, offset.y, offset.z});
              }();
    const std::uint64_t complement_slot =
        index + 1u < factor::kResidentRailCount
            ? layout.rails[index + 1u]
            : [&] {
                const factor::PhysicalOffset offset =
                    factor::physical_offset(index + 1u);
                return grown.physical_slot(
                    {offset.x, offset.y, offset.z});
              }();
    entries.push_back({slot, value});
    entries.push_back({complement_slot, ~value});
  }
  return entries;
}

class Cloud {
 public:
  explicit Cloud(GrownAdult& grown)
      : grown_(grown), layout_(make_layout(grown)) {
    try {
      factor::require_cuda(
          cudaMalloc(&contact_bytes_, factor::kMaxContactBytes),
          "allocate grown-cloud contact");
      factor::require_cuda(
          cudaMalloc(&expected_bytes_, factor::kMaxContactBytes),
          "allocate grown-cloud expected contact");
      factor::require_cuda(
          cudaMalloc(&contact_receipt_, sizeof(*contact_receipt_)),
          "allocate grown-cloud contact receipt");
      factor::require_cuda(cudaMalloc(&prediction_, sizeof(*prediction_)),
                           "allocate grown-cloud prediction");
      factor::require_cuda(
          cudaMalloc(&lesion_receipt_, sizeof(*lesion_receipt_)),
          "allocate grown-cloud lesion receipt");
      factor::require_cuda(cudaMalloc(&ohm_, sizeof(*ohm_)),
                           "allocate grown-cloud OHM witness");
      factor::require_cuda(cudaMalloc(&resources_, sizeof(*resources_)),
                           "allocate grown-cloud resource witness");
      factor::require_cuda(cudaMalloc(&topology_, sizeof(*topology_)),
                           "allocate grown-cloud topology witness");
      factor::require_cuda(cudaMalloc(&situation_, sizeof(*situation_)),
                           "allocate grown-cloud situation witness");
      SiteWord layout_version = 0u;
      factor::require_cuda(
          cudaMemcpy(
              &layout_version,
              grown_.device_words() +
                  layout_.rails[factor::global_rail(
                      factor::kCloudLayoutVersion, 0u)],
              sizeof(layout_version), cudaMemcpyDeviceToHost),
          "read grown-cloud layout version");
      if (layout_version != factor::kCloudLayoutVersionValue) {
        throw std::runtime_error(
            "grown-cloud checkpoint layout is incompatible");
      }
      grown_.configure_cloud_factor(layout_, contact_receipt_);
    } catch (...) {
      release_buffers();
      throw;
    }
  }

  Cloud(const Cloud&) = delete;
  Cloud& operator=(const Cloud&) = delete;

  ~Cloud() {
    destroy_graphs();
    grown_.detach_cloud_factor_receipt();
    release_buffers();
  }

 private:
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
      factor::require_cuda(cudaGraphCreate(&graph.graph, 0u), operation);
      factor::require_cuda(cudaGraphAddKernelNode(
                               &graph.node, graph.graph, nullptr, 0u, &params),
                           operation);
      factor::require_cuda(cudaGraphInstantiate(
                               &graph.executable, graph.graph, nullptr,
                               nullptr, 0u),
                           operation);
    } else {
      factor::require_cuda(cudaGraphExecKernelNodeSetParams(
                               graph.executable, graph.node, &params),
                           operation);
    }
    factor::require_cuda(cudaGraphLaunch(graph.executable, 0), operation);
  }

  void destroy_graphs() noexcept {
    stage_contact_graph_.destroy();
    predict_graph_.destroy();
    lesion_graph_.destroy();
    ohm_graph_.destroy();
    resource_graph_.destroy();
    topology_graph_.destroy();
    situation_graph_.destroy();
  }

  void release_buffers() noexcept {
    (void)cudaFree(lesion_receipt_);
    (void)cudaFree(ohm_);
    (void)cudaFree(resources_);
    (void)cudaFree(topology_);
    (void)cudaFree(situation_);
    (void)cudaFree(prediction_);
    (void)cudaFree(contact_receipt_);
    (void)cudaFree(expected_bytes_);
    (void)cudaFree(contact_bytes_);
    lesion_receipt_ = nullptr;
    ohm_ = nullptr;
    resources_ = nullptr;
    topology_ = nullptr;
    situation_ = nullptr;
    prediction_ = nullptr;
    contact_receipt_ = nullptr;
    expected_bytes_ = nullptr;
    contact_bytes_ = nullptr;
  }

 public:
  factor::ContactReceipt contact(std::span<const std::uint8_t> bytes) {
    if (bytes.size() > factor::kMaxContactBytes) {
      throw std::invalid_argument("grown-cloud contact exceeds resident cap");
    }
    if (!bytes.empty()) {
      factor::require_cuda(
          cudaMemcpy(contact_bytes_, bytes.data(), bytes.size(),
                     cudaMemcpyHostToDevice),
          "copy grown-cloud contact");
    }
    // ⚠ A CONTACT IS NOT AN INTERVENTION AND NOT A BOUNDARY TRANSACTION. It is
    // the organism receiving input, so it is not the experimenter's; but it goes
    // through NO declared port and carries no MembraneReceipt, so calling it a
    // boundary transaction would be the overclaim requirement 2 exists to
    // prevent. It is a resident stage until it earns a port.
    grown_.resident_stage(ResidentStageReason::cloud_contact,
                          [&](SiteWord* words) {
                            const std::uint32_t count =
                                static_cast<std::uint32_t>(bytes.size());
                            void* arguments[] = {
                                kernel_argument(words),
                                kernel_argument(layout_),
                                kernel_argument(contact_bytes_),
                                kernel_argument(count)};
                            launch_graph(
                                stage_contact_graph_,
                                reinterpret_cast<void*>(
                                    factor::stage_contact_kernel),
                                arguments, "launch cloud_F_v1 contact");
                     });
    synchronize("stage cloud_F_v1 contact");
    grown_.develop(1u);
    return copy(contact_receipt_);
  }

  // The returned resident scratch is ordered on the caller's default stream
  // and remains valid only until this Cloud predicts again or is destroyed.
  [[nodiscard]] factor::PredictionWitness* predict_device(
      std::span<const std::uint8_t> bytes, std::uint32_t horizon) {
    if (bytes.size() > factor::kMaxContactBytes) {
      throw std::invalid_argument("grown-cloud probe exceeds resident cap");
    }
    if (!bytes.empty()) {
      factor::require_cuda(
          cudaMemcpy(contact_bytes_, bytes.data(), bytes.size(),
                     cudaMemcpyHostToDevice),
          "copy grown-cloud probe");
    }
    const SiteWord* words = grown_.device_words();
    const std::uint32_t count = static_cast<std::uint32_t>(bytes.size());
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                         kernel_argument(contact_bytes_), kernel_argument(count),
                         kernel_argument(horizon), kernel_argument(prediction_)};
    launch_graph(predict_graph_, reinterpret_cast<void*>(factor::predict_kernel),
                 arguments, "launch cloud_F_v1 prediction");
    return prediction_;
  }

  factor::PredictionWitness predict(std::span<const std::uint8_t> bytes,
                                    std::uint32_t horizon) {
    (void)predict_device(bytes, horizon);
    synchronize("run cloud_F_v1 prediction");
    return copy(prediction_);
  }

  factor::LesionReceipt lesion(std::uint32_t family,
                               std::uint32_t source_mask,
                               std::uint32_t max_bits = 0xffffffffu) {
    if (family > 4u) {
      throw std::invalid_argument("grown-cloud lesion family invalid");
    }
    grown_.intervene(InterventionReason::cloud_lesion, [&](SiteWord* words) {
      const std::uint32_t restore = 0u;
      void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                           kernel_argument(source_mask), kernel_argument(family),
                           kernel_argument(max_bits), kernel_argument(restore),
                           kernel_argument(lesion_receipt_)};
      launch_graph(lesion_graph_, reinterpret_cast<void*>(factor::lesion_kernel),
                   arguments, "launch cloud_F_v1 lesion");
    });
    synchronize("run cloud_F_v1 lesion");
    return copy(lesion_receipt_);
  }

  factor::LesionReceipt restore_lesion(std::uint32_t family,
                                       std::uint32_t source_mask) {
    if (family > 4u) {
      throw std::invalid_argument("grown-cloud lesion family invalid");
    }
    grown_.intervene(InterventionReason::cloud_lesion, [&](SiteWord* words) {
      const std::uint32_t max_bits = 0xffffffffu;
      const std::uint32_t restore = 1u;
      void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                           kernel_argument(source_mask), kernel_argument(family),
                           kernel_argument(max_bits), kernel_argument(restore),
                           kernel_argument(lesion_receipt_)};
      launch_graph(lesion_graph_, reinterpret_cast<void*>(factor::lesion_kernel),
                   arguments, "launch cloud_F_v1 lesion restore");
    });
    synchronize("restore cloud_F_v1 lesion");
    return copy(lesion_receipt_);
  }

  factor::OhmWitness ohm(std::span<const std::uint8_t> probe,
                          std::span<const std::uint8_t> expected,
                          std::uint32_t horizon) {
    if (probe.size() > factor::kMaxContactBytes ||
        expected.size() > factor::kMaxContactBytes) {
      throw std::invalid_argument("grown-cloud OHM contact exceeds resident cap");
    }
    if (!probe.empty()) {
      factor::require_cuda(
          cudaMemcpy(contact_bytes_, probe.data(), probe.size(),
                     cudaMemcpyHostToDevice),
          "copy grown-cloud OHM probe");
    }
    if (!expected.empty()) {
      factor::require_cuda(
          cudaMemcpy(expected_bytes_, expected.data(), expected.size(),
                     cudaMemcpyHostToDevice),
          "copy grown-cloud OHM expected contact");
    }
    const SiteWord* words = grown_.device_words();
    const std::uint32_t probe_count =
        static_cast<std::uint32_t>(probe.size());
    const std::uint32_t expected_count =
        static_cast<std::uint32_t>(expected.size());
    void* arguments[] = {
        kernel_argument(words), kernel_argument(layout_),
        kernel_argument(contact_bytes_), kernel_argument(probe_count),
        kernel_argument(expected_bytes_), kernel_argument(expected_count),
        kernel_argument(horizon), kernel_argument(ohm_)};
    launch_graph(ohm_graph_, reinterpret_cast<void*>(factor::ohm_kernel),
                 arguments, "launch cloud_F_v1 OHM propagation");
    synchronize("run cloud_F_v1 OHM propagation");
    return copy(ohm_);
  }

  factor::ResourceWitness resources() {
    const SiteWord* words = grown_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                         kernel_argument(resources_)};
    launch_graph(resource_graph_, reinterpret_cast<void*>(factor::resource_kernel),
                 arguments, "launch cloud_F_v1 resources");
    synchronize("read cloud_F_v1 resources");
    return copy(resources_);
  }

  factor::TopologyWitness topology() {
    const SiteWord* words = grown_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                         kernel_argument(topology_)};
    launch_graph(topology_graph_, reinterpret_cast<void*>(factor::topology_kernel),
                 arguments, "launch cloud_F_v1 topology");
    synchronize("read cloud_F_v1 topology");
    return copy(topology_);
  }

  factor::SituationWitness situation(
      std::span<const std::uint8_t> probe = {}) {
    if (probe.size() > factor::kMaxContactBytes) {
      throw std::invalid_argument(
          "grown-cloud situation probe exceeds resident cap");
    }
    if (!probe.empty()) {
      factor::require_cuda(
          cudaMemcpy(contact_bytes_, probe.data(), probe.size(),
                     cudaMemcpyHostToDevice),
          "copy grown-cloud situation probe");
    }
    const SiteWord* words = grown_.device_words();
    const std::uint8_t* probe_bytes =
        probe.empty() ? nullptr : contact_bytes_;
    const std::uint32_t count = static_cast<std::uint32_t>(probe.size());
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_),
                         kernel_argument(probe_bytes), kernel_argument(count),
                         kernel_argument(situation_)};
    launch_graph(situation_graph_, reinterpret_cast<void*>(factor::situation_kernel),
                 arguments, "launch cloud_F_v1 situation");
    synchronize("read cloud_F_v1 situation state");
    return copy(situation_);
  }

  [[nodiscard]] const factor::DeviceLayout& layout() const { return layout_; }

 private:
  template <typename T>
  static T copy(const T* device) {
    T host{};
    factor::require_cuda(cudaMemcpy(&host, device, sizeof(host),
                                    cudaMemcpyDeviceToHost),
                         "copy grown-cloud receipt");
    return host;
  }

  static void synchronize(const char* operation) {
    factor::require_cuda(cudaGetLastError(), operation);
    factor::require_cuda(cudaDeviceSynchronize(), operation);
  }

  GrownAdult& grown_;
  factor::DeviceLayout layout_{};
  KernelGraph stage_contact_graph_;
  KernelGraph predict_graph_;
  KernelGraph lesion_graph_;
  KernelGraph ohm_graph_;
  KernelGraph resource_graph_;
  KernelGraph topology_graph_;
  KernelGraph situation_graph_;
  std::uint8_t* contact_bytes_ = nullptr;
  std::uint8_t* expected_bytes_ = nullptr;
  factor::ContactReceipt* contact_receipt_ = nullptr;
  factor::PredictionWitness* prediction_ = nullptr;
  factor::LesionReceipt* lesion_receipt_ = nullptr;
  factor::OhmWitness* ohm_ = nullptr;
  factor::ResourceWitness* resources_ = nullptr;
  factor::TopologyWitness* topology_ = nullptr;
  factor::SituationWitness* situation_ = nullptr;
};

}  // namespace substrate::bcc32::grown_cloud_tissue

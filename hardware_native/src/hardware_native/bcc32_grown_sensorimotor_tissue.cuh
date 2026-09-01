#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/bcc32_developmental_adult.cuh"
#include "hardware_native/bcc32_grown_sensorimotor_factor.cuh"

namespace substrate::bcc32::grown_sensorimotor_tissue {

namespace factor = grown_sensorimotor_factor;
using developmental_adult::GrownAdult;
using substrate::bcc32::InterventionReason;
using developmental_adult::StateEntry;
using factor::ConsequenceReceipt;
using factor::DeviceInputs;
using factor::DeviceLayout;
using factor::LesionReceipt;
using factor::PredictionReceipt;
using factor::RelationCensus;
using factor::RelationEndpointView;
using factor::Rail;
using factor::TransformReceipt;
using factor::kActionCount;
using factor::kMaxContactBytes;
using factor::kNoAction;
using factor::kPhysicalRailCount;

inline DeviceLayout make_layout(const GrownAdult& grown) {
  DeviceLayout layout{};
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index) {
    const factor::PhysicalOffset offset = factor::physical_offset(index);
    layout.rails[index] =
        grown.physical_slot({offset.x, offset.y, offset.z});
  }
  for (std::uint32_t index = 0u;
       index < grown_instance_basin_factor::kPhysicalRailCount; ++index) {
    const grown_instance_basin_factor::PhysicalOffset offset =
        grown_instance_basin_factor::physical_offset(index);
    layout.context.rails[index] =
        grown.physical_slot({offset.x, offset.y, offset.z});
  }
  return layout;
}

inline std::vector<StateEntry> founder_entries(const GrownAdult& grown) {
  const DeviceLayout layout = make_layout(grown);
  std::vector<StateEntry> entries;
  entries.reserve(kPhysicalRailCount);
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; index += 2u) {
    SiteWord value = 0u;
    if (index < factor::kResidentPhysicalRailCount) {
      const Rail rail = static_cast<Rail>(index / 2u);
      if (rail == factor::kRouteEnabled0 ||
          rail == factor::kRouteEnabled1 ||
          rail == factor::kSomaticPath0 ||
          rail == factor::kSomaticPath1 ||
          rail == factor::kReafferencePath ||
          rail == factor::kRemoteControl) {
        value = 0xffffffffu;
      } else if (rail == factor::kFactorMarker) {
        value = factor::kFactorMarkerValue;
      } else if (rail == factor::kLayoutVersion) {
        value = factor::kLayoutVersionValue;
      }
    }
    entries.push_back({layout.rails[index], value});
    entries.push_back({layout.rails[index + 1u], ~value});
  }
  return entries;
}

inline std::vector<StateEntry> motor_surface_founder_entries(
    const GrownAdult& grown) {
  RawByteRails motor_surface{};
  motor_surface.zero =
      with_carriers(motor_surface.zero,
                    developmental_adult::kMotorIdleCarrierPattern);
  motor_surface.one =
      with_carriers(motor_surface.one,
                    developmental_adult::kMotorIdleCarrierPattern);
  std::vector<StateEntry> entries;
  entries.reserve(2u);
  entries.push_back(
      {grown.boundary_port_slot(developmental_adult::kRawMotorZeroPort),
       motor_surface.zero});
  entries.push_back(
      {grown.boundary_port_slot(developmental_adult::kRawMotorOnePort),
       motor_surface.one});
  return entries;
}

class Tissue {
 public:
  explicit Tissue(GrownAdult& adult)
      : adult_(adult), layout_(make_layout(adult)) {
    try {
      require_cuda(cudaMalloc(&layout_device_, sizeof(layout_)),
                   "allocate sensorimotor layout");
      require_cuda(cudaMemcpy(layout_device_, &layout_, sizeof(layout_),
                              cudaMemcpyHostToDevice),
                   "upload sensorimotor layout");
      require_cuda(cudaMalloc(&cue_, kMaxContactBytes),
                   "allocate cue bytes");
      require_cuda(cudaMalloc(&actual_reafference_, kMaxContactBytes),
                   "allocate actual reafference bytes");
      require_cuda(cudaMalloc(&actual_internal_, kMaxContactBytes),
                   "allocate actual internal bytes");
      host_inputs_.actual_reafference = actual_reafference_;
      host_inputs_.actual_internal = actual_internal_;
      host_inputs_.cue = cue_;
      require_cuda(cudaMalloc(&inputs_, sizeof(*inputs_)),
                   "allocate sensorimotor inputs");
      require_cuda(cudaMalloc(&prediction_, sizeof(*prediction_)),
                   "allocate prediction receipt");
      require_cuda(cudaMalloc(&consequence_, sizeof(*consequence_)),
                   "allocate consequence receipt");
      require_cuda(cudaMalloc(&transform_, sizeof(*transform_)),
                   "allocate transform receipt");
      require_cuda(cudaMalloc(&lesion_, sizeof(*lesion_)),
                   "allocate lesion receipt");
      require_cuda(cudaMalloc(&matter_, sizeof(*matter_)),
                   "allocate matter receipt");
      require_cuda(cudaMalloc(&relation_census_, sizeof(*relation_census_)),
                   "allocate relation census");
      require_cuda(cudaMalloc(&relation_endpoint_view_,
                              sizeof(*relation_endpoint_view_)),
                   "allocate relation endpoint view");
      upload_inputs();
      adult_.configure_sensorimotor_factor(
          layout_, inputs_, prediction_, consequence_, transform_);
    } catch (...) {
      release_buffers();
      throw;
    }
  }

  Tissue(const Tissue&) = delete;
  Tissue& operator=(const Tissue&) = delete;

  ~Tissue() {
    destroy_graphs();
    adult_.detach_sensorimotor_factor_buffers();
    release_buffers();
  }

  void prepare_resident(std::span<const std::uint8_t> cue) {
    if (prediction_pending_ || world_staged_) {
      throw std::logic_error(
          "sensorimotor prediction requires an idle contact");
    }
    copy_bytes(cue_, cue, "copy cue bytes");
    host_inputs_.cue_count = static_cast<std::uint32_t>(cue.size());
    host_inputs_.staged = 1u;
    host_inputs_.transform_mode = 0u;
    upload_inputs();
    adult_.develop(1u);
    // Resident callers intentionally defer observing the selected relation.
    // The pending flag keeps the ordinary action transaction live until a
    // later host observation refines it or cancellation reverses this tick.
    prediction_pending_ = true;
    pending_action_ = kNoAction;
  }

  PredictionReceipt observe_prediction() const { return copy(prediction_); }

  PredictionReceipt prepare(std::span<const std::uint8_t> cue) {
    prepare_resident(cue);
    const PredictionReceipt result = observe_prediction();
    prediction_pending_ = result.action != kNoAction;
    pending_action_ = result.action;
    return result;
  }

  void stage_consequence(std::span<const std::uint8_t> reafference,
                         std::span<const std::uint8_t> internal) {
    if (!prediction_pending_ || world_staged_) {
      throw std::logic_error(
          "sensorimotor consequence requires one eligible resident action");
    }
    copy_bytes(actual_reafference_, reafference, "copy actual reafference");
    copy_bytes(actual_internal_, internal, "copy actual internal contact");
    host_inputs_.actual_reafference_count =
        static_cast<std::uint32_t>(reafference.size());
    host_inputs_.actual_internal_count =
        static_cast<std::uint32_t>(internal.size());
    host_inputs_.staged = 1u;
    host_inputs_.transform_mode = 0u;
    upload_inputs();
    world_staged_ = true;
  }

  void realize_world() {
    if (!prediction_pending_ || !world_staged_) {
      throw std::logic_error(
          "sensorimotor outcome requires one actual staged consequence");
    }
    adult_.develop(1u);
    prediction_pending_ = false;
    pending_action_ = kNoAction;
    world_staged_ = false;
    consequence_ready_ = true;
  }

  ConsequenceReceipt settle() {
    if (!consequence_ready_) {
      throw std::logic_error(
          "sensorimotor consequence was not advanced by ordinary develop");
    }
    consequence_ready_ = false;
    const ConsequenceReceipt result = copy(consequence_);
    host_inputs_.staged = 0u;
    host_inputs_.actual_reafference_count = 0u;
    host_inputs_.actual_internal_count = 0u;
    upload_inputs();
    return result;
  }

  // The pointer is valid only between stage_consequence/realize_world and
  // settle.  Callers use it to admit the same returned contact to another
  // device learner without creating a host-side answer copy.
  const std::uint8_t* actual_reafference_device() const {
    if (!world_staged_ && !consequence_ready_)
      throw std::logic_error("actual reafference is not currently staged");
    return actual_reafference_;
  }

  std::uint32_t actual_reafference_count() const {
    if (!world_staged_ && !consequence_ready_)
      throw std::logic_error("actual reafference count is not currently staged");
    return host_inputs_.actual_reafference_count;
  }

  void cancel_prediction() {
    if (!prediction_pending_) return;
    adult_.reverse(1u);
    host_inputs_.staged = 0u;
    host_inputs_.transform_mode = 0u;
    upload_inputs();
    prediction_pending_ = false;
    pending_action_ = kNoAction;
    world_staged_ = false;
  }

  ConsequenceReceipt develop_empty(std::uint32_t ticks) {
    if (world_staged_) {
      throw std::logic_error(
          "empty development cannot bypass an actual consequence");
    }
    ConsequenceReceipt last{};
    for (std::uint32_t tick = 0u; tick < ticks; ++tick) {
      adult_.develop(1u);
      last = copy(consequence_);
      if (last.expired != 0u) {
        prediction_pending_ = false;
        pending_action_ = kNoAction;
        host_inputs_.staged = 0u;
        upload_inputs();
      }
    }
    return last;
  }

  TransformReceipt learn_transform(std::uint8_t input,
                                   std::uint8_t observed) {
    require_idle_transform();
    host_inputs_.transform_input = input;
    host_inputs_.transform_observed = observed;
    host_inputs_.transform_mode = 1u;
    upload_inputs();
    adult_.develop(1u);
    return copy(transform_);
  }

  TransformReceipt predict_transform(std::uint8_t input) {
    require_idle_transform();
    host_inputs_.transform_input = input;
    host_inputs_.transform_observed = 0u;
    host_inputs_.transform_mode = 2u;
    upload_inputs();
    adult_.develop(1u);
    return copy(transform_);
  }

  void withdraw_transform_source() {
    require_idle_transform();
    host_inputs_.transform_input = 0u;
    host_inputs_.transform_observed = 0u;
    host_inputs_.transform_mode = 0u;
    upload_inputs();
  }

  LesionReceipt set_reafference(bool enabled) {
    return set_path(factor::kReafferencePath, enabled);
  }

  LesionReceipt set_somatic(std::uint32_t action, bool enabled) {
    return set_path(action == 0u ? factor::kSomaticPath0
                                 : factor::kSomaticPath1,
                    enabled);
  }

  LesionReceipt set_action(std::uint32_t action, bool enabled) {
    return set_path(action == 0u ? factor::kRouteEnabled0
                                 : factor::kRouteEnabled1,
                    enabled);
  }

  LesionReceipt set_remote(bool enabled) {
    return set_path(factor::kRemoteControl, enabled);
  }

  LesionReceipt lesion_relation(std::uint32_t slot) {
    if (slot >= factor::kRelationSlotCount) {
      throw std::out_of_range("sensorimotor relation lesion is outside count");
    }
    if (prediction_pending_) {
      throw std::logic_error(
          "cannot lesion relation during eligible prediction");
    }
    // ⛔ WAS A DIRECT KERNEL LAUNCH ON THE WORLD POINTER. A lesion is an
    // experimenter's manipulation, not something the organism does, so it goes
    // through the adult's intervention broker: counted, epoch-stamped, and the
    // pointer never leaves the operation.
    adult_.intervene(InterventionReason::lesion_relation,
                     [&](SiteWord* words) {
                       void* arguments[] = {
                           kernel_argument(words), kernel_argument(layout_device_),
                           kernel_argument(slot), kernel_argument(lesion_)};
                       launch_graph(
                           lesion_relation_graph_,
                           "launch sensorimotor relation-lesion graph",
                           reinterpret_cast<void*>(factor::lesion_relation_kernel),
                           dim3(1u), dim3(1u), arguments);
                     });
    synchronize("lesion sensorimotor relation");
    return copy(lesion_);
  }

  LesionReceipt perturb_matched_remote(std::uint32_t physical_changed_bits) {
    if (prediction_pending_) {
      throw std::logic_error(
          "cannot perturb remote tissue during eligible prediction");
    }
    adult_.intervene(InterventionReason::matched_remote_perturbation,
                     [&](SiteWord* words) {
                       void* arguments[] = {
                           kernel_argument(words), kernel_argument(layout_device_),
                           kernel_argument(physical_changed_bits),
                           kernel_argument(lesion_)};
                       launch_graph(
                           matched_remote_graph_,
                           "launch matched-remote perturbation graph",
                           reinterpret_cast<void*>(
                               factor::matched_remote_perturbation_kernel),
                           dim3(1u), dim3(1u), arguments);
                     });
    synchronize("perturb matched remote sensorimotor tissue");
    return copy(lesion_);
  }

  RelationCensus relation_census() {
    const SiteWord* words = adult_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_device_),
                         kernel_argument(relation_census_)};
    launch_graph(relation_census_graph_, "launch relation-census graph",
                 reinterpret_cast<void*>(factor::relation_census_kernel),
                 dim3(1u), dim3(1u), arguments);
    synchronize("census sensorimotor relations");
    return copy(relation_census_);
  }

  RelationEndpointView relation_endpoint_view() {
    const SiteWord* words = adult_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_device_),
                         kernel_argument(relation_endpoint_view_)};
    launch_graph(endpoint_view_graph_, "launch relation-endpoint graph",
                 reinterpret_cast<void*>(factor::relation_endpoint_view_kernel),
                 dim3(1u), dim3(1u), arguments);
    synchronize("view sensorimotor relation endpoints");
    return copy(relation_endpoint_view_);
  }

  std::uint32_t matter() {
    const SiteWord* words = adult_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_device_),
                         kernel_argument(matter_)};
    launch_graph(census_graph_, "launch sensorimotor census graph",
                 reinterpret_cast<void*>(factor::census_kernel), dim3(1u),
                 dim3(1u), arguments);
    synchronize("census sensorimotor matter");
    return copy(matter_);
  }

  std::uint32_t matter_with_context() {
    const SiteWord* words = adult_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_device_),
                         kernel_argument(matter_)};
    launch_graph(census_with_context_graph_,
                 "launch sensorimotor context-census graph",
                 reinterpret_cast<void*>(factor::census_with_context_kernel),
                 dim3(1u), dim3(1u), arguments);
    synchronize("census sensorimotor and instance matter");
    return copy(matter_);
  }

  std::uint32_t resident_motor_action() {
    const SiteWord* words = adult_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_device_),
                         kernel_argument(matter_)};
    launch_graph(motor_probe_graph_, "launch resident motor-probe graph",
                 reinterpret_cast<void*>(factor::motor_probe_kernel), dim3(1u),
                 dim3(1u), arguments);
    synchronize("probe resident sensorimotor action");
    return copy(matter_);
  }

  std::uint32_t remote_probe() {
    const SiteWord* words = adult_.device_words();
    void* arguments[] = {kernel_argument(words), kernel_argument(layout_device_),
                         kernel_argument(matter_)};
    launch_graph(remote_probe_graph_, "launch remote-probe graph",
                 reinterpret_cast<void*>(factor::remote_probe_kernel),
                 dim3(1u), dim3(1u), arguments);
    synchronize("probe remote control route");
    return copy(matter_);
  }

  [[nodiscard]] const DeviceLayout& layout() const { return layout_; }

  [[nodiscard]] const DeviceLayout* device_layout() const {
    return layout_device_;
  }

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

  static void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string(operation) + ": " +
                               cudaGetErrorString(status));
    }
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

  static void copy_bytes(std::uint8_t* destination,
                         std::span<const std::uint8_t> bytes,
                         const char* operation) {
    if (bytes.size() > kMaxContactBytes) {
      throw std::invalid_argument("sensorimotor contact exceeds resident cap");
    }
    if (!bytes.empty()) {
      require_cuda(cudaMemcpy(destination, bytes.data(), bytes.size(),
                              cudaMemcpyHostToDevice),
                   operation);
    }
  }

  template <typename T>
  static T copy(const T* device) {
    T host{};
    require_cuda(cudaMemcpy(&host, device, sizeof(host),
                            cudaMemcpyDeviceToHost),
                 "copy sensorimotor receipt");
    return host;
  }

  void upload_inputs() {
    require_cuda(cudaMemcpy(inputs_, &host_inputs_, sizeof(host_inputs_),
                            cudaMemcpyHostToDevice),
                 "upload represented sensorimotor contacts");
  }

  void require_idle_transform() const {
    if (prediction_pending_ || consequence_ready_) {
      throw std::logic_error(
          "bit-lane transformation cannot bypass eligible action credit");
    }
  }

  void destroy_graphs() {
    lesion_relation_graph_.destroy();
    matched_remote_graph_.destroy();
    relation_census_graph_.destroy();
    endpoint_view_graph_.destroy();
    census_graph_.destroy();
    census_with_context_graph_.destroy();
    motor_probe_graph_.destroy();
    remote_probe_graph_.destroy();
    set_path_graph_.destroy();
  }

  LesionReceipt set_path(Rail rail, bool enabled) {
    if (prediction_pending_) {
      throw std::logic_error(
          "cannot lesion sensorimotor route during eligible prediction");
    }
    adult_.intervene(InterventionReason::route_ablation,
                     [&](SiteWord* words) {
                       const std::uint32_t enabled_value = enabled ? 1u : 0u;
                       void* arguments[] = {
                           kernel_argument(words), kernel_argument(layout_device_),
                           kernel_argument(rail), kernel_argument(enabled_value),
                           kernel_argument(lesion_)};
                       launch_graph(set_path_graph_, "launch sensorimotor path graph",
                                    reinterpret_cast<void*>(factor::set_path_kernel),
                                    dim3(1u), dim3(1u), arguments);
                     });
    synchronize("change sensorimotor path");
    return copy(lesion_);
  }

  void release_buffers() noexcept {
    (void)cudaFree(matter_);
    (void)cudaFree(relation_census_);
    (void)cudaFree(relation_endpoint_view_);
    (void)cudaFree(lesion_);
    (void)cudaFree(consequence_);
    (void)cudaFree(transform_);
    (void)cudaFree(prediction_);
    (void)cudaFree(inputs_);
    (void)cudaFree(actual_internal_);
    (void)cudaFree(actual_reafference_);
    (void)cudaFree(cue_);
    (void)cudaFree(layout_device_);
    matter_ = nullptr;
    relation_census_ = nullptr;
    relation_endpoint_view_ = nullptr;
    lesion_ = nullptr;
    consequence_ = nullptr;
    transform_ = nullptr;
    prediction_ = nullptr;
    inputs_ = nullptr;
    actual_internal_ = nullptr;
    actual_reafference_ = nullptr;
    cue_ = nullptr;
    layout_device_ = nullptr;
  }

  GrownAdult& adult_;
  DeviceLayout layout_{};
  DeviceLayout* layout_device_ = nullptr;
  DeviceInputs host_inputs_{};
  std::uint8_t* cue_ = nullptr;
  std::uint8_t* actual_reafference_ = nullptr;
  std::uint8_t* actual_internal_ = nullptr;
  DeviceInputs* inputs_ = nullptr;
  PredictionReceipt* prediction_ = nullptr;
  ConsequenceReceipt* consequence_ = nullptr;
  TransformReceipt* transform_ = nullptr;
  LesionReceipt* lesion_ = nullptr;
  std::uint32_t* matter_ = nullptr;
  RelationCensus* relation_census_ = nullptr;
  RelationEndpointView* relation_endpoint_view_ = nullptr;
  bool world_staged_ = false;
  bool prediction_pending_ = false;
  std::uint32_t pending_action_ = kNoAction;
  bool consequence_ready_ = false;
  KernelGraph lesion_relation_graph_;
  KernelGraph matched_remote_graph_;
  KernelGraph relation_census_graph_;
  KernelGraph endpoint_view_graph_;
  KernelGraph census_graph_;
  KernelGraph census_with_context_graph_;
  KernelGraph motor_probe_graph_;
  KernelGraph remote_probe_graph_;
  KernelGraph set_path_graph_;
};

}  // namespace substrate::bcc32::grown_sensorimotor_tissue

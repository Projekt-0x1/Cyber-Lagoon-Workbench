#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/bcc32_grown_form_trajectory.cuh"
#include "hardware_native/bcc32_grown_sensorimotor_tissue.cuh"

namespace substrate::bcc32::grown_directional_clause {

using substrate::bcc32::ResidentStageReason;

namespace adult = developmental_adult;
namespace form = grown_form_trajectory;
namespace sensorimotor = grown_sensorimotor_tissue;
namespace factor = grown_sensorimotor_factor;
using adult::GrownAdult;
using substrate::bcc32::RawByteDecode;
using substrate::bcc32::RawByteRails;
using substrate::bcc32::SiteWord;

inline constexpr std::uint32_t kEntityFormBytes = 3u;
inline constexpr std::uint32_t kClauseInputBytes = 6u;
inline constexpr std::uint32_t kCaptureForms = kClauseInputBytes / kEntityFormBytes;
inline constexpr std::uint32_t kCaptureThreads = kCaptureForms;
inline constexpr std::uint32_t kMaxClauseOutputBytes = form::kMaxOutputPhases;
inline constexpr std::uint32_t kGenerateThreads = kMaxClauseOutputBytes;
inline constexpr std::uint32_t kCaptureMask = form::kDirectionalCaptureMask;

struct CaptureReceipt {
  std::uint32_t valid = 0u;
  std::uint32_t direction_valid = 0u;
  SiteWord changed = 0u;
  SiteWord unchanged = 0u;
  std::uint32_t parallel_forms_eligible = 0u;
  std::uint32_t parallel_forms_completed = 0u;
  std::uint32_t parallel_form_mask = 0u;
  std::uint64_t parallel_launches = 0u;
};

struct ClauseReceipt {
  std::uint32_t valid = 0u;
  std::uint32_t direction_valid = 0u;
  std::uint32_t changed_form_valid = 0u;
  std::uint32_t unchanged_form_valid = 0u;
  std::uint32_t clause_valid = 0u;
  std::uint32_t output_count = 0u;
  std::uint32_t failed_phase = 0xffffffffu;
  std::uint32_t phase_conflict = 0u;
  SiteWord changed = 0u;
  SiteWord unchanged = 0u;
  std::uint8_t changed_form[kEntityFormBytes]{};
  std::uint8_t unchanged_form[kEntityFormBytes]{};
  std::uint8_t input[kClauseInputBytes]{};
  std::uint8_t output[kMaxClauseOutputBytes]{};
  std::uint32_t parallel_phases_eligible = 0u;
  std::uint32_t parallel_phases_completed = 0u;
  std::uint32_t parallel_phase_mask = 0u;
  std::uint64_t parallel_launches = 0u;
};

struct EmissionReceipt {
  std::uint32_t valid = 0u;
  std::uint32_t phase = 0xffffffffu;
  std::uint8_t value = 0u;
};

__device__ inline bool qualified_direction(
    const SiteWord* words, const factor::DeviceLayout& layout,
    SiteWord* changed, SiteWord* unchanged) {
  const std::uint32_t slot = factor::read_value(
      words, layout, factor::kActiveRelationSlot);
  if (slot >= factor::kRelationSlotCount) return false;
  const SiteWord context = factor::read_value(
      words, layout, factor::kActiveContext);
  const SiteWord cue = factor::read_value(words, layout, factor::kActiveCue);
  const SiteWord motor = factor::read_value(words, layout, factor::kMotor);
  const std::uint32_t action = factor::read_value(
      words, layout,
      factor::relation_rail(slot, factor::kRelationAction)) & 1u;
  const factor::DirectionalEvidence current =
      factor::derive_directional_evidence(words, layout);
  const auto relation_value = [&](factor::RelationField field) {
    return factor::read_value(words, layout, factor::relation_rail(slot, field));
  };
  const bool qualified =
      relation_value(factor::kRelationOccupied) != 0u &&
      relation_value(factor::kRelationContext) == context &&
      relation_value(factor::kRelationCue) == cue &&
      motor == (1u << action) &&
      relation_value(factor::kRelationPositive) != 0u &&
      relation_value(factor::kRelationNegative) == 0u &&
      current.changed == relation_value(factor::kRelationChanged) &&
      current.unchanged == relation_value(factor::kRelationUnchanged);
  if (!qualified) return false;
  if (changed != nullptr) *changed = current.changed;
  if (unchanged != nullptr) *unchanged = current.unchanged;
  return true;
}

__device__ inline bool appearance_for_mask(
    const SiteWord* words, const factor::DeviceLayout& layout, SiteWord mask,
    SiteWord* appearance) {
  if (__popc(mask) != 1) return false;
  const std::uint32_t basin =
      static_cast<std::uint32_t>(__ffs(static_cast<int>(mask)) - 1);
  if (basin >= factor::instance::kBasinCount ||
      factor::instance::read_field(
          words, layout.context,
          factor::instance::basin_field(basin, factor::instance::kActive)) ==
          0u ||
      factor::instance::read_field(
          words, layout.context,
          factor::instance::basin_field(basin,
                                         factor::instance::kMissingAge)) != 0u)
    return false;
  *appearance = factor::instance::read_field(
      words, layout.context,
      factor::instance::basin_field(basin, factor::instance::kAppearance));
  return *appearance != 0u;
}

__device__ inline bool reconstruct_form(
    const SiteWord* words, const form::DeviceLayout& layout,
    SiteWord appearance, std::uint8_t* output) {
  const std::uint64_t active = form::appearance_coalition_mask(appearance);
  std::uint32_t output_count = 0u;
  for (std::uint32_t region = 0u; region < form::situation::kRegionCount;
       ++region) {
    if ((active & (1ull << region)) == 0u) continue;
    const SiteWord lengths = form::read_value(
        words, layout, region, form::kLengthField);
    if (__popc(lengths) != 1u) return false;
    const std::uint32_t count =
        static_cast<std::uint32_t>(__ffs(static_cast<int>(lengths)));
    if (count != kEntityFormBytes) return false;
    output_count = count;
  }
  if (output_count != kEntityFormBytes) return false;

  const std::uint8_t probe[kEntityFormBytes] = {0u, 0u, 0u};
  for (std::uint32_t phase = 0u; phase < kEntityFormBytes; ++phase) {
    bool valid = false;
    std::uint8_t candidate = 0u;
    for (std::uint32_t region = 0u; region < form::situation::kRegionCount;
         ++region) {
      if ((active & (1ull << region)) == 0u) continue;
      std::uint8_t local = 0u;
      bool mapped = false;
      if (!form::generate_region_phase(words, layout, region, phase, probe,
                                       kEntityFormBytes, &local, &mapped))
        continue;
      if (valid && local != candidate) return false;
      candidate = local;
      valid = true;
    }
    if (!valid) return false;
    output[phase] = candidate;
  }
  return true;
}

static __global__ void capture_directional_input_kernel(
    const SiteWord* words, const factor::DeviceLayout* sensor_layout,
    const form::DeviceLayout* form_layout, std::uint8_t* captured,
    std::uint32_t* valid_mask, CaptureReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x >= kCaptureThreads) return;
  __shared__ std::uint32_t direction_valid_shared;
  __shared__ SiteWord changed_shared;
  __shared__ SiteWord unchanged_shared;
  __shared__ SiteWord appearances_shared[kCaptureForms];
  __shared__ std::uint32_t appearance_mask_shared;
  __shared__ std::uint32_t completed_mask_shared;

  if (threadIdx.x == 0u) {
    direction_valid_shared = 0u;
    changed_shared = 0u;
    unchanged_shared = 0u;
    appearance_mask_shared = 0u;
    completed_mask_shared = 0u;
    for (std::uint32_t index = 0u; index < kClauseInputBytes; ++index)
      captured[index] = 0u;
    if (valid_mask != nullptr) *valid_mask &= ~kCaptureMask;
    if (sensor_layout != nullptr && form_layout != nullptr) {
      SiteWord changed = 0u;
      SiteWord unchanged = 0u;
      if (qualified_direction(words, *sensor_layout, &changed, &unchanged)) {
        direction_valid_shared = 1u;
        changed_shared = changed;
        unchanged_shared = unchanged;
      }
    }
  }
  __syncthreads();

  if (direction_valid_shared != 0u) {
    const std::uint32_t form_index = threadIdx.x;
    const SiteWord mask = form_index == 0u ? changed_shared : unchanged_shared;
    SiteWord appearance = 0u;
    if (appearance_for_mask(words, *sensor_layout, mask, &appearance)) {
      appearances_shared[form_index] = appearance;
      atomicOr(&appearance_mask_shared, 1u << form_index);
    }
  }
  __syncthreads();

  if (appearance_mask_shared == kCaptureMask) {
    const std::uint32_t form_index = threadIdx.x;
    if (reconstruct_form(words, *form_layout, appearances_shared[form_index],
                         captured + form_index * kEntityFormBytes))
      atomicOr(&completed_mask_shared, 1u << form_index);
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    CaptureReceipt local{};
    local.direction_valid = direction_valid_shared;
    local.changed = changed_shared;
    local.unchanged = unchanged_shared;
    local.parallel_forms_eligible =
        direction_valid_shared != 0u ? kCaptureForms : 0u;
    local.parallel_forms_completed = __popc(completed_mask_shared);
    local.parallel_form_mask = completed_mask_shared;
    local.parallel_launches = 1u;
    local.valid = completed_mask_shared == kCaptureMask ? 1u : 0u;
    if (local.valid != 0u && valid_mask != nullptr) *valid_mask |= kCaptureMask;
    if (receipt != nullptr) *receipt = local;
  }
}

static __global__ void generate_clause_kernel(
    const SiteWord* words, const factor::DeviceLayout* sensor_layout,
    const form::DeviceLayout* form_layout, ClauseReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x >= kGenerateThreads) return;
  __shared__ ClauseReceipt shared_receipt;
  __shared__ std::uint32_t setup_valid_shared;
  __shared__ std::uint32_t output_count_shared;
  __shared__ std::uint8_t phase_valid_shared[kMaxClauseOutputBytes];
  __shared__ std::uint8_t phase_conflict_shared[kMaxClauseOutputBytes];
  __shared__ std::uint8_t phase_value_shared[kMaxClauseOutputBytes];

  if (threadIdx.x == 0u) {
    shared_receipt = ClauseReceipt{};
    setup_valid_shared = 0u;
    output_count_shared = 0u;
    for (std::uint32_t phase = 0u; phase < kMaxClauseOutputBytes; ++phase) {
      phase_valid_shared[phase] = 0u;
      phase_conflict_shared[phase] = 0u;
      phase_value_shared[phase] = 0u;
    }
    if (sensor_layout != nullptr && form_layout != nullptr) {
      SiteWord changed = 0u;
      SiteWord unchanged = 0u;
      if (qualified_direction(words, *sensor_layout, &changed, &unchanged)) {
        shared_receipt.direction_valid = 1u;
        shared_receipt.changed = changed;
        shared_receipt.unchanged = unchanged;
        SiteWord changed_appearance = 0u;
        SiteWord unchanged_appearance = 0u;
        shared_receipt.changed_form_valid = appearance_for_mask(
            words, *sensor_layout, changed, &changed_appearance) &&
                                               reconstruct_form(
                                                   words, *form_layout,
                                                   changed_appearance,
                                                   shared_receipt.changed_form);
        shared_receipt.unchanged_form_valid = appearance_for_mask(
            words, *sensor_layout, unchanged, &unchanged_appearance) &&
                                                 reconstruct_form(
                                                     words, *form_layout,
                                                     unchanged_appearance,
                                                     shared_receipt.unchanged_form);
        if (shared_receipt.changed_form_valid != 0u &&
            shared_receipt.unchanged_form_valid != 0u) {
          for (std::uint32_t index = 0u; index < kEntityFormBytes; ++index) {
            shared_receipt.input[index] = shared_receipt.changed_form[index];
            shared_receipt.input[index + kEntityFormBytes] =
                shared_receipt.unchanged_form[index];
          }

          const std::uint64_t active = form::directional_clause_coalition_mask();
          std::uint32_t output_count = 0u;
          bool lengths_valid = true;
          for (std::uint32_t region = 0u;
               region < form::situation::kRegionCount; ++region) {
            if ((active & (1ull << region)) == 0u) continue;
            const SiteWord lengths = form::read_value(
                words, *form_layout, region, form::kLengthField);
            if (__popc(lengths) != 1u) {
              lengths_valid = false;
              break;
            }
            const std::uint32_t count =
                static_cast<std::uint32_t>(__ffs(static_cast<int>(lengths)));
            if (count == 0u || count > kMaxClauseOutputBytes ||
                (output_count != 0u && output_count != count)) {
              lengths_valid = false;
              break;
            }
            output_count = count;
          }
          if (lengths_valid && output_count != 0u &&
              output_count <= kMaxClauseOutputBytes) {
            // Retain the unanimous resident length even when a later byte
            // phase abstains; this is causal-stage evidence, not a validity
            // shortcut.
            shared_receipt.output_count = output_count;
            output_count_shared = output_count;
            setup_valid_shared = 1u;
          }
        }
      }
    }
  }
  __syncthreads();

  if (setup_valid_shared != 0u && threadIdx.x < output_count_shared) {
    const std::uint32_t phase = threadIdx.x;
    const std::uint64_t active = form::directional_clause_coalition_mask();
    bool valid = false;
    std::uint8_t candidate = 0u;
    std::uint8_t conflict = 0u;
    for (std::uint32_t region = 0u; region < form::situation::kRegionCount;
         ++region) {
      if ((active & (1ull << region)) == 0u) continue;
      std::uint8_t local_value = 0u;
      bool mapped = false;
      if (!form::generate_region_phase(
              words, *form_layout, region, phase, shared_receipt.input,
              kClauseInputBytes, &local_value, &mapped))
        continue;
      if (valid && local_value != candidate) {
        conflict = 1u;
        continue;
      }
      candidate = local_value;
      valid = true;
    }
    phase_valid_shared[phase] = valid && conflict == 0u ? 1u : 0u;
    phase_conflict_shared[phase] = conflict;
    phase_value_shared[phase] = candidate;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    shared_receipt.parallel_phases_eligible = output_count_shared;
    shared_receipt.parallel_launches = 1u;
    std::uint32_t completed = 0u;
    std::uint32_t phase_mask = 0u;
    bool all_valid = setup_valid_shared != 0u;
    for (std::uint32_t phase = 0u; phase < output_count_shared; ++phase) {
      if (phase_valid_shared[phase] != 0u) {
        shared_receipt.output[phase] = phase_value_shared[phase];
        ++completed;
        phase_mask |= 1u << phase;
      } else {
        all_valid = false;
        if (shared_receipt.failed_phase == 0xffffffffu) {
          shared_receipt.failed_phase = phase;
          shared_receipt.phase_conflict = phase_conflict_shared[phase];
        }
      }
    }
    shared_receipt.parallel_phases_completed = completed;
    shared_receipt.parallel_phase_mask = phase_mask;
    shared_receipt.clause_valid = all_valid ? 1u : 0u;
    shared_receipt.valid = all_valid ? 1u : 0u;
    if (receipt != nullptr) *receipt = shared_receipt;
  }
}

static __global__ void emit_next_kernel(
    SiteWord* words, std::uint64_t motor_zero_slot, std::uint64_t motor_one_slot,
    const ClauseReceipt* clause, std::uint32_t* cursor,
    EmissionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  EmissionReceipt local{};
  const std::uint32_t phase = *cursor;
  local.phase = phase;
  RawByteRails motor{words[motor_zero_slot], words[motor_one_slot]};
  if (clause != nullptr && clause->valid != 0u &&
      clause->output_count != 0u &&
      clause->output_count <= kMaxClauseOutputBytes &&
      phase < clause->output_count) {
    local.valid = 1u;
    local.value = clause->output[phase];
    motor = with_raw_byte_carriers(motor, local.value);
    *cursor = phase + 1u;
  } else {
    motor.zero = with_carriers(motor.zero, 0x0fu);
    motor.one = with_carriers(motor.one, 0x0fu);
  }
  words[motor_zero_slot] = motor.zero;
  words[motor_one_slot] = motor.one;
  if (receipt != nullptr) *receipt = local;
}

class DirectionalClauseBridge {
 public:
  DirectionalClauseBridge(GrownAdult& grown, sensorimotor::Tissue& sensor,
                          form::FormTrajectory& form)
      : grown_(grown), sensor_(sensor), form_(form) {
    require_cuda(cudaMalloc(&capture_receipt_, sizeof(*capture_receipt_)),
                 "allocate directional capture receipt");
    require_cuda(cudaMalloc(&clause_receipt_, sizeof(*clause_receipt_)),
                 "allocate directional clause receipt");
    require_cuda(cudaMalloc(&emission_receipt_, sizeof(*emission_receipt_)),
                 "allocate directional emission receipt");
    require_cuda(cudaMalloc(&cursor_, sizeof(*cursor_)),
                 "allocate directional emission cursor");
    build_graphs();
  }

  ~DirectionalClauseBridge() {
    if (emit_graph_ != nullptr) cudaGraphExecDestroy(emit_graph_);
    if (generate_graph_ != nullptr) cudaGraphExecDestroy(generate_graph_);
    if (capture_graph_ != nullptr) cudaGraphExecDestroy(capture_graph_);
    cudaFree(cursor_);
    cudaFree(emission_receipt_);
    cudaFree(clause_receipt_);
    cudaFree(capture_receipt_);
  }

  DirectionalClauseBridge(const DirectionalClauseBridge&) = delete;
  DirectionalClauseBridge& operator=(const DirectionalClauseBridge&) = delete;

  form::LearnReceipt learn_form(
      std::span<const std::uint8_t, kEntityFormBytes> output) {
    form_.bind_unique_active_instance(sensor_.device_layout());
    const std::array<std::uint8_t, form::kFormProbeCount> probe{0u, 0u, 0u};
    return form_.learn(
        std::span<const std::uint8_t>(probe.data(), probe.size()), output);
  }

  CaptureReceipt capture_input() {
    grown_.resident_stage(
        ResidentStageReason::form_edge, [&](SiteWord*) {
          launch_graph(capture_graph_, "launch directional capture graph");
        });
    synchronize("capture directional form input");
    return copy(capture_receipt_);
  }

  form::credit::Receipt teach_clause(
      std::span<const std::uint8_t> output,
      std::span<const std::uint8_t> reafference,
      std::span<const std::uint8_t> internal) {
    if (output.empty())
      throw std::invalid_argument(
          "directional clause requires a nonempty learned surface");
    if (output.size() > kMaxClauseOutputBytes)
      throw std::invalid_argument(
          "directional clause learned surface exceeds bounded output capacity");
    form_.bind_directional_clause(sensor_.device_layout());
    const CaptureReceipt captured = capture_input();
    if (captured.valid == 0u)
      throw std::logic_error("directional clause teaching lacked device forms");
    sensor_.stage_consequence(reafference, internal);
    sensor_.realize_world();
    const factor::ConsequenceReceipt consequence = sensor_.settle();
    if (consequence.processed == 0u)
      throw std::logic_error("directional clause teaching lacked consequence");
    form_.stage_captured_input(kClauseInputBytes, kCaptureMask);
    form_.ordinary_develop(1u);
    form_.stage_output(output);
    form_.ordinary_develop(1u);
    return form_.credit_receipt();
  }

  ClauseReceipt generate() {
    grown_.resident_stage(
        ResidentStageReason::form_edge, [&](SiteWord*) {
          require_cuda(cudaMemset(cursor_, 0, sizeof(*cursor_)),
                       "reset directional emission cursor");
          launch_graph(generate_graph_, "launch directional generation graph");
        });
    synchronize("generate directional clause");
    return copy(clause_receipt_);
  }

  RawByteDecode emit_next() {
    // Same shape as form_trajectory's motor staging: it lands on the declared
    // motor port slots and extract_motor_raw_byte performs the legacy reciprocal
    // boundary exchange. That API discards its MembraneReceipt, however, and
    // bcc32_membrane.hpp explicitly excludes that receipt from descended
    // production provenance. This is therefore measured motor contact, not a
    // canonical public-egress receipt.
    grown_.resident_stage(
        ResidentStageReason::stage_motor_byte, [&](SiteWord* words) {
          (void)words;
          launch_graph(emit_graph_, "launch directional emission graph");
        });
    synchronize("stage next directional clause byte");
    const EmissionReceipt emission = copy(emission_receipt_);
    if (emission.valid == 0u) return {};
    return grown_.extract_motor_raw_byte();
  }

  std::vector<std::uint8_t> emit_all() {
    const ClauseReceipt clause = copy(clause_receipt_);
    if (clause.valid == 0u || clause.clause_valid == 0u ||
        clause.output_count == 0u ||
        clause.output_count > kMaxClauseOutputBytes)
      throw std::logic_error("directional clause has no bounded learned surface");
    std::vector<std::uint8_t> result;
    result.reserve(clause.output_count);
    for (std::uint32_t phase = 0u; phase < clause.output_count; ++phase) {
      const RawByteDecode byte = emit_next();
      if (!byte.valid) throw std::logic_error("directional clause motor abstained");
      result.push_back(byte.value);
    }
    return result;
  }

 private:
  static void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " +
                               cudaGetErrorString(status));
  }

  static void synchronize(const char* operation) {
    require_cuda(cudaGetLastError(), operation);
    require_cuda(cudaDeviceSynchronize(), operation);
  }

  template <typename T>
  static void* kernel_argument(T& value) {
    return const_cast<void*>(static_cast<const void*>(&value));
  }

  static cudaGraphExec_t instantiate_kernel_graph(
      const char* operation, void* function, dim3 grid, dim3 block,
      void** arguments) {
    cudaGraph_t graph = nullptr;
    require_cuda(cudaGraphCreate(&graph, 0u), operation);

    cudaKernelNodeParams params{};
    params.func = function;
    params.gridDim = grid;
    params.blockDim = block;
    params.sharedMemBytes = 0u;
    params.kernelParams = arguments;

    cudaGraphNode_t node = nullptr;
    cudaError_t status = cudaGraphAddKernelNode(
        &node, graph, nullptr, 0u, &params);
    if (status != cudaSuccess) {
      cudaGraphDestroy(graph);
      require_cuda(status, operation);
    }

    cudaGraphExec_t executable = nullptr;
    status = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u);
    if (status != cudaSuccess) {
      cudaGraphDestroy(graph);
      require_cuda(status, operation);
    }
    status = cudaGraphDestroy(graph);
    if (status != cudaSuccess) {
      cudaGraphExecDestroy(executable);
      require_cuda(status, operation);
    }
    return executable;
  }

  static void launch_graph(cudaGraphExec_t graph, const char* operation) {
    require_cuda(cudaGraphLaunch(graph, 0), operation);
  }

  void build_graphs() {
    const SiteWord* words = grown_.device_words();
    SiteWord* mutable_words = nullptr;
    grown_.resident_stage(
        ResidentStageReason::stage_motor_byte,
        [&](SiteWord* staged_words) { mutable_words = staged_words; });
    const factor::DeviceLayout* sensor_layout = sensor_.device_layout();
    const form::DeviceLayout* form_layout = &form_.layout();
    std::uint8_t* captured = form_.device_captured_input();
    std::uint32_t* valid_mask = form_.device_capture_valid();
    CaptureReceipt* capture_receipt = capture_receipt_;
    void* capture_arguments[] = {
        kernel_argument(words), kernel_argument(sensor_layout),
        kernel_argument(form_layout), kernel_argument(captured),
        kernel_argument(valid_mask), kernel_argument(capture_receipt)};
    capture_graph_ = instantiate_kernel_graph(
        "instantiate directional capture graph",
        reinterpret_cast<void*>(capture_directional_input_kernel),
        dim3(1u), dim3(kCaptureThreads), capture_arguments);

    ClauseReceipt* clause_receipt = clause_receipt_;
    void* generate_arguments[] = {
        kernel_argument(words), kernel_argument(sensor_layout),
        kernel_argument(form_layout), kernel_argument(clause_receipt)};
    generate_graph_ = instantiate_kernel_graph(
        "instantiate directional generation graph",
        reinterpret_cast<void*>(generate_clause_kernel), dim3(1u),
        dim3(kGenerateThreads), generate_arguments);

    const std::uint64_t motor_zero_slot =
        grown_.boundary_port_slot(adult::kRawMotorZeroPort);
    const std::uint64_t motor_one_slot =
        grown_.boundary_port_slot(adult::kRawMotorOnePort);
    const ClauseReceipt* clause = clause_receipt_;
    std::uint32_t* cursor = cursor_;
    EmissionReceipt* emission_receipt = emission_receipt_;
    void* emit_arguments[] = {
        kernel_argument(mutable_words), kernel_argument(motor_zero_slot),
        kernel_argument(motor_one_slot), kernel_argument(clause),
        kernel_argument(cursor), kernel_argument(emission_receipt)};
    emit_graph_ = instantiate_kernel_graph(
        "instantiate directional emission graph",
        reinterpret_cast<void*>(emit_next_kernel), dim3(1u), dim3(1u),
        emit_arguments);
  }

  template <typename T>
  static T copy(const T* device) {
    T host{};
    require_cuda(cudaMemcpy(&host, device, sizeof(host), cudaMemcpyDeviceToHost),
                 "copy directional clause receipt");
    return host;
  }

  cudaGraphExec_t capture_graph_ = nullptr;
  cudaGraphExec_t generate_graph_ = nullptr;
  cudaGraphExec_t emit_graph_ = nullptr;
  GrownAdult& grown_;
  sensorimotor::Tissue& sensor_;
  form::FormTrajectory& form_;
  CaptureReceipt* capture_receipt_ = nullptr;
  ClauseReceipt* clause_receipt_ = nullptr;
  EmissionReceipt* emission_receipt_ = nullptr;
  std::uint32_t* cursor_ = nullptr;
};

}  // namespace substrate::bcc32::grown_directional_clause

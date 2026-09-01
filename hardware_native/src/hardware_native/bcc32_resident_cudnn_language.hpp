#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace substrate::bcc32::resident_language {

inline constexpr std::size_t kContextWidth = 8u;

// The geometry is deliberately fixed to the measured consumer-GPU training
// shape.  It keeps graph topology stable and leaves all examples and targets
// in the resident byte tape rather than in host-side training structures.
struct Config {
  std::uint32_t train_steps = 700u;
  std::uint32_t batch = 96u;
  std::uint32_t sequence = 192u;
  float learning_rate = 2.0e-3f;
  std::size_t tape_capacity = 1u << 20u;
};

enum class LesionKind : std::uint32_t {
  recurrent = 0u,
  matched_embedding_readout = 1u,
  context_projection = 2u,
  source_tape = 3u,
};

struct Receipt {
  float first_loss = 0.0f;
  float final_loss = 0.0f;
  float heldout_loss = 0.0f;
  float unigram_loss = 0.0f;
  std::vector<std::uint8_t> output_bytes;
  std::uint64_t host_bootstrap_launches = 0u;
  std::uint32_t graph_instantiated_device_launch = 0u;
  std::uint32_t graph_tail_launch = 0u;
  std::uint32_t custom_weight_gradient = 1u;
  std::uint32_t trained_steps = 0u;
  float context_gradient_l1 = 0.0f;
};

// Opaque integration seam.  The enclosing persistent runtime may retain these
// handles and schedule the graph from its own device-owned epoch.  The fields
// are intentionally untyped here: this header does not expose model memory or
// a semantic mutation API.
struct DeviceLaunchHandle {
  void* training_graph_exec = nullptr;
  void* generation_graph_exec = nullptr;
  void* device_state = nullptr;
  void* prompt_bytes = nullptr;
  void* prompt_size = nullptr;
  void* requested_output_size = nullptr;
  void* output_bytes = nullptr;
  void* output_size = nullptr;
  // Device receipt scalar: the last completed training target.
  void* trained_steps = nullptr;
  // Immutable founder target. The parent must use writable_train_steps for
  // same-adult online extension rather than changing this field.
  void* configured_train_steps = nullptr;
  void* tape_bytes = nullptr;
  // Device-owned mutable tape size. The parent may append into the reserved
  // tape/context arrays and publish this scalar from its own graph.
  void* tape_size = nullptr;
  void* train_size = nullptr;
  void* heldout_start = nullptr;
  void* byte_counts = nullptr;
  // Opaque device-owned context seam. The parent may write the current
  // generation/training context and per-byte snapshots from its own graph;
  // no semantic labels or model storage are exposed.
  void* context_vector = nullptr;
  void* context_tape = nullptr;
  void* tape_capacity = nullptr;
  void* train_start = nullptr;
  // Device progress scalar. This is distinct from the last completed receipt.
  void* step = nullptr;
  // Mutable target consumed by the resident graph tail. Raising it and
  // relaunching training_graph_exec continues the same adult online.
  void* writable_train_steps = nullptr;
};

static_assert(std::is_standard_layout_v<DeviceLaunchHandle>);
static_assert(offsetof(DeviceLaunchHandle, tape_size) <
              offsetof(DeviceLaunchHandle, context_vector));
static_assert(offsetof(DeviceLaunchHandle, context_tape) <
              offsetof(DeviceLaunchHandle, tape_capacity));
static_assert(offsetof(DeviceLaunchHandle, train_start) <
              offsetof(DeviceLaunchHandle, step));
static_assert(offsetof(DeviceLaunchHandle, step) <
              offsetof(DeviceLaunchHandle, writable_train_steps));

class ResidentGruLanguage final {
 public:
  struct Impl;

  explicit ResidentGruLanguage(std::uint64_t founder,
                               Config config = Config{});
  ~ResidentGruLanguage();

  ResidentGruLanguage(const ResidentGruLanguage&) = delete;
  ResidentGruLanguage& operator=(const ResidentGruLanguage&) = delete;
  ResidentGruLanguage(ResidentGruLanguage&&) = delete;
  ResidentGruLanguage& operator=(ResidentGruLanguage&&) = delete;

  // Raw sensory ingress.  The caller supplies bytes only; targets are made
  // from the resident tape by device code during a training graph step.
  void present_raw(std::span<const std::uint8_t> bytes);

  // Generic numeric bootstrap for contextual tape segments.  Context is a
  // bounded physical vector repeated beside each byte; it carries no labels.
  void present_contextual(std::span<const std::uint8_t> bytes,
                          std::span<const float> context);

  // One host bootstrap launches the complete resident training graph.  The
  // graph tail relaunches itself until Config::train_steps is reached.
  void train_autonomous();

  // Device-side autoregressive production from raw prompt bytes.  The return
  // value is a passive copy of generated boundary bytes.
  [[nodiscard]] std::vector<std::uint8_t> generate(
      std::span<const std::uint8_t> prompt, std::size_t max_bytes);

  [[nodiscard]] std::vector<std::uint8_t> generate_with_context(
      std::span<const std::uint8_t> prompt,
      std::span<const float> context, std::size_t max_bytes);

  // Test-only physical intervention. It zeros a bounded resident parameter or
  // withdraws the raw source tape; it is excluded from normal learning.
  void apply_test_lesion(LesionKind kind, std::size_t offset,
                         std::size_t count);

  // Integration-only handle exchange.  The parent graph is an opaque CUDA
  // graph-exec pointer supplied by the enclosing persistent runtime; after
  // the final training epoch the resident graph tail-launches that parent.
  [[nodiscard]] DeviceLaunchHandle device_launch_handle() const noexcept;
  void attach_return_graph(void* parent_graph_exec);

  [[nodiscard]] Receipt read_receipt() const;
  [[nodiscard]] std::size_t tape_size() const noexcept { return tape_size_; }

 private:
  Impl* impl_ = nullptr;
  std::size_t tape_size_ = 0u;
};

}  // namespace substrate::bcc32::resident_language

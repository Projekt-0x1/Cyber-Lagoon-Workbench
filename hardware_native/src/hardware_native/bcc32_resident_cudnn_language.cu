#include "bcc32_resident_cudnn_language.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace substrate::bcc32::resident_language {
namespace {

constexpr int kVocab = 256;
constexpr int kInput = 96;
constexpr int kHidden = 384;
constexpr int kGates = 3 * kHidden;
constexpr int kReadout = kVocab;
constexpr int kBatch = 96;
constexpr int kSequence = 192;
constexpr int kContext = static_cast<int>(kContextWidth);
constexpr std::size_t kMaxGeneration = 4096u;
constexpr std::uint32_t kTrainingSentinel = 0xffffffffu;
constexpr std::uint32_t kEvaluationSentinel = 0xfffffffeu;
void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + ": cublas status " +
                             std::to_string(static_cast<int>(status)));
}

struct DeviceReceipt {
  float first_loss = 0.0f;
  float final_loss = 0.0f;
  float heldout_loss = 0.0f;
  float unigram_loss = 0.0f;
  std::uint32_t output_size = 0u;
  std::uint32_t trained_steps = 0u;
  float context_gradient_l1 = 0.0f;
};

struct DeviceState {
  std::uint8_t* tape = nullptr;
  float* context_tape = nullptr;
  std::uint32_t tape_size = 0u;
  std::uint32_t train_size = 0u;
  std::uint32_t train_start = 0u;
  std::uint32_t heldout_start = 0u;
  std::uint32_t tape_capacity = 0u;
  std::uint32_t step = 0u;
  std::uint32_t train_steps = 0u;
  std::uint32_t configured_train_steps = 0u;
  float learning_rate = 0.0f;
  std::uint64_t founder = 0u;
  unsigned long long* counts = nullptr;
  float* embedding = nullptr;
  float* w_ih = nullptr;
  float* w_context = nullptr;
  float* w_hh = nullptr;
  float* bias = nullptr;
  float* readout = nullptr;
  float* readout_bias = nullptr;
  float* m_embedding = nullptr;
  float* v_embedding = nullptr;
  float* m_w_ih = nullptr;
  float* v_w_ih = nullptr;
  float* m_w_context = nullptr;
  float* v_w_context = nullptr;
  float* m_w_hh = nullptr;
  float* v_w_hh = nullptr;
  float* m_bias = nullptr;
  float* v_bias = nullptr;
  float* m_readout = nullptr;
  float* v_readout = nullptr;
  float* m_readout_bias = nullptr;
  float* v_readout_bias = nullptr;
  float* d_embedding = nullptr;
  float* d_w_ih = nullptr;
  float* d_w_context = nullptr;
  float* d_w_hh = nullptr;
  float* d_bias = nullptr;
  float* d_readout = nullptr;
  float* d_readout_bias = nullptr;
  float* x = nullptr;
  float* context_x = nullptr;
  std::uint8_t* targets = nullptr;
  float* h = nullptr;
  float* gates = nullptr;
  float* recurrent_n = nullptr;
  float* pre = nullptr;
  float* logits = nullptr;
  float* d_logits = nullptr;
  float* d_h_out = nullptr;
  float* d_h_next = nullptr;
  float* d_q = nullptr;
  float* d_pre = nullptr;
  float* d_x = nullptr;
  float* sample_loss = nullptr;
  float* step_loss = nullptr;
  float* eval_loss = nullptr;
  float* unigram_loss = nullptr;
  float* context_gradient_l1 = nullptr;
  DeviceReceipt* receipt = nullptr;
  std::uint8_t* generation_prompt = nullptr;
  std::uint32_t generation_prompt_size = 0u;
  std::uint32_t generation_output_size = 0u;
  std::uint8_t* output = nullptr;
  std::uint32_t output_capacity = 0u;
  float* live_context = nullptr;
  cudaGraphExec_t return_graph = nullptr;
};

__device__ std::uint32_t mix32(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31u;
  return static_cast<std::uint32_t>(value);
}

__global__ void seed_parameter(float* values, std::size_t count,
                               std::uint64_t seed, float scale) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    const std::uint32_t bits = mix32(seed + i * 0x9e3779b97f4a7c15ULL);
    const float unit = static_cast<float>(bits & 0x00ffffffu) /
                       static_cast<float>(0x01000000u);
    values[i] = (unit * 2.0f - 1.0f) * scale;
  }
}

__global__ void seed_normal_parameter(float* values, std::size_t count,
                                      std::uint64_t seed, float scale) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    const float u1 = (static_cast<float>(mix32(seed + 2u * i)) + 1.0f) /
                     4294967297.0f;
    const float u2 = (static_cast<float>(mix32(seed + 2u * i + 1u)) + 0.5f) /
                     4294967296.0f;
    values[i] = scale * sqrtf(-2.0f * logf(u1)) *
                cosf(6.2831853071795864769f * u2);
  }
}

__global__ void count_tape(const std::uint8_t* tape, std::uint32_t start,
                           std::uint32_t end, unsigned long long* counts) {
  for (std::uint32_t i = start + blockIdx.x * blockDim.x + threadIdx.x;
       i < end; i += blockDim.x * gridDim.x)
      atomicAdd(&counts[tape[i]], 1ULL);
}

__device__ std::uint32_t evaluation_begin(const DeviceState* state) {
  return state->heldout_start < state->tape_size ? state->heldout_start : 0u;
}

__device__ std::uint32_t batch_source(const DeviceState* state,
                                      std::uint32_t base, int t, int b) {
  const bool training = base == kTrainingSentinel;
  const bool evaluation = base == kEvaluationSentinel;
  const std::uint32_t begin = training ? state->train_start
      : evaluation ? evaluation_begin(state)
      : base;
  const std::uint32_t end = training ? state->train_size : state->tape_size;
  if (end <= begin) return begin;
  const std::uint32_t range = end - begin;
  if (range <= static_cast<std::uint32_t>(kSequence + 1))
    return begin + (static_cast<std::uint32_t>(t) % range);
  const std::uint32_t starts = range - static_cast<std::uint32_t>(kSequence);
  const std::uint64_t epoch = training ? state->step : 0u;
  const std::uint32_t start = mix32(state->founder ^
                                    (epoch * 0x9e3779b97f4a7c15ULL) ^
                                    (static_cast<std::uint64_t>(b) *
                                     0xbf58476d1ce4e5b9ULL)) % starts;
  return begin + start + static_cast<std::uint32_t>(t);
}

__global__ void pack_batch(const DeviceState* state, std::uint32_t base) {
  const std::uint32_t total = static_cast<std::uint32_t>(kBatch * kSequence);
  const bool training = base == kTrainingSentinel;
  const bool evaluation = base == kEvaluationSentinel;
  const std::uint32_t range_start = training ? state->train_start
      : evaluation ? evaluation_begin(state)
      : base;
  const std::uint32_t range_end = training ? state->train_size : state->tape_size;
  if (range_end <= range_start) return;
  const std::uint32_t range_size = range_end - range_start;
  if (range_size == 0u) return;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total; index += blockDim.x * gridDim.x) {
    const std::uint32_t t = index / kBatch;
    const std::uint32_t b = index % kBatch;
    const std::uint32_t source = batch_source(state, base, t, b);
    const std::uint8_t value = state->tape[source];
    const std::uint32_t next = source + 1u < range_end ? source + 1u : range_start;
    const std::uint8_t target = state->tape[next];
    state->targets[index] = target;
    for (int i = 0; i < kInput; ++i)
      state->x[(index * kInput) + static_cast<std::uint32_t>(i)] =
          state->embedding[static_cast<std::size_t>(value) * kInput + i];
    for (int i = 0; i < kContext; ++i)
      state->context_x[(index * kContext) + static_cast<std::uint32_t>(i)] =
          state->context_tape[static_cast<std::size_t>(source) * kContext + i];
  }
}

__global__ void fill_context_tape(float* context_tape, std::uint32_t offset,
                                  std::uint32_t count, const float* context) {
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    for (int component = 0; component < kContext; ++component)
      context_tape[(static_cast<std::size_t>(offset) + index) * kContext +
                   component] = context[component];
  }
}

__global__ void gru_forward_gate(DeviceState* state, int t) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kBatch * kHidden; index += blockDim.x * gridDim.x) {
    const int b = index / kHidden;
    const int j = index % kHidden;
    const std::size_t gate_base = static_cast<std::size_t>(t) * kBatch * kGates +
                                  static_cast<std::size_t>(b) * kGates;
    float az = state->pre[gate_base + j] +
               state->d_pre[gate_base + j] + state->bias[j];
    float ar = state->pre[gate_base + kHidden + j] +
               state->d_pre[gate_base + kHidden + j] +
               state->bias[kHidden + j];
    float an = state->pre[gate_base + 2 * kHidden + j] +
               state->bias[2 * kHidden + j];
    const std::size_t context_base =
        (static_cast<std::size_t>(t) * kBatch + static_cast<std::size_t>(b)) *
        kContext;
    for (int i = 0; i < kContext; i += 4) {
      const float4 value =
          *reinterpret_cast<const float4*>(state->context_x + context_base + i);
      const float4 weight_z =
          *reinterpret_cast<const float4*>(state->w_context + j * kContext + i);
      const float4 weight_r = *reinterpret_cast<const float4*>(
          state->w_context + (kHidden + j) * kContext + i);
      const float4 weight_n = *reinterpret_cast<const float4*>(
          state->w_context + (2 * kHidden + j) * kContext + i);
      az += weight_z.x * value.x;
      ar += weight_r.x * value.x;
      an += weight_n.x * value.x;
      az += weight_z.y * value.y;
      ar += weight_r.y * value.y;
      an += weight_n.y * value.y;
      az += weight_z.z * value.z;
      ar += weight_r.z * value.z;
      an += weight_n.z * value.z;
      az += weight_z.w * value.w;
      ar += weight_r.w * value.w;
      an += weight_n.w * value.w;
    }
    const float z = 1.0f / (1.0f + __expf(-az));
    const float r = 1.0f / (1.0f + __expf(-ar));
    const float candidate = an + r * state->d_pre[gate_base + 2 * kHidden + j];
    const float n = tanhf(candidate);
    const std::size_t gate = gate_base + j;
    const float previous = state->h[static_cast<std::size_t>(t) * kBatch *
                                    kHidden + static_cast<std::size_t>(b) *
                                    kHidden + j];
    state->gates[gate] = z;
    state->gates[gate + kHidden] = r;
    state->gates[gate + 2 * kHidden] = n;
    state->recurrent_n[static_cast<std::size_t>(t) * kBatch * kHidden +
                       static_cast<std::size_t>(b) * kHidden + j] =
        state->d_pre[gate_base + 2 * kHidden + j];
    state->h[static_cast<std::size_t>(t + 1) * kBatch * kHidden +
             static_cast<std::size_t>(b) * kHidden + j] =
        (1.0f - z) * n + z * previous;
  }
}

__global__ void loss_gradient(DeviceState* state) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kBatch * kSequence; index += blockDim.x * gridDim.x) {
    const std::size_t offset = static_cast<std::size_t>(index) * kVocab;
    float maximum = -3.402823466e+38F;
    for (int v = 0; v < kVocab; ++v)
      maximum = fmaxf(maximum, state->logits[offset + v]);
    float normalizer = 0.0f;
    for (int v = 0; v < kVocab; ++v)
      normalizer += __expf(state->logits[offset + v] - maximum);
    const float target_logit = state->logits[offset + state->targets[index]];
    state->sample_loss[index] = -target_logit + maximum + logf(normalizer);
    for (int v = 0; v < kVocab; ++v)
      state->d_logits[offset + v] =
          (__expf(state->logits[offset + v] - maximum) / normalizer) /
          static_cast<float>(kBatch * kSequence);
    state->d_logits[offset + state->targets[index]] -=
        1.0f / static_cast<float>(kBatch * kSequence);
  }
}

__global__ void reduce_loss(DeviceState* state) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < kBatch * kSequence; ++i)
      total += state->sample_loss[i];
    *state->step_loss = total / static_cast<float>(kBatch * kSequence);
    if (state->step == 0u) state->receipt->first_loss = *state->step_loss;
    state->receipt->final_loss = *state->step_loss;
  }
}

__global__ void gru_backward_z_n(DeviceState* state, int t) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kBatch * kHidden; index += blockDim.x * gridDim.x) {
    const int b = index / kHidden;
    const int j = index % kHidden;
    const std::size_t base_h = static_cast<std::size_t>(t) * kBatch * kHidden +
                               static_cast<std::size_t>(b) * kHidden;
    const std::size_t base_gate = static_cast<std::size_t>(t) * kBatch * kGates +
                                  static_cast<std::size_t>(b) * kGates;
    const float z = state->gates[base_gate + j];
    const float n = state->gates[base_gate + 2 * kHidden + j];
    const float previous = state->h[base_h + j];
    const float total = state->d_h_out[base_h + j] +
                        state->d_h_next[b * kHidden + j];
    const float d_z = total * (previous - n) * z * (1.0f - z);
    const float d_n = total * (1.0f - z) * (1.0f - n * n);
    state->d_pre[base_gate + j] = d_z;
    state->d_pre[base_gate + 2 * kHidden + j] = d_n;
    state->pre[base_gate + j] = d_z;
  }
}

__global__ void gru_backward_r_gate(DeviceState* state, int t) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kBatch * kHidden; index += blockDim.x * gridDim.x) {
    const int b = index / kHidden;
    const int k = index % kHidden;
    const std::size_t base_gate = static_cast<std::size_t>(t) * kBatch * kGates +
                                  static_cast<std::size_t>(b) * kGates;
    const float r = state->gates[base_gate + kHidden + k];
    const float recurrent_n =
        state->recurrent_n[static_cast<std::size_t>(t) * kBatch * kHidden +
                           static_cast<std::size_t>(b) * kHidden + k];
    const float d_n = state->d_pre[base_gate + 2 * kHidden + k];
    const float d_pre_r = d_n * recurrent_n * r * (1.0f - r);
    state->d_pre[base_gate + kHidden + k] = d_pre_r;
    state->pre[base_gate + kHidden + k] = d_pre_r;
    state->pre[base_gate + 2 * kHidden + k] = d_n * r;
  }
}

__global__ void gru_backward_direct(DeviceState* state, int t) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kBatch * kHidden; index += blockDim.x * gridDim.x) {
    const int b = index / kHidden;
    const int k = index % kHidden;
    const std::size_t base_h = static_cast<std::size_t>(t) * kBatch * kHidden +
                               static_cast<std::size_t>(b) * kHidden;
    const std::size_t base_gate = static_cast<std::size_t>(t) * kBatch * kGates +
                                  static_cast<std::size_t>(b) * kGates;
    const float z = state->gates[base_gate + k];
    const float total_grad = state->d_h_out[base_h + k] +
                             state->d_h_next[b * kHidden + k];
    state->d_q[b * kHidden + k] = total_grad * z;
  }
}

__global__ void gru_backward_add_direct(DeviceState* state) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kBatch * kHidden; index += blockDim.x * gridDim.x) {
    state->d_h_next[index] += state->d_q[index];
  }
}

__global__ void embedding_gradient(const DeviceState* state, int t) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kBatch * kInput; index += blockDim.x * gridDim.x) {
    const int b = index / kInput;
    const int i = index % kInput;
    const std::size_t sequence_index = static_cast<std::size_t>(t) * kBatch + b;
    const std::uint8_t byte =
        state->tape[batch_source(state, kTrainingSentinel, t, b)];
    atomicAdd(&state->d_embedding[static_cast<std::size_t>(byte) * kInput + i],
              state->d_x[sequence_index * kInput + i]);
  }
}

__global__ void bias_gradient(DeviceState* state) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < kGates;
       index += blockDim.x * gridDim.x) {
    float total = 0.0f;
    for (int i = index; i < kBatch * kSequence * kGates;
         i += kGates)
      total += state->d_pre[i];
    state->d_bias[index] = total;
  }
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < kReadout;
       index += blockDim.x * gridDim.x) {
    float total = 0.0f;
    for (int i = index; i < kBatch * kSequence * kReadout;
         i += kReadout)
      total += state->d_logits[i];
    state->d_readout_bias[index] = total;
  }
}

__global__ void context_projection_gradient(DeviceState* state) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < static_cast<std::size_t>(kGates) * kContext;
       index += blockDim.x * gridDim.x) {
    const int gate = static_cast<int>(index / kContext);
    const int context_index = static_cast<int>(index % kContext);
    float total = 0.0f;
    for (int t = 0; t < kSequence; ++t) {
      for (int b = 0; b < kBatch; ++b) {
        const std::size_t sequence_index =
            static_cast<std::size_t>(t) * kBatch + b;
        total += state->context_x[sequence_index * kContext + context_index] *
                 state->d_pre[sequence_index * kGates + gate];
      }
    }
    state->d_w_context[index] = total;
    atomicAdd(state->context_gradient_l1, fabsf(total));
  }
}

__device__ void adam_one(float* weight, float* first, float* second,
                         const float* gradient, std::size_t index,
                         float rate, float decay) {
  const float g = gradient[index];
  const float m = 0.9f * first[index] + 0.1f * g;
  const float v = 0.999f * second[index] + 0.001f * g * g;
  first[index] = m;
  second[index] = v;
  weight[index] -= rate * decay * weight[index];
  weight[index] -= rate * m / (sqrtf(v) + 1.0e-8f);
}

__global__ void adam_step(DeviceState* state) {
  const float step = static_cast<float>(state->step + 1u);
  const float corrected_rate = state->learning_rate *
      sqrtf(1.0f - powf(0.999f, step)) /
      (1.0f - powf(0.9f, step));
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(kVocab) * kInput;
       i += blockDim.x * gridDim.x)
    adam_one(state->embedding, state->m_embedding, state->v_embedding,
             state->d_embedding, i, corrected_rate, 1.0e-4f);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(kGates) * kInput;
       i += blockDim.x * gridDim.x)
    adam_one(state->w_ih, state->m_w_ih, state->v_w_ih, state->d_w_ih, i,
             corrected_rate, 1.0e-4f);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(kGates) * kContext;
       i += blockDim.x * gridDim.x)
    adam_one(state->w_context, state->m_w_context, state->v_w_context,
             state->d_w_context, i, corrected_rate, 1.0e-4f);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(kGates) * kHidden;
       i += blockDim.x * gridDim.x)
    adam_one(state->w_hh, state->m_w_hh, state->v_w_hh, state->d_w_hh, i,
             corrected_rate, 1.0e-4f);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kGates;
       i += blockDim.x * gridDim.x)
    adam_one(state->bias, state->m_bias, state->v_bias, state->d_bias, i,
             corrected_rate, 0.0f);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < static_cast<std::size_t>(kReadout) * kHidden;
       i += blockDim.x * gridDim.x)
    adam_one(state->readout, state->m_readout, state->v_readout,
             state->d_readout, i, corrected_rate, 1.0e-4f);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kReadout;
       i += blockDim.x * gridDim.x)
    adam_one(state->readout_bias, state->m_readout_bias,
             state->v_readout_bias, state->d_readout_bias, i,
             corrected_rate, 0.0f);
}

__global__ void train_tail(DeviceState* state, cudaGraphExec_t self) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ++state->step;
    if (state->step < state->train_steps) {
      (void)cudaGraphLaunch(self, cudaStreamGraphTailLaunch);
    } else {
      state->receipt->trained_steps = state->step;
      state->receipt->context_gradient_l1 = *state->context_gradient_l1;
      if (state->return_graph != nullptr)
        (void)cudaGraphLaunch(state->return_graph, cudaStreamGraphTailLaunch);
    }
  }
}

__global__ void unigram_loss_kernel(const DeviceState* state) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (state->tape_size <= state->heldout_start) {
      *state->unigram_loss = 0.0f;
      return;
    }
    const std::uint32_t train_span =
        state->train_size > state->train_start
            ? state->train_size - state->train_start
            : 0u;
    const float denominator = static_cast<float>(train_span + kVocab);
    float total = 0.0f;
    std::uint32_t count = 0u;
    for (std::uint32_t i = state->heldout_start; i < state->tape_size; ++i) {
      const std::uint64_t frequency = state->counts[state->tape[i]];
      total -= logf((static_cast<float>(frequency) + 1.0f) / denominator);
      ++count;
    }
    *state->unigram_loss = count == 0u ? 0.0f : total / count;
  }
}

__global__ void reduce_eval_loss(DeviceState* state) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < kBatch * kSequence; ++i)
      total += state->sample_loss[i];
    *state->eval_loss = total / static_cast<float>(kBatch * kSequence);
    state->receipt->heldout_loss = *state->eval_loss;
    state->receipt->unigram_loss = *state->unigram_loss;
  }
}

__global__ void lesion_kernel(DeviceState* state, std::uint32_t kind,
                              std::uint32_t offset, std::uint32_t count) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    if (kind == static_cast<std::uint32_t>(LesionKind::recurrent)) {
      const std::size_t position =
          (static_cast<std::size_t>(offset) + i) %
          (static_cast<std::size_t>(kGates) * kHidden);
      state->w_hh[position] = 0.0f;
    } else if (kind == static_cast<std::uint32_t>(LesionKind::context_projection)) {
      const std::size_t position =
          (static_cast<std::size_t>(offset) + i) %
          (static_cast<std::size_t>(kGates) * kContext);
      state->w_context[position] = 0.0f;
    } else if (kind ==
               static_cast<std::uint32_t>(LesionKind::source_tape)) {
      const std::uint32_t position = offset + i;
      if (position < state->tape_size) {
        state->tape[position] = 0u;
        for (int component = 0; component < kContext; ++component)
          state->context_tape[static_cast<std::size_t>(position) * kContext +
                              component] = 0.0f;
      }
    } else {
      const std::size_t embedding_position =
          (static_cast<std::size_t>(offset) + i) %
          (static_cast<std::size_t>(kVocab) * kInput);
      const std::size_t readout_position =
          (static_cast<std::size_t>(offset) + i) %
          (static_cast<std::size_t>(kReadout) * kHidden);
      state->embedding[embedding_position] = 0.0f;
      state->readout[readout_position] = 0.0f;
    }
  }
}

__global__ void generate_kernel(DeviceState* state) {
  const std::uint8_t* prompt = state->generation_prompt;
  const std::uint32_t prompt_size = state->generation_prompt_size;
  std::uint8_t* output = state->output;
  const std::uint32_t output_size = state->generation_output_size;
  if (prompt_size == 0u || output_size == 0u) return;
  extern __shared__ float shared[];
  float* previous = shared;
  float* current = previous + kHidden;
  float* gates = current + kHidden;
  float* logits = gates + kGates;
  for (int i = threadIdx.x; i < kHidden; i += blockDim.x)
    previous[i] = 0.0f;
  __syncthreads();
  const std::uint32_t total_steps = prompt_size + output_size;
  for (std::uint32_t t = 0u; t < total_steps; ++t) {
    const bool warming = t < prompt_size;
    const std::uint32_t generated = t - prompt_size;
    const std::uint8_t input = warming
        ? prompt[t]
        : (generated == 0u ? prompt[prompt_size - 1u]
                           : output[generated - 1u]);
    for (int j = threadIdx.x; j < kHidden; j += blockDim.x) {
      float az = state->bias[j];
      float ar = state->bias[kHidden + j];
      float an = state->bias[2 * kHidden + j];
      for (int i = 0; i < kInput; i += 4) {
        const float4 value = *reinterpret_cast<const float4*>(
            state->embedding + static_cast<std::size_t>(input) * kInput + i);
        const float4 weight_z =
            *reinterpret_cast<const float4*>(state->w_ih + j * kInput + i);
        const float4 weight_r = *reinterpret_cast<const float4*>(
            state->w_ih + (kHidden + j) * kInput + i);
        const float4 weight_n = *reinterpret_cast<const float4*>(
            state->w_ih + (2 * kHidden + j) * kInput + i);
        az += weight_z.x * value.x;
        ar += weight_r.x * value.x;
        an += weight_n.x * value.x;
        az += weight_z.y * value.y;
        ar += weight_r.y * value.y;
        an += weight_n.y * value.y;
        az += weight_z.z * value.z;
        ar += weight_r.z * value.z;
        an += weight_n.z * value.z;
        az += weight_z.w * value.w;
        ar += weight_r.w * value.w;
        an += weight_n.w * value.w;
      }
      for (int i = 0; i < kContext; i += 4) {
        const float4 value =
            *reinterpret_cast<const float4*>(state->live_context + i);
        const float4 weight_z =
            *reinterpret_cast<const float4*>(state->w_context + j * kContext + i);
        const float4 weight_r = *reinterpret_cast<const float4*>(
            state->w_context + (kHidden + j) * kContext + i);
        const float4 weight_n = *reinterpret_cast<const float4*>(
            state->w_context + (2 * kHidden + j) * kContext + i);
        az += weight_z.x * value.x;
        ar += weight_r.x * value.x;
        an += weight_n.x * value.x;
        az += weight_z.y * value.y;
        ar += weight_r.y * value.y;
        an += weight_n.y * value.y;
        az += weight_z.z * value.z;
        ar += weight_r.z * value.z;
        an += weight_n.z * value.z;
        az += weight_z.w * value.w;
        ar += weight_r.w * value.w;
        an += weight_n.w * value.w;
      }
      float recurrent_z = 0.0f;
      float recurrent_r = 0.0f;
      for (int i = 0; i < kHidden; i += 4) {
        const float4 weight_z =
            *reinterpret_cast<const float4*>(state->w_hh + j * kHidden + i);
        const float4 weight_r = *reinterpret_cast<const float4*>(
            state->w_hh + (kHidden + j) * kHidden + i);
        recurrent_z += weight_z.x * previous[i];
        recurrent_r += weight_r.x * previous[i];
        recurrent_z += weight_z.y * previous[i + 1];
        recurrent_r += weight_r.y * previous[i + 1];
        recurrent_z += weight_z.z * previous[i + 2];
        recurrent_r += weight_r.z * previous[i + 2];
        recurrent_z += weight_z.w * previous[i + 3];
        recurrent_r += weight_r.w * previous[i + 3];
      }
      const float z = 1.0f / (1.0f + __expf(-(az + recurrent_z)));
      const float r = 1.0f / (1.0f + __expf(-(ar + recurrent_r)));
      for (int i = 0; i < kHidden; i += 4) {
        const float4 weight_n = *reinterpret_cast<const float4*>(
            state->w_hh + (2 * kHidden + j) * kHidden + i);
        an += weight_n.x * (r * previous[i]);
        an += weight_n.y * (r * previous[i + 1]);
        an += weight_n.z * (r * previous[i + 2]);
        an += weight_n.w * (r * previous[i + 3]);
      }
      const float n = tanhf(an);
      gates[j] = z;
      gates[kHidden + j] = r;
      gates[2 * kHidden + j] = n;
      current[j] = (1.0f - z) * n + z * previous[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      constexpr int kTop = 20;
      float top_values[kTop];
      int top_indices[kTop];
      for (int rank = 0; rank < kTop; ++rank) {
        top_values[rank] = -3.402823466e+38F;
        top_indices[rank] = 0;
      }
      for (int v = 0; v < kReadout; ++v) {
        float value = state->readout_bias[v];
        for (int j = 0; j < kHidden; ++j)
          value += state->readout[v * kHidden + j] * current[j];
        logits[v] = value;
      }
      for (int v = 0; v < kReadout; ++v) {
        const float value = logits[v];
        if (value > top_values[kTop - 1]) {
          int rank = kTop - 1;
          while (rank > 0 && value > top_values[rank - 1]) {
            top_values[rank] = top_values[rank - 1];
            top_indices[rank] = top_indices[rank - 1];
            --rank;
          }
          top_values[rank] = value;
          top_indices[rank] = v;
        }
      }
      if (!warming) {
        float mass = 0.0f;
        const float maximum = top_values[0];
        for (int rank = 0; rank < kTop; ++rank) {
          top_values[rank] = __expf((top_values[rank] - maximum) / 0.65f);
          mass += top_values[rank];
        }
        const std::uint32_t bits = mix32(
            state->founder ^ (static_cast<std::uint64_t>(generated + 1u) *
                              0x9e3779b97f4a7c15ULL));
        const float draw = (static_cast<float>(bits) + 0.5f) /
                           4294967296.0f * mass;
        float cumulative = 0.0f;
        int selected = top_indices[kTop - 1];
        for (int rank = 0; rank < kTop; ++rank) {
          cumulative += top_values[rank];
          if (draw <= cumulative) {
            selected = top_indices[rank];
            break;
          }
        }
        output[generated] = static_cast<std::uint8_t>(selected);
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kHidden; i += blockDim.x)
      previous[i] = current[i];
    __syncthreads();
  }
  if (threadIdx.x == 0) state->receipt->output_size = output_size;
}

__global__ void generation_tail(DeviceState* state) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && state->return_graph != nullptr)
    (void)cudaGraphLaunch(state->return_graph, cudaStreamGraphTailLaunch);
}

}  // namespace

struct ResidentGruLanguage::Impl {
  static constexpr std::size_t kCublasWorkspaceBytes = 32u << 20u;
  Config config{};
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  void* cublas_workspace = nullptr;
  DeviceState* state = nullptr;
  DeviceState host_state{};
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaGraph_t generation_graph = nullptr;
  cudaGraphExec_t generation_graph_exec = nullptr;
  cudaGraph_t evaluation_graph = nullptr;
  cudaGraphExec_t evaluation_graph_exec = nullptr;
  bool graph_device_launch = false;
  bool graph_tail_launch = false;
  std::size_t counted_train_size = 0u;
  std::uint64_t host_launches = 0u;
  std::size_t tape_size = 0u;
};

namespace {

template <typename T>
void device_alloc(T** pointer, std::size_t count, const char* what) {
  check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer),
                        count * sizeof(T)), what);
}

template <typename T>
void device_free(T*& pointer) noexcept {
  if (pointer != nullptr) {
    (void)cudaFree(pointer);
    pointer = nullptr;
  }
}

void gemm(cublasHandle_t handle, cublasOperation_t left,
          cublasOperation_t right, int m, int n, int k, const float* a,
          int lda, const float* b, int ldb, float* c, int ldc, float beta) {
  static const float alpha = 1.0f;
  check_cublas(cublasSgemm(handle, left, right, m, n, k, &alpha, a, lda, b,
                           ldb, &beta, c, ldc),
               "captured row-major GEMM");
}

void allocate_model(ResidentGruLanguage::Impl& impl, std::uint64_t founder) {
  DeviceState blank{};
  blank.founder = founder;
  blank.train_steps = impl.config.train_steps;
  blank.configured_train_steps = impl.config.train_steps;
  blank.learning_rate = impl.config.learning_rate;
  blank.tape_capacity = static_cast<std::uint32_t>(impl.config.tape_capacity);
  device_alloc(&blank.tape, impl.config.tape_capacity,
               "allocate resident byte tape");
  device_alloc(&blank.context_tape,
               impl.config.tape_capacity * static_cast<std::size_t>(kContext),
               "allocate resident context tape");
  device_alloc(&blank.counts, kVocab, "allocate byte counts");
  device_alloc(&blank.embedding, static_cast<std::size_t>(kVocab) * kInput,
               "allocate embedding");
  device_alloc(&blank.w_ih, static_cast<std::size_t>(kGates) * kInput,
               "allocate input GRU weights");
  device_alloc(&blank.w_context, static_cast<std::size_t>(kGates) * kContext,
               "allocate context projection weights");
  device_alloc(&blank.w_hh, static_cast<std::size_t>(kGates) * kHidden,
               "allocate recurrent GRU weights");
  device_alloc(&blank.bias, kGates, "allocate GRU bias");
  device_alloc(&blank.readout, static_cast<std::size_t>(kReadout) * kHidden,
               "allocate readout");
  device_alloc(&blank.readout_bias, kReadout, "allocate readout bias");
#define ALLOC_OPTIMIZER(name, count)                                           \
  device_alloc(&blank.m_##name, count, "allocate first optimizer moment");   \
  device_alloc(&blank.v_##name, count, "allocate second optimizer moment");  \
  device_alloc(&blank.d_##name, count, "allocate parameter gradient")
  ALLOC_OPTIMIZER(embedding, static_cast<std::size_t>(kVocab) * kInput);
  ALLOC_OPTIMIZER(w_ih, static_cast<std::size_t>(kGates) * kInput);
  ALLOC_OPTIMIZER(w_context, static_cast<std::size_t>(kGates) * kContext);
  ALLOC_OPTIMIZER(w_hh, static_cast<std::size_t>(kGates) * kHidden);
  ALLOC_OPTIMIZER(bias, kGates);
  ALLOC_OPTIMIZER(readout, static_cast<std::size_t>(kReadout) * kHidden);
  ALLOC_OPTIMIZER(readout_bias, kReadout);
#undef ALLOC_OPTIMIZER
  device_alloc(&blank.x, static_cast<std::size_t>(kSequence) * kBatch * kInput,
               "allocate batch input");
  device_alloc(&blank.context_x,
               static_cast<std::size_t>(kSequence) * kBatch * kContext,
               "allocate batch context");
  device_alloc(&blank.targets, static_cast<std::size_t>(kSequence) * kBatch,
               "allocate device targets");
  device_alloc(&blank.h, static_cast<std::size_t>(kSequence + 1) * kBatch *
                              kHidden,
               "allocate hidden trajectory");
  device_alloc(&blank.gates, static_cast<std::size_t>(kSequence) * kBatch *
                                  kGates,
               "allocate GRU gates");
  device_alloc(&blank.recurrent_n,
               static_cast<std::size_t>(kSequence) * kBatch * kHidden,
               "allocate recurrent candidate trajectory");
  device_alloc(&blank.pre, static_cast<std::size_t>(kSequence) * kBatch *
                              kGates,
               "allocate GRU preactivation");
  device_alloc(&blank.logits, static_cast<std::size_t>(kSequence) * kBatch *
                                  kReadout,
               "allocate logits");
  device_alloc(&blank.d_logits, static_cast<std::size_t>(kSequence) * kBatch *
                                    kReadout,
               "allocate logit gradient");
  device_alloc(&blank.d_h_out, static_cast<std::size_t>(kSequence) * kBatch *
                                   kHidden,
               "allocate hidden gradient");
  device_alloc(&blank.d_h_next, static_cast<std::size_t>(kBatch) * kHidden,
               "allocate recurrent gradient");
  device_alloc(&blank.d_q, static_cast<std::size_t>(kBatch) * kHidden,
               "allocate recurrent gate gradient");
  device_alloc(&blank.d_pre, static_cast<std::size_t>(kSequence) * kBatch *
                                  kGates,
               "allocate gate gradient");
  device_alloc(&blank.d_x, static_cast<std::size_t>(kSequence) * kBatch * kInput,
               "allocate input gradient");
  device_alloc(&blank.sample_loss, static_cast<std::size_t>(kSequence) * kBatch,
               "allocate sample loss");
  device_alloc(&blank.step_loss, 1u, "allocate step loss");
  device_alloc(&blank.eval_loss, 1u, "allocate evaluation loss");
  device_alloc(&blank.unigram_loss, 1u, "allocate unigram loss");
  device_alloc(&blank.context_gradient_l1, 1u,
               "allocate context gradient metric");
  device_alloc(&blank.receipt, 1u, "allocate device receipt");
  device_alloc(&blank.generation_prompt, kMaxGeneration,
               "allocate generation prompt");
  device_alloc(&blank.output, kMaxGeneration, "allocate generated output");
  device_alloc(&blank.live_context, kContext, "allocate live context vector");
  blank.output_capacity = static_cast<std::uint32_t>(kMaxGeneration);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&impl.state), sizeof(DeviceState)),
             "allocate resident language state");
  check_cuda(cudaMemcpy(impl.state, &blank, sizeof(blank),
                        cudaMemcpyHostToDevice),
             "initialize resident language state");
  impl.host_state = blank;
  seed_normal_parameter<<<128, 256>>>(
      blank.embedding, static_cast<std::size_t>(kVocab) * kInput,
      founder ^ 0x11u, 1.0f);
  seed_parameter<<<128, 256>>>(blank.w_ih,
                               static_cast<std::size_t>(kGates) * kInput,
                               founder ^ 0x22u, 0.051031f);
  seed_parameter<<<128, 256>>>(blank.w_context,
                               static_cast<std::size_t>(kGates) * kContext,
                               founder ^ 0x2au, 0.051031f);
  seed_parameter<<<128, 256>>>(blank.w_hh,
                               static_cast<std::size_t>(kGates) * kHidden,
                               founder ^ 0x33u, 0.051031f);
  seed_parameter<<<128, 256>>>(blank.readout,
                               static_cast<std::size_t>(kReadout) * kHidden,
                               founder ^ 0x44u, 0.051031f);
  check_cuda(cudaGetLastError(), "seed resident GRU parameters");
  seed_parameter<<<1, 256>>>(blank.bias, kGates, founder ^ 0x55u, 0.051031f);
  seed_parameter<<<1, 256>>>(blank.readout_bias, kReadout, founder ^ 0x66u,
                             0.051031f);
  check_cuda(cudaMemset(blank.m_embedding, 0,
                        static_cast<std::size_t>(kVocab) * kInput *
                            sizeof(float)),
             "clear embedding first moment");
  check_cuda(cudaMemset(blank.v_embedding, 0,
                        static_cast<std::size_t>(kVocab) * kInput *
                            sizeof(float)),
             "clear embedding second moment");
  check_cuda(cudaMemset(blank.m_w_ih, 0,
                        static_cast<std::size_t>(kGates) * kInput *
                            sizeof(float)),
             "clear input first moment");
  check_cuda(cudaMemset(blank.v_w_ih, 0,
                        static_cast<std::size_t>(kGates) * kInput *
                            sizeof(float)),
             "clear input second moment");
  check_cuda(cudaMemset(blank.m_w_context, 0,
                        static_cast<std::size_t>(kGates) * kContext *
                            sizeof(float)),
             "clear context first moment");
  check_cuda(cudaMemset(blank.v_w_context, 0,
                        static_cast<std::size_t>(kGates) * kContext *
                            sizeof(float)),
             "clear context second moment");
  check_cuda(cudaMemset(blank.m_w_hh, 0,
                        static_cast<std::size_t>(kGates) * kHidden *
                            sizeof(float)),
             "clear recurrent first moment");
  check_cuda(cudaMemset(blank.v_w_hh, 0,
                        static_cast<std::size_t>(kGates) * kHidden *
                            sizeof(float)),
             "clear recurrent second moment");
  check_cuda(cudaMemset(blank.m_bias, 0, kGates * sizeof(float)),
             "clear bias first moment");
  check_cuda(cudaMemset(blank.v_bias, 0, kGates * sizeof(float)),
             "clear bias second moment");
  check_cuda(cudaMemset(blank.m_readout, 0,
                        static_cast<std::size_t>(kReadout) * kHidden *
                            sizeof(float)),
             "clear readout first moment");
  check_cuda(cudaMemset(blank.v_readout, 0,
                        static_cast<std::size_t>(kReadout) * kHidden *
                            sizeof(float)),
             "clear readout second moment");
  check_cuda(cudaMemset(blank.m_readout_bias, 0, kReadout * sizeof(float)),
             "clear readout bias first moment");
  check_cuda(cudaMemset(blank.v_readout_bias, 0, kReadout * sizeof(float)),
             "clear readout bias second moment");
  check_cuda(cudaMemset(blank.d_bias, 0, kGates * sizeof(float)),
             "clear bias gradient");
  check_cuda(cudaMemset(blank.d_readout_bias, 0, kReadout * sizeof(float)),
             "clear readout bias gradient");
  check_cuda(cudaMemset(blank.counts, 0, kVocab * sizeof(std::uint64_t)),
             "clear byte counts");
  check_cuda(cudaMemset(blank.context_tape, 0,
                        impl.config.tape_capacity *
                            static_cast<std::size_t>(kContext) * sizeof(float)),
             "clear resident context tape");
  check_cuda(cudaMemset(blank.live_context, 0, kContext * sizeof(float)),
             "clear live context vector");
  check_cuda(cudaMemset(blank.context_gradient_l1, 0, sizeof(float)),
             "clear context gradient metric");
  check_cuda(cudaMemset(blank.d_w_context, 0,
                        static_cast<std::size_t>(kGates) * kContext *
                            sizeof(float)),
             "clear context gradient");
  check_cuda(cudaMemset(blank.receipt, 0, sizeof(DeviceReceipt)),
             "clear resident receipt");
}

void capture_training_graph(ResidentGruLanguage::Impl& impl) {
  check_cublas(cublasSetStream(impl.cublas, impl.stream),
               "bind cuBLAS stream");
  // cublasSetStream resets the handle workspace to its allocator-backed
  // default. Rebind resident storage afterward so capture contains no
  // mem-alloc/free nodes, which device-launchable CUDA graphs reject.
  check_cublas(cublasSetWorkspace(impl.cublas, impl.cublas_workspace,
                                  ResidentGruLanguage::Impl::kCublasWorkspaceBytes),
               "rebind resident cuBLAS workspace after stream");
  check_cuda(cudaStreamBeginCapture(impl.stream, cudaStreamCaptureModeGlobal),
             "begin resident GRU graph capture");
  pack_batch<<<256, 256, 0, impl.stream>>>(impl.state, kTrainingSentinel);
  check_cuda(cudaGetLastError(), "capture tape packing");
  check_cuda(cudaMemsetAsync(impl.host_state.h, 0,
                             static_cast<std::size_t>(kSequence + 1) * kBatch *
                                 kHidden * sizeof(float),
                             impl.stream),
             "capture hidden reset");
  check_cuda(cudaMemsetAsync(impl.host_state.d_embedding, 0,
                             static_cast<std::size_t>(kVocab) * kInput *
                                 sizeof(float),
                             impl.stream),
             "capture embedding gradient reset");
  check_cuda(cudaMemsetAsync(impl.host_state.d_w_ih, 0,
                             static_cast<std::size_t>(kGates) * kInput *
                                 sizeof(float),
                             impl.stream),
             "capture input gradient reset");
  check_cuda(cudaMemsetAsync(impl.host_state.d_w_context, 0,
                             static_cast<std::size_t>(kGates) * kContext *
                                 sizeof(float),
                             impl.stream),
             "capture context gradient reset");
  check_cuda(cudaMemsetAsync(impl.host_state.context_gradient_l1, 0,
                             sizeof(float), impl.stream),
             "capture context gradient metric reset");
  check_cuda(cudaMemsetAsync(impl.host_state.d_w_hh, 0,
                             static_cast<std::size_t>(kGates) * kHidden *
                                 sizeof(float),
                             impl.stream),
             "capture recurrent gradient reset");
  check_cuda(cudaMemsetAsync(impl.host_state.d_readout, 0,
                             static_cast<std::size_t>(kReadout) * kHidden *
                                 sizeof(float),
                             impl.stream),
             "capture readout gradient reset");
  check_cuda(cudaMemsetAsync(impl.host_state.d_h_next, 0,
                             static_cast<std::size_t>(kBatch) * kHidden *
                                 sizeof(float),
                             impl.stream),
             "capture recurrent state gradient reset");

  static const float zero = 0.0f;
  for (int t = 0; t < kSequence; ++t) {
    const float* x = impl.host_state.x + static_cast<std::size_t>(t) * kBatch * kInput;
    const float* previous =
        impl.host_state.h + static_cast<std::size_t>(t) * kBatch * kHidden;
    float* pre = impl.host_state.pre + static_cast<std::size_t>(t) * kBatch * kGates;
    gemm(impl.cublas, CUBLAS_OP_T, CUBLAS_OP_N, kGates, kBatch, kInput,
         impl.host_state.w_ih, kInput, x, kInput, pre, kGates, zero);
    float* recurrent = impl.host_state.d_pre +
                       static_cast<std::size_t>(t) * kBatch * kGates;
    gemm(impl.cublas, CUBLAS_OP_T, CUBLAS_OP_N, kGates, kBatch, kHidden,
         impl.host_state.w_hh, kHidden, previous, kHidden, recurrent, kGates,
         zero);
    gru_forward_gate<<<256, 256, 0, impl.stream>>>(impl.state, t);
    check_cuda(cudaGetLastError(), "capture custom GRU forward gate");
  }
  for (int t = 0; t < kSequence; ++t) {
    const float* current =
        impl.host_state.h + static_cast<std::size_t>(t + 1) * kBatch * kHidden;
    float* logits = impl.host_state.logits + static_cast<std::size_t>(t) * kBatch *
                                             kReadout;
    gemm(impl.cublas, CUBLAS_OP_T, CUBLAS_OP_N, kReadout, kBatch, kHidden,
         impl.host_state.readout, kHidden, current, kHidden, logits, kReadout,
         zero);
  }
  loss_gradient<<<256, 256, 0, impl.stream>>>(impl.state);
  check_cuda(cudaGetLastError(), "capture cross entropy gradient");
  DeviceState* captured_state = impl.state;
  void* reduce_loss_arguments[] = {&captured_state};
  check_cuda(cudaLaunchKernel(
                 reinterpret_cast<const void*>(reduce_loss),
                 dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u}, reduce_loss_arguments,
                 0u, impl.stream),
             "capture loss reduction");
  check_cuda(cudaGetLastError(), "capture loss reduction");
  for (int t = 0; t < kSequence; ++t) {
    const float* d_logits = impl.host_state.d_logits +
                            static_cast<std::size_t>(t) * kBatch * kReadout;
    float* d_h = impl.host_state.d_h_out +
                 static_cast<std::size_t>(t) * kBatch * kHidden;
    gemm(impl.cublas, CUBLAS_OP_N, CUBLAS_OP_N, kHidden, kBatch, kReadout,
         impl.host_state.readout, kHidden, d_logits, kReadout, d_h, kHidden,
         zero);
  }
  for (int t = kSequence - 1; t >= 0; --t) {
    gru_backward_z_n<<<256, 256, 0, impl.stream>>>(impl.state, t);
    gru_backward_r_gate<<<256, 256, 0, impl.stream>>>(impl.state, t);
    gru_backward_direct<<<256, 256, 0, impl.stream>>>(impl.state, t);
    const float* d_recurrent = impl.host_state.pre +
                               static_cast<std::size_t>(t) * kBatch * kGates;
    gemm(impl.cublas, CUBLAS_OP_N, CUBLAS_OP_N, kHidden, kBatch, kGates,
         impl.host_state.w_hh, kHidden, d_recurrent, kGates,
         impl.host_state.d_h_next, kHidden, zero);
    gru_backward_add_direct<<<256, 256, 0, impl.stream>>>(impl.state);
    check_cuda(cudaGetLastError(), "capture custom GRU backward gates");
    const float* d_pre = impl.host_state.d_pre +
                         static_cast<std::size_t>(t) * kBatch * kGates;
    const float* x = impl.host_state.x + static_cast<std::size_t>(t) * kBatch * kInput;
    const float* previous =
        impl.host_state.h + static_cast<std::size_t>(t) * kBatch * kHidden;
    float* d_x = impl.host_state.d_x + static_cast<std::size_t>(t) * kBatch * kInput;
    gemm(impl.cublas, CUBLAS_OP_N, CUBLAS_OP_N, kInput, kBatch, kGates,
         impl.host_state.w_ih, kInput, d_pre, kGates, d_x, kInput, zero);
    gemm(impl.cublas, CUBLAS_OP_N, CUBLAS_OP_T, kInput, kGates, kBatch,
         x, kInput, d_pre, kGates, impl.host_state.d_w_ih, kInput, 1.0f);
    gemm(impl.cublas, CUBLAS_OP_N, CUBLAS_OP_T, kHidden, kGates, kBatch,
         previous, kHidden, d_recurrent, kGates, impl.host_state.d_w_hh,
         kHidden, 1.0f);
    embedding_gradient<<<256, 256, 0, impl.stream>>>(impl.state, t);
    check_cuda(cudaGetLastError(), "capture embedding gradient");
    const float* d_logits = impl.host_state.d_logits +
                            static_cast<std::size_t>(t) * kBatch * kReadout;
    const float* current =
        impl.host_state.h + static_cast<std::size_t>(t + 1) * kBatch * kHidden;
    gemm(impl.cublas, CUBLAS_OP_N, CUBLAS_OP_T, kHidden, kReadout, kBatch,
         current, kHidden, d_logits, kReadout, impl.host_state.d_readout,
         kHidden, 1.0f);
  }
  context_projection_gradient<<<256, 256, 0, impl.stream>>>(impl.state);
  check_cuda(cudaGetLastError(), "capture context projection gradient");
  bias_gradient<<<256, 256, 0, impl.stream>>>(impl.state);
  check_cuda(cudaGetLastError(), "capture bias gradients");
  adam_step<<<256, 256, 0, impl.stream>>>(impl.state);
  check_cuda(cudaGetLastError(), "capture resident AdamW update");
  cudaGraphExec_t captured_self = nullptr;
  void* train_tail_arguments[] = {&captured_state, &captured_self};
  check_cuda(cudaLaunchKernel(
                 reinterpret_cast<const void*>(train_tail),
                 dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u}, train_tail_arguments,
                 0u, impl.stream),
             "capture resident graph tail");
  check_cuda(cudaGetLastError(), "capture resident graph tail");
  check_cuda(cudaStreamEndCapture(impl.stream, &impl.graph),
             "end resident GRU graph capture");
  std::size_t node_count = 0u;
  check_cuda(cudaGraphGetNodes(impl.graph, nullptr, &node_count),
             "count resident graph nodes");
  if (node_count == 0u) throw std::runtime_error("resident graph is empty");
  std::vector<cudaGraphNode_t> nodes(node_count);
  check_cuda(cudaGraphGetNodes(impl.graph, nodes.data(), &node_count),
             "read resident graph nodes");
  const cudaError_t instantiate_status = cudaGraphInstantiateWithFlags(
      &impl.graph_exec, impl.graph, cudaGraphInstantiateFlagDeviceLaunch);
  if (instantiate_status != cudaSuccess) {
    std::array<std::size_t, 16> type_counts{};
    for (const cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
      if (cudaGraphNodeGetType(node, &type) == cudaSuccess &&
          static_cast<std::size_t>(type) < type_counts.size()) {
        ++type_counts[static_cast<std::size_t>(type)];
      }
    }
    std::fprintf(stderr,
                 "resident graph device instantiate failed nodes=%zu "
                 "kernel=%zu memcpy=%zu memset=%zu host=%zu child=%zu empty=%zu "
                 "event_record=%zu event_wait=%zu ext_signal=%zu ext_wait=%zu "
                 "mem_alloc=%zu mem_free=%zu conditional=%zu\n",
                 node_count, type_counts[cudaGraphNodeTypeKernel],
                 type_counts[cudaGraphNodeTypeMemcpy],
                 type_counts[cudaGraphNodeTypeMemset],
                 type_counts[cudaGraphNodeTypeHost],
                 type_counts[cudaGraphNodeTypeGraph],
                 type_counts[cudaGraphNodeTypeEmpty],
                 type_counts[cudaGraphNodeTypeEventRecord],
                 type_counts[cudaGraphNodeTypeWaitEvent],
                 type_counts[cudaGraphNodeTypeExtSemaphoreSignal],
                 type_counts[cudaGraphNodeTypeExtSemaphoreWait],
                 type_counts[cudaGraphNodeTypeMemAlloc],
                 type_counts[cudaGraphNodeTypeMemFree],
                 type_counts[cudaGraphNodeTypeConditional]);
    check_cuda(instantiate_status, "instantiate device-launchable resident graph");
  }
  cudaGraphNode_t tail_node = nodes.back();
  cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
  check_cuda(cudaGraphNodeGetType(tail_node, &node_type),
             "inspect resident tail node");
  if (node_type != cudaGraphNodeTypeKernel)
    throw std::runtime_error("resident graph tail is not a kernel node");
  cudaKernelNodeParams tail_params{};
  check_cuda(cudaGraphKernelNodeGetParams(tail_node, &tail_params),
             "read resident tail parameters");
  void* tail_args[] = {&impl.state, &impl.graph_exec};
  tail_params.kernelParams = tail_args;
  check_cuda(cudaGraphExecKernelNodeSetParams(impl.graph_exec, tail_node,
                                               &tail_params),
             "bind resident graph self handle");
  check_cuda(cudaGraphUpload(impl.graph_exec, impl.stream),
             "upload resident device graph");
  check_cuda(cudaStreamSynchronize(impl.stream),
             "synchronize resident graph upload");
  impl.graph_device_launch = true;
  impl.graph_tail_launch = true;
}

void capture_generation_graph(ResidentGruLanguage::Impl& impl) {
  check_cuda(cudaStreamBeginCapture(impl.stream, cudaStreamCaptureModeGlobal),
             "begin resident generation graph capture");
  generate_kernel<<<1, kHidden, (2 * kHidden + kGates + kReadout) *
                                  sizeof(float),
                    impl.stream>>>(impl.state);
  check_cuda(cudaGetLastError(), "capture resident generation");
  DeviceState* generation_state = impl.state;
  void* generation_tail_arguments[] = {&generation_state};
  check_cuda(cudaLaunchKernel(
                 reinterpret_cast<const void*>(generation_tail),
                 dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                 generation_tail_arguments, 0u, impl.stream),
             "capture resident generation return");
  check_cuda(cudaGetLastError(), "capture resident generation return");
  check_cuda(cudaStreamEndCapture(impl.stream, &impl.generation_graph),
             "end resident generation graph capture");
  check_cuda(cudaGraphInstantiateWithFlags(
                 &impl.generation_graph_exec, impl.generation_graph,
                 cudaGraphInstantiateFlagDeviceLaunch),
             "instantiate device-launchable generation graph");
  check_cuda(cudaGraphUpload(impl.generation_graph_exec, impl.stream),
             "upload resident generation graph");
  check_cuda(cudaStreamSynchronize(impl.stream),
             "synchronize generation graph upload");
}

void copy_state_scalar(DeviceState* state, std::size_t offset,
                       const void* value, std::size_t bytes,
                       cudaStream_t stream, const char* what) {
  check_cuda(cudaMemcpyAsync(reinterpret_cast<std::uint8_t*>(state) + offset,
                             value, bytes, cudaMemcpyHostToDevice, stream),
             what);
}

void launch_custom_forward(ResidentGruLanguage::Impl& impl) {
  static const float zero = 0.0f;
  for (int t = 0; t < kSequence; ++t) {
    const float* x = impl.host_state.x + static_cast<std::size_t>(t) * kBatch * kInput;
    const float* previous =
        impl.host_state.h + static_cast<std::size_t>(t) * kBatch * kHidden;
    float* pre = impl.host_state.pre + static_cast<std::size_t>(t) * kBatch * kGates;
    gemm(impl.cublas, CUBLAS_OP_T, CUBLAS_OP_N, kGates, kBatch, kInput,
         impl.host_state.w_ih, kInput, x, kInput, pre, kGates, zero);
    float* recurrent = impl.host_state.d_pre +
                       static_cast<std::size_t>(t) * kBatch * kGates;
    gemm(impl.cublas, CUBLAS_OP_T, CUBLAS_OP_N, kGates, kBatch, kHidden,
         impl.host_state.w_hh, kHidden, previous, kHidden, recurrent, kGates,
         zero);
    gru_forward_gate<<<256, 256, 0, impl.stream>>>(impl.state, t);
  }
  for (int t = 0; t < kSequence; ++t) {
    const float* current =
        impl.host_state.h + static_cast<std::size_t>(t + 1) * kBatch * kHidden;
    float* logits = impl.host_state.logits + static_cast<std::size_t>(t) * kBatch *
                                             kReadout;
    gemm(impl.cublas, CUBLAS_OP_T, CUBLAS_OP_N, kReadout, kBatch, kHidden,
         impl.host_state.readout, kHidden, current, kHidden, logits, kReadout,
         zero);
  }
  loss_gradient<<<256, 256, 0, impl.stream>>>(impl.state);
}

// Held-out evaluation used to issue ~200 host kernel launches per call (a
// pack + memset + per-timestep forward pass over kSequence steps + two
// scalar reductions), all outside any captured graph. heldout_start and
// tape_size are read live from device state inside pack_batch/batch_source
// via kEvaluationSentinel (see evaluation_begin above), so — exactly like
// the training graph's kTrainingSentinel range — this graph can be captured
// once and relaunched as the resident tape and heldout boundary grow.
void capture_evaluation_graph(ResidentGruLanguage::Impl& impl) {
  check_cuda(cudaStreamBeginCapture(impl.stream, cudaStreamCaptureModeGlobal),
             "begin resident evaluation graph capture");
  pack_batch<<<256, 256, 0, impl.stream>>>(impl.state, kEvaluationSentinel);
  check_cuda(cudaGetLastError(), "capture heldout tape packing");
  check_cuda(cudaMemsetAsync(impl.host_state.h, 0,
                             static_cast<std::size_t>(kSequence + 1) * kBatch *
                                 kHidden * sizeof(float),
                             impl.stream),
             "capture heldout hidden reset");
  launch_custom_forward(impl);
  DeviceState* evaluation_state = impl.state;
  void* unigram_loss_arguments[] = {&evaluation_state};
  check_cuda(cudaLaunchKernel(
                 reinterpret_cast<const void*>(unigram_loss_kernel),
                 dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u}, unigram_loss_arguments,
                 0u, impl.stream),
             "capture heldout unigram loss");
  void* reduce_eval_loss_arguments[] = {&evaluation_state};
  check_cuda(cudaLaunchKernel(
                 reinterpret_cast<const void*>(reduce_eval_loss),
                 dim3{1u, 1u, 1u}, dim3{1u, 1u, 1u},
                 reduce_eval_loss_arguments, 0u, impl.stream),
             "capture heldout loss reduction");
  check_cuda(cudaGetLastError(), "capture heldout loss reduction");
  check_cuda(cudaStreamEndCapture(impl.stream, &impl.evaluation_graph),
             "end resident evaluation graph capture");
  check_cuda(cudaGraphInstantiateWithFlags(&impl.evaluation_graph_exec,
                                           impl.evaluation_graph, 0),
             "instantiate resident evaluation graph");
  check_cuda(cudaGraphUpload(impl.evaluation_graph_exec, impl.stream),
             "upload resident evaluation graph");
  check_cuda(cudaStreamSynchronize(impl.stream),
             "synchronize evaluation graph upload");
}

void run_evaluation(ResidentGruLanguage::Impl& impl) {
  check_cuda(cudaGraphLaunch(impl.evaluation_graph_exec, impl.stream),
             "launch resident evaluation graph");
  check_cuda(cudaStreamSynchronize(impl.stream),
             "synchronize heldout evaluation");
}

void publish_tape_bounds(ResidentGruLanguage::Impl& impl) {
  const std::uint32_t total = static_cast<std::uint32_t>(impl.tape_size);
  const std::uint32_t train =
      total >= 2u * kSequence + 1u ? (total * 9u) / 10u : total;
  const std::uint32_t train_start = 0u;
  const std::uint32_t heldout = train < total ? train : total;
  copy_state_scalar(impl.state, offsetof(DeviceState, tape_size), &total,
                    sizeof(total), impl.stream, "publish tape size");
  copy_state_scalar(impl.state, offsetof(DeviceState, train_size), &train,
                    sizeof(train), impl.stream, "publish train size");
  copy_state_scalar(impl.state, offsetof(DeviceState, train_start),
                    &train_start, sizeof(train_start), impl.stream,
                    "publish training range start");
  copy_state_scalar(impl.state, offsetof(DeviceState, heldout_start),
                    &heldout, sizeof(heldout), impl.stream,
                    "publish heldout boundary");
  if (train > impl.counted_train_size) {
    count_tape<<<256, 256, 0, impl.stream>>>(
        impl.host_state.tape,
        static_cast<std::uint32_t>(impl.counted_train_size), train,
        impl.host_state.counts);
    check_cuda(cudaGetLastError(), "count appended resident tape bytes");
    impl.counted_train_size = train;
  }
}

void destroy_impl(ResidentGruLanguage::Impl& impl) noexcept {
  if (impl.stream != nullptr) (void)cudaStreamSynchronize(impl.stream);
  if (impl.generation_graph_exec != nullptr)
    (void)cudaGraphExecDestroy(impl.generation_graph_exec);
  if (impl.generation_graph != nullptr)
    (void)cudaGraphDestroy(impl.generation_graph);
  if (impl.evaluation_graph_exec != nullptr)
    (void)cudaGraphExecDestroy(impl.evaluation_graph_exec);
  if (impl.evaluation_graph != nullptr)
    (void)cudaGraphDestroy(impl.evaluation_graph);
  if (impl.graph_exec != nullptr) (void)cudaGraphExecDestroy(impl.graph_exec);
  if (impl.graph != nullptr) (void)cudaGraphDestroy(impl.graph);
  device_free(impl.host_state.tape);
  device_free(impl.host_state.context_tape);
  device_free(impl.host_state.counts);
  device_free(impl.host_state.embedding);
  device_free(impl.host_state.w_ih);
  device_free(impl.host_state.w_context);
  device_free(impl.host_state.w_hh);
  device_free(impl.host_state.bias);
  device_free(impl.host_state.readout);
  device_free(impl.host_state.readout_bias);
  device_free(impl.host_state.m_embedding);
  device_free(impl.host_state.v_embedding);
  device_free(impl.host_state.m_w_ih);
  device_free(impl.host_state.v_w_ih);
  device_free(impl.host_state.m_w_context);
  device_free(impl.host_state.v_w_context);
  device_free(impl.host_state.m_w_hh);
  device_free(impl.host_state.v_w_hh);
  device_free(impl.host_state.m_bias);
  device_free(impl.host_state.v_bias);
  device_free(impl.host_state.m_readout);
  device_free(impl.host_state.v_readout);
  device_free(impl.host_state.m_readout_bias);
  device_free(impl.host_state.v_readout_bias);
  device_free(impl.host_state.d_embedding);
  device_free(impl.host_state.d_w_ih);
  device_free(impl.host_state.d_w_context);
  device_free(impl.host_state.d_w_hh);
  device_free(impl.host_state.d_bias);
  device_free(impl.host_state.d_readout);
  device_free(impl.host_state.d_readout_bias);
  device_free(impl.host_state.x);
  device_free(impl.host_state.context_x);
  device_free(impl.host_state.targets);
  device_free(impl.host_state.h);
  device_free(impl.host_state.gates);
  device_free(impl.host_state.recurrent_n);
  device_free(impl.host_state.pre);
  device_free(impl.host_state.logits);
  device_free(impl.host_state.d_logits);
  device_free(impl.host_state.d_h_out);
  device_free(impl.host_state.d_h_next);
  device_free(impl.host_state.d_q);
  device_free(impl.host_state.d_pre);
  device_free(impl.host_state.d_x);
  device_free(impl.host_state.sample_loss);
  device_free(impl.host_state.step_loss);
  device_free(impl.host_state.eval_loss);
  device_free(impl.host_state.unigram_loss);
  device_free(impl.host_state.context_gradient_l1);
  device_free(impl.host_state.receipt);
  device_free(impl.host_state.output);
  device_free(impl.host_state.generation_prompt);
  device_free(impl.host_state.live_context);
  device_free(reinterpret_cast<std::uint8_t*&>(impl.cublas_workspace));
  if (impl.state != nullptr) (void)cudaFree(impl.state);
  if (impl.cublas != nullptr) (void)cublasDestroy(impl.cublas);
  if (impl.stream != nullptr) (void)cudaStreamDestroy(impl.stream);
}

}  // namespace

ResidentGruLanguage::ResidentGruLanguage(std::uint64_t founder, Config config)
    : impl_(new Impl), tape_size_(0u) {
  impl_->config = config;
  if (config.batch != kBatch || config.sequence != kSequence ||
      config.train_steps == 0u || config.tape_capacity < 2u ||
      config.tape_capacity > std::numeric_limits<std::uint32_t>::max()) {
    delete impl_;
    impl_ = nullptr;
    throw std::invalid_argument(
        "resident GRU requires batch=96, sequence=192, positive train_steps "
        "and a uint32 resident tape capacity");
  }
  try {
    check_cuda(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking),
               "create resident language stream");
    check_cublas(cublasCreate(&impl_->cublas), "create resident cuBLAS handle");
    check_cuda(cudaMalloc(&impl_->cublas_workspace,
                          Impl::kCublasWorkspaceBytes),
               "allocate resident cuBLAS workspace");
    check_cublas(cublasSetWorkspace(impl_->cublas, impl_->cublas_workspace,
                                    Impl::kCublasWorkspaceBytes),
                 "bind resident cuBLAS workspace");
    allocate_model(*impl_, founder);
    check_cuda(cudaDeviceSynchronize(), "synchronize resident founder");
    capture_training_graph(*impl_);
    capture_generation_graph(*impl_);
    capture_evaluation_graph(*impl_);
  } catch (...) {
    destroy_impl(*impl_);
    delete impl_;
    impl_ = nullptr;
    throw;
  }
}

ResidentGruLanguage::~ResidentGruLanguage() {
  if (impl_ != nullptr) {
    destroy_impl(*impl_);
    delete impl_;
    impl_ = nullptr;
  }
}

void ResidentGruLanguage::present_raw(std::span<const std::uint8_t> bytes) {
  if (bytes.empty()) return;
  if (bytes.size() > impl_->config.tape_capacity - tape_size_)
    throw std::length_error("resident byte tape capacity exceeded");
  check_cuda(cudaMemcpyAsync(impl_->host_state.tape + tape_size_, bytes.data(),
             bytes.size(), cudaMemcpyHostToDevice,
                             impl_->stream),
             "append raw resident tape");
  check_cuda(cudaMemsetAsync(
                 impl_->host_state.context_tape + tape_size_ * kContext, 0,
                 bytes.size() * static_cast<std::size_t>(kContext) *
                     sizeof(float),
                 impl_->stream),
             "append neutral resident context");
  check_cuda(cudaMemsetAsync(impl_->host_state.live_context, 0,
                             kContext * sizeof(float), impl_->stream),
             "clear resident live context");
  tape_size_ += bytes.size();
  impl_->tape_size = tape_size_;
  publish_tape_bounds(*impl_);
  // Small boundary contacts arrive continuously. Keep their copies and count
  // kernels ordered on the resident stream, but do not force 4,096 host/device
  // round trips for a 1 MiB life history. The full-tape boundary is the only
  // cross-stream handoff used by the persistent parent, so synchronize there.
  if (tape_size_ == impl_->config.tape_capacity)
    check_cuda(cudaStreamSynchronize(impl_->stream),
               "seal resident raw tape");
}

void ResidentGruLanguage::present_contextual(
    std::span<const std::uint8_t> bytes, std::span<const float> context) {
  if (bytes.empty()) return;
  if (context.size() != kContextWidth)
    throw std::invalid_argument("resident context vector has the wrong width");
  if (bytes.size() > impl_->config.tape_capacity - tape_size_)
    throw std::length_error("resident byte tape capacity exceeded");

  check_cuda(cudaMemcpyAsync(impl_->host_state.tape + tape_size_, bytes.data(),
                             bytes.size(), cudaMemcpyHostToDevice,
                             impl_->stream),
             "append contextual resident tape");
  check_cuda(cudaMemcpyAsync(impl_->host_state.live_context, context.data(),
                             kContext * sizeof(float), cudaMemcpyHostToDevice,
                             impl_->stream),
             "publish resident live context");
  fill_context_tape<<<256, 256, 0, impl_->stream>>>(
      impl_->host_state.context_tape, static_cast<std::uint32_t>(tape_size_),
      static_cast<std::uint32_t>(bytes.size()), impl_->host_state.live_context);
  check_cuda(cudaGetLastError(), "expand contextual resident snapshots");
  // The expanded bootstrap segment is temporary host storage. This is one
  // segment boundary synchronization, never a per-byte training round trip.
  check_cuda(cudaStreamSynchronize(impl_->stream),
             "synchronize contextual resident append");
  tape_size_ += bytes.size();
  impl_->tape_size = tape_size_;
  publish_tape_bounds(*impl_);
}

void ResidentGruLanguage::train_autonomous() {
  if (tape_size_ < 2u) throw std::runtime_error("resident tape is empty");
  if (impl_->tape_size < kSequence + 1u)
    throw std::runtime_error("resident tape is shorter than training sequence");
  const std::uint32_t zero = 0u;
  copy_state_scalar(impl_->state, offsetof(DeviceState, step), &zero,
                    sizeof(zero), impl_->stream, "reset resident train step");
  check_cuda(cudaGraphLaunch(impl_->graph_exec, impl_->stream),
             "bootstrap resident training graph");
  ++impl_->host_launches;
  check_cuda(cudaStreamSynchronize(impl_->stream),
             "synchronize autonomous resident training");
  run_evaluation(*impl_);
}

std::vector<std::uint8_t> ResidentGruLanguage::generate(
    std::span<const std::uint8_t> prompt, std::size_t max_bytes) {
  std::array<float, kContext> neutral{};
  return generate_with_context(prompt, neutral, max_bytes);
}

std::vector<std::uint8_t> ResidentGruLanguage::generate_with_context(
    std::span<const std::uint8_t> prompt, std::span<const float> context,
    std::size_t max_bytes) {
  if (context.size() != kContextWidth)
    throw std::invalid_argument("resident context vector has the wrong width");
  if (max_bytes > kMaxGeneration) max_bytes = kMaxGeneration;
  if (prompt.size() > kMaxGeneration) prompt = prompt.first(kMaxGeneration);
  check_cuda(cudaMemcpyAsync(impl_->host_state.live_context, context.data(),
                             kContext * sizeof(float), cudaMemcpyHostToDevice,
                             impl_->stream),
             "upload resident generation context");
  check_cuda(cudaMemcpyAsync(impl_->host_state.generation_prompt, prompt.data(),
                             prompt.size(),
                             cudaMemcpyHostToDevice, impl_->stream),
             "upload generation prompt");
  const std::uint32_t prompt_size = static_cast<std::uint32_t>(prompt.size());
  const std::uint32_t output_size = static_cast<std::uint32_t>(max_bytes);
  copy_state_scalar(impl_->state, offsetof(DeviceState, generation_prompt_size),
                    &prompt_size, sizeof(prompt_size), impl_->stream,
                    "publish generation prompt length");
  copy_state_scalar(impl_->state, offsetof(DeviceState, generation_output_size),
                    &output_size, sizeof(output_size), impl_->stream,
                    "publish generation output length");
  check_cuda(cudaGraphLaunch(impl_->generation_graph_exec, impl_->stream),
             "launch resident generation graph");
  check_cuda(cudaStreamSynchronize(impl_->stream),
             "synchronize resident generation");
  std::vector<std::uint8_t> result(max_bytes);
  check_cuda(cudaMemcpy(result.data(), impl_->host_state.output, max_bytes,
                        cudaMemcpyDeviceToHost),
             "read generated bytes");
  return result;
}

void ResidentGruLanguage::apply_test_lesion(LesionKind kind, std::size_t offset,
                                             std::size_t count) {
  if (count == 0u) return;
  if (count > 1u << 20u || offset > std::numeric_limits<std::uint32_t>::max())
    throw std::out_of_range("resident lesion exceeds bounded test seam");
  lesion_kernel<<<256, 256, 0, impl_->stream>>>(
      impl_->state, static_cast<std::uint32_t>(kind),
      static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(count));
  check_cuda(cudaGetLastError(), "apply resident physical lesion");
  check_cuda(cudaStreamSynchronize(impl_->stream),
             "synchronize resident physical lesion");
}

DeviceLaunchHandle ResidentGruLanguage::device_launch_handle() const noexcept {
  auto* base = reinterpret_cast<std::uint8_t*>(impl_->state);
  auto* receipt = reinterpret_cast<std::uint8_t*>(impl_->host_state.receipt);
  DeviceLaunchHandle handle{};
  handle.training_graph_exec = reinterpret_cast<void*>(impl_->graph_exec);
  handle.generation_graph_exec =
      reinterpret_cast<void*>(impl_->generation_graph_exec);
  handle.device_state = reinterpret_cast<void*>(impl_->state);
  handle.prompt_bytes = reinterpret_cast<void*>(impl_->host_state.generation_prompt);
  handle.prompt_size = base + offsetof(DeviceState, generation_prompt_size);
  handle.requested_output_size =
      base + offsetof(DeviceState, generation_output_size);
  handle.output_bytes = reinterpret_cast<void*>(impl_->host_state.output);
  handle.output_size = receipt + offsetof(DeviceReceipt, output_size);
  handle.trained_steps = receipt + offsetof(DeviceReceipt, trained_steps);
  handle.configured_train_steps =
      base + offsetof(DeviceState, configured_train_steps);
  handle.tape_bytes = reinterpret_cast<void*>(impl_->host_state.tape);
  handle.tape_size = base + offsetof(DeviceState, tape_size);
  handle.train_size = base + offsetof(DeviceState, train_size);
  handle.heldout_start = base + offsetof(DeviceState, heldout_start);
  handle.byte_counts = reinterpret_cast<void*>(impl_->host_state.counts);
  handle.context_vector = reinterpret_cast<void*>(impl_->host_state.live_context);
  handle.context_tape = reinterpret_cast<void*>(impl_->host_state.context_tape);
  handle.tape_capacity = base + offsetof(DeviceState, tape_capacity);
  handle.train_start = base + offsetof(DeviceState, train_start);
  handle.step = base + offsetof(DeviceState, step);
  handle.writable_train_steps = base + offsetof(DeviceState, train_steps);
  return handle;
}

void ResidentGruLanguage::attach_return_graph(void* parent_graph_exec) {
  cudaGraphExec_t graph = reinterpret_cast<cudaGraphExec_t>(parent_graph_exec);
  copy_state_scalar(impl_->state, offsetof(DeviceState, return_graph), &graph,
                    sizeof(graph), impl_->stream, "attach resident return graph");
  check_cuda(cudaStreamSynchronize(impl_->stream),
             "synchronize resident return graph attachment");
}

Receipt ResidentGruLanguage::read_receipt() const {
  check_cuda(cudaStreamSynchronize(impl_->stream),
             "synchronize resident receipt");
  DeviceReceipt device{};
  check_cuda(cudaMemcpy(&device, impl_->host_state.receipt, sizeof(device),
                        cudaMemcpyDeviceToHost),
             "read resident receipt");
  Receipt result;
  result.first_loss = device.first_loss;
  result.final_loss = device.final_loss;
  result.heldout_loss = device.heldout_loss;
  result.unigram_loss = device.unigram_loss;
  result.host_bootstrap_launches = impl_->host_launches;
  result.graph_instantiated_device_launch = impl_->graph_device_launch ? 1u : 0u;
  result.graph_tail_launch = impl_->graph_tail_launch ? 1u : 0u;
  result.trained_steps = device.trained_steps;
  result.context_gradient_l1 = device.context_gradient_l1;
  if (device.output_size != 0u) {
    result.output_bytes.resize(device.output_size);
    check_cuda(cudaMemcpy(result.output_bytes.data(), impl_->host_state.output,
                          device.output_size, cudaMemcpyDeviceToHost),
               "read resident output bytes");
  }
  return result;
}

}  // namespace substrate::bcc32::resident_language

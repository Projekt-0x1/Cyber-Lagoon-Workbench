#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_b3_factor_amplitude_matter.cuh"

// This is intentionally an arithmetic migration oracle, not a route owner.
// It preserves the complete fixed-point factor population and exposes no
// route, row, or semantic key supplied by the host.
namespace bcc32_b3_factor_regional_contact {

namespace factor = bcc32_b3_factor_amplitude_matter;
using substrate::bcc32::SiteWord;

constexpr std::uint32_t kBatch = factor::kBatch;
constexpr std::uint32_t kHidden = factor::kHidden;
constexpr std::uint32_t kInput = factor::kInput;
constexpr std::uint32_t kFactorFamilies = factor::kFactorFamilies;
constexpr std::uint32_t kFactorWords = factor::kFactorWords;
constexpr std::uint32_t kPopulationWords = factor::kPopulationWords;
constexpr std::uint32_t kGradientFamilies = 3u;
constexpr std::uint32_t kContactShift = 5u + 20u - 18u;
constexpr std::uint32_t kGradientShift = 18u + 4u - 20u;
constexpr std::int64_t kSignalLimit = (std::int64_t{1} << 30u) - 1ll;
constexpr std::int64_t kMaximumPreProduct = std::int64_t{128} * kSignalLimit;
constexpr std::uint32_t kDeltaWords = kFactorFamilies * kFactorWords;
constexpr std::uint32_t kInputGradientWords = kGradientFamilies * kHidden * kInput;
constexpr std::uint32_t kRecurrentGradientWords = kGradientFamilies * kHidden * kHidden;

static_assert(kContactShift == 7u, "factor contact fixed-point shift changed");
static_assert(kGradientShift == 2u, "factor gradient fixed-point shift changed");
static_assert(kMaximumPreProduct > 0ll &&
                  kMaximumPreProduct <= INT64_MAX / 2ll,
              "int64 must contain -128 times the bounded residual signal");
static_assert(kPopulationWords == 64512u, "canonical factor population changed");
static_assert(kDeltaWords == 49152u, "factor contact shape changed");

struct ContactReceipt {
  unsigned long long contact_count = 0ull;
  unsigned long long clamp_count = 0ull;
  unsigned long long invalid_amplitude_count = 0ull;
  unsigned long long pre_product_overflow_count = 0ull;
};

struct DeviceContactView {
  const SiteWord* population = nullptr;
  const std::int32_t* signal = nullptr;
  std::int8_t* contacted_delta = nullptr;
  std::int64_t* input_gradient = nullptr;
  std::int64_t* recurrent_gradient = nullptr;
  ContactReceipt* receipt = nullptr;
};

__host__ __device__ inline std::int64_t signed_round_divide_pow2(
    std::int64_t value, std::uint32_t shift) {
  if (shift == 0u)
    return value;
  const std::uint64_t magnitude =
      value < 0ll ? static_cast<std::uint64_t>(-(value + 1ll)) + 1ull
                  : static_cast<std::uint64_t>(value);
  const std::uint64_t rounded =
      (magnitude + (std::uint64_t{1} << (shift - 1u))) >> shift;
  return value < 0ll ? -static_cast<std::int64_t>(rounded)
                     : static_cast<std::int64_t>(rounded);
}

__device__ inline std::int32_t decode_or_zero(const SiteWord word,
                                               ContactReceipt* receipt) {
  const std::int32_t amplitude = factor::decode_amplitude(word);
  if (amplitude != factor::kInvalidAmplitude)
    return amplitude;
  if (receipt != nullptr)
    atomicAdd(&receipt->invalid_amplitude_count, 1ull);
  return 0;
}

__device__ inline std::int8_t contact_delta(std::int32_t amplitude,
                                             std::int32_t signal,
                                             ContactReceipt* receipt) {
  const std::int64_t product = static_cast<std::int64_t>(amplitude) *
                               static_cast<std::int64_t>(signal);
  // The static bound above makes this branch unreachable for valid inputs. It
  // is retained as a device receipt rather than narrowing before multiplication.
  if (product > kMaximumPreProduct || product < -kMaximumPreProduct) {
    if (receipt != nullptr)
      atomicAdd(&receipt->pre_product_overflow_count, 1ull);
    return 0;
  }
  const std::int64_t rounded = signed_round_divide_pow2(product, kContactShift);
  if (rounded > 127ll) {
    if (receipt != nullptr)
      atomicAdd(&receipt->clamp_count, 1ull);
    return 127;
  }
  if (rounded < -127ll) {
    if (receipt != nullptr)
      atomicAdd(&receipt->clamp_count, 1ull);
    return -127;
  }
  return static_cast<std::int8_t>(rounded);
}

static __global__ void factor_contact_kernel(DeviceContactView view) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kDeltaWords || view.population == nullptr || view.signal == nullptr ||
      view.contacted_delta == nullptr)
    return;
  const std::uint32_t local = index % kFactorWords;
  const std::int32_t amplitude = decode_or_zero(view.population[index], view.receipt);
  view.contacted_delta[index] = contact_delta(amplitude, view.signal[local], view.receipt);
  if (view.receipt != nullptr)
    atomicAdd(&view.receipt->contact_count, 1ull);
}

static __global__ void input_gradient_kernel(DeviceContactView view) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kInputGradientWords || view.population == nullptr ||
      view.contacted_delta == nullptr || view.input_gradient == nullptr)
    return;
  const std::uint32_t input_index = index % kInput;
  const std::uint32_t row = index / kInput;
  const std::uint32_t family = row / kHidden;
  const std::uint32_t hidden = row % kHidden;
  std::int64_t sum = 0ll;
  for (std::uint32_t batch = 0u; batch < kBatch; ++batch) {
    const std::uint32_t delta_index = family * kFactorWords + batch * kHidden + hidden;
    const std::uint32_t input_word = factor::family_offset(kFactorFamilies) +
                                     batch * kInput + input_index;
    sum += static_cast<std::int64_t>(view.contacted_delta[delta_index]) *
           static_cast<std::int64_t>(decode_or_zero(view.population[input_word], view.receipt));
  }
  view.input_gradient[index] = signed_round_divide_pow2(sum, kGradientShift);
}

static __global__ void recurrent_gradient_kernel(DeviceContactView view) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kRecurrentGradientWords || view.population == nullptr ||
      view.contacted_delta == nullptr || view.recurrent_gradient == nullptr)
    return;
  const std::uint32_t previous_index = index % kHidden;
  const std::uint32_t row = index / kHidden;
  const std::uint32_t gradient_family = row / kHidden;
  const std::uint32_t hidden = row % kHidden;
  // Recurrent candidates occupy factor family three; input candidates do not.
  const std::uint32_t factor_family = gradient_family == 2u ? 3u : gradient_family;
  std::int64_t sum = 0ll;
  for (std::uint32_t batch = 0u; batch < kBatch; ++batch) {
    const std::uint32_t delta_index = factor_family * kFactorWords +
                                      batch * kHidden + hidden;
    const std::uint32_t previous_word = factor::family_offset(kFactorFamilies + 1u) +
                                        batch * kHidden + previous_index;
    sum += static_cast<std::int64_t>(view.contacted_delta[delta_index]) *
           static_cast<std::int64_t>(decode_or_zero(view.population[previous_word], view.receipt));
  }
  view.recurrent_gradient[index] = signed_round_divide_pow2(sum, kGradientShift);
}

}  // namespace bcc32_b3_factor_regional_contact

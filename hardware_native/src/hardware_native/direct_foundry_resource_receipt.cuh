#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_RESOURCE_RECEIPT_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_RESOURCE_RECEIPT_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_network {

// Context-bound engineering measurements. They compare execution cost only;
// no score or field here can express truth, meaning, participation or credit.
struct DirectFoundryResourceReceiptV1 {
  std::uint32_t abi_version = 1u;
  std::uint32_t sample_count = 0u;
  std::uint64_t source_latency_ns = 0u;
  std::uint64_t candidate_latency_ns = 0u;
  std::uint64_t source_active_work = 0u;
  std::uint64_t candidate_active_work = 0u;
  std::uint64_t source_memory_bytes = 0u;
  std::uint64_t candidate_memory_bytes = 0u;
  std::uint64_t source_precision_error_q32 = 0u;
  std::uint64_t candidate_precision_error_q32 = 0u;
  std::uint64_t source_failure_cost = 0u;
  std::uint64_t candidate_failure_cost = 0u;
  DirectSha256Address source_candidate{};
  DirectSha256Address candidate{};
  DirectSha256Address task{};
  DirectSha256Address guard{};
  DirectSha256Address body_regime{};
  DirectSha256Address evaluator{};
  DirectSha256Address resource_regime{};
  DirectSha256Address benchmark{};
  DirectSha256Address reproducibility{};
};
static_assert(std::is_trivially_copyable_v<DirectFoundryResourceReceiptV1> &&
              std::is_standard_layout_v<DirectFoundryResourceReceiptV1>);

#if defined(__CUDACC__)
#define DIRECT_FOUNDRY_RESOURCE_HD __host__ __device__
#else
#define DIRECT_FOUNDRY_RESOURCE_HD
#endif

DIRECT_FOUNDRY_RESOURCE_HD inline bool direct_foundry_resource_address_nonzero(
    const DirectSha256Address& address) {
  std::uint8_t any = 0u;
  for (std::uint8_t byte : address.byte) any |= byte;
  return any != 0u;
}

DIRECT_FOUNDRY_RESOURCE_HD inline void direct_foundry_resource_update_u32(
    detail::DirectSha256State* state, std::uint32_t value) {
  const std::uint8_t bytes[4] = {
      static_cast<std::uint8_t>(value),
      static_cast<std::uint8_t>(value >> 8u),
      static_cast<std::uint8_t>(value >> 16u),
      static_cast<std::uint8_t>(value >> 24u)};
  state->update(bytes, sizeof(bytes));
}

DIRECT_FOUNDRY_RESOURCE_HD inline void direct_foundry_resource_update_u64(
    detail::DirectSha256State* state, std::uint64_t value) {
  std::uint8_t bytes[8];
  for (std::uint32_t i = 0u; i < 8u; ++i)
    bytes[i] = static_cast<std::uint8_t>(value >> (i * 8u));
  state->update(bytes, sizeof(bytes));
}

DIRECT_FOUNDRY_RESOURCE_HD inline DirectSha256Address
direct_foundry_resource_receipt_address(
    const DirectFoundryResourceReceiptV1& receipt) {
  static constexpr char kDomain[] = "0x1-direct-foundry-resource-receipt-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_foundry_resource_update_u32(&state, receipt.abi_version);
  direct_foundry_resource_update_u32(&state, receipt.sample_count);
  const std::uint64_t measures[] = {
      receipt.source_latency_ns,
      receipt.candidate_latency_ns,
      receipt.source_active_work,
      receipt.candidate_active_work,
      receipt.source_memory_bytes,
      receipt.candidate_memory_bytes,
      receipt.source_precision_error_q32,
      receipt.candidate_precision_error_q32,
      receipt.source_failure_cost,
      receipt.candidate_failure_cost};
  for (std::uint64_t value : measures)
    direct_foundry_resource_update_u64(&state, value);
  const DirectSha256Address* roots[] = {
      &receipt.source_candidate, &receipt.candidate,       &receipt.task,
      &receipt.guard,            &receipt.body_regime,     &receipt.evaluator,
      &receipt.resource_regime,  &receipt.benchmark,       &receipt.reproducibility};
  for (const DirectSha256Address* root : roots)
    state.update(root->byte, sizeof(root->byte));
  return state.finish();
}

DIRECT_FOUNDRY_RESOURCE_HD inline bool
direct_foundry_resource_receipt_valid(
    const DirectFoundryResourceReceiptV1& receipt) {
  if (receipt.abi_version != 1u || receipt.sample_count == 0u ||
      receipt.source_latency_ns == 0u || receipt.candidate_latency_ns == 0u ||
      receipt.source_active_work == 0u ||
      receipt.candidate_active_work == 0u ||
      receipt.source_memory_bytes == 0u ||
      receipt.candidate_memory_bytes == 0u)
    return false;
  const DirectSha256Address* roots[] = {
      &receipt.source_candidate, &receipt.candidate,       &receipt.task,
      &receipt.guard,            &receipt.body_regime,     &receipt.evaluator,
      &receipt.resource_regime,  &receipt.benchmark,       &receipt.reproducibility};
  for (const DirectSha256Address* root : roots)
    if (!direct_foundry_resource_address_nonzero(*root)) return false;
  const bool no_worse =
      receipt.candidate_latency_ns <= receipt.source_latency_ns &&
      receipt.candidate_active_work <= receipt.source_active_work &&
      receipt.candidate_memory_bytes <= receipt.source_memory_bytes &&
      receipt.candidate_precision_error_q32 <=
          receipt.source_precision_error_q32 &&
      receipt.candidate_failure_cost <= receipt.source_failure_cost;
  const bool strictly_better =
      receipt.candidate_latency_ns < receipt.source_latency_ns ||
      receipt.candidate_active_work < receipt.source_active_work ||
      receipt.candidate_memory_bytes < receipt.source_memory_bytes ||
      receipt.candidate_precision_error_q32 <
          receipt.source_precision_error_q32 ||
      receipt.candidate_failure_cost < receipt.source_failure_cost;
  return no_worse && strictly_better;
}

#undef DIRECT_FOUNDRY_RESOURCE_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_FOUNDRY_RESOURCE_RECEIPT_CUH

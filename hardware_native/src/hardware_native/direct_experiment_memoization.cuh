#ifndef HARDWARE_NATIVE_DIRECT_EXPERIMENT_MEMOIZATION_CUH
#define HARDWARE_NATIVE_DIRECT_EXPERIMENT_MEMOIZATION_CUH

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hardware_native/direct_experiment_identity.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kDirectExperimentMemoKeyAbiV2 = 2u;

// Ephemeral foundry index key. DirectExperimentIdentityV1 binds Gamma,
// environment, body, and protocol; executor identity completes the exact
// same-state/same-executor replay boundary required before reuse.
struct DirectExperimentMemoKeyV2 {
  DirectExperimentIdentityV1 experiment{};
  DirectSha256Address executor{};
  DirectSha256Address root{};
};

struct DirectExperimentMemoEntryV2 {
  DirectExperimentMemoKeyV2 key{};
  DirectSha256Address trajectory{};
  DirectSha256Address receipt{};
};

inline bool direct_address_is_nonzero(const DirectSha256Address& address) {
  std::uint8_t any = 0u;
  for (std::uint8_t byte : address.byte) any |= byte;
  return any != 0u;
}

inline bool direct_experiment_memo_key(const DirectExperimentIdentityV1& experiment,
                                       const void* executor_bytes,
                                       std::size_t executor_size,
                                       DirectExperimentMemoKeyV2* out) {
  if (out == nullptr || !direct_address_is_nonzero(experiment.root) ||
      executor_size == 0u || executor_bytes == nullptr)
    return false;

  out->experiment = experiment;
  if (!direct_sha256_content_address(executor_bytes, executor_size, &out->executor))
    return false;

  static constexpr char kDomain[] = "0x1-direct-experiment-memo-key-v2";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  const std::uint32_t tag = kDirectExperimentMemoKeyAbiV2;
  state.update(&tag, sizeof(tag));
  state.update(experiment.root.byte, sizeof(experiment.root.byte));
  state.update(out->executor.byte, sizeof(out->executor.byte));
  out->root = state.finish();
  return true;
}

inline bool same_direct_experiment_memo_key(const DirectExperimentMemoKeyV2& left,
                                            const DirectExperimentMemoKeyV2& right) {
  return left.root == right.root && left.experiment.root == right.experiment.root &&
         left.executor == right.executor;
}

// Rebuildable linear index over canonical receipts. It owns no evidence and
// confers no good/bad, promotion, reward, or resident authority.
class DirectExperimentMemoization {
 public:
  bool lookup(const DirectExperimentMemoKeyV2& key,
              DirectExperimentMemoEntryV2* out) const {
    if (out == nullptr) return false;
    for (const DirectExperimentMemoEntryV2& entry : entries_)
      if (same_direct_experiment_memo_key(entry.key, key)) {
        *out = entry;
        return true;
      }
    return false;
  }

  bool remember(const DirectExperimentMemoEntryV2& candidate) {
    if (!direct_address_is_nonzero(candidate.key.root) ||
        !direct_address_is_nonzero(candidate.trajectory) ||
        !direct_address_is_nonzero(candidate.receipt))
      return false;
    for (const DirectExperimentMemoEntryV2& entry : entries_) {
      if (entry.key.root != candidate.key.root) continue;
      return same_direct_experiment_memo_key(entry.key, candidate.key) &&
             entry.trajectory == candidate.trajectory && entry.receipt == candidate.receipt;
    }
    entries_.push_back(candidate);
    return true;
  }

  std::size_t size() const { return entries_.size(); }
  void clear() { entries_.clear(); }

 private:
  std::vector<DirectExperimentMemoEntryV2> entries_;
};

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_EXPERIMENT_MEMOIZATION_CUH

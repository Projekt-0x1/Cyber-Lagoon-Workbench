inline void learn_distributed_motor_bytes(AdultState& state,
                                          const std::uint8_t* device_bytes,
                                          std::uint32_t byte_count) {
  if (!state.distributed_motor_enabled || byte_count == 0u) return;
  const distributed_motor::DeviceStateView view = distributed_motor_view(state);
  for (std::uint32_t begin = 0u; begin < byte_count;
       begin += kDistributedMotorScratchSteps) {
    const std::uint32_t count =
        std::min(kDistributedMotorScratchSteps, byte_count - begin);
    cuda_require(distributed_motor::learn_raw(
                     view, device_bytes + begin, count, 0u),
                 "learn distributed raw-event sequence");
  }
}

inline void assimilate_distributed_contact(AdultState& state,
                                           const std::uint8_t* host_bytes,
                                           std::uint32_t byte_count) {
  if (!state.distributed_motor_enabled || host_bytes == nullptr || byte_count == 0u)
    return;
  DeviceArray<std::uint8_t> device_bytes(byte_count);
  cuda_require(cudaMemcpy(device_bytes.get(), host_bytes, byte_count,
                          cudaMemcpyHostToDevice),
               "upload continuous distributed contact");
  learn_distributed_motor_bytes(state, device_bytes.get(), byte_count);
  cuda_require(cudaDeviceSynchronize(), "complete continuous distributed contact");
}

struct TrainReport {
  std::uint32_t unit_occurrences;
  std::uint32_t unique_units;
  std::uint32_t unique_bigrams;
  std::uint32_t unique_trigrams;
  std::uint32_t online_bigrams;
  std::uint32_t online_trigrams;
  std::uint32_t online_associations;
  std::uint32_t online_conditioned_transitions;
  std::uint32_t online_episode_units;
  std::uint32_t mass_budget;
  std::uint32_t mass_reserve;
  std::uint32_t occupied_mass;
  std::uint32_t ledger_ok;
  std::uint32_t boundary_bytes[kBoundaryCount];
  std::uint32_t motor_context[kMotorWords];
  std::uint32_t motor_completion[kCompositionUnits];
  proposition_tissue::TissueScalars proposition_tissue{};
  ConstructionAssociationReceipt construction_association{};
  resident_credit::BankScalars conditioned_credit{};
  std::uint32_t ordered_binding_capacity = 0u;
  std::uint32_t ordered_construction_capacity = 0u;
  std::uint32_t construction_association_capacity = 0u;
  std::uint32_t relation_triple_capacity = 0u;
  std::size_t resident_bytes;
};

using EfferenceReport = ResidentEfferenceState;
using InteractionShadowReport = ResidentInteractionShadowState;

__device__ __forceinline__ std::uint32_t mix32(std::uint32_t x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

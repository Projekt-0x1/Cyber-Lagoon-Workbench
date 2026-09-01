#ifndef HARDWARE_NATIVE_DIRECT_EXPERIMENT_IDENTITY_CUH
#define HARDWARE_NATIVE_DIRECT_EXPERIMENT_IDENTITY_CUH

#include <cstddef>
#include <cstdint>

#include "hardware_native/direct_content_address.cuh"
#include "hardware_native/direct_network_brain.cuh"

#if defined(__CUDACC__)
#define DIRECT_EXPERIMENT_IDENTITY_HD __host__ __device__
#else
#define DIRECT_EXPERIMENT_IDENTITY_HD
#endif

namespace substrate::direct_network {

inline constexpr std::uint32_t kDirectExperimentIdentityAbiV1 = 1u;

// gh #1290, i.exact_experiment_identity: one SHA-256 digest over the four
// inputs an experiment must pin -- the genome (whose authored Gamma carries
// the developmental clock), the developmental environment, the body
// geometry, and the opaque test-protocol bytes. The identity is computable
// before birth; two identical experiments hash identically and perturbing
// any one leg moves exactly its component address and the root.
struct DirectExperimentIdentityV1 {
  DirectSha256Address genome{};
  DirectSha256Address development{};
  DirectSha256Address body{};
  DirectSha256Address protocol{};
  DirectSha256Address root{};
};

DIRECT_EXPERIMENT_IDENTITY_HD inline bool direct_experiment_identity(
    const DirectGenomeV1& genome, const DirectDevelopmentEnvironmentV1& development,
    const DirectBodyManifestV1& body, const void* protocol_bytes,
    std::size_t protocol_size, DirectExperimentIdentityV1* out) {
  if (out == nullptr || (protocol_size != 0u && protocol_bytes == nullptr)) return false;
  if (development.constraint_count > kMaxDevelopmentConstraints ||
      body.binding_count > kMaxBoundaryPorts)
    return false;
  if (!direct_sha256_genome_address(genome, &out->genome)) return false;

  // Same in-use byte rules as canonical_environment_root/canonical_body_root:
  // unused trailing capacity has no authority over the identity.
  detail::DirectSha256State state{};
  state.update(&development,
               offsetof(DirectDevelopmentEnvironmentV1, constraints) +
                   sizeof(DevelopmentEnvironmentConstraint) * development.constraint_count);
  out->development = state.finish();

  state = detail::DirectSha256State{};
  state.update(&body, offsetof(DirectBodyManifestV1, bindings) +
                           sizeof(BoundaryPortBinding) * body.binding_count);
  out->body = state.finish();

  if (!direct_sha256_content_address(protocol_bytes, protocol_size, &out->protocol))
    return false;

  state = detail::DirectSha256State{};
  const std::uint32_t tag = kDirectExperimentIdentityAbiV1;
  state.update(&tag, sizeof(tag));
  for (const DirectSha256Address* component :
       {&out->genome, &out->development, &out->body, &out->protocol})
    state.update(component->byte, sizeof(component->byte));
  out->root = state.finish();
  return true;
}

}  // namespace substrate::direct_network

#undef DIRECT_EXPERIMENT_IDENTITY_HD

#endif

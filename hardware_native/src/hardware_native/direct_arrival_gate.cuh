// h.verify_arrival -- cryptographic and physical verification that sensory
// inputs arrived intact through the skull boundary. The sender seals the
// pending ingress staging range before the membrane ships it; the receiver
// recomputes the digest over what actually landed in the device queue and,
// on mismatch, scrubs the landed bytes and retracts the publication so
// unverified bytes can never reach the resident.
#ifndef HARDWARE_NATIVE_DIRECT_ARRIVAL_GATE_CUH
#define HARDWARE_NATIVE_DIRECT_ARRIVAL_GATE_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "direct_adult_core.cuh"
#include "direct_content_address.cuh"

namespace substrate::direct_adult_core {

struct DirectArrivalSealV1 {
  direct_network::DirectSha256Address payload_digest{};
  std::uint32_t first_event = 0u;
  std::uint32_t event_count = 0u;
};

struct DirectArrivalGateReceiptV1 {
  std::uint32_t verified = 0u;
  std::uint32_t digest_mismatch = 0u;
  std::uint32_t scrubbed_events = 0u;
};

// Sender side: content-address the exact staging bytes the membrane is about
// to ship. Empty pending range yields an empty seal (event_count 0).
DirectArrivalSealV1 seal_sensory_arrival_v1(const DirectAdultRuntime* runtime);

// Receiver side: recompute the digest over the landed queue range and enforce
// the boundary -- on mismatch the landed events are scrubbed and the
// publication cursor is retracted before any resident step can observe them.
// Stream-ordered after the flush that shipped the range; must be called
// before stepping the adult.
DirectArrivalGateReceiptV1 enforce_sensory_arrival_v1(
    DirectAdultRuntime* runtime, const DirectArrivalSealV1& seal,
    const direct_network::DirectSha256Address* declared_digest_device);

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ARRIVAL_GATE_CUH

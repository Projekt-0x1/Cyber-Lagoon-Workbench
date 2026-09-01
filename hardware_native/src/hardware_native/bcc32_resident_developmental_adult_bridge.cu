#include "bcc32_resident_developmental_adult_bridge.hpp"

#include "bcc32_developmental_adult.cuh"

namespace substrate::bcc32::developmental_adult {

GrownAdult* make_resident_founder_grown_adult(
    std::int64_t aperture_edge_chunks) {
  return new GrownAdult(kFounderHash, FounderVariant::intact,
                        aperture_edge_chunks);
}

ContentAddress resident_genesis_manifest_identity(const GrownAdult* adult) {
  return adult->genesis_manifest_identity();
}

device_ordinary_f_timeline::DeviceOrdinaryFTimeline*
claim_resident_ordinary_f_timeline(GrownAdult* adult, std::uint32_t capacity,
                                   std::uint32_t history_depth) {
  return adult->claim_device_ordinary_f_timeline(capacity, history_depth)
      .release();
}

void destroy_resident_grown_adult(GrownAdult* adult) { delete adult; }

}  // namespace substrate::bcc32::developmental_adult

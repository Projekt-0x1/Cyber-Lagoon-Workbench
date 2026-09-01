#include "bcc32_developmental_adult_bridge.hpp"

#include "bcc32_developmental_adult.cuh"

namespace substrate::bcc32::developmental_adult {

GrownAdult* make_founder_grown_adult() {
  return new GrownAdult(kFounderHash);
}

DevelopmentalHash founder_hash() { return kFounderHash; }

ContentAddress genesis_manifest_identity(const GrownAdult* adult) {
  return adult->genesis_manifest_identity();
}

device_ordinary_f_timeline::DeviceOrdinaryFTimeline*
claim_ordinary_f_timeline(GrownAdult* adult) {
  return adult->claim_device_ordinary_f_timeline().release();
}

void destroy_grown_adult(GrownAdult* adult) { delete adult; }

}  // namespace substrate::bcc32::developmental_adult

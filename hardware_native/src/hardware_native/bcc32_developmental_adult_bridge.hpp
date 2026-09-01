#pragma once

// Opaque host surface over GrownAdult for TUs that must not compile the
// developmental kernel library carried by bcc32_developmental_adult.cuh.
// The operations below are the exact member calls they name, compiled once
// in bcc32_developmental_adult_bridge.cu.

#include "bcc32_developmental_population_seed.cuh"
#include "bcc32_provenance.hpp"

namespace substrate::bcc32::device_ordinary_f_timeline {
class DeviceOrdinaryFTimeline;
}

namespace substrate::bcc32::developmental_adult {

class GrownAdult;

[[nodiscard]] GrownAdult* make_founder_grown_adult();
[[nodiscard]] DevelopmentalHash founder_hash();
[[nodiscard]] ContentAddress genesis_manifest_identity(
    const GrownAdult* adult);
[[nodiscard]] device_ordinary_f_timeline::DeviceOrdinaryFTimeline*
claim_ordinary_f_timeline(GrownAdult* adult);
void destroy_grown_adult(GrownAdult* adult);

}  // namespace substrate::bcc32::developmental_adult

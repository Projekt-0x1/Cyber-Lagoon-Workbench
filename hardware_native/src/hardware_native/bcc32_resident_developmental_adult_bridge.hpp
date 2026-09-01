#pragma once

#include "bcc32_provenance.hpp"
#include <cstdint>

namespace substrate::bcc32::device_ordinary_f_timeline {
class DeviceOrdinaryFTimeline;
}
namespace substrate::bcc32::developmental_adult {
class GrownAdult;

[[nodiscard]] GrownAdult* make_resident_founder_grown_adult(
    std::int64_t aperture_edge_chunks);
[[nodiscard]] ContentAddress resident_genesis_manifest_identity(
    const GrownAdult* adult);
[[nodiscard]] device_ordinary_f_timeline::DeviceOrdinaryFTimeline*
claim_resident_ordinary_f_timeline(GrownAdult* adult, std::uint32_t capacity,
                                   std::uint32_t history_depth);
void destroy_resident_grown_adult(GrownAdult* adult);
}  // namespace substrate::bcc32::developmental_adult

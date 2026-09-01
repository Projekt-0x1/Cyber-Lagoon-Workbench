#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "hardware_native/bcc32_developmental_adult.cuh"
#include "hardware_native/bcc32_resident_edge_bank.cuh"

// Companion to bcc32_resident_edge_bank.cuh, split out exactly as
// bcc32_grown_instance_tissue.cuh is split from bcc32_grown_instance_basin.cuh:
// this file depends on the full GrownAdult definition (physical_slot(),
// StateEntry, attach_founder_matter()) so it cannot be included FROM
// bcc32_developmental_adult.cuh itself without a cycle. Only test/consumer
// code includes this file.
namespace substrate::bcc32::resident_edge_bank_tissue {

namespace factor = resident_edge_bank;
using substrate::bcc32::SiteWord;
using developmental_adult::GrownAdult;
using developmental_adult::StateEntry;

inline factor::DeviceLayout make_layout(const GrownAdult& grown) {
  factor::DeviceLayout layout{};
  for (std::uint32_t index = 0u; index < factor::kPhysicalRailCount;
       ++index) {
    const factor::PhysicalOffset offset = factor::physical_offset(index);
    layout.rails[index] =
        grown.physical_slot({offset.x, offset.y, offset.z});
  }
  return layout;
}

inline std::vector<StateEntry> founder_entries(const GrownAdult& grown) {
  const factor::DeviceLayout layout = make_layout(grown);
  std::vector<StateEntry> entries;
  entries.reserve(factor::kResidentPhysicalRailCount);
  for (std::uint32_t index = 0u;
       index < factor::kResidentPhysicalRailCount; index += 2u) {
    const std::uint32_t field = index / 2u;
    SiteWord value = 0u;
    if (field == factor::kFactorMarker) value = factor::kFactorMarkerValue;
    if (field == factor::kLayoutVersion) value = factor::kLayoutVersionValue;
    entries.push_back({layout.rails[index], value});
    entries.push_back({layout.rails[index + 1u], ~value});
  }
  return entries;
}

}  // namespace substrate::bcc32::resident_edge_bank_tissue

#pragma once

#include "bcc32_provenance.hpp"

namespace substrate::bcc32 {

// Canonical SHA-256 address of the fixed BCC-32 word ABI, geometry, executable
// gate descriptors, primitive truth tables, factor order, and streaming map. This is
// provenance metadata, never matter.
[[nodiscard]] ContentAddress canonical_law_identity();

// Residency and chunking identify an execution profile, not a physics epoch.
// Different Gold Rush hardware profiles may continue one law-bound lineage.
[[nodiscard]] ContentAddress execution_profile_identity(
    std::uint32_t aperture_chunks);
[[nodiscard]] ContentAddress canonical_execution_profile_identity();

} // namespace substrate::bcc32

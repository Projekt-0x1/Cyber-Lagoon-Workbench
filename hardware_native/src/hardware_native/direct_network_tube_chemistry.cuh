#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_TUBE_CHEMISTRY_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_TUBE_CHEMISTRY_CUH

#include <cstdint>

#include "hardware_native/direct_network_brain.cuh"

#if defined(__CUDACC__)
#define DIRECT_TUBE_CHEMISTRY_HD __host__ __device__
#else
#define DIRECT_TUBE_CHEMISTRY_HD
#endif

namespace substrate::direct_network {

// gh #1294, c.contextual_tube_chemistry: the local chemical state of an
// axonal routing tube modulates how strongly it transmits and how plastic it
// is. The channel rides in the chemotype identity word itself -- both
// endpoints express their tube chemistry in the top byte, bits 24..31 -- so
// Gamma authors it through the same field that already addresses constructor
// rules, and a lawful #930 chemotype write re-modulates the tube without host
// rebinding. The gains are a pure local function of the endpoint pair: no
// global state, no storage, and no authority over participation, credit
// sign, or evidence.
//
// Inert by construction on every pre-existing genome: surveyed chemotype
// identities stay below bit 20, so every unauthored tube maps to the exact
// neutral gain and mul_q16(x, kQ16One) == x bitwise.
inline constexpr std::int32_t kTubeChemistryNeutralQ16 = 1 << 16;

struct DirectTubeChemistryQ16 {
  std::int32_t conductance_gain_q16;
  std::int32_t plasticity_gain_q16;
};

DIRECT_TUBE_CHEMISTRY_HD inline std::uint32_t direct_tube_chemistry_byte(std::uint32_t chemotype) {
  return (chemotype >> 24u) & 0xFFu;
}

DIRECT_TUBE_CHEMISTRY_HD inline DirectTubeChemistryQ16 direct_tube_chemistry_q16(
    std::uint32_t source_chemotype, std::uint32_t target_chemotype) {
  const std::int32_t source_byte =
      static_cast<std::int32_t>(direct_tube_chemistry_byte(source_chemotype));
  const std::int32_t target_byte =
      static_cast<std::int32_t>(direct_tube_chemistry_byte(target_chemotype));
  // A tube is chemically live only when both endpoints author the byte;
  // one-sided or absent authoring leaves the tube at the exact neutral gain.
  if (source_byte == 0 || target_byte == 0)
    return DirectTubeChemistryQ16{kTubeChemistryNeutralQ16, kTubeChemistryNeutralQ16};

  // Signed product of the departures from the authored midpoint: like
  // polarities amplify the tube, opposite polarities dampen it. Byte operands
  // bound the product to [-16256, 16256], so both sums stay inside +-25% of
  // neutral by construction; the plasticity moment moves at one quarter rate.
  const std::int32_t affinity = (source_byte - 128) * (target_byte - 128);
  return DirectTubeChemistryQ16{kTubeChemistryNeutralQ16 + affinity,
                                kTubeChemistryNeutralQ16 + (affinity >> 2)};
}

}  // namespace substrate::direct_network

#undef DIRECT_TUBE_CHEMISTRY_HD

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_TUBE_CHEMISTRY_CUH

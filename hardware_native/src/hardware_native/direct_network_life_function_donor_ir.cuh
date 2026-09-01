#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_LIFE_FUNCTION_DONOR_IR_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_LIFE_FUNCTION_DONOR_IR_CUH

#include "hardware_native/direct_network_life_function.cuh"

namespace substrate::direct_network {

// Historical/donor-only entry for regression contracts that still construct a
// Gamma POD directly. It is not the canonical Direct compiler boundary and
// must not be used to advance adult capability claims. The canonical entry is
// `compile_direct_brain(const DirectGenomeV1&, ...)`.
DirectBirthReceiptV1 compile_donor_gamma_ir_direct_brain(
    const GammaV1& lowered_gamma, const DirectBodyManifestV1& body,
    const DirectDevelopmentEnvironmentV1& environment, DirectBrain* out_brain,
    DirectCompileOptions options = {});

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_LIFE_FUNCTION_DONOR_IR_CUH

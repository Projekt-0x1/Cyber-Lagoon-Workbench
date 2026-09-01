#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_PHASE_WIRING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_PHASE_WIRING_CUH

namespace substrate::direct_adult_core {

struct DirectAdultRuntime;

// Thin host-stepped phase boundary for continuing Adult-owned language plasticity.
// These functions carry no semantic routing; their implementation consumes only
// the runtime's ordinary exact-history/action/efference state.
void launch_resident_language_assimilation_phase(DirectAdultRuntime* runtime);
void launch_resident_language_motor_finalize_phase(DirectAdultRuntime* runtime);

}  // namespace substrate::direct_adult_core

#endif

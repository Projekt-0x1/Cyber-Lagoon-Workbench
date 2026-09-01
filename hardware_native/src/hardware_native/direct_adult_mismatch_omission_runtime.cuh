#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MISMATCH_OMISSION_RUNTIME_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MISMATCH_OMISSION_RUNTIME_CUH

#include "hardware_native/direct_adult_mismatch_omission.cuh"

struct alignas(8) ResidentMismatchOmissionRuntime {
  ResidentMismatchOmissionFrontier frontier;
};
static_assert(std::is_standard_layout_v<ResidentMismatchOmissionRuntime> &&
              std::is_trivial_v<ResidentMismatchOmissionRuntime> &&
              std::has_unique_object_representations_v<
                  ResidentMismatchOmissionRuntime>);

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_MISMATCH_OMISSION_RUNTIME_CUH

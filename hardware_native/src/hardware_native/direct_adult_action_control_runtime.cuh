#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ACTION_CONTROL_RUNTIME_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ACTION_CONTROL_RUNTIME_CUH
#include <type_traits>
#include "hardware_native/direct_causal_program.cuh"
#include "hardware_native/direct_adult_somatic_marker.cuh"
#include "hardware_native/direct_adult_volitional_veto.cuh"
namespace substrate::direct_adult_core {
struct DirectAdultActionControlRuntimeBlock {
  direct_causal_program::ProgramBank programs;
  direct_network::DirectSomaticMarkerState somatic;
  direct_network::ResidentPersistentCommitment commitment;
};
static_assert(std::is_trivial_v<DirectAdultActionControlRuntimeBlock> &&
              std::is_standard_layout_v<DirectAdultActionControlRuntimeBlock>);
}
#endif

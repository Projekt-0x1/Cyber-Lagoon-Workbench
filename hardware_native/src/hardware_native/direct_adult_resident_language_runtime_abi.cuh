#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_RUNTIME_ABI_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_RUNTIME_ABI_CUH

#include <cstddef>

struct CUstream_st;

namespace substrate::direct_network {

struct DirectExactHistoryHotPage;
struct DirectResidentLanguageRuntimeBlock;

// Host-side ownership ABI for the continuing Adult's language-plasticity
// factor.  Core/checkpoint code intentionally sees only this opaque boundary;
// the heavy device-language mechanism headers stay out of always-on Adult Core
// translation units.
std::size_t direct_resident_language_runtime_storage_bytes() noexcept;
DirectResidentLanguageRuntimeBlock* create_direct_resident_language_runtime();
void destroy_direct_resident_language_runtime(
    DirectResidentLanguageRuntimeBlock* state) noexcept;

__device__ void direct_resident_language_assimilate_owned(
    DirectResidentLanguageRuntimeBlock* state,
    const DirectExactHistoryHotPage* history);
void launch_direct_resident_language_assimilation(
    DirectResidentLanguageRuntimeBlock* state,
    const DirectExactHistoryHotPage* history, CUstream_st* stream);

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_RUNTIME_ABI_CUH

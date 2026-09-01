#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CORE_CONSTANTS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CORE_CONSTANTS_CUH
#include <cstdint>
#if defined(__CUDACC__)
#define DIRECT_ADULT_HD __host__ __device__
#else
#define DIRECT_ADULT_HD
#endif
namespace substrate::direct_adult_core {
using Word = std::uint32_t;
inline constexpr std::uint32_t kInvalidIndex = 0xffffffffu;
inline constexpr std::uint64_t kInvalidTicket = 0xffffffffffffffffULL;
inline constexpr std::int32_t kQ16One = 1 << 16;
inline constexpr std::int32_t kQ16Half = 1 << 15;
inline constexpr std::int32_t kMinConductanceQ16 = 1 << 8;
inline constexpr std::int32_t kMaxConductanceQ16 = 4 << 16;
// Active compact-relation depth is independent of physical ancestry, terminal,
// packet, trajectory and action-link capacities. It charges only recursively
// reused resident motor-ground morphology.
inline constexpr std::uint32_t kMaxActiveCompositionDepth = 3u;
// One shared birth/runtime resource ABI. Life Function seeds these exact pool
// bounds and the adult runtime reserves the same quantities; changing a
// physical capacity here therefore cannot leave a stale birth-time literal.
inline constexpr std::uint32_t kMaxLiveEligibilityRecords = 16384u;
inline constexpr std::uint32_t kMaxIngressQueueSize = 4096u;
inline constexpr std::uint32_t kMaxAsynchronousTickets = 2048u;
inline constexpr std::uint64_t kAdultEligibilityResourceCapacity =
    kMaxLiveEligibilityRecords;
inline constexpr std::uint64_t kAdultPacketResourceCapacity =
    kMaxIngressQueueSize;
inline constexpr std::uint64_t kAdultTicketResourceCapacity =
    2ull * kMaxAsynchronousTickets;
inline constexpr std::uint64_t kAdultEligibilityResourceBytesPerUnit = 72u;
inline constexpr std::uint64_t kAdultPacketResourceBytesPerUnit = 32u;
inline constexpr std::uint64_t kAdultTicketResourceBytesPerUnit = 48u;
}  // namespace substrate::direct_adult_core
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_CORE_CONSTANTS_CUH

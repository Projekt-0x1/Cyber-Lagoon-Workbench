// Patch 0003 companion to bcc32_network_matter.cuh: page-bucketed spatial
// addressing for NetworkNode storage. A node's identity is its fixed
// coordinate, not a separately stored field or a device pointer -- the
// coordinate deterministically selects a page, and within that page a
// device-safe atomic claim assigns a slot. This is the representation
// patch 0004's Life Function graph builds its "page bitmaps -> page
// populations -> deterministic page offsets -> node-index compacted
// frontier" construction tick on top of (patch program section "Sparse
// frontier"); this patch only owns coordinate-to-page addressing and
// bounded, fail-closed slot reservation within one page.

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_PAGE_DIRECTORY_CUH
#define HARDWARE_NATIVE_BCC32_NETWORK_PAGE_DIRECTORY_CUH

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define BCC32_NETWORK_PAGE_HD __host__ __device__
#else
#define BCC32_NETWORK_PAGE_HD
#endif

namespace substrate::bcc32::network_recipe {

// Fixed capacity per page: bounded device work per page touch, matching
// bcc32_network_matter.cuh's kMaximum{Parents,Children} discipline of
// paying for bounded storage rather than an unbounded one.
inline constexpr std::uint32_t kPageNodes = 1024;

// Pages per axis and sites per page per axis. kPageGridExtent^3 total pages
// tile a (kPageGridExtent * kPageSiteExtent)^3 coordinate volume; a
// coordinate outside that volume wraps (modulo) rather than being rejected,
// since a Genome's seed/field coordinates (patch 0002) are themselves
// bounded only by the recipe author, not by this addressing scheme.
inline constexpr std::uint32_t kPageGridExtent = 32;
inline constexpr std::uint32_t kPageSiteExtent = 64;
inline constexpr std::uint32_t kPageCount =
    kPageGridExtent * kPageGridExtent * kPageGridExtent;
inline constexpr std::uint32_t kMaxNetworkNodes = kPageCount * kPageNodes;

BCC32_NETWORK_PAGE_HD inline std::uint32_t page_index_for_coordinate(
    const std::uint32_t (&coordinate)[3]) {
  const std::uint32_t px = (coordinate[0] / kPageSiteExtent) % kPageGridExtent;
  const std::uint32_t py = (coordinate[1] / kPageSiteExtent) % kPageGridExtent;
  const std::uint32_t pz = (coordinate[2] / kPageSiteExtent) % kPageGridExtent;
  return (pz * kPageGridExtent + py) * kPageGridExtent + px;
}

// One entry per page: how many of that page's kPageNodes slots are
// currently claimed. This is the whole directory entry for patch 0003 --
// patch 0004 adds the per-page occupancy bitmap and prefix-scan machinery
// needed to compact a frontier deterministically; this patch only needs
// enough to bound and fail closed on a single page's capacity.
struct PageDirectoryEntry {
  std::uint32_t node_count;
};
static_assert(std::is_standard_layout_v<PageDirectoryEntry> &&
                  std::is_trivial_v<PageDirectoryEntry>,
              "PageDirectoryEntry must be a fixed-width POD for device residency");

enum class PageReservationError : std::uint32_t {
  kNone = 0,
  kPageFull,
};

// Device-safe (atomic) claim of the next free slot within one page. Fails
// closed with kPageFull -- and leaves page.node_count exactly where it was
// before the call -- when the page is already at kPageNodes capacity; it
// never overflows into a neighboring page (that would silently break the
// coordinate-determines-page invariant every reader depends on) and never
// clamps the slot index to something already in use.
BCC32_NETWORK_PAGE_HD inline PageReservationError reserve_node_slot(
    PageDirectoryEntry& page, std::uint32_t& out_slot_in_page) {
#if defined(__CUDA_ARCH__)
  const std::uint32_t slot = atomicAdd(&page.node_count, 1u);
#else
  const std::uint32_t slot = page.node_count++;
#endif
  if (slot >= kPageNodes) {
#if defined(__CUDA_ARCH__)
    atomicSub(&page.node_count, 1u);
#else
    --page.node_count;
#endif
    out_slot_in_page = kPageNodes;  // sentinel: no valid slot
    return PageReservationError::kPageFull;
  }
  out_slot_in_page = slot;
  return PageReservationError::kNone;
}

BCC32_NETWORK_PAGE_HD inline std::uint32_t global_node_index(std::uint32_t page_id,
                                                                std::uint32_t slot_in_page) {
  return page_id * kPageNodes + slot_in_page;
}

}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_PAGE_DIRECTORY_CUH

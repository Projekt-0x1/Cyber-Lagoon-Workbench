#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>

// RWR0 resident matter checkpoint identity beyond the 1,024-Record ceiling
// (Linear 0X1-147, plan docs/audits/2026-08-14-rwr0-resident-matter-page-
// scaling-plan.md §3.6(a)).
//
// `ResidentRewriteState` is trivially copyable, and the existing checkpoint
// contracts (bcc32_cuda_resident_dormant_discourse_contract.cu line 489,
// bcc32_cuda_resident_pending_means_production_contract.cu lines 437, 855,
// 876, 1065) define "checkpoint" as a raw std::memcpy of
// sizeof(ResidentRewriteState) into a byte array and back. That definition
// is correct only while directory.live_page_count == 1 -- which is every
// case those tests currently construct. Once population grows past page 0
// (ResidentPageDirectory::pages[], causal_rewrite_universe.cuh,
// grow_resident_pages), a raw byte image of the struct captures the
// *pointer values* in pages[], not the RecordPage bytes they reference.
// Restoring that image produces a state that ALIASES the checkpoint
// source's live out-of-line pages instead of owning an independent copy --
// on a real save/restart (a different process, or the same process after
// the source pages are freed) those pointers are dangling, and even
// same-process the "restored" copy silently shares mutable state with the
// source for every page beyond page 0. Either way, a raw sizeof(...) image
// is not a real checkpoint the moment a second page exists; it only reads
// as one because the existing tests never grow past page 0 to exercise it.
//
// This header adds the explicit, versioned page-walk save/restore pair
// plan §3.6(a) recommends: a small header (format tag + live_page_count)
// followed by the struct's fixed-size body, followed by each live
// out-of-line page's raw bytes in page order (1, 2, ...). Restore never
// trusts the pointer bytes embedded in the fixed-size body it just copied
// in -- every out-of-line page is reallocated fresh (mirroring
// grow_resident_pages's own zero-init shape) and then overwritten with the
// real saved page bytes, and a format-version mismatch is rejected
// explicitly rather than reinterpreted.
//
// This is additive: it does not touch the existing raw-sizeof checkpoint
// tests (they remain correct, pinned single-page coverage) or
// copy_rewrite_state (bcc32_resident_rewrite_runtime.cu's internal
// close-work/action-return staging copy, already hardened separately to
// fault-close on live_page_count > 1 rather than alias).
namespace substrate::bcc32::causal_rewrite {

inline constexpr std::uint32_t kResidentCheckpointFormatVersion = 1u;

struct ResidentCheckpointHeader {
  std::uint32_t format_version = kResidentCheckpointFormatVersion;
  std::uint32_t live_page_count = 1u;
};

// Exact byte size of the page-walk image `state`'s current population would
// serialize to.
inline std::size_t resident_checkpoint_image_size(
    const ResidentRewriteState& state) {
  return sizeof(ResidentCheckpointHeader) + sizeof(ResidentRewriteState) +
         static_cast<std::size_t>(state.directory.live_page_count - 1u) *
             sizeof(RecordPage);
}

// Writes the versioned page-walk image of `state` into `out`, which must
// point at a buffer of at least resident_checkpoint_image_size(state)
// bytes. Returns the number of bytes written.
inline std::size_t save_resident_checkpoint(const ResidentRewriteState& state,
                                             std::byte* out) {
  ResidentCheckpointHeader header{};
  header.live_page_count = state.directory.live_page_count;
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  std::memcpy(out, &state, sizeof(state));
  out += sizeof(state);
  for (std::uint32_t page = 1u; page < state.directory.live_page_count;
       ++page) {
    std::memcpy(out, state.directory.pages[page - 1u], sizeof(RecordPage));
    out += sizeof(RecordPage);
  }
  return resident_checkpoint_image_size(state);
}

// Restores `*out_state` from a versioned page-walk image previously
// produced by save_resident_checkpoint. Returns false and leaves
// `*out_state` untouched on a format-version mismatch, a truncated or
// oversized image, an out-of-range live_page_count, or an out-of-line page
// allocation failure -- it never silently reinterprets bytes it does not
// recognize as this format, and it never dereferences a pointer carried in
// from the image's fixed-size body (those bytes are foreign/stale by
// construction the instant this crosses a real save/restart boundary; they
// are discarded and every out-of-line page is reallocated fresh before
// `*out_state` is written).
inline bool load_resident_checkpoint(const std::byte* in, std::size_t in_size,
                                      ResidentRewriteState* out_state) {
  if (in_size <
      sizeof(ResidentCheckpointHeader) + sizeof(ResidentRewriteState))
    return false;
  ResidentCheckpointHeader header{};
  std::memcpy(&header, in, sizeof(header));
  if (header.format_version != kResidentCheckpointFormatVersion) return false;
  if (header.live_page_count < 1u ||
      header.live_page_count > kMaxResidentPages)
    return false;
  const std::size_t expected =
      sizeof(ResidentCheckpointHeader) + sizeof(ResidentRewriteState) +
      static_cast<std::size_t>(header.live_page_count - 1u) *
          sizeof(RecordPage);
  if (in_size != expected) return false;

  ResidentRewriteState staged{};
  std::memcpy(&staged, in + sizeof(header), sizeof(staged));
  // Discard every pointer the raw copy above just carried in -- reset to a
  // clean single-page directory, then reallocate each out-of-line page from
  // scratch and fill it from the real saved bytes that follow.
  for (std::uint32_t page = 0u; page < kMaxResidentPages - 1u; ++page)
    staged.directory.pages[page] = nullptr;
  staged.directory.live_page_count = 1u;
  const std::byte* cursor =
      in + sizeof(header) + sizeof(ResidentRewriteState);
  for (std::uint32_t page = 1u; page < header.live_page_count; ++page) {
    if (!grow_resident_pages(&staged)) return false;
    std::memcpy(staged.directory.pages[page - 1u], cursor, sizeof(RecordPage));
    cursor += sizeof(RecordPage);
  }
  *out_state = staged;
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite

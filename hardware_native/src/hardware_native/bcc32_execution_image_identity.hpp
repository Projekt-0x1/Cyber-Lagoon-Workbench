#pragma once

// SEALED EXECUTION, PIECE ONE: BIND THE IMAGE THAT IS ACTUALLY RUNNING.
//
// `canonical_law_identity()` hashes the netlist, word ABI, gate descriptors,
// primitive truth tables, factor order and streaming map. That is an identity
// for the ABSTRACT LAW, and it was treated as the provenance half of the
// anti-host guarantee. It is not sufficient, and the counterexample is short:
//
//     __device__ PatchMailbox g_patch;        // host: cudaMemcpyToSymbolAsync
//     next = lawful_local_update(world, i);
//     if (g_patch.epoch == epoch && g_patch.site == i)
//         next = g_patch.replacement;
//
// That write has causal radius ZERO, lives in a device target, links no
// forbidden host symbol, and changes neither netlist nor truth tables nor
// factor order -- so `canonical_law_identity()` stays green while the host
// chooses which site, which value and which epoch. The locality census cannot
// see it either, because it IS local. Both halves of the guarantee pass while
// the guarantee is violated (`START.md` 0.12, retraction of `4db2ec81616e`).
//
// ⭐ WHAT THE ADVERSARY CANNOT AVOID: it has to be COMPILED IN. The mailbox
// symbol, the comparison against it, and the host-side `cudaMemcpyToSymbol`
// call all have to exist in the shipped image. So an identity over the running
// image is not one more inspection -- it is the one binding a compiled-in
// backdoor must break.
//
// This file supplies that identity. It is one of three pieces named in
// `START.md` 0.12, and on its own it is NOT the sealed-execution guarantee:
//
//   1. image identity          <- this file
//   2. exclusive ownership of organism memory by the persistent kernel,
//      with the membrane as the only host-visible allocation
//   3. the launch graph, kernel parameters and device symbol contents
//
// ⚠ Honest limits, stated so they are not assumed away. This hashes the host
// executable image. It does NOT cover a CUDA module JIT-compiled or loaded from
// a separate cubin at run time, DMA from another process, another CUDA context
// writing the same allocation, or a replaced driver. An owner with root can
// always substitute the binary; the scientific claim is that the ORDINARY
// executable path lacks the authority, so cheating requires replacing a
// hash-bound image rather than calling a convenient method.

#include "bcc32_provenance.hpp"

namespace substrate::bcc32 {

// SHA-256 over the bytes of the executable image currently running, with its
// exact length. Throws if the image cannot be read -- an unreadable image is a
// failed seal, never a pass.
[[nodiscard]] ContentAddress running_image_identity();

// The abstract law and the image that implements it, bound together. This is
// what a receipt should carry: `canonical_law_identity()` alone certifies a law
// that the running code may or may not be executing.
[[nodiscard]] ContentAddress sealed_execution_identity();

}  // namespace substrate::bcc32

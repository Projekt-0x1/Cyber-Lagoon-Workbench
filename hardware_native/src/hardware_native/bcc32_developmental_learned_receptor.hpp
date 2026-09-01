#pragma once

#include "bcc32_developmental_append.hpp"

namespace substrate::bcc32 {

class ReferenceLattice;

// Teacher-withdrawn resident routing.  Authority digits 0/1 are learned only
// by developmental_append.  This factor never changes them: it pushes the
// selected A/B leg into transient journal digits 2..7, transposes that leg's
// physical port vacancy onto the common processive inlet, and pops the tag on
// inverse so mixed A/B histories remain injective.
void apply_k_developmental_learned_receptor(ReferenceLattice& lattice,
                                             bool inverse);

}  // namespace substrate::bcc32

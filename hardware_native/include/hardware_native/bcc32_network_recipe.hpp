// Compatibility re-export. The canonical fixed-width Genome Gamma ABI now
// lives in the Direct lane as hardware_native/direct_network_recipe.hpp: it
// is the type the species genome the seed atlas authors actually lowers
// into, and the superseded BCC lattice is a consumer of it, not its owner.
// The dependency arrow points bcc32 -> direct, never the reverse.
//
// This shim exists only so the ~85 remaining bcc32 translation units that
// spell substrate::bcc32::network_recipe::* keep compiling while that lane
// is retired. New code includes the direct_ header and names the direct_
// namespace. Delete this file when the last bcc32 consumer is gone.

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_RECIPE_HPP
#define HARDWARE_NATIVE_BCC32_NETWORK_RECIPE_HPP

#include "hardware_native/direct_network_recipe.hpp"

#define BCC32_NETWORK_RECIPE_HD DIRECT_NETWORK_RECIPE_HD

namespace substrate::bcc32::network_recipe {
using namespace ::substrate::direct_network::recipe;
}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_RECIPE_HPP

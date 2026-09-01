#include "bcc32_device_resident_frontier.cuh"

#include <type_traits>

namespace substrate::bcc32::device_resident_frontier {

static_assert(std::is_trivially_copyable_v<ProductionState>);
static_assert(std::is_trivially_copyable_v<ProductionSnapshot>);

}  // namespace substrate::bcc32::device_resident_frontier

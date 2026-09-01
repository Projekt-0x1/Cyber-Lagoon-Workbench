#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_CONSTRUCTION_ECONOMY_INL
#define HARDWARE_NATIVE_DIRECT_NETWORK_CONSTRUCTION_ECONOMY_INL

inline constexpr std::uint32_t kRouteConstructionCostShift = 5u;
inline constexpr std::uint32_t kRouteConstructionCostMask = 0x7ffu << kRouteConstructionCostShift;

DIRECT_NETWORK_HD inline std::uint32_t construction_route_cost(
    const DirectNode& source, const DirectNode& target) {
  const std::uint64_t distance =
      static_cast<std::uint64_t>(source.coordinate[0] > target.coordinate[0]
          ? source.coordinate[0] - target.coordinate[0]
          : target.coordinate[0] - source.coordinate[0]) +
      static_cast<std::uint64_t>(source.coordinate[1] > target.coordinate[1]
          ? source.coordinate[1] - target.coordinate[1]
          : target.coordinate[1] - source.coordinate[1]) +
      static_cast<std::uint64_t>(source.coordinate[2] > target.coordinate[2]
          ? source.coordinate[2] - target.coordinate[2]
          : target.coordinate[2] - source.coordinate[2]);
  const std::uint64_t cost = 1u + distance / 32u + source.active_route_count +
                             target.active_in_degree;
  return static_cast<std::uint32_t>(cost > 0x7ffu ? 0x7ffu : cost);
}

DIRECT_NETWORK_HD inline std::uint32_t encode_route_construction_cost(
    std::uint32_t flags, std::uint32_t cost) {
  const std::uint32_t bounded = cost == 0u ? 1u : (cost > 0x7ffu ? 0x7ffu : cost);
  return (flags & ~kRouteConstructionCostMask) |
         (bounded << kRouteConstructionCostShift);
}

DIRECT_NETWORK_HD inline std::uint32_t decode_route_construction_cost(
    std::uint32_t flags) {
  const std::uint32_t cost = (flags & kRouteConstructionCostMask) >>
                             kRouteConstructionCostShift;
  return cost == 0u ? 1u : cost;
}

#endif

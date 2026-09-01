#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SENSOR_PORT_INDEX_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SENSOR_PORT_INDEX_CUH

// Included mid-TU, after <algorithm>/<cstdint> and the DirectAdultRuntime
// type surface: kInvalidIndex resolves through the enclosing
// substrate::direct_adult_core scope, matching direct_adult_motor_affect_helpers.cuh.

// Stage-time admission index for direct_adult_route_transport_runtime.inl.
// One batch stages against one frozen born-port table, so the per-event table
// scan collapses onto this channel-sorted copy of the sensor ports built once
// per call. Birth refuses two sensor-role bindings on one channel
// (direct_boundary_bindings_share_a_sensor_channel) and caps the table at
// kMaxBoundaryPorts entries, so a channel span holds a single port; walking
// the whole span keeps the exact matches==1 verdict even for a table that
// evaded that refusal.
struct SensorPortEntry {
  std::uint32_t channel;
  std::uint32_t node;
  std::uint32_t port_index;
};

// Exact restatement of the retired per-event scan over host_boundary_ports: a
// port matches when its channel equals event_channel and event_node is the
// any-node wildcard or equals the port's node; the contact admits only when
// exactly one port matches, and resolves to that port's table index.
inline std::uint32_t sorted_sensor_port_index(const SensorPortEntry* ports,
                                              std::uint32_t port_count,
                                              std::uint32_t event_node,
                                              std::uint32_t event_channel) {
  const SensorPortEntry* span_begin = std::lower_bound(
      ports, ports + port_count, event_channel,
      [](const SensorPortEntry& port, std::uint32_t channel) {
        return port.channel < channel;
      });
  std::uint32_t result = kInvalidIndex, matches = 0u;
  for (const SensorPortEntry* port = span_begin;
       port != ports + port_count && port->channel == event_channel; ++port) {
    if (event_node == kInvalidIndex || port->node == event_node) {
      result = port->port_index;
      ++matches;
    }
  }
  return matches == 1u ? result : kInvalidIndex;
}

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_SENSOR_PORT_INDEX_CUH

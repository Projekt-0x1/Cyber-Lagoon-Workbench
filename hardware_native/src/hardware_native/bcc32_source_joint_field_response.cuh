#pragma once

// A bounded field response grown from the resident source/joint ecology.
//
// This helper deliberately has no field identifier, coordinate-labelled
// target, or abstraction/fitness scalar.  Each response lane is owned by an
// existing sparse-source route and receives only the raw contact feature for
// that owner plus the resident joint prediction/credit routed by the same
// owner pair.  The lane index is an implementation aperture, not a semantic
// address supplied by the host.

#include "bcc32_sparse_source_joint.cuh"

#include <cstdint>

namespace substrate::bcc32::source_joint_field_response {

inline constexpr std::int32_t kResponseLimit = 8192;

struct Metrics {
  std::int32_t response_l1 = 0;
  std::int32_t residual_l1 = 0;
  std::int32_t peak_response = 0;
  std::int32_t peak_residual = 0;
  std::uint32_t active_routes = 0u;
  std::uint32_t contact_routes = 0u;
  std::uint32_t responsive_routes = 0u;
  std::uint64_t owner_mix = 0u;
  std::int32_t developmental_pressure_l1 = 0;
  std::int32_t functional_horizon = 0;
  std::int32_t functional_integration = 0;
  std::int32_t functional_delay = 0;
  std::int32_t functional_spatial_tv_l1 = 0;
};

__device__ __forceinline__ std::int32_t clamp_response(std::int64_t value) {
  return static_cast<std::int32_t>(
      value < -kResponseLimit
          ? -kResponseLimit
          : value > kResponseLimit ? kResponseLimit : value);
}

__device__ __forceinline__ std::int32_t abs_response(std::int32_t value) {
  return value < 0 ? -value : value;
}

// Compute one distributed response field from current raw geometry and the
// already-grown source/joint routes.  No host target or route selector enters
// this function.  The two output arrays remain device-private to the kernel.
__device__ void evaluate(
    const sparse_source_joint::State* state,
    const sparse_source_joint::RawContact* contacts,
    std::uint32_t contact_count, std::uint32_t cell_count,
    std::int32_t* response, std::int32_t* residual, Metrics* metrics) {
  *metrics = {};
  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    response[index] = 0;
    residual[index] = 0;
  }
  if (cell_count == 0u) return;

  for (std::uint32_t route_index = 0u;
       route_index < sparse_source_joint::kRouteCapacity; ++route_index) {
    const sparse_source_joint::SparseRoute& route = state->routes[route_index];
    if (route.live == 0u) continue;
    ++metrics->active_routes;
    std::int64_t route_response = route.credit;
    std::int64_t route_residual = route.credit;
    // Developmental pressure is resident physical surface history, not bits
    // of a route hash. Reuse deepens the local route's mature pressure.
    const std::int32_t developmental =
        static_cast<std::int32_t>(route.surface_offset) +
        static_cast<std::int32_t>(route.contact_count > 0u
                                      ? route.contact_count - 1u
                                      : 0u);
    route_response += developmental;
    route_residual += developmental / 2;
    metrics->developmental_pressure_l1 += abs_response(developmental);
    metrics->functional_horizon +=
        static_cast<std::int32_t>(route.eligibility_age +
                                  route.contact_count + 1u);
    metrics->functional_delay +=
        static_cast<std::int32_t>(route.contact_count > 0u
                                      ? route.contact_count - 1u
                                      : 0u);
    bool contacted = false;
    for (std::uint32_t contact_index = 0u; contact_index < contact_count;
         ++contact_index) {
      const sparse_source_joint::RawContact& contact = contacts[contact_index];
      if (sparse_source_joint::owner_for(contact) != route.owner) continue;
      contacted = true;
      route_response += sparse_source_joint::operand_feature(contact);
      route_residual += sparse_source_joint::raw_projection(contact);
    }
    if (contacted) {
      ++metrics->contact_routes;
      metrics->functional_integration +=
          static_cast<std::int32_t>(route.contact_count + 1u);
    }
    const std::uint32_t lane = route_index % cell_count;
    response[lane] = clamp_response(static_cast<std::int64_t>(response[lane]) +
                                     route_response);
    residual[lane] = clamp_response(static_cast<std::int64_t>(residual[lane]) +
                                     route_residual);
    metrics->owner_mix = sparse_source_joint::mix_owner(
        metrics->owner_mix ^ 0x243f6a8885a308d3ull, route.owner);
    metrics->owner_mix = sparse_source_joint::mix_owner(metrics->owner_mix,
                                                        route.route);
  }

  for (std::uint32_t joint_index = 0u;
       joint_index < sparse_source_joint::kJointCapacity; ++joint_index) {
    const sparse_source_joint::SparseJoint& joint = state->joints[joint_index];
    // Only the assembly selected by the current resident raw history is
    // expressed into the field. Older contextual joints remain resident
    // matter, but cannot leak their response into the active history.
    if (joint.live == 0u || joint.interaction_enabled == 0u ||
        joint.context_key != state->context_key)
      continue;
    ++metrics->responsive_routes;
    const std::uint32_t lane =
        static_cast<std::uint32_t>(joint.route % cell_count);
    response[lane] = clamp_response(
        static_cast<std::int64_t>(response[lane]) + joint.prediction);
    residual[lane] = clamp_response(
        static_cast<std::int64_t>(residual[lane]) + joint.credit);
    metrics->owner_mix = sparse_source_joint::mix_owner(metrics->owner_mix,
                                                        joint.first_owner);
    metrics->owner_mix = sparse_source_joint::mix_owner(metrics->owner_mix,
                                                        joint.second_owner);
    metrics->functional_integration += 2;
    metrics->functional_delay +=
        static_cast<std::int32_t>(joint.eligibility_age + 1u);
  }

  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    const std::int32_t lane_response = response[index];
    const std::int32_t lane_residual = residual[index];
    metrics->response_l1 += abs_response(lane_response);
    metrics->residual_l1 += abs_response(lane_residual);
    if (abs_response(lane_response) > metrics->peak_response)
      metrics->peak_response = abs_response(lane_response);
    if (abs_response(lane_residual) > metrics->peak_residual)
      metrics->peak_residual = abs_response(lane_residual);
    if (index != 0u)
      metrics->functional_spatial_tv_l1 += abs_response(
          response[index] - response[index - 1u]);
  }
}

}  // namespace substrate::bcc32::source_joint_field_response

#pragma once

#include <cstdint>

#include "bcc32_developmental_append.hpp"
#include "bcc32_geometry.cuh"

namespace substrate::bcc32 {

class ReferenceLattice;

inline constexpr std::uint32_t kDevelopmentalCreditServiceRingCount = 36u;
inline constexpr std::uint32_t kDevelopmentalCreditServiceMaxPeriod = 22u;
inline constexpr std::uint32_t kDevelopmentalCreditServiceEnableSite = 17u;

// Site17 is a structural phenotype owner and may simultaneously carry lawful
// traffic in the orthogonal carrier byte.  Only its non-carrier structure can
// enable service: transient carrier vacancies must not suppress an owner, and
// no carrier pattern can manufacture one on Q or a wrong structural base.
[[nodiscard]] __host__ __device__ constexpr bool
developmental_credit_service_enable_word_matches(
    SiteWord word, std::uint32_t marker, std::uint32_t path,
    std::uint32_t waste) {
  return (word & ~kCarrierMask) ==
         (developmental_append_product_word(
              kDevelopmentalCreditServiceEnableSite, marker, path, waste,
              0u) &
          ~kCarrierMask);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_leg(std::uint32_t ring) {
  return ring & 1u;
}

[[nodiscard]] __host__ __device__ constexpr bool
developmental_credit_service_reject_source_ring(std::uint32_t ring) {
  return ring >= 6u && ring < 16u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_reject_source_bank(std::uint32_t ring) {
  return ring < 6u ? 0u : (ring - 6u) / 2u;
}

[[nodiscard]] __host__ __device__ constexpr bool
developmental_credit_service_reject_teacher_ring(std::uint32_t ring) {
  return ring >= 16u && ring < 26u;
}

[[nodiscard]] __host__ __device__ constexpr bool
developmental_credit_service_reject_clock_ring(std::uint32_t ring) {
  return ring >= 26u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_reject_bank(std::uint32_t ring) {
  if (developmental_credit_service_reject_source_ring(ring))
    return (ring - 6u) / 2u;
  if (developmental_credit_service_reject_teacher_ring(ring))
    return (ring - 16u) / 2u;
  return ring < 26u ? 0u : (ring - 26u) / 2u;
}

[[nodiscard]] __host__ __device__ constexpr bool
developmental_credit_service_teacher_ring(std::uint32_t ring) {
  return ring >= 2u && ring < 4u;
}

[[nodiscard]] __host__ __device__ constexpr bool
developmental_credit_service_reject_ring(std::uint32_t ring) {
  return developmental_credit_service_reject_teacher_ring(ring);
}

[[nodiscard]] __host__ __device__ constexpr bool
developmental_credit_service_clock_ring(std::uint32_t ring) {
  return (ring >= 4u && ring < 6u) ||
         developmental_credit_service_reject_clock_ring(ring);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_first_length(std::uint32_t ring) {
  constexpr std::uint8_t kFirst[kDevelopmentalCreditServiceRingCount]{
      7, 7, 9, 1, 1, 10, 1, 8, 1, 8, 1, 8, 1, 8, 8, 8, 9, 9,
      9, 9, 9, 1, 9, 1, 9, 1, 1, 10, 1, 10, 10, 10, 10, 10, 1, 10};
  return ring < kDevelopmentalCreditServiceRingCount ? kFirst[ring] : 1u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_second_length(std::uint32_t ring) {
  constexpr std::uint8_t kSecond[kDevelopmentalCreditServiceRingCount]{
      2, 2, 1, 9, 10, 1, 8, 1, 8, 1, 8, 1, 8, 1, 1, 1, 1, 1,
      1, 1, 1, 9, 1, 9, 1, 9, 10, 1, 10, 1, 1, 1, 1, 1, 10, 1};
  return ring < kDevelopmentalCreditServiceRingCount ? kSecond[ring] : 1u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_period(std::uint32_t ring) {
  return 2u * (developmental_credit_service_first_length(ring) +
               developmental_credit_service_second_length(ring));
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_incoming(std::uint32_t ring,
                                      std::uint32_t marker,
                                      std::uint32_t path,
                                      std::uint32_t waste) {
  if (developmental_credit_service_teacher_ring(ring) ||
      developmental_credit_service_reject_ring(ring))
    return developmental_append_spent_teacher_basis(
        developmental_credit_service_leg(ring), marker, path, waste);
  if (ring < 2u || developmental_credit_service_reject_source_ring(ring))
    return developmental_append_spent_source_basis(
        developmental_credit_service_leg(ring), marker, path, waste);
  return developmental_append_clock_basis(
      developmental_credit_service_leg(ring), marker, path, waste);
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_outgoing(std::uint32_t ring,
                                      std::uint32_t marker,
                                      std::uint32_t path,
                                      std::uint32_t waste) {
  // Directions are stored in the role frame (marker,path,waste,free), then
  // mapped through the live S4 frame.  The sign bit is the ordinary BCC
  // negative-direction bit, so this one table is shared by CPU and CUDA.
  constexpr std::uint8_t kRoleDirection[kDevelopmentalCreditServiceRingCount]{
      6, 0, 7, 0, 1, 0, 0, 0, 2, 4, 6, 7, 4, 7, 0, 4, 0, 4,
      2, 4, 7, 6, 2, 0, 3, 2, 5, 3, 4, 0, 2, 3, 2, 0, 0, 3};
  const std::uint32_t role_direction =
      ring < kDevelopmentalCreditServiceRingCount ? kRoleDirection[ring] : 0u;
  const std::uint32_t role = role_direction & 3u;
  const std::uint32_t free =
      developmental_append_free_basis(marker, path, waste);
  const std::uint32_t basis =
      role == 0u ? marker : role == 1u ? path : role == 2u ? waste : free;
  return basis + (role_direction & 4u);
}

[[nodiscard]] __host__ __device__ constexpr Int3
developmental_credit_service_ingress(std::uint32_t ring,
                                     std::uint32_t marker,
                                     std::uint32_t path,
                                     std::uint32_t waste) {
  const DevelopmentalAppendOffset offset =
      developmental_credit_service_reject_source_ring(ring)
          ? developmental_append_reject_source_ingress_offset(
                developmental_credit_service_leg(ring),
                developmental_credit_service_reject_source_bank(ring))
          : developmental_credit_service_reject_teacher_ring(ring)
              ? developmental_append_reject_teacher_ingress_offset(
                    developmental_credit_service_leg(ring),
                    developmental_credit_service_reject_bank(ring))
          : developmental_credit_service_reject_clock_ring(ring)
              ? developmental_append_reject_clock_ingress_offset(
                    developmental_credit_service_leg(ring),
                    developmental_credit_service_reject_bank(ring))
          : developmental_credit_service_teacher_ring(ring)
          ? developmental_append_accepted_teacher_ingress_offset(
                developmental_credit_service_leg(ring))
          : (developmental_credit_service_clock_ring(ring)
                        ? developmental_append_clock_ingress_offset(
                              developmental_credit_service_leg(ring))
                        : developmental_append_accepted_source_ingress_offset(
                              developmental_credit_service_leg(ring)));
  const Int3 marker_step =
      direction_offset(static_cast<Direction>(marker));
  const Int3 path_step = direction_offset(static_cast<Direction>(path));
  const Int3 waste_step = direction_offset(static_cast<Direction>(waste));
  return {marker_step.x * offset.marker + path_step.x * offset.path +
              waste_step.x * offset.waste,
          marker_step.y * offset.marker + path_step.y * offset.path +
              waste_step.y * offset.waste,
          marker_step.z * offset.marker + path_step.z * offset.path +
              waste_step.z * offset.waste};
}

[[nodiscard]] __host__ __device__ constexpr Int3
developmental_credit_service_corner(std::uint32_t ring,
                                    std::uint32_t corner,
                                    std::uint32_t marker,
                                    std::uint32_t path,
                                    std::uint32_t waste) {
  const Int3 ingress = developmental_credit_service_ingress(
      ring, marker, path, waste);
  const std::uint32_t incoming = developmental_credit_service_incoming(
      ring, marker, path, waste);
  const std::uint32_t outgoing = developmental_credit_service_outgoing(
      ring, marker, path, waste);
  const Int3 first = direction_offset(static_cast<Direction>(outgoing));
  const Int3 second = direction_offset(
      static_cast<Direction>(incoming ^ 4u));
  const std::int32_t first_length = static_cast<std::int32_t>(
      developmental_credit_service_first_length(ring));
  const std::int32_t second_length = static_cast<std::int32_t>(
      developmental_credit_service_second_length(ring));
  if (corner == 0u) return ingress;
  if (corner == 1u)
    return ingress + Int3{first.x * first_length, first.y * first_length,
                          first.z * first_length};
  if (corner == 2u)
    return ingress +
           Int3{first.x * first_length + second.x * second_length,
                first.y * first_length + second.y * second_length,
                first.z * first_length + second.z * second_length};
  return ingress + Int3{second.x * second_length,
                        second.y * second_length,
                        second.z * second_length};
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_corner_incoming(
    std::uint32_t ring, std::uint32_t corner, std::uint32_t marker,
    std::uint32_t path, std::uint32_t waste) {
  const std::uint32_t incoming = developmental_credit_service_incoming(
      ring, marker, path, waste);
  const std::uint32_t outgoing = developmental_credit_service_outgoing(
      ring, marker, path, waste);
  const std::uint32_t directions[4u]{outgoing, incoming ^ 4u,
                                    outgoing ^ 4u, incoming};
  return corner == 0u ? incoming : directions[corner - 1u];
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_credit_service_corner_outgoing(
    std::uint32_t ring, std::uint32_t corner, std::uint32_t marker,
    std::uint32_t path, std::uint32_t waste) {
  const std::uint32_t incoming = developmental_credit_service_incoming(
      ring, marker, path, waste);
  const std::uint32_t outgoing = developmental_credit_service_outgoing(
      ring, marker, path, waste);
  const std::uint32_t directions[4u]{outgoing, incoming ^ 4u,
                                    outgoing ^ 4u, incoming};
  return directions[corner];
}

// The differentiated adult owner supplies routing authority; site 17 is a
// separately lesionable enable.  K_service only transposes carrier channels at
// the sixteen declared ring corners.  Ordinary S_P supplies motion and phase.
void apply_k_developmental_credit_service(ReferenceLattice& lattice,
                                          bool inverse);

}  // namespace substrate::bcc32

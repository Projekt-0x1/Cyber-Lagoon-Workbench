#pragma once

// Current-F developmental composition for a multi-contact receiver.
//
// Two persistent processive prediction bodies are queried by autonomous t0
// launcher matter.  The zero state emits on a negative rail; the one state
// emits on a positive rail.  A complete S4 transform maps the second source's
// positive rail from basis 1 to basis 2.  Equal ballistic distances bring the
// two distinct positive carrier holes to one member of the existing stable
// signed-synapse region.  differentiated_signed_synapse is the only operation
// that writes the preregistered marker-basis B for their union.  The other
// three inputs settle to distinct ordinary non-marker words; this is a
// two-input/four-state marker-specific integrator, not a Boolean-state codec.
//
// A founder hash selects one generic outer S4 orientation and one of two
// viable route lengths.  No literal hash is compared; no hash bit selects an
// answer, input, receiver word, or intervention.  Sources and receiver are
// seeded/developmentally instantiated matter, not endogenously recruited or
// learned tissue.

#include <array>
#include <cstdint>

#include "bcc32_processive_prediction_query_launcher_seed.cuh"
#include "bcc32_signed_synapse_region_seed.cuh"

namespace substrate::bcc32::grown_multicontact_integrator {

using FounderHash = std::uint32_t;

inline constexpr std::array<FounderHash, 3u> kPreregisteredFounderHashes{
    {0x31c0a101u, 0x31c0b202u, 0x31c0c304u}};
inline constexpr FounderHash kHeldOutFounderHash = 0x9e3779b9u;
inline constexpr std::uint8_t kInputA = 0x1u;
inline constexpr std::uint8_t kInputB = 0x2u;
inline constexpr std::uint32_t kSourceCount = 2u;
inline constexpr std::uint32_t kRegionSiteCount = kSignedRegionSeedSiteCount;
inline constexpr std::uint32_t kSourceAnatomySiteCount = kProcessiveWeightSitesPerCell;
inline constexpr std::uint32_t kSourceLauncherSiteCount =
    kProcessivePredictionQueryLauncherSeedSiteCount;
inline constexpr std::uint32_t kFounderSiteCount =
    kRegionSiteCount + kSourceCount * (kSourceAnatomySiteCount + kSourceLauncherSiteCount);

using Permutation = std::array<std::uint32_t, 4u>;

struct Grammar {
  Permutation outer{{0u, 1u, 2u, 3u}};
  std::uint32_t output_distance = 8u;
};

[[nodiscard]] constexpr Grammar grammar(FounderHash hash) {
  const std::uint32_t rotation = hash & 0x3u;
  Grammar result{};
  for (std::uint32_t basis = 0u; basis < 4u; ++basis)
    result.outer[basis] = (basis + rotation) & 0x3u;
  result.output_distance = 8u + ((hash >> 2u) & 0x1u);
  return result;
}

[[nodiscard]] constexpr Permutation compose(Permutation outer, Permutation inner) {
  Permutation result{};
  for (std::uint32_t basis = 0u; basis < 4u; ++basis)
    result[basis] = outer[inner[basis]];
  return result;
}

[[nodiscard]] constexpr Permutation source_permutation(FounderHash hash, std::uint32_t source) {
  const Permutation identity{{0u, 1u, 2u, 3u}};
  const Permutation swap_paths{{0u, 2u, 1u, 3u}};
  return compose(grammar(hash).outer, source == 0u ? identity : swap_paths);
}

[[nodiscard]] constexpr SiteWord permute_word(SiteWord word, const Permutation& permutation) {
  SiteWord result = 0u;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    const std::uint32_t image = permutation[basis];
    if ((word & carrier_bit(basis)) != 0u)
      result |= carrier_bit(image);
    if ((word & carrier_bit(basis + 4u)) != 0u)
      result |= carrier_bit(image + 4u);
    if ((word & face_bit(basis)) != 0u)
      result |= face_bit(image);
    if ((word & face_bit(basis + 4u)) != 0u)
      result |= face_bit(image + 4u);
    if ((word & owned_bond_bit(basis)) != 0u)
      result |= owned_bond_bit(image);
    if ((word & channel_bit(kConformationShift, basis)) != 0u)
      result |= channel_bit(kConformationShift, image);
    if ((word & channel_bit(kReactiveShift, basis)) != 0u)
      result |= channel_bit(kReactiveShift, image);
    if ((word & energy_bit(basis)) != 0u)
      result |= energy_bit(image);
  }
  return result;
}

[[nodiscard]] constexpr Int3 permute_coordinate(Int3 coordinate, const Permutation& permutation) {
  const std::int32_t coefficients[3] = {coordinate.x, coordinate.y, coordinate.z};
  Int3 result{};
  for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
    const Int3 image = basis_offset(static_cast<Basis>(permutation[axis]));
    result.x += coefficients[axis] * image.x;
    result.y += coefficients[axis] * image.y;
    result.z += coefficients[axis] * image.z;
  }
  return result;
}

[[nodiscard]] constexpr Int3 add(Int3 left, Int3 right) {
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}

[[nodiscard]] constexpr Int3 subtract(Int3 left, Int3 right) {
  return {left.x - right.x, left.y - right.y, left.z - right.z};
}

[[nodiscard]] constexpr Int3 scale(Int3 value, std::int32_t factor) {
  return {value.x * factor, value.y * factor, value.z * factor};
}

[[nodiscard]] constexpr DevelopmentalSeedSite site(Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x), static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

[[nodiscard]] constexpr Int3 base_target() {
  const auto origins = signed_region_origins(kSignedSynapseRegionSeedHash);
  // Branch 2 uses the third non-marker basis.  Source paths 1 and 2 therefore
  // approach laterally and do not occupy the branch-to-downstream probe ray.
  return {origins[2].x, origins[2].y, origins[2].z};
}

[[nodiscard]] constexpr Int3 receiver_coordinate(FounderHash hash) {
  return permute_coordinate(base_target(), grammar(hash).outer);
}

[[nodiscard]] constexpr Int3 consumer_coordinate(FounderHash) {
  return {0, 0, 0};
}

[[nodiscard]] constexpr std::uint32_t marker_basis(FounderHash hash) {
  return grammar(hash).outer[0u];
}

[[nodiscard]] constexpr std::uint32_t source_basis(FounderHash hash, std::uint32_t source) {
  return source_permutation(hash, source)[1u];
}

[[nodiscard]] constexpr std::uint32_t consumer_lane(FounderHash hash) {
  return grammar(hash).outer[3u];
}

[[nodiscard]] constexpr std::uint32_t first_arrival_tick(FounderHash hash) {
  return grammar(hash).output_distance + 1u;
}

[[nodiscard]] constexpr std::uint32_t second_arrival_tick(FounderHash hash) {
  return first_arrival_tick(hash) + 3u;
}

[[nodiscard]] constexpr ProcessivePredictionQueryLauncherSeedHash launcher_hash() {
  return make_processive_prediction_query_launcher_seed_hash(
      1u, 3u, kProcessivePredictionQueryPulseCount, kProcessivePredictionQueryGene);
}

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, kFounderSiteCount> founder_seed(
    FounderHash hash, std::uint8_t input_mask) {
  std::array<DevelopmentalSeedSite, kFounderSiteCount> result{};
  std::uint32_t write = 0u;
  const Grammar selected = grammar(hash);

  for (const DevelopmentalSeedSite& raw :
       signed_synapse_region_seed(kSignedSynapseRegionSeedHash)) {
    const Int3 coordinate = permute_coordinate({raw.x, raw.y, raw.z}, selected.outer);
    result[write++] = site(coordinate, permute_word(raw.word, selected.outer));
  }

  const auto anatomy = processive_prediction_projection_seed();
  const auto launchers = processive_prediction_query_launcher_seed(launcher_hash());
  const Int3 threshold = processive_prediction_projection_body(0u);
  const Int3 target = base_target();
  for (std::uint32_t source = 0u; source < kSourceCount; ++source) {
    const Permutation inner =
        source == 0u ? Permutation{{0u, 1u, 2u, 3u}} : Permutation{{0u, 2u, 1u, 3u}};
    const Permutation full = compose(selected.outer, inner);
    const std::uint32_t base_path = source == 0u ? 1u : 2u;
    const Int3 body = subtract(target, scale(direction_offset(static_cast<Direction>(base_path)),
                                             static_cast<std::int32_t>(selected.output_distance)));
    const Int3 transformed_body = permute_coordinate(body, selected.outer);
    for (std::uint32_t index = 0u; index < kSourceAnatomySiteCount; ++index) {
      const Int3 relative =
          subtract({anatomy[index].x, anatomy[index].y, anatomy[index].z}, threshold);
      SiteWord word = anatomy[index].word;
      if (index == 0u && (input_mask & (1u << source)) != 0u)
        word = processive_weight_one_word(kProcessivePredictionProjectionSeedHash);
      result[write++] =
          site(add(transformed_body, permute_coordinate(relative, full)), permute_word(word, full));
    }
    for (const DevelopmentalSeedSite& raw : launchers) {
      const Int3 relative = subtract({raw.x, raw.y, raw.z}, threshold);
      result[write++] = site(add(transformed_body, permute_coordinate(relative, full)),
                             permute_word(raw.word, full));
    }
  }
  return result;
}

[[nodiscard]] constexpr bool generic_geometry_is_valid(FounderHash hash) {
  return source_basis(hash, 0u) != source_basis(hash, 1u) &&
         source_basis(hash, 0u) != marker_basis(hash) &&
         source_basis(hash, 1u) != marker_basis(hash) && grammar(hash).output_distance >= 8u;
}

static_assert(valid_processive_prediction_query_launcher_hash(launcher_hash()));
static_assert(generic_geometry_is_valid(kPreregisteredFounderHashes[0]));
static_assert(generic_geometry_is_valid(kPreregisteredFounderHashes[1]));
static_assert(generic_geometry_is_valid(kPreregisteredFounderHashes[2]));
static_assert(generic_geometry_is_valid(kHeldOutFounderHash));

}  // namespace substrate::bcc32::grown_multicontact_integrator

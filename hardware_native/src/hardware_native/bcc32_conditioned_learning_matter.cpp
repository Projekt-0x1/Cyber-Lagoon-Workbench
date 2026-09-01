#include "bcc32_conditioned_learning_matter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <istream>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "bcc32_law.cuh"
#include "bcc32_processive_credit_return_seed.cuh"
#include "bcc32_reciprocal_tape.cuh"

namespace substrate::bcc32 {
namespace {

constexpr std::uint64_t kCheckpointMagic = 0x31544d4c43323342ull;
constexpr std::uint32_t kCheckpointVersion = 5u;
constexpr std::uint32_t kWeightLength =
    kProcessiveCreditReturnStageCount;
constexpr std::uint32_t kCreditSettleTicks =
    kProcessiveCreditReturnSettleTicks;
constexpr std::uint32_t kProbeBasis = 1u;
constexpr std::uint32_t kMaximumCheckpointEntries = 33'554'432u;
constexpr std::uint64_t kMaximumSitesPerEntry = 1u << 24u;
constexpr std::uint32_t kActiveBatchChunksPerRoute = 8u;
constexpr std::int32_t kActiveBatchPitchX = 160;
constexpr std::int32_t kActiveBatchPitchY = 188;
constexpr std::int32_t kActiveBatchPitchZ = 76;

struct RouteBounds {
  std::int32_t min_x = 0;
  std::int32_t max_x = 0;
  std::int32_t min_y = 0;
  std::int32_t max_y = 0;
  std::int32_t min_z = 0;
  std::int32_t max_z = 0;
};

constexpr RouteBounds processive_credit_return_bounds() {
  const auto seed =
      processive_credit_return_seed(kProcessiveCreditReturnSeedHash);
  RouteBounds result{
      seed[0].x, seed[0].x, seed[0].y,
      seed[0].y, seed[0].z, seed[0].z,
  };
  for (std::uint32_t index = 1u;
       index < kProcessiveCreditReturnSeedSiteCount; ++index) {
    result.min_x = std::min(
        result.min_x, static_cast<std::int32_t>(seed[index].x));
    result.max_x = std::max(
        result.max_x, static_cast<std::int32_t>(seed[index].x));
    result.min_y = std::min(
        result.min_y, static_cast<std::int32_t>(seed[index].y));
    result.max_y = std::max(
        result.max_y, static_cast<std::int32_t>(seed[index].y));
    result.min_z = std::min(
        result.min_z, static_cast<std::int32_t>(seed[index].z));
    result.max_z = std::max(
        result.max_z, static_cast<std::int32_t>(seed[index].z));
  }
  return result;
}

constexpr RouteBounds kRouteBounds =
    processive_credit_return_bounds();
constexpr std::uint32_t kRouteStrideX =
    static_cast<std::uint32_t>(
        kRouteBounds.max_x - kRouteBounds.min_x + 1);
constexpr std::uint32_t kRouteStrideY =
    static_cast<std::uint32_t>(
        kRouteBounds.max_y - kRouteBounds.min_y + 1);
constexpr std::uint32_t kRouteStrideZ =
    static_cast<std::uint32_t>(
        kRouteBounds.max_z - kRouteBounds.min_z + 1);
constexpr std::uint32_t kRoutesPerChunkX =
    kChunkEdge / kRouteStrideX;
constexpr std::uint32_t kRoutesPerChunkY =
    kChunkEdge / kRouteStrideY;
constexpr std::uint32_t kRoutesPerChunkZ =
    kChunkEdge / kRouteStrideZ;
constexpr std::uint32_t kRoutesPerChunk =
    kRoutesPerChunkX * kRoutesPerChunkY * kRoutesPerChunkZ;

static_assert(
    processive_weight_length(processive_credit_return_weight_hash(
        kProcessiveCreditReturnSeedHash)) == kWeightLength);
static_assert(kProcessiveCreditReturnSeedSiteCount == 48u);
static_assert(kRoutesPerChunk == 110u);

template <typename T>
void write_plain(std::ostream& output, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!output) throw std::runtime_error("conditioned matter checkpoint write failed");
}

template <typename T>
T read_plain(std::istream& input) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) throw std::runtime_error("truncated conditioned matter checkpoint");
  return value;
}

void write_component(std::ostream& output, const CoordinateComponent& value) {
  const std::string bytes = canonical_coordinate_component(value);
  if (bytes.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("oversized conditioned matter coordinate");
  write_plain(output, static_cast<std::uint32_t>(bytes.size()));
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) throw std::runtime_error("conditioned matter coordinate write failed");
}

CoordinateComponent read_component(std::istream& input) {
  const std::uint32_t size = read_plain<std::uint32_t>(input);
  if (size == 0u || size > 4096u)
    throw std::runtime_error("invalid conditioned matter coordinate length");
  std::string bytes(size, '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input) throw std::runtime_error("truncated conditioned matter coordinate");
  try {
    if ((bytes.front() != 'p' && bytes.front() != 'n') ||
        bytes.find_first_not_of("0123456789", 1u) != std::string::npos)
      throw std::runtime_error("bad coordinate spelling");
    CoordinateComponent value(bytes.substr(1u));
    return bytes.front() == 'n' ? -value : value;
  } catch (...) {
    throw std::runtime_error("invalid conditioned matter coordinate");
  }
}

Z3Coordinate seed_coordinate(const DevelopmentalSeedSite& site) {
  return {site.x, site.y, site.z};
}

Z3Coordinate translated_by(const Z3Coordinate& origin, Int3 delta) {
  return {origin.x + delta.x, origin.y + delta.y, origin.z + delta.z};
}

Z3Coordinate translated_by(const Z3Coordinate& origin,
                           const Z3Coordinate& delta) {
  return {origin.x + delta.x, origin.y + delta.y, origin.z + delta.z};
}

// Which 100^3 chunk the `index`-th occupied chunk sits in.
//
// WHY THIS IS NOT A FREE CHOICE. The paged executor's resident window is the UNION of
// per-chunk halo dilations (`resident_window`, bcc32_transition.cu:134) -- it does NOT form a
// global bounding box. Every occupied chunk therefore contributes its own 3x3x3 neighbourhood,
// and what a layout can do is make those neighbourhoods OVERLAP. A straight line of n chunks
// shares halo only along the line, giving a window of 9n + 18; the same n chunks in a compact
// cube give (ceil(cbrt(n)) + 2)^3. Measured on the real seed: 1,000 occupied chunks cost 9,018
// resident chunks in a line against 1,728 in a cube, a 5.22x difference in aperture for
// identical matter, and the ratio grows without bound in n. At the declared bank of 524,288
// routes it is ~168 GB against ~27 GB -- the difference between violating the disclosed
// <=100 GB envelope and sitting well inside it.
//
// So chunks fill a cube SHELL BY SHELL. Shell-filling rather than a fixed-side odometer is the
// point: the bank is sparsely bound, so every PREFIX of slots must be compact, not only a full
// bank. For chunk c, k is the smallest side with k^3 > c, and the chunk lands on one of the
// three new faces of that shell.
//
// See docs/audits/2026-07-28-route-population-capacity.md and
// docs/audits/2026-07-28-conditioned-route-chunk-layout.md.
struct ChunkTriple {
  std::uint64_t x = 0u, y = 0u, z = 0u;
};

ChunkTriple chunk_shell_coordinate(std::uint64_t chunk) {
  std::uint64_t side = 1u;
  while (side * side * side <= chunk) ++side;
  const std::uint64_t inner = side - 1u;
  std::uint64_t offset = chunk - inner * inner * inner;
  if (offset < side * side)  // the x == inner face, complete
    return {inner, offset / side, offset % side};
  offset -= side * side;
  if (offset < inner * side)  // the y == inner face, minus the edge the x face already took
    return {offset / side, inner, offset % side};
  offset -= inner * side;     // the z == inner face, minus both edges
  return {offset / inner, offset % inner, inner};
}

Z3Coordinate route_origin(std::uint32_t slot) {
  const std::uint32_t chunk = slot / kRoutesPerChunk;
  std::uint32_t local = slot % kRoutesPerChunk;
  const std::uint32_t x = local % kRoutesPerChunkX;
  local /= kRoutesPerChunkX;
  const std::uint32_t y = local % kRoutesPerChunkY;
  const std::uint32_t z = local / kRoutesPerChunkY;
  const std::int64_t local_x =
      static_cast<std::int64_t>(x) * kRouteStrideX -
      kRouteBounds.min_x;
  const std::int64_t local_y =
      static_cast<std::int64_t>(y) * kRouteStrideY -
      kRouteBounds.min_y;
  const std::int64_t local_z =
      static_cast<std::int64_t>(z) * kRouteStrideZ -
      kRouteBounds.min_z;
  const ChunkTriple placed = chunk_shell_coordinate(chunk);
  return {
      CoordinateComponent(placed.x) * kChunkEdge + local_x,
      CoordinateComponent(placed.y) * kChunkEdge + local_y,
      CoordinateComponent(placed.z) * kChunkEdge + local_z,
  };
}

const std::vector<Z3Coordinate>& route_coordinates() {
  static const std::vector<Z3Coordinate> result = [] {
    std::vector<Z3Coordinate> coordinates;
    const auto sites =
        processive_credit_return_seed(kProcessiveCreditReturnSeedHash);
    coordinates.reserve(kProcessiveCreditReturnSeedSiteCount);
    for (std::uint32_t index = 0u;
         index < kProcessiveCreditReturnSeedSiteCount; ++index) {
      coordinates.push_back(seed_coordinate(sites[index]));
    }
    std::sort(coordinates.begin(), coordinates.end(), CoordinateLess{});
    return coordinates;
  }();
  return result;
}

const std::vector<Z3Coordinate>& active_route_coordinates() {
  static const std::vector<Z3Coordinate> result = [] {
    std::vector<Z3Coordinate> coordinates = route_coordinates();
    for (std::uint32_t stage = 0u; stage < kWeightLength; ++stage) {
      const Int3 positive = processive_positive_credit_return_port(
          kProcessiveCreditReturnSeedHash, stage);
      const Int3 negative = processive_negative_credit_return_port(
          kProcessiveCreditReturnSeedHash, stage);
      coordinates.push_back({positive.x, positive.y, positive.z});
      coordinates.push_back({negative.x, negative.y, negative.z});
    }
    std::sort(coordinates.begin(), coordinates.end(), CoordinateLess{});
    coordinates.erase(
        std::unique(coordinates.begin(), coordinates.end()),
        coordinates.end());
    return coordinates;
  }();
  return result;
}

struct StoredSite {
  SiteCoord coordinate{};
  SiteWord word = kQ;
};

SiteWord read_world(const WorldStore& world, const SiteCoord& at) {
  return world.read_site(at);
}

void write_world(WorldStore* world, const SiteCoord& at, SiteWord word) {
  std::string error;
  if (!world->write_site(at, word, &error))
    throw std::runtime_error(error.empty() ? "conditioned world write failed"
                                           : error);
}

std::vector<StoredSite> world_support(const WorldStore& world) {
  std::vector<StoredSite> result;
  result.reserve(world.support().non_quiescent_sites);
  for (const auto& [chunk, words] : world.chunks()) {
    for (std::uint64_t index = 0u; index < words.size(); ++index) {
      const SiteWord word = words[index];
      if (word == kQ) continue;
      const std::uint64_t plane = kChunkEdge * kChunkEdge;
      const std::uint32_t local_x =
          static_cast<std::uint32_t>(index / plane);
      const std::uint64_t remainder = index % plane;
      const std::uint32_t local_y =
          static_cast<std::uint32_t>(remainder / kChunkEdge);
      const std::uint32_t local_z =
          static_cast<std::uint32_t>(remainder % kChunkEdge);
      result.push_back({
          {chunk.x * kChunkEdge + local_x,
           chunk.y * kChunkEdge + local_y,
           chunk.z * kChunkEdge + local_z},
          word,
      });
    }
  }
  std::sort(result.begin(), result.end(),
            [](const StoredSite& left, const StoredSite& right) {
              return CoordinateLess{}(left.coordinate, right.coordinate);
            });
  return result;
}

boost::multiprecision::cpp_int world_delta_n_q(const WorldStore& world) {
  boost::multiprecision::cpp_int result = 0;
  const std::int32_t quiescent = std::popcount(kQ);
  for (const auto& [coordinate, words] : world.chunks()) {
    (void)coordinate;
    for (const SiteWord word : words) {
      result += static_cast<std::int32_t>(std::popcount(word)) - quiescent;
    }
  }
  return result;
}

std::int32_t word_delta_n_q(SiteWord word) {
  return static_cast<std::int32_t>(std::popcount(word)) -
         static_cast<std::int32_t>(std::popcount(kQ));
}

// The seed bit `lesion_stage` used to introduce from nowhere immediately
// before its exchanges. Making it the escrow's vacuum from birth means an
// unlesioned slot already carries it, so route_conserved_delta is exact at
// every stage rather than jumping the first time a stage is lesioned.
SiteWord conditioned_lesion_escrow_vacuum() {
  // Derived once: the seed hash is fixed for the whole bank, and
  // `route_is_lesioned` runs on every admitted credit.
  static const SiteWord vacuum = [] {
    const ProcessiveWeightRegionSeedHash weight_hash =
        processive_credit_return_weight_hash(
            kProcessiveCreditReturnSeedHash);
    return kQ | energy_bit(processive_weight_marker(weight_hash));
  }();
  return vacuum;
}

// Any escrow slot away from vacuum means this route carries known physical
// damage; identical to the old scalar `lesion_escrow != kQ` check, just
// checked across every stage instead of one collapsing word.
bool route_is_lesioned(
    const std::array<SiteWord, kProcessiveCreditReturnStageCount>& escrow) {
  const SiteWord vacuum = conditioned_lesion_escrow_vacuum();
  return std::any_of(escrow.begin(), escrow.end(),
                     [vacuum](SiteWord word) { return word != vacuum; });
}

void require_canonical_boundary(
    const std::array<SiteWord, kProcessiveCreditReturnStageCount>& input,
    const std::array<SiteWord, kProcessiveCreditReturnStageCount>& output) {
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(
          kProcessiveCreditReturnSeedHash);
  const SiteWord input_word =
      kQ ^ carrier_bit(processive_weight_path(weight_hash));
  const SiteWord output_word =
      kQ ^ carrier_bit(processive_weight_marker(weight_hash));
  bool reached_unused = false;
  for (std::uint32_t stage = 0u;
       stage < kProcessiveCreditReturnStageCount; ++stage) {
    const bool used = input[stage] == kQ && output[stage] == output_word;
    const bool unused =
        input[stage] == input_word && output[stage] == kQ;
    if (!used && !unused)
      throw std::runtime_error(
          "conditioned credit boundary tape was not canonical");
    if (unused) {
      reached_unused = true;
    } else if (reached_unused) {
      throw std::runtime_error(
          "conditioned credit boundary tape was not a prefix");
    }
  }
}

boost::multiprecision::cpp_int complete_delta(
    const WorldStore& world,
    const std::array<SiteWord, kProcessiveCreditReturnStageCount>& input,
    const std::array<SiteWord, kProcessiveCreditReturnStageCount>& output) {
  boost::multiprecision::cpp_int result = world_delta_n_q(world);
  for (const SiteWord word : input) result += word_delta_n_q(word);
  for (const SiteWord word : output) result += word_delta_n_q(word);
  return result;
}

Z3Coordinate active_batch_origin(std::uint32_t index) {
  return {
      CoordinateComponent(index % 5u) * kActiveBatchPitchX,
      CoordinateComponent((index / 5u) % 5u) * kActiveBatchPitchY,
      CoordinateComponent(index / 25u) * kActiveBatchPitchZ,
  };
}

void pack_active_route(WorldStore* batch, const WorldStore& route,
                       const Z3Coordinate& origin) {
  for (const StoredSite& site : world_support(route))
    write_world(batch, translated_by(origin, site.coordinate), site.word);
}

WorldStore unpack_active_route(const WorldStore& batch,
                               const Z3Coordinate& origin) {
  WorldStore route;
  for (const Z3Coordinate& relative : active_route_coordinates()) {
    const SiteWord word = read_world(batch, translated_by(origin, relative));
    if (word != kQ) write_world(&route, relative, word);
  }
  return route;
}

std::uint64_t mix64(std::uint64_t hash, std::uint64_t value) {
  hash ^= value;
  hash *= 1099511628211ull;
  return hash;
}

std::uint64_t mix_component(std::uint64_t hash,
                            const CoordinateComponent& value) {
  const std::string bytes = canonical_coordinate_component(value);
  for (const unsigned char byte : bytes) hash = mix64(hash, byte);
  return mix64(hash, 0xffu);
}

}  // namespace

Z3Coordinate conditioned_route_origin(std::uint32_t slot) {
  return route_origin(slot);
}

std::uint32_t conditioned_routes_per_chunk() { return kRoutesPerChunk; }

ConditionedLearningMatter::ConditionedLearningMatter(
    std::size_t organ_capacity,
    std::shared_ptr<ConditionedMatterExecutor> executor)
    : executor_(std::move(executor)) {
  if (organ_capacity > kMaximumCheckpointEntries)
    throw std::runtime_error("conditioned matter capacity overflow");
  if (organ_capacity != 0u && executor_ == nullptr)
    throw std::runtime_error("conditioned matter physical executor missing");
  capacity_ = organ_capacity;
}

std::vector<ConditionedLearningMatter::Entry>::iterator
ConditionedLearningMatter::find(const ConditionedMatterKey& key) {
  return std::find_if(entries_.begin(), entries_.end(),
                      [&key](const Entry& entry) {
                        return entry.bound && entry.key == key;
                      });
}

std::vector<ConditionedLearningMatter::Entry>::const_iterator
ConditionedLearningMatter::find(const ConditionedMatterKey& key) const {
  return std::find_if(entries_.begin(), entries_.end(),
                      [&key](const Entry& entry) {
                        return entry.bound && entry.key == key;
                      });
}

bool ConditionedLearningMatter::contains(
    const ConditionedMatterKey& key) const {
  const auto found = find(key);
  return found != entries_.end() && found->key == key;
}

ConditionedLearningMatter::Entry ConditionedLearningMatter::seeded_entry(
    const ConditionedMatterKey& key, std::uint32_t slot) const {
  Entry entry;
  entry.key = key;
  entry.bound = true;
  entry.slot = slot;
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(
          kProcessiveCreditReturnSeedHash);
  entry.input_tapes.fill(
      kQ ^ carrier_bit(processive_weight_path(weight_hash)));
  entry.output_tapes.fill(kQ);
  entry.lesion_escrow.fill(conditioned_lesion_escrow_vacuum());
  return entry;
}

WorldStore ConditionedLearningMatter::seeded_route_world() const {
  WorldStore route;
  const auto sites =
      processive_credit_return_seed(kProcessiveCreditReturnSeedHash);
  for (std::uint32_t index = 0u;
       index < kProcessiveCreditReturnSeedSiteCount; ++index) {
    const Z3Coordinate at = seed_coordinate(sites[index]);
    if (read_world(route, at) != kQ)
      throw std::runtime_error("conditioned weight seed overlapped itself");
    write_world(&route, at, sites[index].word);
  }
  return route;
}

WorldStore ConditionedLearningMatter::read_route_world(
    const Entry& entry) const {
  WorldStore route;
  const Z3Coordinate origin = route_origin(entry.slot);
  for (const Z3Coordinate& relative : route_coordinates()) {
    const SiteWord word =
        read_world(world_, translated_by(origin, relative));
    if (word != kQ) write_world(&route, relative, word);
  }
  return route;
}

void ConditionedLearningMatter::write_route_world(
    const Entry& entry, const WorldStore& route) {
  const std::vector<StoredSite> support = world_support(route);
  const std::vector<Z3Coordinate>& allowed = route_coordinates();
  if (support.size() != allowed.size())
    throw std::runtime_error(
        "conditioned route escaped its packed support extent");
  for (std::size_t index = 0u; index < support.size(); ++index) {
    if (!(support[index].coordinate == allowed[index]))
      throw std::runtime_error(
          "conditioned route changed its packed support coordinates");
  }
  const Z3Coordinate origin = route_origin(entry.slot);
  for (const Z3Coordinate& relative : allowed) {
    write_world(&world_, translated_by(origin, relative),
                read_world(route, relative));
  }
}

void ConditionedLearningMatter::advance_world(
    WorldStore* world, bool inverse, std::uint32_t supersteps,
    std::uint64_t* page_count) const {
  if (executor_ == nullptr)
    throw std::runtime_error("conditioned matter physical executor missing");
  std::string error;
  if (!executor_->advance(world, inverse, supersteps, page_count, &error)) {
    throw std::runtime_error(
        error.empty() ? "conditioned paged transition failed" : error);
  }
}

ConditionedLearningMatter::ConsumeReceipt
ConditionedLearningMatter::consume_with_receipt(
    std::span<const ConditionedMatterCredit> credits) {
  // Source-event order is part of the contact.  Equal-event positive credit is
  // applied before negative credit, matching the bridge's fixed 2*event slots.
  const std::size_t required = required_new_entries(credits);
  if (required > remaining_capacity())
    throw std::runtime_error("conditioned physical organ bank exhausted");
  entries_.reserve(entries_.size() + required);
  ConsumeReceipt receipt{};
  receipt.requested = static_cast<std::uint32_t>(credits.size());

  struct PendingRoute {
    Entry entry{};
    WorldStore world{};
    bool existing = false;
  };
  std::vector<PendingRoute> pending;
  pending.reserve(credits.size());
  std::uint32_t staged_next_slot = next_slot_;

  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(kProcessiveCreditReturnSeedHash);
  const auto sites = processive_weight_region_seed(weight_hash);
  const std::uint32_t path = processive_weight_path(weight_hash);
  const std::uint32_t marker = processive_weight_marker(weight_hash);
  constexpr std::uint64_t kChunkBytes =
      static_cast<std::uint64_t>(kChunkEdge) * kChunkEdge * kChunkEdge *
      sizeof(SiteWord);
  const std::uint64_t aperture_chunks =
      std::max<std::uint64_t>(1u, executor_->aperture_bytes() / kChunkBytes);
  const std::size_t wave_limit = static_cast<std::size_t>(
      std::max<std::uint64_t>(
          1u, aperture_chunks / kActiveBatchChunksPerRoute));

  auto route_for = [&](const ConditionedMatterCredit& credit)
      -> PendingRoute* {
    auto route = std::find_if(
        pending.begin(), pending.end(), [&credit](const PendingRoute& item) {
          return item.entry.key == credit.key;
        });
    if (route != pending.end()) return &*route;
    const auto existing = find(credit.key);
    if (existing == entries_.end()) {
      const Entry entry = seeded_entry(credit.key, staged_next_slot++);
      pending.push_back({entry, seeded_route_world(), false});
    } else {
      pending.push_back({*existing, read_route_world(*existing), true});
    }
    return &pending.back();
  };

  struct StagedContact {
    PendingRoute* route = nullptr;
    bool positive = false;
    Entry entry_before{};
    WorldStore world_before{};
    boost::multiprecision::cpp_int before_contact = 0;
  };

  auto stage_contact = [&](const ConditionedMatterCredit& credit,
                           PendingRoute* route,
                           StagedContact* staged) -> bool {
    const bool positive = credit.polarity > 0;
    const std::uint32_t endpoint_stage =
        positive ? 0u : kWeightLength - 1u;
    const Z3Coordinate endpoint = seed_coordinate(
        sites[endpoint_stage * kProcessiveWeightSitesPerCell]);
    const boost::multiprecision::cpp_int before_contact = complete_delta(
        route->world, route->entry.input_tapes, route->entry.output_tapes);
    const Entry entry_before = route->entry;
    const WorldStore world_before = route->world;
    SiteWord body = read_world(route->world, endpoint);
    std::uint32_t entry_exchanges = 0u;
    if (positive) {
      const SiteWord bit = carrier_bit(path);
      for (std::uint32_t stage = 0u; stage < kWeightLength; ++stage) {
        const SiteWord body_before = body;
        const SiteWord tape_before = route->entry.input_tapes[stage];
        reciprocal_quantum_exchange(
            body, bit, route->entry.input_tapes[stage], bit);
        entry_exchanges +=
            body != body_before ||
                    route->entry.input_tapes[stage] != tape_before
                ? 1u
                : 0u;
      }
    } else {
      for (std::uint32_t reverse_stage = kWeightLength;
           reverse_stage > 0u; --reverse_stage) {
        const std::uint32_t stage = reverse_stage - 1u;
        const SiteWord body_before = body;
        const SiteWord tape_before = route->entry.output_tapes[stage];
        reciprocal_quantum_exchange(
            body, carrier_bit(path + 4u), route->entry.output_tapes[stage],
            carrier_bit(marker));
        entry_exchanges +=
            body != body_before ||
                    route->entry.output_tapes[stage] != tape_before
                ? 1u
                : 0u;
      }
    }
    if (entry_exchanges == 0u) {
      ++receipt.abstained;
      return false;
    }
    if (entry_exchanges != 1u)
      throw std::runtime_error(
          "conditioned fixed boundary admitted multiple credits");
    write_world(&route->world, endpoint, body);
    if (complete_delta(route->world, route->entry.input_tapes,
                       route->entry.output_tapes) != before_contact) {
      throw std::runtime_error(
          "conditioned credit entry violated represented matter");
    }
    *staged = {
        route, positive, entry_before, world_before, before_contact};
    return true;
  };

  auto collect_return = [&](StagedContact* staged) {
    PendingRoute* route = staged->route;
    if (complete_delta(route->world, route->entry.input_tapes,
                       route->entry.output_tapes) !=
        staged->before_contact) {
      throw std::runtime_error(
          "conditioned learning changed represented matter");
    }
    const SiteWord return_world_bit = carrier_bit(marker);
    std::uint32_t return_exchanges = 0u;
    for (std::uint32_t port_index = 0u; port_index < kWeightLength;
         ++port_index) {
      const std::uint32_t stage =
          staged->positive ? port_index
                           : kWeightLength - port_index - 1u;
      const Int3 port =
          staged->positive
              ? processive_positive_credit_return_port(
                    kProcessiveCreditReturnSeedHash, stage)
              : processive_negative_credit_return_port(
                    kProcessiveCreditReturnSeedHash, stage);
      const Z3Coordinate return_coordinate{port.x, port.y, port.z};
      SiteWord return_word = read_world(route->world, return_coordinate);
      if (return_word != (kQ ^ return_world_bit)) continue;
      const SiteWord return_before = return_word;
      if (staged->positive) {
        const SiteWord tape_before = route->entry.output_tapes[stage];
        reciprocal_quantum_exchange(
            return_word, return_world_bit, route->entry.output_tapes[stage],
            return_world_bit);
        return_exchanges +=
            return_word != return_before ||
                    route->entry.output_tapes[stage] != tape_before
                ? 1u
                : 0u;
      } else {
        const SiteWord tape_before = route->entry.input_tapes[stage];
        reciprocal_quantum_exchange(
            return_word, return_world_bit, route->entry.input_tapes[stage],
            carrier_bit(path));
        return_exchanges +=
            return_word != return_before ||
                    route->entry.input_tapes[stage] != tape_before
                ? 1u
                : 0u;
      }
      write_world(&route->world, return_coordinate, return_word);
    }
    if (return_exchanges != 1u) {
      if (route_is_lesioned(staged->entry_before.lesion_escrow)) {
        route->entry = staged->entry_before;
        route->world = staged->world_before;
        ++receipt.abstained;
        return;
      }
      throw std::runtime_error(
          "conditioned spent credit missed its fixed return port");
    }
    ++receipt.fixed_returns;
    if (complete_delta(route->world, route->entry.input_tapes,
                       route->entry.output_tapes) !=
        staged->before_contact) {
      throw std::runtime_error(
          "conditioned spent-carrier return violated represented matter");
    }
    ++receipt.conserved_contacts;
    require_canonical_boundary(route->entry.input_tapes,
                               route->entry.output_tapes);
    route->entry.completed_supersteps += kCreditSettleTicks;
    ++receipt.admitted;
  };

  std::size_t cursor = 0u;
  while (cursor < credits.size()) {
    const std::uint32_t source_event = credits[cursor].source_event;
    std::size_t event_end = cursor + 1u;
    while (event_end < credits.size() &&
           credits[event_end].source_event == source_event) {
      ++event_end;
    }
    while (cursor < event_end) {
      std::vector<StagedContact> wave;
      std::vector<ConditionedMatterKey> wave_keys;
      wave.reserve(std::min<std::size_t>(wave_limit, event_end - cursor));
      wave_keys.reserve(wave.capacity());
      while (cursor < event_end && wave.size() < wave_limit) {
        const ConditionedMatterCredit& credit = credits[cursor];
        if (std::find(wave_keys.begin(), wave_keys.end(), credit.key) !=
            wave_keys.end()) {
          break;
        }
        wave_keys.push_back(credit.key);
        PendingRoute* route = route_for(credit);
        StagedContact staged;
        ++cursor;
        if (stage_contact(credit, route, &staged))
          wave.push_back(std::move(staged));
      }
      if (wave.empty()) continue;

      WorldStore batch;
      for (std::size_t index = 0u; index < wave.size(); ++index)
        pack_active_route(
            &batch, wave[index].route->world,
            active_batch_origin(static_cast<std::uint32_t>(index)));
      const std::uint64_t expected_support =
          wave.size() * route_coordinates().size();
      if (batch.support().non_quiescent_sites != expected_support)
        throw std::runtime_error(
            "conditioned active batch routes overlapped before execution");
      std::uint64_t pages = 0u;
      advance_world(&batch, false, kCreditSettleTicks, &pages);
      if (pages == 0u)
        throw std::runtime_error(
            "conditioned paged transition executed no pages");
      std::vector<WorldStore> evolved;
      evolved.reserve(wave.size());
      std::uint64_t extracted_support = 0u;
      for (std::size_t index = 0u; index < wave.size(); ++index) {
        evolved.push_back(unpack_active_route(
            batch, active_batch_origin(static_cast<std::uint32_t>(index))));
        extracted_support += evolved.back().support().non_quiescent_sites;
      }
      if (batch.support().non_quiescent_sites != extracted_support)
        throw std::runtime_error(
            "conditioned active batch produced unassigned support");
      ++receipt.executor_waves;
      receipt.peak_wave_routes = std::max(
          receipt.peak_wave_routes,
          static_cast<std::uint32_t>(wave.size()));
      for (std::size_t index = 0u; index < wave.size(); ++index) {
        wave[index].route->world = std::move(evolved[index]);
        collect_return(&wave[index]);
      }
    }
  }

  for (PendingRoute& route : pending) {
    write_route_world(route.entry, route.world);
    if (route.existing) {
      *find(route.entry.key) = route.entry;
    } else {
      entries_.insert(
          std::lower_bound(
              entries_.begin(), entries_.end(), route.entry.key,
              [](const Entry& candidate,
                 const ConditionedMatterKey& wanted) {
                return candidate.key < wanted;
              }),
          route.entry);
      ++bound_count_;
    }
  }
  next_slot_ = staged_next_slot;
  return receipt;
}

std::size_t ConditionedLearningMatter::required_new_entries(
    std::span<const ConditionedMatterCredit> credits) const {
  for (std::size_t index = 1u; index < credits.size(); ++index) {
    if (credits[index].source_event < credits[index - 1u].source_event)
      throw std::runtime_error("conditioned credit order is not deterministic");
  }
  std::vector<ConditionedMatterKey> unbound;
  unbound.reserve(credits.size());
  for (const ConditionedMatterCredit& credit : credits) {
    if (credit.polarity != 1 && credit.polarity != -1)
      throw std::runtime_error("conditioned credit polarity must be signed");
    if (contains(credit.key)) continue;
    const auto insertion =
        std::lower_bound(unbound.begin(), unbound.end(), credit.key);
    if (insertion == unbound.end() || !(*insertion == credit.key))
      unbound.insert(insertion, credit.key);
  }
  return unbound.size();
}

void ConditionedLearningMatter::consume(
    const ConditionedMatterCredit& credit) {
  const std::array<ConditionedMatterCredit, 1u> contact{{credit}};
  consume(contact);
}

ConditionedLearningMatter::ConsumeReceipt
ConditionedLearningMatter::consume_event_batch(
    std::span<const ConditionedMatterDeviceCredit> events) {
  std::vector<ConditionedMatterCredit> credits;
  credits.reserve(events.size());
  for (const ConditionedMatterDeviceCredit& event : events) {
    if (event.valid == 0u) continue;
    credits.push_back({{event.anchor, event.previous, event.next},
                       event.polarity, event.source_event});
  }
  return consume_with_receipt(credits);
}

ConditionedLearningMatter::ConsumeReceipt
ConditionedLearningMatter::consume_device_batch(
    const ConditionedMatterDeviceCredit* device_credits,
    std::uint32_t count) {
  if (count != 0u && device_credits == nullptr)
    throw std::runtime_error("conditioned device credit batch is null");
  std::vector<ConditionedMatterDeviceCredit> events(count);
  if (count != 0u) {
    const cudaError_t status = cudaMemcpy(
        events.data(), device_credits,
        static_cast<std::size_t>(count) * sizeof(events.front()),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
      throw std::runtime_error(
          std::string("copy conditioned device credit batch: ") +
          cudaGetErrorString(status));
  }
  return consume_event_batch(events);
}

std::uint32_t ConditionedLearningMatter::conductance(
    const ConditionedMatterKey& key) const {
  const auto found = find(key);
  if (found == entries_.end()) return 0u;

  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(
          kProcessiveCreditReturnSeedHash);
  const auto sites = processive_weight_region_seed(weight_hash);
  const std::uint32_t marker = processive_weight_marker(weight_hash);
  std::uint32_t result = 0u;
  for (std::uint32_t stage = 0u; stage < kWeightLength; ++stage) {
    WorldStore probe = read_route_world(*found);
    const Z3Coordinate target =
        seed_coordinate(sites[stage * kProcessiveWeightSitesPerCell]);
    std::array<SiteWord, 3u> tapes{};
    SiteWord target_word = read_world(probe, target);
    std::uint32_t tape_index = 0u;
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
      if (basis == marker) continue;
      const SiteWord bit = carrier_bit(basis + 4u);
      tapes[tape_index] = kQ ^ bit;
      reciprocal_quantum_exchange(target_word, bit, tapes[tape_index], bit);
      ++tape_index;
    }
    write_world(&probe, target, target_word);
    const WorldStore before = probe;
    std::uint64_t forward_pages = 0u;
    advance_world(&probe, false, 1u, &forward_pages);
    const Z3Coordinate positive_output = translated_by(
        target, direction_offset(static_cast<Direction>(kProbeBasis)));
    result += !bit_is_set(read_world(probe, positive_output),
                          carrier_bit(kProbeBasis));
    std::uint64_t inverse_pages = 0u;
    advance_world(&probe, true, 1u, &inverse_pages);
    if (forward_pages == 0u || inverse_pages == 0u ||
        !probe.equals(before)) {
      throw std::runtime_error("conditioned conductance query did not uncompute");
    }
  }
  return result;
}

std::vector<ConditionedMatterKey>
ConditionedLearningMatter::inventory_keys() const {
  std::vector<ConditionedMatterKey> result;
  result.reserve(bound_count_);
  for (const Entry& entry : entries_) {
    if (entry.bound) result.push_back(entry.key);
  }
  return result;
}

std::vector<std::uint32_t> ConditionedLearningMatter::conductances(
    std::span<const ConditionedMatterKey> keys) const {
  std::vector<std::uint32_t> result;
  result.reserve(keys.size());
  for (const ConditionedMatterKey& key : keys)
    result.push_back(conductance(key));
  return result;
}

void ConditionedLearningMatter::publish_conductance_device(
    const ConditionedMatterDeviceKey* device_keys, std::uint32_t count,
    std::uint32_t* device_conductance) const {
  if (count != 0u && (device_keys == nullptr || device_conductance == nullptr))
    throw std::runtime_error("conditioned device conductance batch is null");
  std::vector<ConditionedMatterDeviceKey> keys(count);
  if (count != 0u) {
    const cudaError_t read_status = cudaMemcpy(
        keys.data(), device_keys,
        static_cast<std::size_t>(count) * sizeof(keys.front()),
        cudaMemcpyDeviceToHost);
    if (read_status != cudaSuccess)
      throw std::runtime_error(
          std::string("read conditioned device inventory: ") +
          cudaGetErrorString(read_status));
  }
  std::vector<std::uint32_t> values;
  values.reserve(count);
  for (const ConditionedMatterDeviceKey& key : keys)
    values.push_back(conductance({key.anchor, key.previous, key.next}));
  if (count != 0u) {
    const cudaError_t write_status = cudaMemcpy(
        device_conductance, values.data(),
        static_cast<std::size_t>(count) * sizeof(values.front()),
        cudaMemcpyHostToDevice);
    if (write_status != cudaSuccess)
      throw std::runtime_error(
          std::string("publish conditioned device conductance: ") +
          cudaGetErrorString(write_status));
  }
}

void ConditionedLearningMatter::lesion_stage(
    const ConditionedMatterKey& key, std::uint32_t stage) {
  if (stage >= kWeightLength)
    throw std::runtime_error("conditioned lesion stage overflow");
  auto found = find(key);
  if (found == entries_.end())
    throw std::runtime_error("conditioned lesion key missing");
  if (found->lesion_escrow[stage] != conditioned_lesion_escrow_vacuum())
    throw std::runtime_error("conditioned stage already lesioned");
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(
          kProcessiveCreditReturnSeedHash);
  const auto sites = processive_weight_region_seed(weight_hash);
  const std::uint32_t marker = processive_weight_marker(weight_hash);
  const Z3Coordinate at =
      seed_coordinate(sites[stage * kProcessiveWeightSitesPerCell]);
  WorldStore route = read_route_world(*found);
  SiteWord body = read_world(route, at);
  reciprocal_quantum_exchange(body, energy_bit(marker),
                              found->lesion_escrow[stage], energy_bit(marker));
  reciprocal_quantum_exchange(body, owned_bond_bit(marker),
                              found->lesion_escrow[stage],
                              owned_bond_bit(marker));
  write_world(&route, at, body);
  write_route_world(*found, route);
}

void ConditionedLearningMatter::restore_lesioned_stage(
    const ConditionedMatterKey& key, std::uint32_t stage) {
  if (stage >= kWeightLength)
    throw std::runtime_error("conditioned restore stage out of range");
  auto found = find(key);
  if (found == entries_.end())
    throw std::runtime_error("conditioned restore key missing");
  if (found->lesion_escrow[stage] ==
      conditioned_lesion_escrow_vacuum()) {
    throw std::runtime_error(
        "conditioned restore stage was not lesioned");
  }
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(
          kProcessiveCreditReturnSeedHash);
  const auto sites = processive_weight_region_seed(weight_hash);
  const std::uint32_t marker = processive_weight_marker(weight_hash);
  const Z3Coordinate at =
      seed_coordinate(sites[stage * kProcessiveWeightSitesPerCell]);
  WorldStore route = read_route_world(*found);
  SiteWord body = read_world(route, at);
  reciprocal_quantum_exchange(
      body, owned_bond_bit(marker), found->lesion_escrow[stage],
      owned_bond_bit(marker));
  reciprocal_quantum_exchange(
      body, energy_bit(marker), found->lesion_escrow[stage],
      energy_bit(marker));
  if (found->lesion_escrow[stage] !=
      conditioned_lesion_escrow_vacuum()) {
    throw std::runtime_error(
        "conditioned restore did not empty stage escrow");
  }
  write_world(&route, at, body);
  write_route_world(*found, route);
}

std::array<SiteWord, kProcessiveCreditReturnStageCount>
ConditionedLearningMatter::lesion_escrow(const ConditionedMatterKey& key) const {
  const auto found = find(key);
  if (found == entries_.end())
    throw std::runtime_error("conditioned lesion key missing");
  return found->lesion_escrow;
}

DeltaNQ ConditionedLearningMatter::route_conserved_delta(
    const ConditionedMatterKey& key) const {
  const auto found = find(key);
  if (found == entries_.end())
    throw std::runtime_error("conditioned lesion key missing");
  DeltaNQ result = world_delta_n_q(read_route_world(*found));
  for (const SiteWord word : found->lesion_escrow) result += word_delta_n_q(word);
  return result;
}

void ConditionedLearningMatter::save(std::ostream& output) const {
  write_plain(output, kCheckpointMagic);
  write_plain(output, kCheckpointVersion);
  write_plain(output, kProcessiveCreditReturnSeedHash);
  if (capacity_ > kMaximumCheckpointEntries ||
      entries_.size() > capacity_ || bound_count_ != entries_.size())
    throw std::runtime_error("conditioned matter bank extent overflow");
  write_plain(output, static_cast<std::uint32_t>(capacity_));
  write_plain(output, static_cast<std::uint32_t>(bound_count_));
  for (const Entry& entry : entries_) {
    write_plain(output, static_cast<std::uint8_t>(entry.bound ? 1u : 0u));
    write_plain(output, entry.key);
    write_plain(output, entry.input_tapes);
    write_plain(output, entry.output_tapes);
    write_plain(output, entry.lesion_escrow);
    write_plain(output, entry.completed_supersteps);
    const std::vector<StoredSite> support =
        world_support(read_route_world(entry));
    write_plain(output, static_cast<std::uint64_t>(support.size()));
    for (const StoredSite& site : support) {
      write_component(output, site.coordinate.x);
      write_component(output, site.coordinate.y);
      write_component(output, site.coordinate.z);
      write_plain(output, site.word);
    }
  }
}

ConditionedLearningMatter ConditionedLearningMatter::load(
    std::istream& input,
    std::shared_ptr<ConditionedMatterExecutor> executor) {
  if (read_plain<std::uint64_t>(input) != kCheckpointMagic ||
      read_plain<std::uint32_t>(input) != kCheckpointVersion ||
      read_plain<ProcessiveCreditReturnSeedHash>(input) !=
          kProcessiveCreditReturnSeedHash) {
    throw std::runtime_error("incompatible conditioned matter checkpoint");
  }
  const std::uint32_t capacity = read_plain<std::uint32_t>(input);
  const std::uint32_t bound_count = read_plain<std::uint32_t>(input);
  if (capacity > kMaximumCheckpointEntries || bound_count > capacity)
    throw std::runtime_error("oversized conditioned matter checkpoint");
  ConditionedLearningMatter result(capacity, std::move(executor));
  result.entries_.reserve(bound_count);
  for (std::uint32_t index = 0u; index < bound_count; ++index) {
    Entry entry;
    const std::uint8_t bound = read_plain<std::uint8_t>(input);
    if (bound > 1u)
      throw std::runtime_error("invalid conditioned matter binding flag");
    entry.bound = bound != 0u;
    entry.key = read_plain<ConditionedMatterKey>(input);
    entry.input_tapes =
        read_plain<std::array<SiteWord,
                              kProcessiveCreditReturnStageCount>>(input);
    entry.output_tapes =
        read_plain<std::array<SiteWord,
                              kProcessiveCreditReturnStageCount>>(input);
    require_canonical_boundary(entry.input_tapes, entry.output_tapes);
    entry.lesion_escrow =
        read_plain<std::array<SiteWord,
                              kProcessiveCreditReturnStageCount>>(input);
    entry.completed_supersteps = read_plain<std::uint64_t>(input);
    entry.slot = result.next_slot_++;
    WorldStore route;
    const std::uint64_t site_count = read_plain<std::uint64_t>(input);
    if (site_count > kMaximumSitesPerEntry)
      throw std::runtime_error("oversized conditioned matter entry");
    for (std::uint64_t site = 0u; site < site_count; ++site) {
      const Z3Coordinate at{read_component(input), read_component(input),
                            read_component(input)};
      const SiteWord word = read_plain<SiteWord>(input);
      if (word == kQ)
        throw std::runtime_error("quiescent conditioned checkpoint site");
      write_world(&route, at, word);
    }
    result.write_route_world(entry, route);
    result.entries_.push_back(std::move(entry));
  }
  result.bound_count_ = bound_count;
  const std::size_t observed_bound = static_cast<std::size_t>(std::count_if(
      result.entries_.begin(), result.entries_.end(),
      [](const Entry& entry) { return entry.bound; }));
  if (observed_bound != result.bound_count_)
    throw std::runtime_error("conditioned matter bound-count mismatch");
  for (std::size_t left = 0u; left < result.entries_.size(); ++left) {
    if (!result.entries_[left].bound) continue;
    for (std::size_t right = left + 1u; right < result.entries_.size(); ++right) {
      if (result.entries_[right].bound &&
          result.entries_[left].key == result.entries_[right].key)
        throw std::runtime_error("duplicate conditioned matter binding");
    }
  }
  return result;
}

std::uint64_t ConditionedLearningMatter::physical_hash() const {
  std::uint64_t hash = 1469598103934665603ull;
  hash = mix64(hash, kProcessiveCreditReturnSeedHash);
  hash = mix64(hash, capacity_);
  for (const Entry& entry : entries_) {
    hash = mix64(hash, entry.bound ? 1u : 0u);
    hash = mix64(hash, entry.key.anchor);
    hash = mix64(hash, entry.key.previous);
    hash = mix64(hash, entry.key.next);
    for (const SiteWord word : entry.input_tapes)
      hash = mix64(hash, word);
    for (const SiteWord word : entry.output_tapes)
      hash = mix64(hash, word);
    for (const SiteWord word : entry.lesion_escrow)
      hash = mix64(hash, word);
    for (const StoredSite& site :
         world_support(read_route_world(entry))) {
      hash = mix_component(hash, site.coordinate.x);
      hash = mix_component(hash, site.coordinate.y);
      hash = mix_component(hash, site.coordinate.z);
      hash = mix64(hash, site.word);
    }
  }
  return hash;
}

}  // namespace substrate::bcc32

#include "bcc32_viability.hpp"

#include "bcc32_law_identity.hpp"
#include "bcc32_law.cuh"
#include "bcc32_types.cuh"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>

namespace substrate::bcc32::viability {

std::vector<Port> transcribe_bond_restore_ports(const std::vector<ReferenceSite>& seed,
                                                std::uint32_t horizon) {
  std::vector<Port> ports;
  for (const ReferenceSite& site : seed) {
    if ((site.word & kOwnedBondMask) != 0u) {
      ports.push_back(Port{site.coordinate, kOwnedBondMask, horizon});
    }
  }
  return ports;
}

namespace {

// Steps `arrangement` through F, collecting the masked reading at every port at
// every tick from 1 up to that port's declared tick.
BaselineTrace trace(const std::vector<ReferenceSite>& arrangement,
                    const std::vector<Port>& ports) {
  std::uint32_t horizon = 0u;
  for (const Port& port : ports) {
    horizon = std::max(horizon, port.tick);
  }

  BaselineTrace readings(ports.size());
  for (std::size_t index = 0; index < ports.size(); ++index) {
    readings[index].reserve(ports[index].tick);
  }

  ReferenceLattice lattice(arrangement);
  const DeltaNQ conserved = delta_n_q(lattice);
  for (std::uint32_t tick = 1u; tick <= horizon; ++tick) {
    apply_superstep(lattice);
    if (delta_n_q(lattice) != conserved) {
      throw std::runtime_error("BCC32 viability trajectory violated DeltaN_Q");
    }
    for (std::size_t index = 0; index < ports.size(); ++index) {
      if (tick <= ports[index].tick) {
        readings[index].push_back(lattice.read(ports[index].coordinate) &
                                  ports[index].observed_mask);
      }
    }
  }
  return readings;
}

}  // namespace

BaselineTrace baseline_trace(const std::vector<ReferenceSite>& arrangement,
                             const std::vector<Port>& ports) {
  return trace(arrangement, ports);
}

bool viable(const std::vector<ReferenceSite>& perturbed, const std::vector<Port>& ports,
            const BaselineTrace& baseline) {
  return trace(perturbed, ports) == baseline;
}

NullDraw word_shuffle_null(const std::vector<ReferenceSite>& seed, std::uint64_t rng_seed) {
  NullDraw draw;

  // A support with fewer than two distinct words cannot be non-trivially shuffled.
  bool distinct = false;
  for (const ReferenceSite& site : seed) {
    if (site.word != seed.front().word) {
      distinct = true;
    }
  }
  if (!distinct) {
    draw.degenerate = true;
    draw.arrangement = seed;  // degenerate; the caller MUST check draw.degenerate
    return draw;
  }

  std::mt19937_64 rng(rng_seed);
  std::vector<SiteWord> words;
  words.reserve(seed.size());
  for (const ReferenceSite& site : seed) {
    words.push_back(site.word);
  }

  std::vector<ReferenceSite> shuffled = seed;
  for (;;) {
    std::shuffle(words.begin(), words.end(), rng);
    for (std::size_t index = 0; index < seed.size(); ++index) {
      shuffled[index].word = words[index];
    }
    if (shuffled != seed) {
      draw.arrangement = std::move(shuffled);
      return draw;
    }
    ++draw.redraws;
  }
}

DensityCount enumerate_cpu(const std::vector<ReferenceSite>& arrangement,
                           const std::vector<Port>& ports, Alphabet alphabet,
                           const std::vector<SiteWord>& catalog) {
  if (alphabet == Alphabet::full) {
    // 2^32 per site is a GPU job. Refuse rather than silently sample: a quietly
    // sampled number reported as exhaustive is the failure this whole design exists
    // to prevent.
    std::abort();
  }

  const BaselineTrace baseline = baseline_trace(arrangement, ports);
  DensityCount count;
  count.baseline_delta_n_q = delta_n_q(ReferenceLattice(arrangement));
  count.delta_n_q_trajectories_checked = 1u;

  for (std::size_t index = 0; index < arrangement.size(); ++index) {
    const SiteWord original = arrangement[index].word;

    std::vector<SiteWord> candidates;
    if (alphabet == Alphabet::hamming1) {
      candidates.reserve(32u);
      for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
        candidates.push_back(original ^ (1u << bit));
      }
    } else {
      candidates = catalog;
    }

    std::vector<ReferenceSite> perturbed = arrangement;
    for (const SiteWord candidate : candidates) {
      if (candidate == original) {
        continue;  // identity is not a mutation
      }
      perturbed[index].word = candidate;
      ++count.total;
      ++count.delta_n_q_trajectories_checked;
      if (viable(perturbed, ports, baseline)) {
        ++count.viable;
      }
    }
  }

  return count;
}

std::optional<ContentAddress> sealed_capsule_identity(
    const std::vector<ReferenceSite>& support) {
  Genesis genesis{};
  genesis.metadata.genesis_class = GenesisClass::G1;
  genesis.metadata.artifact_type = ArtifactType::authored_material;
  genesis.metadata.contact_boundary_mask = 0xffu;
  genesis.sites.reserve(support.size());
  for (const ReferenceSite& site : support) {
    const auto in_i32 = [](const CoordinateComponent& value) {
      return value >= std::numeric_limits<std::int32_t>::min() &&
             value <= std::numeric_limits<std::int32_t>::max();
    };
    if (!in_i32(site.coordinate.x) || !in_i32(site.coordinate.y) ||
        !in_i32(site.coordinate.z)) {
      return std::nullopt;
    }
    genesis.sites.push_back(
        {{site.coordinate.x.convert_to<std::int32_t>(),
          site.coordinate.y.convert_to<std::int32_t>(),
          site.coordinate.z.convert_to<std::int32_t>()},
         site.word});
  }
  const std::optional<EncodedGenesis> capsule =
      seal_genesis(std::move(genesis), canonical_law_identity());
  if (!capsule.has_value()) {
    return std::nullopt;
  }
  return genesis_artifact_identity(*capsule);
}

}  // namespace substrate::bcc32::viability
